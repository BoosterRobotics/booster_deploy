from __future__ import annotations

import sys
from time import sleep
import select
import numpy as np
import torch
import mujoco
import mujoco.viewer
from booster_assets import BOOSTER_ASSETS_DIR
from .base_controller import BaseController, ControllerCfg, VelocityCommand


class MujocoController(BaseController):
    def __init__(self, cfg: ControllerCfg):
        super().__init__(cfg)

        mjcf_path = self._expand_assets_placeholder(self.robot.cfg.mjcf_path)
        self.mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.mj_model.opt.timestep = self.cfg.mujoco.physics_dt
        self.decimation = self.cfg.mujoco.decimation
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_resetData(self.mj_model, self.mj_data)

        self.mj_data.qpos = np.concatenate(
            [
                np.array(self.cfg.mujoco.init_pos, dtype=np.float32),
                np.array(self.cfg.mujoco.init_quat, dtype=np.float32),
                self.robot.default_joint_pos.numpy(),
            ]
        )
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # render a second "ghost" robot (kinematic only) without
        # modifying the MuJoCo XML. This uses a second MjData to compute FK from
        # generalized coordinates and draws a duplicated set of geoms via
        # viewer.user_scn.
        self._ghost_mj_data = mujoco.MjData(self.mj_model)
        # Keep ghost initialized to the current simulated pose so it is valid
        # even before any policy calls set_reference_qpos().
        self._ghost_mj_data.qpos[:] = self.mj_data.qpos
        self._ghost_mj_data.qvel[:] = 0.0
        mujoco.mj_forward(self.mj_model, self._ghost_mj_data)
        self._ghost_rgba = np.array(
            self.cfg.mujoco.ghost_rgba, dtype=np.float32)
        self._ghost_scene_option = mujoco.MjvOption()

        # Reference qpos can be set explicitly by the policy.
        self._reference_qpos: np.ndarray | None = None

    def start(self):
        # Clear reference; policy.reset() may set a fresh one.
        self._reference_qpos = None
        return super().start()

    def render_reference_robot(
        self,
        viewer,
        # mj_data: mujoco.MjData,
        *,
        rgba: np.ndarray | None = None,
    ) -> None:
        """Render a kinematic robot pose into viewer.user_scn using mj_data."""
        mujoco.mjv_updateScene(
            self.mj_model,
            self._ghost_mj_data,
            self._ghost_scene_option,
            None,
            viewer.cam,
            int(mujoco.mjtCatBit.mjCAT_DYNAMIC),
            viewer.user_scn,
        )
        if rgba is None:
            rgba = self._ghost_rgba

        for i in range(viewer.user_scn.ngeom):
            viewer.user_scn.geoms[i].rgba[:] = rgba

    def set_reference_qpos(
        self,
        qpos: np.ndarray | torch.Tensor | None,
    ) -> None:
        """Set the reference generalized coordinates (qpos) for ghost rendering.

        Policies should call this each step (or whenever updated). Pass None to
        clear the reference.
        """
        if qpos is None:
            self._reference_qpos = None
            return

        if isinstance(qpos, torch.Tensor):
            qpos_np = qpos.detach().cpu().numpy()
        else:
            qpos_np = np.asarray(qpos)

        qpos_np = qpos_np.astype(np.float32, copy=False).reshape(-1)
        if qpos_np.shape[0] != int(self.mj_model.nq):
            raise ValueError(
                f"reference qpos must have shape (nq,), got {qpos_np.shape} (nq={int(self.mj_model.nq)})"
            )
        self._reference_qpos = qpos_np.copy()
        # FK + offset
        self._ghost_mj_data.qpos[:] = self._reference_qpos
        self._ghost_mj_data.qvel[:] = 0.0
        mujoco.mj_forward(self.mj_model, self._ghost_mj_data)

    def _expand_assets_placeholder(self, path: str) -> str:
        """Replace {BOOSTER_ASSETS_DIR} placeholder in a path string.
        """
        try:
            return path.replace("{BOOSTER_ASSETS_DIR}", str(BOOSTER_ASSETS_DIR))
        except Exception:
            return path

    def update_vel_command(self):
        cmd: VelocityCommand = self.vel_command
        if select.select([sys.stdin], [], [], 0)[0]:
            try:
                parts = sys.stdin.readline().strip().split()
                if len(parts) == 3:
                    (cmd.lin_vel_x, cmd.lin_vel_y, cmd.ang_vel_yaw) = map(float, parts)
                    print(
                        f"Updated command to: x={cmd.lin_vel_x},"
                        f"y={cmd.lin_vel_y}, yaw={cmd.ang_vel_yaw}\n"
                        "Set command (x, y, yaw): ",
                        end="",
                    )
                else:
                    raise ValueError
            except ValueError:
                print(
                    "Invalid input. Enter three numeric values. "
                    "Set command (x, y, yaw): ",
                    end="",
                )

    def update_state(self) -> None:
        dof_pos = self.mj_data.qpos.astype(np.float32)[7:]
        dof_vel = self.mj_data.qvel.astype(np.float32)[6:]
        dof_torque = self.mj_data.qfrc_actuator[6:].astype(np.float32)

        base_pos_w = self.mj_data.qpos.astype(np.float32)[:3]
        base_quat = self.mj_data.qpos.astype(np.float32)[3:7]
        base_lin_vel_b = self.mj_data.qvel.astype(np.float32)[:3]
        base_ang_vel_b = self.mj_data.qvel.astype(np.float32)[3:6]

        self.robot.data.joint_pos = torch.from_numpy(
            dof_pos).to(self.robot.data.device)
        self.robot.data.joint_vel = torch.from_numpy(
            dof_vel).to(self.robot.data.device)
        self.robot.data.feedback_torque = torch.from_numpy(
            dof_torque).to(self.robot.data.device)
        self.robot.data.root_pos_w = torch.from_numpy(
            base_pos_w).to(self.robot.data.device)
        self.robot.data.root_quat_w = torch.from_numpy(
            base_quat).to(self.robot.data.device)
        self.robot.data.root_lin_vel_b = torch.from_numpy(
            base_lin_vel_b).to(self.robot.data.device)
        self.robot.data.root_ang_vel_b = torch.from_numpy(
            base_ang_vel_b).to(self.robot.data.device)

    def log_states(self, dof_targets: np.ndarray) -> None:
        if self.cfg.mujoco.log_states is not None:
            if not hasattr(self, '_states'):
                self._states = {
                    'root_pos_w': [],
                    'root_quat_w': [],
                    'root_lin_vel_b': [],
                    'root_ang_vel_b': [],
                    'joint_pos': [],
                    'joint_vel': [],
                    'joint_torque': [],
                    'dof_targets': [],
                }
            base_pos_w = self.mj_data.qpos.astype(np.float32)[:3]
            base_quat = self.mj_data.qpos.astype(np.float32)[3:7]
            base_lin_vel_b = self.mj_data.qvel.astype(np.float32)[:3]
            base_ang_vel_b = self.mj_data.qvel.astype(np.float32)[3:6]
            dof_pos = self.mj_data.qpos.astype(np.float32)[7:]
            dof_vel = self.mj_data.qvel.astype(np.float32)[6:]
            dof_torque = self.mj_data.qfrc_actuator[6:].astype(np.float32)

            self._states['root_pos_w'].append(base_pos_w)
            self._states['root_quat_w'].append(base_quat)
            self._states['root_lin_vel_b'].append(base_lin_vel_b)
            self._states['root_ang_vel_b'].append(base_ang_vel_b)
            self._states['joint_pos'].append(dof_pos)
            self._states['joint_vel'].append(dof_vel)
            self._states['joint_torque'].append(dof_torque)
            self._states['dof_targets'].append(dof_targets)
            if len(self._states['root_pos_w']) % 100 == 0:
                _states = {k: np.stack(v) for k, v in self._states.items()}
                np.savez(f'{self.cfg.mujoco.log_states}.npz', **_states)
                print(f'saved {self.cfg.mujoco.log_states}.npz '
                      f'at {self._step_count} steps')

    def ctrl_step(self, dof_targets: torch.Tensor):
        dof_targets = dof_targets.cpu().numpy()  # type: ignore
        self.log_states(dof_targets)
        if self.vel_command is not None:
            self.update_vel_command()

        dof_pos = self.mj_data.qpos.astype(np.float32)[7:]
        dof_vel = self.mj_data.qvel.astype(np.float32)[6:]
        kp = self.robot.joint_stiffness.numpy()
        kd = self.robot.joint_damping.numpy()
        # ctrl_limit = [
        #     np.minimum(self.mj_model.actuator_forcerange[:, 0],
        #                self.mj_model.actuator_ctrlrange[:, 0]),
        #     np.maximum(self.mj_model.actuator_forcerange[:, 1],
        #                self.mj_model.actuator_ctrlrange[:, 1]),
        # ]
        ctrl_limit = self.robot.effort_limit.numpy()
        for i in range(self.decimation):
            self.mj_data.ctrl = np.clip(
                kp * (dof_targets - dof_pos) - kd * dof_vel,
                -ctrl_limit,
                ctrl_limit,
            )
            mujoco.mj_step(self.mj_model, self.mj_data)
            dof_pos = self.mj_data.qpos.astype(np.float32)[7:]
            dof_vel = self.mj_data.qvel.astype(np.float32)[6:]

    def run(self):
        with mujoco.viewer.launch_passive(
                self.mj_model, self.mj_data) as viewer:

            self.viewer = viewer
            viewer.cam.elevation = -20
            if self.vel_command is not None:
                print("\nSet command (x, y, yaw): ", end="")
            self.update_state()
            self.start()
            while viewer.is_running() and self.is_running:
                sleep(self.cfg.mujoco.physics_dt * self.cfg.mujoco.decimation)
                self.update_state()
                dof_targets = self.policy_step()
                self.ctrl_step(dof_targets)

                if self.cfg.mujoco.visualize_reference_ghost:
                    # Render kinematic "ghost" robot from generalized coordinates.
                    self.render_reference_robot(
                        viewer,
                        rgba=self._ghost_rgba,
                    )

                self.viewer.cam.lookat[:] = self.mj_data.qpos.astype(np.float32)[0:3]
                self.viewer.sync()
