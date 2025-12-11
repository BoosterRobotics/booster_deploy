# Booster Deploy

Booster Deploy is a lightweight deployment framework that supports running control policies on Booster robots (sim2real), MuJoCo (sim2sim), and Webots (internal sim2sim). The system adopts many well-established designs from IsaacLab to provide modular abstractions, allowing unified policy execution across both simulated and real robotic platforms.


## Prerequisites

| Environment | Notes |
|-------------|-------|
| Booster firmware >= v1.4 | Required for real robot deployments. |
| Python 3.10+ | Already installed on the robot |
| ROS 2 Humble | Required for `/low_state` + `/joint_ctrl` topics. Already installed on the robot. |
| MuJoCo / Webots | Optional; install if you plan to run the respective simulators. |


## Running Deployments

### Add and list tasks:
   1. Create a subfolder under `tasks/` for your task.
   2. Implement a `Policy`/`PolicyCfg` and provide a `ControllerCfg` referencing the policy.
   3. Place policy checkpoints under `models/` and reference the path in the config.
   4. Register your `ControllerCfg` config in the task registry (see existing tasks for the registration pattern).
   5. Check all available tasks:
      ```bash
      python3 scripts/deploy.py --list
      ```

### Run Sim2Sim (MuJoCo)

- Download and install BoosterAssets:
   - Clone the [booster_assets](https://github.com/BoosterRobotics/booster_assets) which contains Booster robot models and resources.
   - Install booster_assets python helper following the instructions in the repository.

- Install Python dependencies on local machine:
   ```
   pip install -r requirements.txt
   ```

- Launch the task in mujoco:
   ```bash
   python scripts/deploy.py --task <TASK_NAME> --mujoco
   ```

### Run Sim2Real (Real Robots)

**IMPORTANT**: Make sure to install [Booster Firmware](https://booster.feishu.cn/wiki/E3q5wF5SnitXZgkY18Uc8odBnXb) >= v1.4 on the robot before proceeding.

- After you finish testing your task with Sim2Sim locally, copy the project to the robot.

- Install Booster Robotic SDK on robot:
   - Clone the latest [Booster Robotics SDK](https://github.com/BoosterRobotics/booster_robotics_sdk) repository into the robot.
   - Follow the build instructions in the SDK repository.
   - **Important**: Make sure to build and install the Python bindings:
     ```bash
     cd booster_robotics_sdk
     mkdir build && cd build
     cmake .. -DBUILD_PYTHON_BINDING=ON
     make -j$(nproc)
     sudo make install
     ```

- Install Python dependencies on the robot:
   ```
   pip install -r requirements.txt
   ```

- SSH into the robot and start the ROS 2 environment by sourcing the provided setup script:
   ```bash
   source /opt/booster/BoosterRos2Interface/install/setup.bash
   ```

- Launch the task on the robot and follow the prompts shown in the command line..
   ```bash
   python3 scripts/deploy.py --task <TASK_NAME>
   ```


## Repository Layout

```
booster_deploy/
├─ booster_deploy/           # Controllers, policies, utilities
├─ scripts/                  # Entry-point scripts (deploy.py)
├─ tasks/                    # Task registry and configs
├─ requirements.txt          # Python dependencies
└─ fastdds_profile.xml       # Default FastDDS settings for ROS 2
```

Key modules:
- `booster_deploy/`: Core module providing a unified abstraction for both simulators and physical robots, and handling communication via ROS 2 (implements a /low_state subscriber and a /low_cmd publisher to bridge policies to hardware).
- `booster_deploy/robots/`: Robot configuration modules. This folder contains booster robot configs by defining a `RobotCfg` describing:
    - joint names and body names
    - default joint positions
    - default joint stiffness (`joint_stiffness`) and damping (`joint_damping`)
    - effort limits
    - `mjcf_path` for MuJoCo model loading
    - `prepare_state` (prepare pose, stiffness and damping used when entering custom mode)

 - `tasks/`: User task definitions and implementations. Each task module contains:
    - `Policy`/`PolicyCfg` class implementing the inference logic;
    - a `ControllerCfg` class describing the task configuration including the policy;
    - registering a task with a `ControllerCfg` instance.

   Typical task layout (example):

   ```text
   tasks/my_task/
   ├─ __init__.py        # registers the task via utils.register.register_task
   ├─ task.py            # Policy and ControllerCfg implementation
   ├─ models/            # optional policy checkpoints
   └─ motions/           # optional motion primitives or recordings
   ```
