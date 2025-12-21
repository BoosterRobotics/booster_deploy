from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.registry import register_task
from .locomotion import (
    K1WalkControllerCfg,
    T1WalkControllerCfg
)

# Register locomotion tasks


@configclass
class T1WalkControllerCfg1(T1WalkControllerCfg):
    '''Human-like walk for T1 robot.'''
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "models/t1_walk.pt"


register_task(
    "t1_walk", T1WalkControllerCfg1())
