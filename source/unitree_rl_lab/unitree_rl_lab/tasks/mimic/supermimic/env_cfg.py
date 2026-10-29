from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.mimic.robots.g1_29dof.dance_102.tracking_env_cfg import RobotEnvCfg as DanceCfg
from unitree_rl_lab.tasks.mimic.robots.g1_29dof.gangnanm_style.tracking_env_cfg import RobotEnvCfg as GangnamCfg
from . import mdp


@configclass
class SuperMimicRewardsCfg:
    tracking = RewardTermCfg(func=mdp.tracking_reward_and_capture, weight=1.)


@configclass
class SuperMimicTerminationsCfg:
    failure = TerminationTermCfg(func=mdp.tracking_failure)
    time_out = TerminationTermCfg(func=isaac_mdp.time_out, time_out=True)
    motion_end = TerminationTermCfg(func=mdp.motion_end, time_out=True)


@configclass
class ProprioceptionCfg(ObservationGroupCfg):
    proprioception = ObservationTermCfg(func=mdp.proprioception)

    def __post_init__(self):
        self.enable_corruption = False  # adapter adds the same noise as imagination
        self.concatenate_terms = True


@configclass
class SuperMimicObservationsCfg:
    policy = ProprioceptionCfg()


def configure_supermimic(cfg):
    cfg.supermimic_capture = True
    cfg.commands.motion.class_type = mdp.SuperMimicMotionCommand
    cfg.commands.motion.anchor_body_name = "pelvis"
    cfg.commands.motion.debug_vis = False
    cfg.rewards = SuperMimicRewardsCfg()
    cfg.terminations = SuperMimicTerminationsCfg()
    cfg.observations = SuperMimicObservationsCfg()
    cfg.events.push_robot = None
    cfg.events.physics_material.func = mdp.randomize_shared_friction
    cfg.events.physics_material.params = {}
    cfg.scene.contact_forces = None  # no contact reward or sensor history required
    cfg.scene.terrain.visual_material = None  # avoid remote texture dependency


@configclass
class G1DanceSuperMimicEnvCfg(DanceCfg):
    supermimic_capture: bool = True
    supermimic_start_frame: int = -1

    def __post_init__(self):
        super().__post_init__()
        configure_supermimic(self)


@configclass
class G1GangnamSuperMimicEnvCfg(GangnamCfg):
    supermimic_capture: bool = True
    supermimic_start_frame: int = -1

    def __post_init__(self):
        super().__post_init__()
        configure_supermimic(self)
