import gymnasium as gym


for task_name, config in (("Dance-102", "G1DanceSuperMimicEnvCfg"), ("Gangnam-Style", "G1GangnamSuperMimicEnvCfg")):
    gym.register(id=f"Unitree-G1-29dof-SuperMimic-{task_name}",
                 entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
                 kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:{config}"})
