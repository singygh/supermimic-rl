from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    """Versioned tensor/units contract; goal_t scores transition t -> t+1."""

    state_dim: int = 3
    obs_dim: int = 3
    action_dim: int = 1
    goal_dim: int = 1
    context_dim: int = 1
    history_length: int = 10
    dt: float = 0.02
    version: str = "toy-point-mass-v1"
    state_layout: str = "position[m], velocity[m/s], previous_action[normalized]"
    frame_convention: str = "1D world axis"
    action_scale: float = 4.0
    joint_names: tuple[str, ...] = ()
    body_names: tuple[str, ...] = ()
    action_scales: tuple[float, ...] = ()

    def __post_init__(self):
        if min(self.state_dim, self.obs_dim, self.action_dim, self.goal_dim, self.history_length) < 1:
            raise ValueError("State, observation, action, goal and history sizes must be positive")
        if self.context_dim < 0 or self.dt <= 0 or self.action_scale <= 0:
            raise ValueError("Invalid context size, dt or action scale")
