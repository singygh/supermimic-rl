from dataclasses import dataclass, field
from typing import Protocol

from torch import Tensor

from supermimic.configs import TaskSpec
from supermimic.storage.batches import TensorBatch


@dataclass
class StepView(TensorBatch):
    """Atomic snapshot; tensors must remain valid after the next simulator step."""

    state: Tensor
    clean_obs: Tensor
    obs_history: Tensor
    goal: Tensor
    context: Tensor
    episode_id: Tensor
    motion_id: Tensor
    frame_id: Tensor


@dataclass
class EnvStep:
    terminal: StepView  # true post-action state BEFORE any automatic reset
    current: StepView  # policy input AFTER automatic reset (if any)
    action_exec: Tensor  # actual normalized action, including simulator clipping
    reward: Tensor
    terminated: Tensor
    truncated: Tensor
    extras: dict = field(default_factory=dict)  # optional simulator episode summaries


class EnvAdapter(Protocol):
    num_envs: int
    device: str
    spec: TaskSpec

    def get_step_view(self) -> StepView: ...
    def reset(self, ids: Tensor | None = None) -> StepView: ...
    def step(self, action_exec: Tensor) -> EnvStep: ...
    def get_context(self) -> Tensor: ...
    def get_full_state(self) -> Tensor: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...
