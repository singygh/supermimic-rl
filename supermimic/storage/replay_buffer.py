from dataclasses import fields

import torch

from .batches import ImaginationSeed, ReplayWindow, TransitionBatch


class ReplayBuffer:
    """Ring of vector steps [capacity, num_envs, ...], storing real data only."""

    def __init__(self, capacity, num_envs, device="cpu", goal_resolver=None):
        if capacity < 1 or num_envs < 1:
            raise ValueError("Replay capacity and num_envs must be positive")
        self.capacity, self.num_envs = capacity, num_envs
        self.device = torch.device(device)
        self.data = {}
        self.goal_resolver = goal_resolver
        self.cursor = self.size = 0

    def append(self, step: TransitionBatch):
        incoming = {f.name: getattr(step, f.name).detach().to(self.device) for f in fields(step)}
        if self.goal_resolver is not None:
            incoming["goal_t"] = incoming["goal_t"][:, :0]  # reconstruct references from frame indices
        for name, value in incoming.items():
            if value.shape[0] != self.num_envs:
                raise ValueError(f"{name}: expected {self.num_envs} environments")
            if name in self.data and (value.shape != self.data[name].shape[1:]
                                      or value.dtype != self.data[name].dtype):
                raise ValueError(f"{name}: replay shape/dtype changed")
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"{name}: nonfinite replay data")
        if not self.data:
            self.data = {n: torch.empty((self.capacity, *v.shape), device=self.device, dtype=v.dtype)
                         for n, v in incoming.items()}
        for name, value in incoming.items():
            self.data[name][self.cursor].copy_(value)
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _indices(self):
        return (torch.arange(self.size, device=self.device) + self.cursor - self.size) % self.capacity

    def valid_starts(self, k):
        """Final transition may terminate/truncate; earlier transitions may not."""
        if k < 1:
            raise ValueError("Window length must be positive")
        if self.size < k:
            return torch.empty((0, 2), device=self.device, dtype=torch.long)
        idx = self._indices()
        if k == 1:
            return torch.ones(self.size, self.num_envs, dtype=torch.bool, device=self.device).nonzero()
        # Prefix sums over invalid adjacent edges avoid [time,k,env,context]
        # materialization, which is prohibitive with thousands of G1 robots.
        episodes = self.data["episode_id"][idx]
        contexts = self.data["context"][idx]
        done = self.data["terminated_tp1"][idx] | self.data["truncated_tp1"][idx]
        invalid = (episodes[1:] != episodes[:-1]) | done[:-1]
        invalid |= (contexts[1:] != contexts[:-1]).any(-1)
        prefix = torch.cat((torch.zeros_like(invalid[:1], dtype=torch.long), invalid.long().cumsum(0)), 0)
        valid = (prefix[k-1:] - prefix[:self.size-k+1]) == 0
        return valid.nonzero()

    def sample_windows(self, k, batch_size):
        valid = self.valid_starts(k)
        if not len(valid):
            raise ValueError(f"No valid {k}-step replay windows; collect more data or shorten k")
        starts, envs = valid[torch.randint(len(valid), (batch_size,), device=self.device)].unbind(1)
        times = self._indices()[starts[:, None] + torch.arange(k, device=self.device)]
        get = lambda key: self.data[key][times, envs[:, None]]
        goals = (get("goal_t") if self.goal_resolver is None else
                 self.goal_resolver(get("frame_id").flatten()).reshape(batch_size, k, -1))
        return ReplayWindow(
            get("state_t")[:, 0], get("action_exec_t"), get("next_state"), get("reward_tp1"),
            get("next_clean_obs"), get("terminated_tp1"), get("truncated_tp1"), goals,
            get("context")[:, 0], torch.ones((batch_size, k), device=self.device),
        )

    def sample_seeds(self, batch_size):
        if not self.size:
            raise ValueError("Cannot sample empty replay")
        times = self._indices()[torch.randint(self.size, (batch_size,), device=self.device)]
        envs = torch.randint(self.num_envs, (batch_size,), device=self.device)
        get = lambda key: self.data[key][times, envs]
        goals = get("goal_t") if self.goal_resolver is None else self.goal_resolver(get("frame_id"))
        return ImaginationSeed(get("state_t"), get("obs_history_t"), goals, get("context"),
                               get("motion_id"), get("frame_id"))

    def state_dict(self):
        # Snapshot chronological spans straight into CPU memory. CUDA advanced
        # indexing + clone would allocate additional full-sized replay tensors.
        data = {}
        oldest = (self.cursor - self.size) % self.capacity
        first = min(self.size, self.capacity - oldest)
        for key, value in self.data.items():
            snapshot = torch.empty((self.size, *value.shape[1:]), dtype=value.dtype, device="cpu")
            snapshot[:first].copy_(value[oldest:oldest + first])
            if first < self.size:
                snapshot[first:].copy_(value[:self.size - first])
            data[key] = snapshot
        return {"capacity": self.capacity, "num_envs": self.num_envs,
                "compact_goals": self.goal_resolver is not None,
                "data": data}

    def load_state_dict(self, state):
        if state["capacity"] != self.capacity or state["num_envs"] != self.num_envs:
            raise ValueError("Replay capacity/num_envs mismatch")
        if state.get("compact_goals", False) != (self.goal_resolver is not None):
            raise ValueError("Replay reference reconstruction configuration mismatch")
        data = state["data"]
        self.size = len(next(iter(data.values()))) if data else 0
        self.cursor = self.size % self.capacity
        self.data = {}
        for name, value in data.items():
            self.data[name] = torch.empty((self.capacity, *value.shape[1:]), dtype=value.dtype,
                                          device=self.device)
            # Direct CPU->CUDA copy avoids a second full-sized GPU allocation.
            self.data[name][:self.size].copy_(value)
