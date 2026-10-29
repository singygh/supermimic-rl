from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn import functional as F

from supermimic.configs import AlgorithmConfig
from supermimic.storage import (ImaginedRollout, TransitionBatch, lambda_returns,
                                survival_objective)


@contextmanager
def frozen_parameters(*modules):
    """Freeze weights and stochastic layers while preserving input Jacobians."""
    params = [(p, p.requires_grad) for module in modules for p in module.parameters()]
    modes = [(submodule, submodule.training) for module in modules for submodule in module.modules()]
    try:
        for parameter, _ in params:
            parameter.requires_grad_(False)
        for module in modules:
            module.eval()
        yield
    finally:
        for parameter, flag in params:
            parameter.requires_grad_(flag)
        for module, mode in modes:
            module.training = mode


class SuperMimicAlgorithm:
    """Real collection -> autoregressive model fit -> imagined BPTT + GAE critic.

    goal_provider(seed, offset) must return the task reference at frame_id+offset,
    including offset H for terminal value bootstrap. Never use future real states.
    """

    def __init__(self, actor, critic, world_model, storage, goal_provider,
                 cfg: AlgorithmConfig | None = None, device="cpu", observation_noise=None):
        self.cfg = cfg or AlgorithmConfig()
        self.device = torch.device(device)
        self.actor, self.critic, self.world_model = [m.to(device) for m in (actor, critic, world_model)]
        self.storage, self.goal_provider = storage, goal_provider
        self.observation_noise = observation_noise
        groups = [{id(p) for p in m.parameters()} for m in (actor, critic, world_model)]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
            raise ValueError("Actor, critic and world model must have separate parameters")
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=self.cfg.actor_lr, betas=self.cfg.actor_betas)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=self.cfg.critic_lr)
        self.world_model_optimizer = torch.optim.Adam(world_model.parameters(), lr=self.cfg.world_model_lr)
        self.num_updates = self.num_collected_steps = 0

    def train_mode(self):
        for module in (self.actor, self.critic, self.world_model):
            module.train()

    def eval_mode(self):
        for module in (self.actor, self.critic, self.world_model):
            module.eval()

    def act(self, obs_history, goal):
        return self.actor(obs_history, goal)

    @torch.no_grad()
    def collect(self, env, n_steps, step_callback=None):
        reward_sum, completed = 0.0, 0
        failures, timeouts = 0, 0
        actor_mode = self.actor.training
        self.actor.eval()
        try:
            for _ in range(n_steps):
                view = env.get_step_view().to(self.device)
                action = self.act(view.obs_history, view.goal)
                action = (action + self.cfg.action_noise_std * torch.randn_like(action)).clamp(-1, 1)
                step = env.step(action.to(env.device))
                terminal = step.terminal.to(self.device)
                # Never use step.current here: automatic resets may have overwritten state/context.
                batch = TransitionBatch(
                    view.state, view.clean_obs, view.obs_history, view.goal,
                    step.action_exec.to(self.device), step.reward.to(self.device),
                    terminal.state, terminal.clean_obs, step.terminated.to(self.device),
                    step.truncated.to(self.device), view.context, view.episode_id,
                    view.motion_id, view.frame_id,
                )
                self.storage.append(batch)
                if step_callback is not None:
                    step_callback(step, view.episode_id)
                reward_sum += step.reward.mean().item()
                completed += (step.terminated | step.truncated).sum().item()
                failures += step.terminated.sum().item()
                timeouts += step.truncated.sum().item()
                self.num_collected_steps += env.num_envs
        finally:
            self.actor.train(actor_mode)
        return {"real_reward": reward_sum / max(n_steps, 1), "completed_episodes": completed,
                "real_terminations": failures, "real_truncations": timeouts}

    def _step_optimizer(self, loss, optimizer, module):
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite optimization loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(module.parameters(), self.cfg.max_grad_norm, error_if_nonfinite=True)
        if module is self.actor and self.cfg.actor_grad_value is not None:
            nn.utils.clip_grad_value_(module.parameters(), self.cfg.actor_grad_value)
        optimizer.step()
        return float(norm)

    def fit_world_model(self, batch):
        batch = batch.to(self.device)
        model, cfg = self.world_model, self.cfg
        state = batch.state_0
        per_step = []
        errors = []
        reward_errors, termination_errors = [], []
        for t in range(batch.actions.shape[1]):
            action, goal = batch.actions[:, t], batch.goals[:, t]
            state = model.transition(state, action, batch.context)  # no teacher forcing or detach
            error = (model.state_error(state, batch.states[:, t]) if hasattr(model, "state_error")
                     else (state - batch.states[:, t]).square().mean(-1))
            loss = cfg.state_loss_weight * error
            if model.learn_observation:
                obs = model.observe_clean(state, batch.context)
                loss = loss + cfg.observation_loss_weight * (obs - batch.clean_obs[:, t]).square().mean(-1)
            reward_error = (model.reward(state, action, goal) - batch.rewards[:, t]).square()
            if model.learn_reward:
                loss = loss + cfg.reward_loss_weight * reward_error
            probability = model.termination_prob(state, goal)
            if model.learn_termination:
                loss = loss + cfg.termination_loss_weight * F.binary_cross_entropy_with_logits(
                    model.termination_logits(state, goal), batch.terminated[:, t].float(), reduction="none")
            per_step.append(loss * batch.valid_mask[:, t])
            errors.append(error.detach())
            reward_errors.append(reward_error.detach())
            termination_errors.append((probability.detach() - batch.terminated[:, t].float()).square())
        # Paper-style sum over time, mean over batch (not mean over time).
        total = torch.stack(per_step, 1).sum(1).mean()
        norm = self._step_optimizer(total, self.world_model_optimizer, model)
        self.world_model_optimizer.zero_grad(set_to_none=True)
        error_tensor = torch.stack(errors, 1)
        return {"world_model_loss": total.item(), "world_model_grad_norm": norm,
                "model_state_mse_1": error_tensor[:, 0].mean().item(),
                f"model_state_mse_{len(errors)}": error_tensor[:, -1].mean().item(),
                "model_reward_mse": torch.stack(reward_errors).mean().item(),
                "termination_brier": torch.stack(termination_errors).mean().item()}

    def imagine(self, seed, horizon=None):
        """Build the complete differentiable graph; only weights are frozen."""
        seed = seed.to(self.device).detached()
        horizon = self.cfg.imagination_horizon if horizon is None else horizon
        if horizon < 1:
            raise ValueError("Imagination horizon must be positive")
        states, actions, rewards, terminations, values, goals = [seed.state_0], [], [], [], [], []
        history = seed.obs_history_0  # real noisy history, already containing current observation
        with frozen_parameters(self.world_model, self.critic):
            for t in range(horizon):
                goal = self.goal_provider(seed, t)
                goals.append(goal)
                values.append(self.critic(states[-1], goal, seed.context))
                action = self.actor(history, goal)
                state = self.world_model.transition(states[-1], action, seed.context)
                reward = self.world_model.reward(state, action, goal)
                probability = self.world_model.termination_prob(state, goal)
                if not torch.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
                    raise ValueError("Termination head must produce finite probabilities in [0, 1]")
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                terminations.append(probability)
                clean = self.world_model.observe_clean(state, seed.context)
                noisy = (self.observation_noise(clean) if self.observation_noise is not None else
                         clean + self.cfg.observation_noise_std * torch.randn_like(clean))
                history = torch.cat((history[:, 1:], noisy[:, None]), dim=1)
            goals.append(self.goal_provider(seed, horizon))
            values.append(self.critic(states[-1], goals[-1], seed.context))
        return ImaginedRollout(torch.stack(states, 1), torch.stack(goals, 1), seed.context,
                               torch.stack(actions, 1), torch.stack(rewards, 1),
                               torch.stack(terminations, 1), torch.stack(values, 1))

    def update_actor(self, rollout):
        objective = survival_objective(rollout.rewards, rollout.termination,
                                       rollout.values[:, -1], self.cfg.gamma).mean()
        self.world_model_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        norm = self._step_optimizer(-objective, self.actor_optimizer, self.actor)
        return {"actor_loss": -objective.item(), "actor_grad_norm": norm,
                "imagined_return": objective.item(),
                "survival_H": (1 - rollout.termination.detach()).prod(1).mean().item()}

    def update_critic(self, rollout):
        # Targets and fitting inputs are fixed for every epoch/minibatch.
        rollout = rollout.detached()
        targets = lambda_returns(rollout.rewards, rollout.termination, rollout.values,
                                 self.cfg.gamma, self.cfg.lam).detach().flatten()
        states = rollout.states[:, :-1].reshape(-1, rollout.states.shape[-1])
        goals = rollout.goals[:, :-1].reshape(-1, rollout.goals.shape[-1])
        context = rollout.context[:, None].expand(-1, rollout.rewards.shape[1], -1)
        context = context.reshape(states.shape[0], rollout.context.shape[-1])
        losses, norms = [], []
        for _ in range(self.cfg.critic_epochs):
            permutation = torch.randperm(len(states), device=self.device)
            for indices in permutation.tensor_split(min(self.cfg.critic_mini_batches, len(states))):
                value = self.critic(states[indices], goals[indices], context[indices])
                loss = F.mse_loss(value, targets[indices])
                norms.append(self._step_optimizer(loss, self.critic_optimizer, self.critic))
                losses.append(loss.item())
        self.critic_optimizer.zero_grad(set_to_none=True)
        return {"critic_loss": sum(losses) / len(losses), "critic_grad_norm": sum(norms) / len(norms)}

    def update(self):
        totals = {}
        for _ in range(self.cfg.updates_per_iteration):
            ratio = 1.
            if self.cfg.lr_decay_updates is not None:
                phase = min(max(self.num_updates - self.cfg.actor_warmup_updates, 0) / self.cfg.lr_decay_updates, 1.)
                ratio = self.cfg.min_lr_ratio + (1 - self.cfg.min_lr_ratio) * .5 * (1 + math.cos(math.pi * phase))
            for optimizer, base in ((self.actor_optimizer, self.cfg.actor_lr),
                                    (self.critic_optimizer, self.cfg.critic_lr),
                                    (self.world_model_optimizer, self.cfg.world_model_lr)):
                for group in optimizer.param_groups:
                    group["lr"] = base * ratio
            metrics = {}
            for _ in range(self.cfg.world_model_batches_per_update):
                window = self.storage.sample_windows(self.cfg.model_horizon, self.cfg.model_batch_size)
                for key, value in self.fit_world_model(window).items():
                    metrics[key] = metrics.get(key, 0.0) + value / self.cfg.world_model_batches_per_update
            if self.num_updates >= self.cfg.actor_warmup_updates:
                rollout = self.imagine(self.storage.sample_seeds(self.cfg.imagination_batch_size))
                critic_rollout = rollout.detached()
                metrics.update(self.update_actor(rollout))
                metrics.update(self.update_critic(critic_rollout))
            metrics["world_model_warmup"] = float(self.num_updates < self.cfg.actor_warmup_updates)
            metrics["learning_rate"] = self.cfg.actor_lr * ratio
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value / self.cfg.updates_per_iteration
            self.num_updates += 1
        return totals

    @torch.no_grad()
    def evaluate_model(self, horizons=(1, 8, 32), batch_size=None):
        """Real replay rollouts, independent of imagined policy return.

        Unavailable horizons are omitted until enough uninterrupted data exists.
        This is an in-replay diagnostic, not a held-out validation score.
        """
        batch_size = batch_size or self.cfg.model_batch_size
        metrics = {}
        mode = self.world_model.training
        self.world_model.eval()
        try:
            for horizon in horizons:
                if not len(self.storage.valid_starts(horizon)):
                    continue
                batch = self.storage.sample_windows(horizon, batch_size).to(self.device)
                state = batch.state_0
                for t in range(horizon):
                    state = self.world_model.transition(state, batch.actions[:, t], batch.context)
                metrics[f"replay_state_mse_{horizon}"] = (state - batch.states[:, -1]).square().mean().item()
                if hasattr(self.world_model, "state_error"):
                    metrics[f"replay_normalized_state_mse_{horizon}"] = self.world_model.state_error(
                        state, batch.states[:, -1]).mean().item()
        finally:
            self.world_model.train(mode)
        return metrics

    def state_dict(self):
        return {name: getattr(self, name).state_dict() for name in (
            "actor", "critic", "world_model", "actor_optimizer", "critic_optimizer", "world_model_optimizer")} | {
            "num_updates": self.num_updates, "num_collected_steps": self.num_collected_steps}

    def load_state_dict(self, state, load_optimizer=True):
        for name in ("actor", "critic", "world_model"):
            getattr(self, name).load_state_dict(state[name])
        if load_optimizer:
            for name in ("actor_optimizer", "critic_optimizer", "world_model_optimizer"):
                getattr(self, name).load_state_dict(state[name])
        self.num_updates, self.num_collected_steps = state["num_updates"], state["num_collected_steps"]
