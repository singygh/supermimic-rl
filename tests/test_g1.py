from copy import deepcopy
from types import SimpleNamespace
import unittest

import torch

from supermimic.algorithms import SuperMimicAlgorithm
from supermimic.configs import AlgorithmConfig
from supermimic.configs.g1 import g1_config
from supermimic.envs.isaaclab_adapter import IsaacLabEnvAdapter
from supermimic.models import Actor, Critic
from supermimic.models.g1_world_model import G1WorldModel
from supermimic.storage import ReplayBuffer, TransitionBatch
from supermimic.tasks.g1 import G1MotionReference, G1TrackingTask, normalize_quat


def fixture():
    task = G1TrackingTask(("pelvis", "torso_link"), tuple(f"joint_{i}" for i in range(29)),
        torch.zeros(29), torch.tensor([[-1., 1.]]).expand(29, 2), torch.ones(29))
    n = 40
    q = torch.zeros(n, 29)
    q[:, 0] = torch.arange(n) * .01
    pos = torch.zeros(n, 2, 3)
    pos[:, :, 2] = 1
    quat = torch.zeros(n, 2, 4)
    quat[..., 0] = 1
    ref = G1MotionReference(task, q, torch.zeros_like(q), pos, quat, torch.zeros_like(pos), torch.zeros_like(pos), 50)
    context = torch.zeros(2, 33)
    context[:, 0] = .7
    state = task.pack_state(q[:2], q[:2] * 0, q[:2] * 0, q[:2] * 0,
                            pos[:2], quat[:2], pos[:2] * 0, pos[:2] * 0)
    return task, ref, context, state


class FakeIsaacEnv:
    """Exercise the public adapter against an in-place auto-reset lifecycle."""
    def __init__(self):
        task, reference, context, self.initial = fixture()
        self.cfg = SimpleNamespace(supermimic_capture=True)
        self.num_envs, self.device = 2, "cpu"
        self.command = SimpleNamespace(time_steps=torch.zeros(2, dtype=torch.long))
        self._supermimic_runtime = SimpleNamespace(task=task, reference=reference,
            context=context, command=self.command, motion_digest="fixture", state=lambda: self.current,
            terminal=None)
        self.current = self.initial.clone()
        self.unwrapped = self
        self.steps = torch.zeros(2, dtype=torch.long)

    def reset(self, env_ids=None):
        ids = torch.arange(2) if env_ids is None else env_ids
        self.current[ids] = self.initial[ids]
        self.command.time_steps[ids] = 0
        self.steps[ids] = 0
        return {}, {}

    def step(self, action):
        runtime = self._supermimic_runtime
        task = runtime.task
        state = self.current.clone()
        state[:, task.slices["previous_action"]] = state[:, task.slices["action"]]
        state[:, task.slices["action"]] = action
        state[:, task.slices["q"]] += .001
        frame = self.command.time_steps.clone()
        self.steps += 1
        term = torch.zeros(2, dtype=torch.bool)
        trunc = self.steps >= 10
        reward = task.reward(state, action, runtime.reference.at(frame))
        runtime.terminal = (state.clone(), frame, action.clone(), term.clone(), trunc.clone())
        self.current = state
        self.command.time_steps += 1
        self.reset(trunc.nonzero().flatten())
        return {}, reward, term, trunc, {}


class G1Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)

    def test_reward_and_termination_reference_alignment(self):
        task, ref, context, state = fixture()
        goal = ref.at(torch.tensor([0, 1]))
        torch.testing.assert_close(task.reward(state, torch.zeros(2, 29), goal), torch.full((2,), 5. * task.spec.dt))
        self.assertFalse(task.terminated(state, goal).any())
        moved = state.clone()
        moved[:, task.slices["pos"]] += torch.tensor([0., 0., .26] * 2)
        self.assertTrue(task.terminated(moved, goal).all())
        self.assertTrue((task.termination_prob(moved, goal) > .5).all())
        # Opposite quaternion signs encode exactly the same rotation.
        opposite = state.clone()
        opposite[:, task.slices["quat"]] *= -1
        torch.testing.assert_close(task.reward(opposite, torch.zeros(2, 29), goal), torch.full((2,), 5. * task.spec.dt))

    def test_calibration_noise_and_action_smoothness(self):
        task, ref, context, state = fixture()
        context[:, 1:30] = .005
        clean = task.observe_clean(state, context)
        torch.testing.assert_close(clean[:, :29], state[:, :29] - .005)
        torch.testing.assert_close(clean[:, 61:64], torch.tensor([[0., 0., -1.]]).expand(2, 3))
        noise = task.add_observation_noise(clean) - clean
        self.assertTrue((noise.abs() <= task.noise_amplitude + 1e-7).all())
        self.assertTrue((noise[:, -29:] == 0).all())
        goal = ref.at(torch.tensor([0, 1]))
        action = torch.ones(2, 29) * .2
        torch.testing.assert_close(task.reward(state, action, goal), torch.full((2,), task.spec.dt * (5 - .1 * 29 * .2**2)))

    def test_future_reference_clamp_and_fps(self):
        task, ref, _, _ = fixture()
        future = ref.at(torch.tensor([0, 39])).reshape(2, 10, -1)
        torch.testing.assert_close(future[0, :, 0], torch.arange(10) * .01)
        torch.testing.assert_close(future[1, :, 0], torch.full((10,), .39))
        with self.assertRaises(ValueError):
            G1MotionReference(task, torch.zeros(2, 29), torch.zeros(2, 29),
                torch.zeros(2, 2, 3), normalize_quat(torch.zeros(2, 2, 4)),
                torch.zeros(2, 2, 3), torch.zeros(2, 2, 3), 60)

    def test_g1_bptt_gradients_and_frozen_world_model(self):
        env = IsaacLabEnvAdapter(FakeIsaacEnv(), observation_noise=False)
        model = G1WorldModel(env.task, (16, 16))
        actor = Actor(env.spec, (16,), (16,), latent_dim=32)
        critic = Critic(env.spec, (16,))
        cfg = AlgorithmConfig(model_batch_size=2, imagination_batch_size=2, critic_epochs=1, critic_mini_batches=1)
        alg = SuperMimicAlgorithm(actor, critic, model, ReplayBuffer(24, 2, goal_resolver=env.reference.at),
            env.reference, cfg, observation_noise=env.add_observation_noise)
        alg.collect(env, 20)
        self.assertEqual(alg.storage.data["goal_t"].shape[-1], 0)
        window = alg.storage.sample_windows(8, 4)
        self.assertEqual(window.goals.shape, (4, 8, env.spec.goal_dim))
        restored = ReplayBuffer(24, 2, goal_resolver=env.reference.at)
        restored.load_state_dict(alg.storage.state_dict())
        torch.manual_seed(5)
        expected = alg.storage.sample_windows(8, 4)
        torch.manual_seed(5)
        actual = restored.sample_windows(8, 4)
        torch.testing.assert_close(expected.goals, actual.goals)
        torch.testing.assert_close(expected.states, actual.states)
        before = deepcopy(model.state_dict())
        rollout = alg.imagine(alg.storage.sample_seeds(2))
        metrics = alg.update_actor(rollout)
        self.assertGreater(metrics["actor_grad_norm"], 0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        for key, value in before.items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
        alg.update_critic(rollout)
        self.assertGreater(alg.fit_world_model(window)["world_model_grad_norm"], 0)

    def test_adapter_terminal_state_and_partial_reset(self):
        env = IsaacLabEnvAdapter(FakeIsaacEnv(), observation_noise=False)
        for _ in range(10):
            result = env.step(torch.ones(2, 29) * .2)
        self.assertTrue(result.truncated.all())
        self.assertTrue((result.terminal.frame_id == 9).all())
        self.assertTrue((result.current.frame_id == 0).all())
        self.assertTrue((result.current.episode_id == result.terminal.episode_id + 1).all())
        torch.testing.assert_close(result.terminal.clean_obs[:, -29:], torch.full((2, 29), .2))
        torch.testing.assert_close(result.current.clean_obs[:, -29:], torch.zeros(2, 29))
        env.step(torch.ones(2, 29))
        history = env.history[1].clone()
        env.reset(torch.tensor([0]))
        torch.testing.assert_close(env.history[1], history)
        self.assertEqual(env.env.command.time_steps[1], 1)

    def test_profile_paper_details(self):
        cfg, architecture = g1_config("paper")
        self.assertEqual(cfg.algorithm.world_model_batches_per_update, 10)
        self.assertEqual(cfg.algorithm.actor_betas, (.9, .95))
        self.assertEqual(cfg.algorithm.actor_grad_value, .1)
        self.assertEqual(architecture["world_model"], (2048, 2048, 2048))

    def test_model_warmup_then_actor_update(self):
        env = IsaacLabEnvAdapter(FakeIsaacEnv(), observation_noise=False)
        cfg = AlgorithmConfig(model_batch_size=2, imagination_batch_size=2, critic_epochs=1,
            critic_mini_batches=1, actor_warmup_updates=1)
        alg = SuperMimicAlgorithm(Actor(env.spec, (16,), (16,), latent_dim=32), Critic(env.spec, (16,)),
            G1WorldModel(env.task, (16,)), ReplayBuffer(24, 2, goal_resolver=env.reference.at),
            env.reference, cfg, observation_noise=env.add_observation_noise)
        alg.collect(env, 10)
        actor_before, critic_before = deepcopy(alg.actor.state_dict()), deepcopy(alg.critic.state_dict())
        warmup = alg.update()
        self.assertNotIn("actor_loss", warmup)
        self.assertNotIn("critic_loss", warmup)
        self.assertGreater(warmup["world_model_grad_norm"], 0)
        for module, before in ((alg.actor, actor_before), (alg.critic, critic_before)):
            for key, value in before.items():
                torch.testing.assert_close(value, module.state_dict()[key], rtol=0, atol=0)
        active = alg.update()
        self.assertIn("actor_loss", active)
        self.assertGreater(active["actor_grad_norm"], 0)

    def test_single_motion_profile(self):
        cfg, architecture = g1_config("single_motion")
        self.assertEqual(cfg.algorithm.actor_warmup_updates, 100)
        self.assertEqual(cfg.algorithm.world_model_batches_per_update, 10)
        self.assertEqual(architecture["world_model"], (512, 512, 512))

    def test_optimizer_schedule_and_two_stage_actor_clipping(self):
        env = IsaacLabEnvAdapter(FakeIsaacEnv(), observation_noise=False)
        cfg = AlgorithmConfig(model_batch_size=2, imagination_batch_size=2, critic_epochs=1,
            critic_mini_batches=1, actor_betas=(.9, .95), actor_grad_value=.1, lr_decay_updates=2)
        alg = SuperMimicAlgorithm(Actor(env.spec, (16,), (16,), latent_dim=32), Critic(env.spec, (16,)),
            G1WorldModel(env.task, (16,)), ReplayBuffer(24, 2, goal_resolver=env.reference.at),
            env.reference, cfg, observation_noise=env.add_observation_noise)
        alg.collect(env, 10)
        alg.num_updates = 2
        metrics = alg.update()
        self.assertAlmostEqual(metrics["learning_rate"], cfg.actor_lr * .03)
        self.assertEqual(alg.actor_optimizer.param_groups[0]["betas"], (.9, .95))
        for parameter in alg.actor.parameters():
            if parameter.grad is not None:
                self.assertLessEqual(parameter.grad.abs().max().item(), .100001)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
