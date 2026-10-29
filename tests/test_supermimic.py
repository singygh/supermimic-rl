from copy import deepcopy
from dataclasses import fields
import tempfile
import unittest

import torch

from supermimic.configs import AlgorithmConfig, RunnerConfig, TaskSpec
from supermimic.envs import ToyAdapter
from supermimic.models import WorldModel
from supermimic.models.heads import ToyObservation, ToyReward, ToyTermination, toy_transition
from supermimic.runners import SuperMimicRunner
from supermimic.storage import ReplayBuffer, lambda_returns, survival_objective


def make_runner(max_episode_steps=64, exact=False):
    torch.manual_seed(7)
    cfg = RunnerConfig(algorithm=AlgorithmConfig(model_batch_size=4, imagination_batch_size=4,
        critic_epochs=2, critic_mini_batches=2, observation_noise_std=0),
        hidden_dims=(16, 16), replay_capacity=48)
    env = ToyAdapter(3, max_episode_steps=max_episode_steps, noise_std=0)
    model = ExactToyModel(env.spec, cfg.hidden_dims, ToyObservation(), ToyReward(), ToyTermination()) if exact else (
        WorldModel(env.spec, cfg.hidden_dims, ToyObservation(), ToyReward(), ToyTermination()))
    return SuperMimicRunner(env, cfg, world_model=model)


class ExactToyModel(WorldModel):
    def transition(self, state, action, context):
        return toy_transition(state, action, context, self.spec.dt, self.spec.action_scale)


class ReturnsTests(unittest.TestCase):
    def test_survival_hand_calculation_and_boundaries(self):
        reward = torch.tensor([[2., 3., 4.]])
        terminal = torch.tensor([5.])
        done = torch.full_like(reward, 0.25)
        expected = 2 + .9 * .75 * 3 + .9**2 * .75**2 * 4 + .9**3 * .75**3 * 5
        self.assertAlmostEqual(survival_objective(reward, done, terminal, .9).item(), expected, places=5)
        self.assertAlmostEqual(survival_objective(reward, done * 0, terminal, 1).item(), 14)
        self.assertEqual(survival_objective(reward, torch.ones_like(done), terminal, .9).item(), 2)

    def test_lambda_zero_one_and_termination(self):
        reward = torch.tensor([[1., 2., 3.]])
        value = torch.tensor([[10., 20., 30., 40.]])
        done = torch.tensor([[0., 1., 0.]])
        torch.testing.assert_close(lambda_returns(reward, done, value, .5, 0), torch.tensor([[11., 2., 23.]]))
        torch.testing.assert_close(lambda_returns(reward, done, value, .5, 1), torch.tensor([[2., 2., 23.]]))

    def test_soft_termination_gradient(self):
        done = torch.full((1, 3), .2, requires_grad=True)
        survival_objective(torch.ones_like(done), done, torch.ones(1), .99).sum().backward()
        self.assertTrue((done.grad < 0).all())


class ReplayTests(unittest.TestCase):
    def test_ring_windows_and_terminal_alignment(self):
        runner = make_runner(max_episode_steps=10)
        runner.alg.collect(runner.env, 75)  # force ring wrap and many auto resets
        replay = runner.alg.storage
        idx = replay._indices()
        for start, env in replay.valid_starts(8).tolist():
            times = idx[start:start + 8]
            episodes = replay.data["episode_id"][times, env]
            self.assertTrue((episodes == episodes[0]).all())
            frames = replay.data["frame_id"][times, env]
            torch.testing.assert_close(frames[1:], frames[:-1] + 1)
            torch.testing.assert_close(replay.data["next_state"][times[:-1], env],
                                       replay.data["state_t"][times[1:], env])
        window = replay.sample_windows(8, 16)
        state = window.state_0
        for t in range(8):
            state = toy_transition(state, window.actions[:, t], window.context)
            torch.testing.assert_close(state, window.states[:, t])
        self.assertFalse(window.truncated[:, :-1].any())
        self.assertTrue(window.truncated[:, -1].any())

    def test_terminal_state_is_not_reset_state(self):
        env = ToyAdapter(2, max_episode_steps=1, noise_std=0)
        before = env.get_step_view()
        result = env.step(torch.ones(2, 1))
        torch.testing.assert_close(result.terminal.state, toy_transition(before.state, torch.ones(2, 1), before.context))
        self.assertTrue(result.truncated.all())
        self.assertFalse(result.terminated.any())
        self.assertTrue((result.current.episode_id == before.episode_id + 1).all())
        torch.testing.assert_close(result.current.obs_history,
            result.current.clean_obs[:, None].expand(-1, 10, -1))
        self.assertFalse(torch.equal(result.terminal.state, result.current.state))

    def test_context_changes_invalidate_windows(self):
        runner = make_runner()
        runner.alg.collect(runner.env, 8)
        replay = runner.alg.storage
        self.assertEqual(len(replay.valid_starts(8)), 3)
        replay.data["context"][3, 0] += 1
        self.assertEqual(len(replay.valid_starts(8)), 2)

    def test_buffer_save_load_and_empty(self):
        runner = make_runner()
        with self.assertRaises(ValueError):
            runner.alg.storage.sample_windows(8, 1)
        runner.alg.collect(runner.env, 60)
        replay = runner.alg.storage
        restored = ReplayBuffer(48, 3)
        restored.load_state_dict(replay.state_dict())
        for key in replay.data:
            torch.testing.assert_close(replay.data[key][replay._indices()], restored.data[key][restored._indices()])
        torch.testing.assert_close(replay.valid_starts(8), restored.valid_starts(8))


class GradientTests(unittest.TestCase):
    def test_transition_action_jacobian_finite_difference(self):
        state = torch.tensor([[.3, .2, .1]], dtype=torch.double)
        context = torch.tensor([[.2]], dtype=torch.double)
        action = torch.tensor([[.4]], dtype=torch.double, requires_grad=True)
        gradient = torch.autograd.grad(toy_transition(state, action, context).square().sum(), action)[0]
        eps = 1e-6
        numerical = (toy_transition(state, action + eps, context).square().sum()
                     - toy_transition(state, action - eps, context).square().sum()) / (2 * eps)
        self.assertAlmostEqual(gradient.item(), numerical.item(), places=8)

    def test_bptt_actor_gradient_finite_difference_and_isolation(self):
        runner = make_runner(exact=True)
        alg = runner.alg
        alg.collect(runner.env, 10)
        for model in (alg.actor, alg.critic, alg.world_model):
            model.double()
        seed = alg.storage.sample_seeds(2)
        for field in fields(seed):
            value = getattr(seed, field.name)
            if value.is_floating_point():
                setattr(seed, field.name, value.double())
        model_before, critic_before = deepcopy(alg.world_model.state_dict()), deepcopy(alg.critic.state_dict())
        parameter = alg.actor.decoder[-1].bias
        def objective():
            rollout = alg.imagine(seed)
            return survival_objective(rollout.rewards, rollout.termination, rollout.values[:, -1], alg.cfg.gamma).mean()
        derivative = torch.autograd.grad(objective(), parameter)[0][0].item()
        eps = 1e-5
        with torch.no_grad():
            parameter[0] += eps
        plus = objective().item()
        with torch.no_grad():
            parameter[0] -= 2 * eps
        minus = objective().item()
        with torch.no_grad():
            parameter[0] += eps
        self.assertGreater(abs(derivative), 1e-6)
        self.assertAlmostEqual(derivative, (plus - minus) / (2 * eps), places=6)
        metrics = alg.update_actor(alg.imagine(seed))
        self.assertGreater(metrics["actor_grad_norm"], 0)
        for module, before in ((alg.world_model, model_before), (alg.critic, critic_before)):
            self.assertTrue(all(p.grad is None for p in module.parameters()))
            for key, value in module.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_late_reward_reaches_first_action(self):
        state = torch.tensor([[.2, 0., 0.]])
        first = torch.tensor([[.3]], requires_grad=True)
        state = toy_transition(state, first, torch.tensor([[.2]]))
        detached = state.detach()
        for _ in range(31):
            state = toy_transition(state, torch.zeros_like(first), torch.tensor([[.2]]))
            detached = toy_transition(detached, torch.zeros_like(first), torch.tensor([[.2]]))
        gradient = torch.autograd.grad(state[:, 0].sum(), first)[0]
        self.assertGreater(gradient.item(), .01)
        self.assertFalse(detached.requires_grad)

    def test_learned_heads_and_optimizer_domains(self):
        runner = make_runner()
        alg = runner.alg
        model = WorldModel(runner.env.spec, (16, 16))
        alg.world_model = model
        alg.world_model_optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        alg.collect(runner.env, 16)
        actor_before = deepcopy(alg.actor.state_dict())
        metrics = alg.fit_world_model(alg.storage.sample_windows(8, 4))
        self.assertGreater(metrics["world_model_grad_norm"], 0)
        for head in (model.dynamics, model.observation_head, model.reward_head, model.termination_head):
            self.assertTrue(any(alg.world_model_optimizer.state[p]["exp_avg"].abs().sum() > 0 for p in head.parameters()))
        for key, value in alg.actor.state_dict().items():
            torch.testing.assert_close(value, actor_before[key])
        rollout = alg.imagine(alg.storage.sample_seeds(4))
        alg.update_actor(rollout)
        actor_before = deepcopy(alg.actor.state_dict())
        alg.update_critic(rollout)
        for key, value in alg.actor.state_dict().items():
            torch.testing.assert_close(value, actor_before[key], rtol=0, atol=0)

    def test_analytic_reward_observation_match_simulator(self):
        runner = make_runner()
        env, model = runner.env, runner.alg.world_model
        view = env.get_step_view()
        step = env.step(torch.full((3, 1), .5))
        torch.testing.assert_close(model.reward(step.terminal.state, step.action_exec, view.goal), step.reward)
        torch.testing.assert_close(model.observe_clean(step.terminal.state, view.context), step.terminal.clean_obs)


class RunnerTests(unittest.TestCase):
    def test_checkpoint_resumes_identically(self):
        original = make_runner()
        original.learn(3)
        with tempfile.TemporaryDirectory() as directory:
            path = directory + "/model.pt"
            original.save(path, {"test": 1})
            expected_metrics = original.learn(1)
            expected_models = deepcopy(original.alg.state_dict())
            restored = make_runner()
            self.assertEqual(restored.load(path), {"test": 1})
            actual_metrics = restored.learn(1)
            for name in ("actor", "critic", "world_model"):
                for key, value in expected_models[name].items():
                    torch.testing.assert_close(value, restored.alg.state_dict()[name][key], rtol=0, atol=0)
            for key in ("actor_loss", "world_model_loss", "critic_loss", "real_reward"):
                self.assertEqual(expected_metrics[key], actual_metrics[key])
            self.assertEqual(restored.current_learning_iteration, 4)

    def test_evaluation_preserves_training_and_deployment_interface(self):
        runner = make_runner()
        runner.learn(1)
        before = runner.env.state_dict()
        random_before = torch.get_rng_state().clone()
        result = runner.evaluate(10)
        self.assertTrue(torch.isfinite(torch.tensor(result["mean_reward"])))
        torch.testing.assert_close(torch.get_rng_state(), random_before)
        for key, value in before["tensors"].items():
            torch.testing.assert_close(value, runner.env.state_dict()["tensors"][key])
        view = runner.env.get_step_view()
        action = runner.get_inference_policy()(view.obs_history, view.goal)
        self.assertEqual(action.shape, (3, 1))
        self.assertFalse(action.requires_grad)

    def test_warmup_skips_update_until_windows_exist(self):
        runner = make_runner()
        runner.cfg.num_steps_per_env = 2
        runner.learn(3)
        self.assertEqual(runner.alg.num_updates, 0)
        runner.learn(1)
        self.assertEqual(runner.alg.num_updates, 1)

    def test_dict_configuration(self):
        runner = SuperMimicRunner(ToyAdapter(2), {"hidden_dims": (16,),
            "algorithm": {"model_batch_size": 2, "imagination_batch_size": 2, "critic_epochs": 1}})
        self.assertIn("actor_loss", runner.learn(1))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
