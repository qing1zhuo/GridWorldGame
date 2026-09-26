"""Independent checks of reward semantics, Bellman evaluation and gradient isolation."""
import unittest
import numpy as np
import torch
from config import config
from env import GridWorld
from evaluate import optimal_reference, policy_values, transition_model
from runner import ActorCriticRunner


class GridTests(unittest.TestCase):
    def test_notebook_exposes_every_config_parameter(self):
        import ast
        import inspect
        import json
        from pathlib import Path
        notebook = json.loads(Path(__file__).with_name('main.ipynb').read_text(encoding='utf-8'))
        calls = []
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                tree = ast.parse(''.join(cell['source']))
                calls.extend(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                             and isinstance(n.func, ast.Name) and n.func.id == 'config')
        self.assertEqual(len(calls), 1)
        self.assertEqual({k.arg for k in calls[0].keywords}, set(inspect.signature(config).parameters))

    def test_custom_schedule_scale_and_solver_limit(self):
        cfg = config(lr=0.1, critic_lr=0.2, lr_decay_steps=4,
                     min_lr_ratio=0.4, reward_scale=20, entropy_coef=0)
        runner = ActorCriticRunner(cfg)
        eye = torch.eye(25)
        metrics = runner.train_batch(eye[[8]],torch.tensor([1]),torch.tensor([10.]),
                                     eye[[9]],torch.tensor([True]))
        self.assertAlmostEqual(metrics['critic_loss'],0.25)
        self.assertAlmostEqual(runner.actor_optimizer.param_groups[0]['lr'],0.075)
        self.assertAlmostEqual(runner.critic_optimizer.param_groups[0]['lr'],0.15)
        runner.updates = 4
        runner.train_batch(eye[[8]],torch.tensor([1]),torch.tensor([10.]),
                          eye[[9]],torch.tensor([True]))
        self.assertAlmostEqual(runner.actor_optimizer.param_groups[0]['lr'],0.04)
        self.assertAlmostEqual(runner.critic_optimizer.param_groups[0]['lr'],0.08)
        with self.assertRaises(RuntimeError):
            optimal_reference(config(value_iteration_max_steps=1))
        for kwargs in [dict(min_lr_ratio=0), dict(lr_decay_steps=0),
                       dict(reward_scale=0), dict(greedy_tolerance=0)]:
            with self.assertRaises(ValueError):
                config(**kwargs)

    def test_all_transitions_against_scalar_specification(self):
        cfg = config()
        env = GridWorld(cfg)
        nxt, rewards, done = transition_model(cfg)
        directions = [(-1,0),(0,1),(1,0),(0,-1),(0,0)]
        for row in range(5):
            for col in range(5):
                for a, (dr,dc) in enumerate(directions):
                    sid = row*5+col
                    if cfg.env.grid[row,col] == 2:
                        expected = (sid,0,True)
                    elif not (0 <= row+dr < 5 and 0 <= col+dc < 5):
                        expected = (sid,-10,False)
                    else:
                        cell = cfg.env.grid[row+dr,col+dc]
                        expected = ((row+dr)*5+col+dc, {0:0,1:-10,2:10}[cell], cell==2)
                    self.assertEqual((nxt[sid,a],rewards[sid,a],done[sid,a]),expected)
                    if cfg.env.grid[row,col] != 2:
                        env.reset((row,col))
                        s,r,d = env.step(a)
                        self.assertEqual((s[0]*5+s[1],r,d),expected)

    def test_reset_and_terminal_contract(self):
        env = GridWorld(config())
        for _ in range(100):
            self.assertNotEqual(env.cfg.env.grid[env.reset()],2)
        with self.assertRaises(ValueError):
            env.reset((1,4))
        env.reset((1,3))
        self.assertEqual(env.step(1),((1,4),10.0,True))
        with self.assertRaises(RuntimeError):
            env.step(4)

    def test_optimal_values_and_ties(self):
        cfg = config()
        ref = optimal_reference(cfg)
        expected = np.array([[6.561,7.29,8.1,9,10], [5.9049,6.561,9,10,0],
                             [5.31441,0,10,9,10], [0,10,0,10,9], [0,9,10,9,8.1]])
        np.testing.assert_allclose(ref['values'].reshape(5,5),expected,atol=1e-8)
        self.assertTrue(ref['optimal_actions'][18,3])  # lower target shortcut
        self.assertEqual(ref['optimal_actions'][15].tolist(),[False,False,True,False,True])
        pi = ref['optimal_actions']/ref['optimal_actions'].sum(1,keepdims=True)
        np.testing.assert_allclose(policy_values(pi,cfg),ref['values'],atol=1e-9)
        stay = np.zeros((25,5)); stay[:,4] = 1
        values = policy_values(stay,cfg)
        self.assertAlmostEqual(values[0],0)
        self.assertAlmostEqual(values[4],-100)

    def test_terminal_bootstrap_and_separate_heads(self):
        cfg = config(entropy_coef=0)
        runner = ActorCriticRunner(cfg)
        eye = torch.eye(25)
        before = runner.model.actor_head.weight.detach().clone()
        with torch.no_grad():
            runner.model.critic_head.weight[0,9] = 1000
        self.assertEqual(runner.model(eye[9])[1].item(),0)
        metrics = runner.train_batch(eye[[8]],torch.tensor([1]),torch.tensor([10.]),
                                     eye[[9]],torch.tensor([True]))
        self.assertAlmostEqual(metrics['critic_loss'],1.0)
        self.assertGreater(runner.model.actor_head.weight[1,8].item(),before[1,8].item())
        np.testing.assert_array_equal(runner.model.actor_head.weight.detach().numpy()[:,0],before.numpy()[:,0])


if __name__ == '__main__':
    unittest.main()
