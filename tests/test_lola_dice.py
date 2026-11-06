"""Checks for the LOLA-DiCE shaping correction."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dbos.methods.baselines import LOLADiceTrainer  # noqa: E402


def make_trainer(inner_lr):
    torch.manual_seed(0)
    np.random.seed(0)
    config = {
        "scenario": "avalon5", "n_envs": 2, "n_steps": 12, "hidden_dim": 32,
        "device": "cpu", "use_wandb": False, "lola_inner_lr": inner_lr, "lola_weight": 1.0,
    }
    return LOLADiceTrainer(config)


def rollout(trainer):
    rnn = torch.zeros(trainer.n_agents, 2, trainer.memory_units)
    beliefs = trainer.belief_tracker.init_belief(trainer.n_envs)
    return trainer.collect_rollout(rnn, beliefs)[0]


def shaper_head_grad(inner_lr):
    trainer = make_trainer(inner_lr)
    buf = rollout(trainer)
    trainer.policy.zero_grad()
    trainer._lola_dice_loss(buf).backward()
    return trainer.policy.policy_head.weight.grad[: trainer.n_actions]


def test_no_opponent_step_gives_no_shaping_gradient():
    grad = shaper_head_grad(inner_lr=0.0)
    assert grad is None or torch.allclose(grad, torch.zeros_like(grad), atol=1e-8)


def test_opponent_step_gives_shaping_gradient():
    grad = shaper_head_grad(inner_lr=1.0)
    assert grad is not None and grad.abs().sum() > 0


def test_episode_terms_restart_at_boundaries():
    trainer = make_trainer(1.0)
    T, ape = 4, trainer.agents_per_env
    B = ape  # one env
    logp = torch.full((T, B), -1.0)
    dones = torch.zeros(T, B, dtype=torch.bool)
    dones[1] = True  # episode ends at t=1, a new one starts at t=2
    step, cum, discount = trainer._lola_episode_terms(logp, dones)
    per_step = -1.0 * ape
    assert torch.allclose(step[:, 0], torch.full((T,), per_step))
    # The first step of the new episode (t=2) is included in its cumulative sum.
    assert torch.allclose(cum[:, 0], torch.tensor([1, 2, 1, 2]) * per_step)
    g = trainer.gamma
    assert torch.allclose(discount[:, 0], torch.tensor([1.0, g, 1.0, g]))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
