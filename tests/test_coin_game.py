"""Checks that Coin Game observers do not share the shaper's policy head."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dbos.methods import BOSTrainer  # noqa: E402
from dbos.methods.baselines import LOLADiceTrainer  # noqa: E402

BASE = {"scenario": "coin_game_multi_fullobs", "n_envs": 4, "n_steps": 12, "hidden_dim": 32,
        "device": "cpu", "use_wandb": False}


def seeded(cls, config):
    torch.manual_seed(0)
    np.random.seed(0)
    return cls(config)


def rollout(trainer):
    rnn = torch.zeros(trainer.n_agents, 2, trainer.memory_units)
    beliefs = trainer.belief_tracker.init_belief(trainer.n_envs)
    return trainer.collect_rollout(rnn, beliefs)


def test_observers_have_their_own_role_and_prior():
    trainer = seeded(BOSTrainer, {**BASE, "lambda_dec": 0.5, "k": 1})
    roles = trainer.env.get_roles().reshape(trainer.n_envs, -1)
    assert set(roles[:, 0].tolist()) <= {0, 1}
    assert (roles[:, 1:] == trainer.env.OBSERVER_ROLE).all()
    assert trainer.n_roles == 3
    prior = trainer._initial_per_observer_beliefs()
    assert torch.allclose(prior[..., :2], torch.full_like(prior[..., :2], 0.5))
    assert torch.all(prior[..., 2] == 0)


def test_dbos_step_does_not_change_observer_policy():
    trainer = seeded(BOSTrainer, {**BASE, "lambda_dec": 0.5, "k": 1})
    buf, rnn = rollout(trainer)[:2]
    obs = torch.from_numpy(trainer._current_obs).unsqueeze(0)
    roles = torch.from_numpy(trainer.env.get_roles()).long().unsqueeze(0)
    with torch.no_grad():
        before = trainer.policy.forward_sequence(obs, rnn, roles=roles)[0]["log_policy"][0].clone()
    trainer.optimizer.zero_grad()
    trainer.belief_planning_loss(buf).backward()
    trainer._mask_deception_gradients()
    with torch.no_grad():
        for p in trainer.policy.parameters():
            if p.grad is not None:
                p -= p.grad
        after = trainer.policy.forward_sequence(obs, rnn, roles=roles)[0]["log_policy"][0]
    change = (after - before).abs().sum(-1).reshape(trainer.n_envs, -1)
    observer_change = change[:, 1:]
    assert torch.all(observer_change < 1e-7)


def test_lola_treats_red_seat_as_shaper():
    trainer = seeded(LOLADiceTrainer, {**BASE, "lola_inner_lr": 1.0})
    buf = rollout(trainer)[0]
    roles = buf.roles[: trainer.n_steps]
    shaper, opponent, shaper_roles = trainer._lola_agent_masks(roles)
    per_env = shaper.reshape(trainer.n_steps, trainer.n_envs, -1)
    assert torch.all(per_env[..., 0] == 1) and torch.all(per_env[..., 1:] == 0)
    assert trainer.env.OBSERVER_ROLE not in shaper_roles
    assert torch.all(shaper + opponent == 1)


def test_lola_no_opponent_step_gives_no_shaping_gradient():
    trainer = seeded(LOLADiceTrainer, {**BASE, "lola_inner_lr": 0.0})
    buf = rollout(trainer)[0]
    trainer.policy.zero_grad()
    trainer._lola_dice_loss(buf).backward()
    grad = trainer.policy.policy_head.weight.grad
    assert grad is None or torch.allclose(grad[: trainer.n_actions], torch.zeros(1), atol=1e-8)
