#!/usr/bin/env python3
"""Unified training entry point for all methods and environments."""

import argparse
import math
import os
import sys
import yaml
import random
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path) as f:
        return expand_paths(yaml.safe_load(f))


def expand_paths(value):
    if isinstance(value, dict):
        return {k: expand_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_paths(v) for v in value]
    if isinstance(value, str):
        defaults = {
            "SCRATCH_ROOT": f"/scratch/gilbreth/{os.environ.get('USER', 'user')}/bos_shaping/neurips/results",
            "PROJECT_DIR": os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        }
        out = value
        out = out.replace(
            "${SCRATCH_ROOT:-/scratch/gilbreth/${USER}/bos_shaping/neurips/results}",
            os.environ.get("SCRATCH_ROOT", defaults["SCRATCH_ROOT"]),
        )
        for key, default in defaults.items():
            out = out.replace("${" + key + "}", os.environ.get(key, default))
            out = out.replace("${" + key + ":-" + default + "}", os.environ.get(key, default))
        return os.path.expanduser(os.path.expandvars(out))
    return value


def apply_timestep_overrides(config):
    """Allow scripts/configs to specify env-step budgets instead of epochs."""
    steps_per_epoch = int(config.get("n_steps", 1)) * int(config.get("n_envs", 1))
    steps_per_epoch = max(steps_per_epoch, 1)

    target_steps = os.environ.get("TARGET_ENV_TIMESTEPS", config.get("target_env_timesteps"))
    if target_steps is not None:
        target_steps = int(float(target_steps))
        config["target_env_timesteps"] = target_steps
        config["n_train_epochs"] = int(math.ceil(target_steps / steps_per_epoch))

    eval_steps = os.environ.get("EVAL_ENV_TIMESTEPS", config.get("eval_env_timesteps"))
    if eval_steps is not None:
        eval_steps = int(float(eval_steps))
        config["eval_env_timesteps"] = eval_steps
        config["eval_interval"] = max(1, int(math.ceil(eval_steps / steps_per_epoch)))

    save_steps = os.environ.get("SAVE_ENV_TIMESTEPS", config.get("save_env_timesteps"))
    if save_steps is not None:
        save_steps = int(float(save_steps))
        config["save_env_timesteps"] = save_steps
        config["save_interval"] = max(1, int(math.ceil(save_steps / steps_per_epoch)))


def main():
    parser = argparse.ArgumentParser(description="Train RL agent with configurable method and environment")
    parser.add_argument("--method", choices=["bos", "ppo", "ppo_belief", "aa", "bbm", "lola_dice"], default="bos",
                        help="Training method (BOS, PPO, belief-value PPO, Advantage Alignment, BBM, or LOLA-DiCE)")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu). Auto-detect if not specified")
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config(args.config)
    apply_timestep_overrides(config)
    config["seed"] = args.seed
    config["use_wandb"] = args.wandb
    if args.device:
        config["device"] = args.device
    else:
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    if args.method == "bos":
        from dbos.methods import BOSTrainer
        trainer_cls = BOSTrainer
    elif args.method == "ppo":
        from dbos.methods.baselines import PPOTrainer
        trainer_cls = PPOTrainer
    elif args.method == "ppo_belief":
        from dbos.methods.baselines import BeliefValuePPOTrainer
        trainer_cls = BeliefValuePPOTrainer
    elif args.method == "aa":
        from dbos.methods.baselines import AdvantageAlignmentTrainer
        trainer_cls = AdvantageAlignmentTrainer
    elif args.method == "bbm":
        from dbos.methods.baselines import BBMTrainer
        trainer_cls = BBMTrainer
    elif args.method == "lola_dice":
        from dbos.methods.baselines import LOLADiceTrainer
        trainer_cls = LOLADiceTrainer
    else:
        raise ValueError(f"Unknown method: {args.method}")

    print(f"Training {args.method.upper()} on {config.get('scenario', 'unknown')} "
          f"(seed={args.seed}, device={config['device']})")

    trainer = trainer_cls(config)
    trainer.train()


if __name__ == "__main__":
    main()
