#!/usr/bin/env python3
"""Train RTG using the PMAlgorithm path with YAML configs."""

import argparse
import math
import os
import random
import sys
from typing import Any, Dict

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.rtg.support import Config, evaluate_model, make_algo, make_env  # noqa: E402


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return expand_paths(yaml.safe_load(f) or {})


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
        out = value.replace(
            "${SCRATCH_ROOT:-/scratch/gilbreth/${USER}/bos_shaping/neurips/results}",
            os.environ.get("SCRATCH_ROOT", defaults["SCRATCH_ROOT"]),
        )
        for key, default in defaults.items():
            out = out.replace("${" + key + "}", os.environ.get(key, default))
        return os.path.expanduser(os.path.expandvars(out))
    return value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_config(raw: Dict[str, Any], run_name: str, seed: int, device: str | None) -> Config:
    args = dict(raw)
    args["mode"] = "train"
    args["run"] = run_name
    args["seed"] = seed
    if device:
        args["device"] = device
    args["train_scenarios"] = args.get("train_scenarios", args.get("train_scenario", "rescue_training"))
    args["eval_scenarios"] = args.get("eval_scenarios", args.get("eval_scenario", "rescue"))
    args["epochs"] = int(args.get("epochs", 300))
    args["parallel_envs"] = int(args.get("parallel_envs", args.get("n_envs", 512)))
    args["n_steps"] = int(args.get("n_steps", 16))
    args["prediction_mode"] = str(args.get("prediction_mode", "off"))
    args["deception_bonus"] = args.get("deception_bonus", "(0,0,0)")
    args["model"] = str(args.get("model", "default"))
    args["save_model"] = str(args.get("save_model", "10"))
    args["verbose"] = int(args.get("verbose", 1))
    algo_params = args.get("algo_params", {})
    if isinstance(algo_params, str):
        algo_params = yaml.safe_load(algo_params) or {}
    float_keys = {
        "learning_rate",
        "adam_epsilon",
        "max_grad_norm",
        "gamma",
        "gae_lambda",
        "entropy_bonus",
        "ppo_epsilon",
        "vf_coef",
        "dm_learning_rate",
        "dm_loss_scale",
    }
    int_keys = {
        "batch_epochs",
        "mini_batch_size",
        "pm_memory_units",
        "pm_out_features",
        "dm_memory_units",
        "dm_out_features",
        "dm_mini_batch_size",
        "dm_max_window_size",
        "deception_bonus_start_timestep",
    }
    for key in [
        "learning_rate",
        "adam_epsilon",
        "batch_epochs",
        "mini_batch_size",
        "max_grad_norm",
        "gamma",
        "gae_lambda",
        "entropy_bonus",
        "ppo_epsilon",
        "vf_coef",
        "pm_memory_units",
        "pm_out_features",
        "pm_lstm_mode",
        "dm_memory_units",
        "dm_out_features",
        "dm_mini_batch_size",
        "dm_max_window_size",
        "dm_learning_rate",
        "dm_lstm_mode",
        "dm_loss_scale",
        "deception_bonus_start_timestep",
    ]:
        if key in args:
            val = args[key]
            if key in float_keys:
                val = float(val)
            elif key in int_keys:
                val = int(val)
            algo_params[key] = val
    args["algo_params"] = algo_params
    args.setdefault("micro_batch_size", "auto")
    args.setdefault("amp", False)
    args.setdefault("split_policy", False)
    args.setdefault("nan_check", False)
    args.setdefault("restore", False)

    config = Config()
    config.setup(args)
    os.makedirs(config.log_folder, exist_ok=True)
    return config


def apply_bos_knobs(algo, raw: Dict[str, Any]) -> None:
    # BOS controls for PMAlgorithm path.
    algo.bos_k = int(raw.get("bos_k", raw.get("k", 0)))
    algo.bos_lambda = float(raw.get("bos_lambda", raw.get("lambda_dec", 0.0)))
    algo.bos_loss = str(raw.get("bos_loss", "v_critic"))
    algo.bos_belief_init = str(raw.get("bos_belief_init", "role_bp"))
    algo.bos_obs_mode = str(raw.get("bos_obs_mode", "auto"))
    algo.bos_obs_resample = str(raw.get("bos_obs_resample", "per_window"))
    algo.bos_start_timestep = int(raw.get("bos_start_timestep", 0))
    algo.bos_ramp_timestep = int(raw.get("bos_ramp_timestep", 0))
    algo.bos_grad_frac = float(raw.get("bos_grad_frac", 0.0))
    algo.bos_grad_allow_amplify = bool(raw.get("bos_grad_allow_amplify", False))
    algo.bos_critic_hidden = int(raw.get("bos_critic_hidden", 256))
    algo.bos_critic_lr = float(raw.get("bos_critic_lr", 1e-3))

    ckpt_root = str(raw.get("checkpoint_dir", "outputs/checkpoints/rtg_paper"))
    ckpt_name = str(raw.get("checkpoint_name", "run"))
    algo.checkpoint_dir = ckpt_root
    algo.checkpoint_name = ckpt_name
    algo.checkpoint_interval = int(raw.get("checkpoint_interval", 10_000_000))


def save_resolved_config(raw: Dict[str, Any], args: argparse.Namespace, config: Config, algo) -> None:
    """Write the exact run config beside periodic checkpoints for mixture eval."""
    run_cfg = dict(raw)
    run_cfg["run_name"] = args.run_name
    run_cfg["seed"] = args.seed
    run_cfg["device"] = config.device
    run_cfg["steps_per_epoch"] = args.steps_per_epoch
    run_cfg["eval_interval"] = int(raw.get("eval_interval", 0))

    ckpt_dir = os.path.join(algo.checkpoint_dir, algo.checkpoint_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(run_cfg, f, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RTG using PMAlgorithm and YAML config.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--run-name", required=True, help="Run name used for log/checkpoint folders.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", default=None, help="cuda/cpu/auto override.")
    parser.add_argument("--prediction-mode", default=None, help="off|action|observation|both")
    parser.add_argument("--bos-k", type=int, default=None, help="Override BOS k.")
    parser.add_argument("--bos-lambda", type=float, default=None, help="Override BOS lambda.")
    parser.add_argument("--deception-bonus", default=None, help="Override deception bonus tuple, e.g. '(0.5,0,0)'.")
    parser.add_argument("--train-scenario", default=None, help="Override train scenario.")
    parser.add_argument("--eval-scenario", default=None, help="Override eval scenario.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--parallel-envs", type=int, default=None, help="Override number of parallel envs.")
    parser.add_argument("--n-steps", type=int, default=None, help="Override rollout steps per update.")
    parser.add_argument("--mini-batch-size", type=int, default=None, help="Override PPO mini-batch size.")
    parser.add_argument("--eval-interval", type=int, default=None, help="Override training-time eval interval; 0 disables.")
    parser.add_argument("--eval-trials", type=int, default=None, help="Override training-time eval trials.")
    parser.add_argument("--checkpoint-name", default=None, help="Override checkpoint folder name.")
    parser.add_argument("--checkpoint-interval", type=int, default=None, help="Override periodic checkpoint interval.")
    parser.add_argument("--steps-per-epoch", type=int, default=1_000_000, help="Environment interactions per epoch.")
    parser.add_argument("--resume-from", default=None, help="Periodic RTG checkpoint to restore before training.")
    parser.add_argument("--target-timesteps", type=int, default=None,
                        help="Total environment interactions to reach after a resume.")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging.")
    args = parser.parse_args()

    raw = load_yaml(args.config)
    if args.prediction_mode is not None:
        raw["prediction_mode"] = args.prediction_mode
    if args.bos_k is not None:
        raw["bos_k"] = args.bos_k
    if args.bos_lambda is not None:
        raw["bos_lambda"] = args.bos_lambda
    if args.deception_bonus is not None:
        raw["deception_bonus"] = args.deception_bonus
    if args.train_scenario is not None:
        raw["train_scenario"] = args.train_scenario
    if args.eval_scenario is not None:
        raw["eval_scenario"] = args.eval_scenario
    if args.epochs is not None:
        raw["epochs"] = args.epochs
    if args.parallel_envs is not None:
        raw["parallel_envs"] = args.parallel_envs
    if args.n_steps is not None:
        raw["n_steps"] = args.n_steps
    if args.mini_batch_size is not None:
        raw["mini_batch_size"] = args.mini_batch_size
    if args.eval_interval is not None:
        raw["eval_interval"] = args.eval_interval
    if args.eval_trials is not None:
        raw["eval_trials"] = args.eval_trials
    # Periodic checkpoints are consumed by mixture eval, so each sweep run needs its own folder.
    raw["checkpoint_name"] = args.checkpoint_name or args.run_name
    if args.checkpoint_interval is not None:
        raw["checkpoint_interval"] = args.checkpoint_interval

    set_seed(args.seed)
    config = build_config(raw, args.run_name, args.seed, args.device)

    if args.wandb:
        import wandb
        wandb.init(project=os.environ.get("WANDB_PROJECT", "neurips_rtg_final"), name=args.run_name, config=raw)

    print(f"Training RTG PMAlgorithm run={args.run_name} seed={args.seed} device={config.device}")
    vec_env = make_env(config.train_scenarios, config.parallel_envs, name="train", log_path=config.log_folder)
    algo = make_algo(vec_env, config)
    apply_bos_knobs(algo, raw)
    if args.resume_from:
        algo.load(args.resume_from)
        # Keep periodic checkpoints measured from the restored interaction count.
        algo._last_checkpoint_t = algo.t
        print(f"Resumed from {args.resume_from} at step={algo.t}")
    save_resolved_config(raw, args, config, algo)

    eval_interval = int(raw.get("eval_interval", 10))
    eval_trials = int(raw.get("eval_trials", 128))
    eval_scenario = str(raw.get("eval_scenario", "rescue"))

    if args.target_timesteps is not None:
        remaining_steps = max(0, int(args.target_timesteps) - int(algo.t))
        epochs_to_run = int(math.ceil(remaining_steps / max(1, int(args.steps_per_epoch))))
    else:
        epochs_to_run = config.epochs

    for epoch in range(epochs_to_run):
        # learn() steps are relative
        algo.learn(
            int(args.steps_per_epoch),
            reset_num_timesteps=(epoch == 0 and not args.resume_from),
        )
        if eval_interval > 0 and ((epoch + 1) % eval_interval == 0):
            eval_dir = os.path.join(config.log_folder, f"eval_ep{epoch + 1:03d}")
            red_s, green_s, blue_s = evaluate_model(algo, eval_scenario, eval_dir, trials=eval_trials)
            print(
                f"[eval] epoch={epoch + 1:03d} scenario={eval_scenario} "
                f"red={red_s:.3f} green={green_s:.3f} blue={blue_s:.3f}"
            )

    final_path = os.path.join(config.log_folder, "model_final.pt")
    algo.save(final_path)
    print(f"Saved final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
