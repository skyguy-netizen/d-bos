#!/usr/bin/env python3
# held-out eval of an RTG checkpoint (self-play)
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs import make_env as make_unified_env  # noqa: E402
from envs.rtg import utils as rtg_utils  # noqa: E402
from envs.rtg import rescue  # noqa: E402
from envs.rtg.support import make_algo, make_env  # noqa: E402
from train_rtg_paper import apply_bos_knobs, build_config, load_yaml, set_seed  # noqa: E402


def evaluate_self_play(algo, eval_scenario: str, trials: int, sub_folder: str):
    os.makedirs(sub_folder, exist_ok=True)
    vec_env = make_unified_env(eval_scenario, trials, name="eval", log_path=sub_folder)
    obs = vec_env.reset()
    rnn_states = algo.get_initial_rnn_state(vec_env.num_envs)
    terminals = np.zeros([len(rnn_states)], dtype=bool)
    vec_env.run_once = True

    team_scores = [None] * trials
    outcomes = [None] * trials
    while not all(terminals):
        with torch.no_grad():
            roles = vec_env.get_roles()
            model_output, new_rnn_states = algo.forward(
                obs=torch.from_numpy(obs),
                rnn_states=rnn_states,
                roles=torch.from_numpy(roles),
            )
            rnn_states[:] = new_rnn_states
            log_policy = model_output["log_policy"].detach().cpu().numpy()
            actions = rtg_utils.sample_action_from_logp(log_policy)
        obs, _, terminals, infos = vec_env.step(actions)
        for i, env in enumerate(vec_env.games):
            if env.round_outcome != "" and outcomes[i] is None:
                team_scores[i] = tuple(env.round_team_scores)
                outcomes[i] = env.round_outcome

    rescue.flush_logs()

    red_scores = [r for r, g, b in team_scores]
    green_scores = [g for r, g, b in team_scores]
    blue_scores = [b for r, g, b in team_scores]
    red_wins = [1.0 if o == "general_killed" else 0.0 for o in outcomes]
    blue_wins = [1.0 if o == "general_rescued" else 0.0 for o in outcomes]
    return {
        "trials": trials,
        "red_score": float(np.mean(red_scores)),
        "green_score": float(np.mean(green_scores)),
        "blue_score": float(np.mean(blue_scores)),
        "red_win_rate": float(np.mean(red_wins)),
        "blue_win_rate": float(np.mean(blue_wins)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True, help="Base YAML used to build the PMAlgorithm config.")
    parser.add_argument("--bos-k", type=int, default=None, help="Override bos_k (e.g. 1/3/5); leave unset for PPO/BBM.")
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--eval-seed", type=int, default=20260729)
    parser.add_argument("--episodes", type=int, default=50, help="Number of self-play trials (episodes).")
    parser.add_argument("--eval-scenario", default=None, help="Override eval scenario; defaults to config's own.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--env-tag", default="rtg_kfuture")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    raw = load_yaml(args.config)
    raw["prediction_mode"] = "action"
    if args.bos_k is not None:
        raw["bos_k"] = args.bos_k

    set_seed(args.eval_seed)
    run_name = f"eval_{args.label}_s{args.train_seed}_{args.eval_seed}"
    config = build_config(raw, run_name, args.eval_seed, args.device)

    vec_env = make_env(config.train_scenarios, 1, name="eval_load")
    algo = make_algo(vec_env, config)
    apply_bos_knobs(algo, raw)
    algo.load(args.checkpoint)

    eval_scenario = args.eval_scenario or str(raw.get("eval_scenario", "rescue"))
    metrics = evaluate_self_play(algo, eval_scenario, args.episodes, config.log_folder)

    result = {
        "method": args.label,
        "environment": args.env_tag,
        "scenario": eval_scenario,
        "train_seed": args.train_seed,
        "eval_seed": args.eval_seed,
        "episodes": args.episodes,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        **metrics,
    }
    result["red_win_rate_mc_se"] = math.sqrt(
        result["red_win_rate"] * (1.0 - result["red_win_rate"]) / args.episodes
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.label}_s{args.train_seed}.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    if args.wandb:
        import wandb

        run = wandb.init(
            entity=os.environ.get("WANDB_ENTITY", "variable_decept"),
            project=os.environ.get("WANDB_PROJECT", "rebuttal_rtg_eval50"),
            name=f"{args.env_tag}_{args.label}_s{args.train_seed}_eval{args.episodes}",
            config={
                "environment": args.env_tag,
                "scenario": eval_scenario,
                "method": args.label,
                "train_seed": args.train_seed,
                "eval_seed": args.eval_seed,
                "episodes": args.episodes,
                "checkpoint": result["checkpoint"],
            },
            tags=[args.env_tag, args.label, "expanded_eval"],
        )
        wandb.log({f"expanded_eval/{key}": value for key, value in result.items()
                   if isinstance(value, (int, float))})
        run.summary.update(result)
        run.finish()

    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
