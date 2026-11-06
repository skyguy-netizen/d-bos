#!/usr/bin/env python3
# held-out eval of a BOSTrainer checkpoint (Avalon, Coin)
import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dbos.methods.baselines import BeliefValuePPOTrainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--eval-seed", type=int, default=20260725)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", default="belief_value_ppo",
                         help="Method label used for the output filename, wandb run name, and "
                              "result['method']. Must be set explicitly when evaluating anything "
                              "other than belief_value_ppo checkpoints, since it is not inferred "
                              "from --checkpoint.")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--env-tag", default="avalon5",
                         help="Cosmetic environment tag for the output JSON / wandb run name / "
                              "tags. Does not affect which environment is actually loaded — that "
                              "comes from the checkpoint's own saved config['scenario'].")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(checkpoint["config"])
    config.update(
        {
            "device": args.device,
            "seed": args.eval_seed,
            "use_wandb": False,
            "save_dir": "",
        }
    )

    set_seed(args.eval_seed)
    trainer = BeliefValuePPOTrainer(config)
    trainer.policy.load_state_dict(checkpoint["policy"], strict=True)
    metrics = trainer.evaluate(n_episodes=args.episodes)

    result = {
        "method": args.label,
        "environment": args.env_tag,
        "scenario": config.get("scenario"),
        "train_seed": args.train_seed,
        "eval_seed": args.eval_seed,
        "episodes": args.episodes,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
    }
    if "red_win_rate" in metrics:
        red_win_rate = float(metrics["red_win_rate"])
        result["red_wins"] = int(round(red_win_rate * args.episodes))
        result["red_win_rate"] = red_win_rate
        result["red_win_rate_mc_se"] = math.sqrt(
            red_win_rate * (1.0 - red_win_rate) / args.episodes
        )
    for key in ("blue_win_rate", "red_score", "blue_score"):
        if key in metrics:
            result[key] = float(metrics[key])
    for key in ("red_scores_raw", "blue_scores_raw"):
        if key in metrics:
            result[key] = metrics[key]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.label}_s{args.train_seed}.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    if args.wandb:
        import wandb

        run = wandb.init(
            entity=os.environ.get("WANDB_ENTITY", "variable_decept"),
            project=os.environ.get(
                "WANDB_PROJECT", "rebuttal_avalon_expanded_eval"
            ),
            name=f"{args.env_tag}_{args.label}_s{args.train_seed}_eval{args.episodes}",
            config={
                "environment": args.env_tag,
                "scenario": config.get("scenario"),
                "method": args.label,
                "train_seed": args.train_seed,
                "eval_seed": args.eval_seed,
                "episodes": args.episodes,
                "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": result["checkpoint_epoch"],
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
