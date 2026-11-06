# usage: python scripts/plots/plot_gradient_diagnostics.py [--refresh]
import argparse
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from common import COLORS, TABLE_SEEDS, fetch_histories, plot_mean_se, save, seed_curves  # noqa: E402

BOS_KEY = "train/bos_grad_norm_final"
PPO_KEY = "train/ppo_grad_norm"
WIN_KEY = "eval/red_win_rate"
PROBE_SEEDS = (42, 43)


def bos_k1_estimated(run):
    cfg = run.config
    if cfg.get("method") == "bos" and cfg.get("k") == 1 and cfg.get("bos_likelihood_obs_mode") == "estimated":
        return "D-BOS k=1 (estimated)"
    return None


def gradient_ratio(refresh):
    data = fetch_histories("avalon_gradient_ratio", "final_adapter_tuned_avalon", bos_k1_estimated,
                           [BOS_KEY, PPO_KEY], TABLE_SEEDS, refresh=refresh)
    runs = data["D-BOS k=1 (estimated)"]
    for series in runs.values():
        series["ratio_pct"] = [
            100.0 * b / p if b is not None and p else None
            for b, p in zip(series[BOS_KEY], series[PPO_KEY])
        ]
    grid, stacked = seed_curves(runs, "ratio_pct")
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_mean_se(ax, grid, stacked, "D-BOS k=1 (estimated)", "tab:purple")
    ax.set_title("Injected D-BOS correction relative to the PPO gradient (Avalon)")
    ax.set_xlabel("Training steps (millions)")
    ax.set_ylabel("D-BOS grad norm / PPO grad norm (%)")
    ax.legend()
    save(fig, "gradient_ratio_avalon.png")


def amplification(refresh):
    def probe_label(run):
        cfg = run.config
        if (cfg.get("method") == "bos" and cfg.get("k") == 1
                and cfg.get("bos_likelihood_obs_mode") != "estimated"
                and cfg.get("bos_grad_frac") == 0.06):
            return "D-BOS k=1, amplified (bos_grad_frac=0.06)"
        return None

    amplified = fetch_histories("avalon_amplified_probe", "rebuttal_avalon_gradfrac", probe_label,
                                [WIN_KEY], PROBE_SEEDS, refresh=refresh)
    ppo = fetch_histories("avalon_ppo_probe_seeds", "final_adapter_tuned_avalon",
                          lambda r: "PPO" if r.config.get("method") == "ppo" else None,
                          [WIN_KEY], PROBE_SEEDS, refresh=refresh)
    probe_runs = amplified["D-BOS k=1, amplified (bos_grad_frac=0.06)"]
    window = min(series["step"][-1] for series in probe_runs.values())
    # Compare PPO over the same steps and seeds as the shorter probe runs.
    ppo_runs = {
        seed: {key: [v for v, s in zip(values, series["step"]) if s <= window]
               if isinstance(values, list) else values for key, values in series.items()}
        for seed, series in ppo["PPO"].items()
    }
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, runs, color in [("PPO", ppo_runs, COLORS["PPO"]),
                               ("D-BOS k=1, amplified (bos_grad_frac=0.06)", probe_runs, "tab:red")]:
        grid, stacked = seed_curves(runs, WIN_KEY)
        plot_mean_se(ax, grid, stacked, label, color)
    ax.set_title("Amplifying the D-BOS gradient (Avalon, 2-seed probe)")
    ax.set_xlabel("Training steps (millions)")
    ax.set_ylabel("Eval spy win rate")
    ax.legend()
    save(fig, "amplification_collapse.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-download W&B histories")
    args = parser.parse_args()
    gradient_ratio(args.refresh)
    amplification(args.refresh)
