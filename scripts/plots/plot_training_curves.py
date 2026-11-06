# usage: python scripts/plots/plot_training_curves.py [--refresh]
import argparse
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from common import TABLE_SEEDS, fetch_histories, plot_mean_se, save, seed_curves

COLORS = {
    "PPO": "tab:blue", "BBM": "tab:red", "Advantage Alignment": "tab:purple",
    "LOLA-DiCE": "tab:brown", "D-BOS k=1": "tab:green", "D-BOS k=3": "tab:olive",
    "D-BOS k=5": "tab:cyan", "D-BOS k=1 (Est.)": "tab:green", "D-BOS k=3 (Est.)": "tab:olive",
    "Belief-ValuePPO": "tab:gray", "Belief-ValuePPO (Est.)": "tab:pink",
}


def method_label(run):
    cfg = run.config
    method = cfg.get("method")
    names = {"ppo": "PPO", "bbm": "BBM", "advantage_alignment": "Advantage Alignment",
             "lola_dice": "LOLA-DiCE"}
    if method in names:
        return names[method]
    if method == "belief_value_ppo":
        estimated = cfg.get("bos_likelihood_obs_mode") == "estimated"
        return "Belief-ValuePPO" + (" (Est.)" if estimated else "")
    if method == "bos" and cfg.get("k") in (1, 3, 5):
        estimated = cfg.get("bos_likelihood_obs_mode") == "estimated"
        return f"D-BOS k={cfg['k']}" + (" (Est.)" if estimated else "")
    return None


def merge(*datasets):
    merged = {}
    for data in datasets:
        for label, by_seed in data.items():
            merged.setdefault(label, {}).update(by_seed)
    return merged


def curve_figure(data, key, labels, title, ylabel, filename):
    fig, ax = plt.subplots(figsize=(6, 4))
    for label in labels:
        if label not in data:
            print(f"  {filename}: no runs for {label}")
            continue
        grid, stacked = seed_curves(data[label], key)
        style = "--" if "(Est.)" in label else "-"
        plot_mean_se(ax, grid, stacked, label, COLORS.get(label, "gray"))
        ax.lines[-1].set_linestyle(style)
    ax.set_title(title)
    ax.set_xlabel("Training steps (millions)")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False)
    save(fig, filename)


def avalon(refresh):
    key = "eval/red_win_rate"
    sources = [
        ("avalon_curves_main", "final_adapter_tuned_avalon"),
        ("avalon_curves_aa", "rebuttal_avalon_advantage_alignment_v2"),
        ("avalon_curves_lola", "camready_avalon_lola_dice_capped"),
        ("avalon_curves_bvppo", "rebuttal_avalon_belief_value_ppo_v2"),
        ("avalon_curves_bvppo_est", "camready_avalon_belief_value_ppo_estimated"),
    ]
    data = merge(*[fetch_histories(cache, project, method_label, [key], TABLE_SEEDS, refresh=refresh)
                   for cache, project in sources])
    curve_figure(data, key, ["PPO", "BBM", "Advantage Alignment", "LOLA-DiCE",
                             "D-BOS k=1", "D-BOS k=3", "D-BOS k=5"],
                 "Avalon: spy win rate during training",
                 "Evaluation win rate", "avalon_training_curves.png")
    curve_figure(data, key, ["PPO", "Belief-ValuePPO", "Belief-ValuePPO (Est.)",
                             "D-BOS k=1 (Est.)", "D-BOS k=3 (Est.)"],
                 "Avalon: estimated mode and belief-critic ablation",
                 "Evaluation win rate", "avalon_training_curves_estimated.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-download W&B histories")
    args = parser.parse_args()
    avalon(args.refresh)
