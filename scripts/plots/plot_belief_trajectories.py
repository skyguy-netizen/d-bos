# usage: python scripts/plots/plot_belief_trajectories.py [--refresh]
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import (COLORS, TABLE_SEEDS, fetch_histories, plot_mean_se, save,  # noqa: E402
                    seed_curves)

RTG_SEEDS = tuple(range(6))


def rtg_label(run):
    cfg = run.config
    if cfg.get("bos_loss") == "nll_1_minus":
        return "BBM"
    if cfg.get("bos_loss") == "v_critic" and "_esti" not in run.name:
        if not cfg.get("bos_lambda"):
            return "PPO"
        if cfg.get("bos_k") == 1:
            return "D-BOS k=1"
    return None


def method_label(run, k=1, estimated=None):
    cfg = run.config
    method = cfg.get("method")
    if method in ("ppo", "bbm"):
        return method.upper()
    if method == "bos" and cfg.get("k") == k:
        is_estimated = cfg.get("bos_likelihood_obs_mode") == "estimated"
        if estimated is None or estimated == is_estimated:
            return f"D-BOS k={k}" + (" (estimated)" if is_estimated else "")
    return None


def color_for(label):
    return COLORS["D-BOS"] if label.startswith("D-BOS") else COLORS.get(label, "gray")


def trajectory_figure(data, key, labels, title, ylabel, filename):
    fig, ax = plt.subplots(figsize=(6, 4))
    for label in labels:
        if label not in data:
            print(f"  {filename}: no runs for {label}")
            continue
        grid, stacked = seed_curves(data[label], key)
        plot_mean_se(ax, grid, stacked, label, color_for(label))
    ax.set_title(title)
    ax.set_xlabel("Training steps (millions)")
    ax.set_ylabel(ylabel)
    ax.legend()
    save(fig, filename)


def rtg(refresh):
    key = "train/b_true"
    data = fetch_histories("rtg_belief", "neurips_rtg_kfuture", rtg_label, [key], RTG_SEEDS,
                           refresh=refresh)
    trajectory_figure(data, key, ["PPO", "BBM", "D-BOS k=1"],
                      "RTG: belief trajectory by method (mean ± SE)",
                      r"$b_{\mathrm{true}}$ (belief in shaper's true role)",
                      "rtg_belief_trajectory.png")


def avalon(refresh):
    key = "train/avalon_resistance_b_true"
    data = fetch_histories("avalon_belief", "final_adapter_tuned_avalon",
                           lambda r: method_label(r, k=1, estimated=True) or
                           ("PPO" if r.config.get("method") == "ppo" else None),
                           [key], TABLE_SEEDS, refresh=refresh)
    trajectory_figure(data, key, ["PPO", "D-BOS k=1 (estimated)"],
                      "Avalon: resistance belief over training (mean ± SE)",
                      r"Resistance belief in shaper's true role ($b_{\mathrm{true}}$)",
                      "avalon_belief_trajectory.png")


def coin(refresh):
    key = "train/b_true"
    # Observers have their own policy head (role 2) in these runs.
    data = fetch_histories("coin_v2_belief", "camready_coin_v2", method_label, [key],
                           TABLE_SEEDS, refresh=refresh)
    trajectory_figure(data, key, ["PPO", "BBM", "D-BOS k=1"],
                      "Coin Game: observer belief over training (mean ± SE)",
                      r"$b_{\mathrm{true}}$ (belief in shaper's true role)",
                      "coingame_belief_trajectory.png")


RELAY_KEYS = ["train/oracle_b_true_observer_1", "train/oracle_b_true_observer_2", "train/blue_score"]


def relay_data(refresh):
    # Observer 1 is the ally and observer 2 the adversary
    # (BeliefRelayEnv.get_last_oracle_observer_beliefs).
    return fetch_histories("belief_relay", "rebuttal_belief_relay",
                           lambda r: method_label(r, k=1), RELAY_KEYS, TABLE_SEEDS,
                           refresh=refresh)


def relay_separation(refresh):
    runs = relay_data(refresh)["D-BOS k=1"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, label, color in [(RELAY_KEYS[0], "Ally observer", "tab:green"),
                              (RELAY_KEYS[1], "Adversary observer", "tab:red")]:
        grid, stacked = seed_curves(runs, key)
        plot_mean_se(ax, grid, stacked, label, color)
    ax.set_title("Belief-relay game: ally vs. adversary belief (D-BOS k=1)")
    ax.set_xlabel("Training steps (millions)")
    ax.set_ylabel(r"$b_{\mathrm{true}}$ (belief in shaper's true type)")
    ax.legend()
    save(fig, "belief_relay_separation.png")


def relay_belief_vs_score(refresh):
    runs = relay_data(refresh)["D-BOS k=1"]
    seeds = sorted(runs, key=int)
    n_cols = 3
    n_rows = int(np.ceil(len(seeds) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.5 * n_rows), sharex=True, sharey=True)
    for ax, seed in zip(np.ravel(axes), seeds):
        series = runs[seed]
        pairs = [(b, s) for b, s in zip(series[RELAY_KEYS[0]], series[RELAY_KEYS[2]])
                 if b is not None and s is not None]
        belief, score = map(np.asarray, zip(*pairs))
        r = np.corrcoef(belief, score)[0, 1]
        ax.scatter(belief, score, s=10, alpha=0.7)
        ax.set_title(f"seed={seed}, r={r:.2f}")
    fig.supxlabel("Ally belief in shaper's true type")
    fig.supylabel("blue_score (observer reward)")
    fig.suptitle(f"Belief vs. observer outcome across training (D-BOS k=1, {len(seeds)} seeds)")
    save(fig, "belief_vs_score_correlation.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-download W&B histories")
    args = parser.parse_args()
    rtg(args.refresh)
    avalon(args.refresh)
    coin(args.refresh)
    relay_separation(args.refresh)
    relay_belief_vs_score(args.refresh)
