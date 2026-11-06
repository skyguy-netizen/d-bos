# usage: python scripts/plots/plot_theorem2.py
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common import save  # noqa: E402
from dbos.theorem2_validation import run_trial  # noqa: E402

R, A, Z_TRUE, N_SEEDS = 4, 3, 0, 8

plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 14,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12})
KS = [1, 2, 3, 4, 5, 6]
EPSILONS = [0.0, 0.25, 0.5, 1.0, 2.0]
FIT_EPS, FIT_K = 1.0, 4


def plain_log_ticks(ax, xticks, yticks):
    """Label a log-log axis with a few plain numbers instead of crowded powers of ten."""
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
    for axis, ticks in ((ax.xaxis, xticks), (ax.yaxis, yticks)):
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())
    ax.set_xticklabels([f"{t:g}" for t in xticks])
    ax.set_yticklabels([f"{t:g}" for t in yticks])


def sweep():
    """Return {(k, eps): (mean realized eps, mean error, mean cosine)}."""
    results = {}
    for k in KS:
        for eps in EPSILONS:
            errs, real_eps, cos = [], [], []
            for s in range(N_SEEDS):
                # Same seeding as theorem2_validation.main().
                err, _, realized, cos_sim = run_trial(R, A, k, eps, Z_TRUE,
                                                      seed=1000 * k + int(100 * eps) + s)
                errs.append(err)
                real_eps.append(realized)
                if not math.isnan(cos_sim):
                    cos.append(cos_sim)
            results[(k, eps)] = (np.mean(real_eps), np.mean(errs),
                                 np.mean(cos) if cos else float("nan"))
    return results


def loglog_fit(x, y):
    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
    return slope, math.exp(intercept)


def main():
    results = sweep()

    ks = np.array(KS, dtype=float)
    err_k = np.array([results[(k, FIT_EPS)][1] for k in KS])
    p, c = loglog_fit(ks, err_k)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(ks, err_k, "o-", label="measured error")
    ax.loglog(ks, c * ks ** p, "--", label=rf"fit: error $\propto k^{{{p:.2f}}}$")
    ax.set_title(r"Gradient error vs. $k$")
    ax.set_xlabel(r"Belief horizon $k$")
    ax.set_ylabel("Gradient error")
    plain_log_ticks(ax, KS, [0.1, 0.2])
    ax.legend()
    save(fig, "theorem2_error_vs_k.png")

    eps_pos = [e for e in EPSILONS if e > 0]
    real_eps = np.array([results[(FIT_K, e)][0] for e in eps_pos])
    err_eps = np.array([results[(FIT_K, e)][1] for e in eps_pos])
    q, c = loglog_fit(real_eps, err_eps)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(real_eps, err_eps, "o-", color="tab:green", label="measured error")
    ax.loglog(real_eps, c * real_eps ** q, "--", color="tab:red",
              label=rf"fit: error $\propto \epsilon^{{{q:.2f}}}$")
    ax.set_title(r"Gradient error vs. $\epsilon$")
    ax.set_xlabel(r"Realized $\epsilon$")
    ax.set_ylabel("Gradient error")
    plain_log_ticks(ax, [0.2, 0.5, 1.0], [0.1, 0.2, 0.4])
    ax.legend()
    save(fig, "theorem2_error_vs_eps.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    for eps in eps_pos:
        ax.plot(KS, [results[(k, eps)][2] for k in KS], "o-", label=rf"$\epsilon$={eps}")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_title(r"Gradient direction vs. $k$")
    ax.set_xlabel(r"Belief horizon $k$")
    ax.set_ylabel("Cosine similarity")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    save(fig, "cosine_similarity_vs_k.png")

    cos_all = np.array([v[2] for (k, e), v in results.items() if e > 0 and not math.isnan(v[2])])
    print(f"k exponent p={p:.3f}, eps exponent q={q:.3f}, "
          f"mean cosine={cos_all.mean():.4f}, positive fraction={(cos_all > 0).mean():.3f}")


if __name__ == "__main__":
    main()
