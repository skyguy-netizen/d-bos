import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import numpy as np 

ENTITY = "variable_decept"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIGURE_DIR = os.path.join(REPO, "paper_results", "figures")
DATA_DIR = os.path.join(REPO, "paper_results", "plot_data")
TABLE_SEEDS = tuple(range(42, 48))

COLORS = {
    "PPO": "tab:blue",
    "BBM": "tab:red",
    "D-BOS": "tab:green",
    "AA": "tab:purple",
    "LOLA-DiCE": "tab:brown",
}


def run_seed(run):
    seed = run.config.get("seed")
    if seed is None and "_s" in run.name:
        # RTG run names carry the seed as a suffix, e.g. rtg_kfuture_bbm_action_s2.
        seed = run.name.rsplit("_s", 1)[1].split()[0]
    return int(seed) if seed is not None else None


def fetch_histories(cache_name, project, label_fn, keys, seeds, samples=500, refresh=False):
    path = os.path.join(DATA_DIR, f"{cache_name}.json")
    if os.path.exists(path) and not refresh:
        with open(path) as f:
            return json.load(f)

    import wandb

    api = wandb.Api(timeout=120)
    newest = {}
    # lazy=False is required: lazily loaded runs have an empty config.
    for run in api.runs(f"{ENTITY}/{project}", lazy=False):
        if run.state != "finished":
            continue
        label, seed = label_fn(run), run_seed(run)
        if label is None or seed not in seeds:
            continue
        previous = newest.get((label, seed))
        if previous is None or run.created_at > previous.created_at:
            newest[(label, seed)] = run

    data = {}
    for (label, seed), run in sorted(newest.items()):
        rows = run.history(keys=list(keys), samples=samples, pandas=False)
        series = {"step": [row["_step"] for row in rows], "run": run.name}
        for key in keys:
            series[key] = [row.get(key) for row in rows]
        data.setdefault(label, {})[str(seed)] = series

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
    return data


def seed_curves(by_seed, key, n_points=75):
    """Interpolate each seed onto a shared step grid. Returns (grid, [S, n])."""
    curves = []
    for series in by_seed.values():
        pairs = [(s, v) for s, v in zip(series["step"], series[key]) if v is not None]
        if len(pairs) >= 2:
            steps, values = map(np.asarray, zip(*pairs))
            curves.append((steps, values))
    if not curves:
        return None, None
    end = min(steps[-1] for steps, _ in curves)
    start = max(steps[0] for steps, _ in curves)
    grid = np.linspace(start, end, n_points)
    stacked = np.stack([np.interp(grid, steps, values) for steps, values in curves])
    return grid, stacked


def plot_mean_se(ax, grid, stacked, label, color, steps_scale=1e6):
    mean = stacked.mean(axis=0)
    se = stacked.std(axis=0, ddof=1) / np.sqrt(len(stacked)) if len(stacked) > 1 else 0 * mean
    x = grid / steps_scale
    ax.plot(x, mean, color=color, label=f"{label} (n={len(stacked)})")
    ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.2)


def save(fig, name):
    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {os.path.relpath(path, REPO)}")
