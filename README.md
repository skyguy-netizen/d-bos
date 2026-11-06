# Differentiable Belief-Based Opponent Shaping (D-BOS)

Research code for the NeurIPS 2026 paper on Differentiable Belief-Based
Opponent Shaping.

## Setup

Python 3.11. Install the pinned dependencies (these match the paper runs):

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
```

## Training

Avalon, Coin Game and the belief-relay game use `runfiles/train.py`:

```bash
python runfiles/train.py --method bos --config configs/avalon/dbos_k1_estimated.yaml --seed 42 --wandb
```

Rescue-the-General uses `runfiles/train_rtg_paper.py` with
`configs/rtg/{ppo,bbm,dbos}.yaml`.

## Acknowledgments

We thank the creators of Rescue-the-General for the environment. All
Rescue-the-General files (including `envs/rtg/*` and
`runfiles/train_rtg_paper.py`) are adapted from
https://github.com/maitchison/RTG.

The baselines in `dbos/methods/bos.py` follow the reference implementations:

- Advantage Alignment: https://github.com/jduquevan/advantage-alignment
- LOLA-DiCE: https://github.com/alshedivat/lola (`lola_dice/`) 

We also thank the creators of the Advantage Alignment repo as well as LOLA-DiCE
