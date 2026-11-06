# closed-form check of Theorem 2: D-BOS belief-gradient error vs k and eps
import itertools
import math

import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)


def enumerate_sequences(A, k):
    return list(itertools.product(range(A), repeat=k))


def belief_trajectory(log_pi_role, seq, prior, floor=0.02):
    # Bayes belief chain with the same floor as bos_belief_floor
    R =prior.shape[0]
    b = prior
    b_min = b.min()
    for a in seq:
        log_lik = log_pi_role[:, a]
        log_post = torch.log(b + 1e-8) + log_lik
        log_post = log_post - log_post.max()
        b = torch.softmax(log_post, dim=-1)
        if floor > 0:
            f = min(floor, 0.49 / R)
            b = (1.0 - f * R) * b + f
            b = b / b.sum()
        b_min = torch.minimum(b_min, b.min())
    return b, b_min


def run_trial(R, A, k, eps_target, z_true, seed):
    torch.manual_seed(seed)
    theta = torch.randn(R, A, requires_grad=True)

    def get_logs(delta=None):
        logits = theta if delta is None else theta + delta
        return torch.log_softmax(logits, dim=-1)

    if eps_target > 0:
        raw = torch.randn(R, A)
        raw = raw / raw.abs().max() * eps_target
    else:
        raw = torch.zeros(R, A)
    delta = raw.detach()

    prior = torch.full((R,), 1.0 / R)
    seqs = enumerate_sequences(A, k)

    log_pi = get_logs()
    log_q = get_logs(delta)

    realized_eps = (log_q[z_true] - log_pi[z_true]).abs().max().item()

    seq_t = torch.tensor(seqs, dtype=torch.long)
    logp_seq = log_pi[z_true][seq_t].sum(dim=-1)
    p_seq = torch.softmax(logp_seq, dim=0) if False else torch.exp(logp_seq - torch.logsumexp(logp_seq, dim=0))

    loss_true_total = 0.0
    loss_appr_total = 0.0
    b_mins = []
    for i, seq in enumerate(seqs):
        w = p_seq[i]
        if w.item() < 1e-12:
            continue
        b_true_final, bmin_t = belief_trajectory(log_pi, seq, prior)
        b_appr_final, bmin_a = belief_trajectory(log_q, seq, prior)
        b_mins.append(min(bmin_t.item(), bmin_a.item()))
        loss_true_total = loss_true_total + w * (-torch.log(b_true_final[z_true] + 1e-8))
        loss_appr_total = loss_appr_total + w * (-torch.log(b_appr_final[z_true] + 1e-8))

    grad_true = torch.autograd.grad(loss_true_total, theta, retain_graph=True)[0]
    grad_appr = torch.autograd.grad(loss_appr_total, theta, retain_graph=True)[0]

    err = (grad_appr - grad_true).norm().item()
    gt, ga = grad_true.flatten(), grad_appr.flatten()
    if gt.norm().item() < 1e-10 or ga.norm().item() < 1e-10:
        cos_sim = float("nan")
    else:
        cos_sim = F.cosine_similarity(ga.unsqueeze(0), gt.unsqueeze(0)).item()
    b_min = min(b_mins) if b_mins else 1.0 / R
    return err, b_min, realized_eps, cos_sim


def main():
    R, A = 4, 3
    z_true = 0
    n_seeds = 8
    ks = [1, 2, 3, 4, 5, 6]
    epsilons = [0.0, 0.25, 0.5, 1.0, 2.0]

    print(f"{'k':>3} {'eps_target':>10} {'realized_eps':>12} {'mean_err':>12} {'std_err':>10} {'mean_b_min':>10} {'mean_cos':>10} {'std_cos':>10}")
    results = []
    for k in ks:
        for eps in epsilons:
            errs, bmins, real_epss, cos_sims = [], [], [], []
            for s in range(n_seeds):
                err, b_min, realized_eps, cos_sim = run_trial(R, A, k, eps, z_true, seed=1000 * k + int(100 * eps) + s)
                errs.append(err)
                bmins.append(b_min)
                real_epss.append(realized_eps)
                if not math.isnan(cos_sim):
                    cos_sims.append(cos_sim)
            errs = np.array(errs)
            mean_real_eps = float(np.mean(real_epss))
            mean_cos = float(np.mean(cos_sims)) if cos_sims else float("nan")
            std_cos = float(np.std(cos_sims)) if cos_sims else float("nan")
            print(f"{k:>3} {eps:>10.2f} {mean_real_eps:>12.4f} {errs.mean():>12.6f} {errs.std():>10.6f} {np.mean(bmins):>10.4f} {mean_cos:>10.4f} {std_cos:>10.4f}")
            results.append((k, eps, mean_real_eps, errs.mean(), errs.std(), mean_cos))

    print()
    # Sanity check: eps=0 -> zero error (approximate model IS the true model)
    zero_eps_errs = [r[3] for r in results if r[1] == 0.0]
    print(f"Sanity check (eps=0, should be ~0): max mean_err = {max(zero_eps_errs):.8f}")

    # Fit exponent of k at fixed eps_target=1.0 (predicted: ~2, quadratic)
    print()
    print("--- Fitting error ~ k^p at eps_target=1.0 (Theorem 2 predicts p=2) ---")
    sub = [(k, e) for (k, eps, real_eps, e, _, _) in results if eps == 1.0 and e > 0]
    if len(sub) >= 2:
        log_k = np.log([k for k, _ in sub])
        log_e = np.log([e for _, e in sub])
        p, c = np.polyfit(log_k, log_e, 1)
        print(f"  fitted exponent p = {p:.3f}  (log-log slope; intercept exp(c)={math.exp(c):.4f})")

    print()
    print("--- Fitting error ~ realized_eps^q at k=4 (Theorem 2 predicts q=1) ---")
    sub = [(real_eps, e) for (k, eps, real_eps, e, _, _) in results if k == 4 and eps > 0 and e > 0]
    if len(sub) >= 2:
        log_e_in = np.log([re for re, _ in sub])
        log_e_out = np.log([e for _, e in sub])
        q, c = np.polyfit(log_e_in, log_e_out, 1)
        print(f"  fitted exponent q = {q:.3f}  (log-log slope; intercept exp(c)={math.exp(c):.4f})")

    print()
    print("--- Gradient DIRECTION check: is the D-BOS gradient correctly-directed despite ToM error? ---")
    print("(cosine similarity between the approximate/D-BOS gradient and the true gradient, eps>0 only)")
    cos_vals = [c for (k, eps, real_eps, e, _, c) in results if eps > 0 and not math.isnan(c)]
    if cos_vals:
        arr = np.array(cos_vals)
        print(f"  n={len(arr)} mean_cos_sim={arr.mean():.4f}  std={arr.std():.4f}  min={arr.min():.4f}  max={arr.max():.4f}")
        frac_positive = float((arr > 0).mean())
        print(f"  fraction with positive cosine similarity: {frac_positive:.3f}")
    # Cosine similarity as k grows at fixed eps_target=1.0 -- does direction degrade with k?
    print()
    print("--- cosine similarity vs k (eps_target=1.0) ---")
    for k in ks:
        sub_c = [c for (kk, eps, real_eps, e, _, c) in results if kk == k and eps == 1.0 and not math.isnan(c)]
        if sub_c:
            print(f"  k={k}: mean_cos_sim={np.mean(sub_c):.4f}")


if __name__ == "__main__":
    main()
