import torch


def bayes_update(prior: torch.Tensor, logits: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    # softmax-Bayes update in log space
    if prior.shape != logits.shape:
        raise ValueError(f"prior/logits shape mismatch: {prior.shape} vs {logits.shape}")
    return torch.softmax(torch.log(prior + eps) + logits, dim=-1)


class BeliefTracker:

    def __init__(self, n_roles: int = 3, device: str = "cpu"):
        self.n_roles = n_roles
        self.device  = device

    def init_belief(self, batch_size: int) -> torch.Tensor:
        return torch.ones(batch_size, self.n_roles, device=self.device) / self.n_roles

    def update(self, belief, role_log_policy, action):
        # b_{t+1}^z ∝ π(a_t | role=z) * b_t^z
        B, R, _ = role_log_policy.shape
        a_idx       = action.unsqueeze(1).expand(B, R).unsqueeze(-1)
        log_lik     = role_log_policy.gather(-1, a_idx).squeeze(-1)
        return bayes_update(belief, log_lik)

    def batch_update_trajectory(self, role_log_policy_seq, actions_seq, init_belief=None):
        T, B, R, _ = role_log_policy_seq.shape
        belief  = init_belief if init_belief is not None else \
                  self.init_belief(B).to(role_log_policy_seq.device)
        beliefs = [belief]
        for t in range(T):
            belief = self.update(belief, role_log_policy_seq[t], actions_seq[t])
            beliefs.append(belief)
        return torch.stack(beliefs, dim=0)  # [T+1, B, n_roles]
