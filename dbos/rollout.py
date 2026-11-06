import numpy as np
import torch


class RolloutBuffer:

    def __init__(self, n_steps, n_agents, n_envs, obs_shape, memory_units, n_roles, n_obs=1):
        self.n_steps  = n_steps
        self.n_agents = n_agents
        self.n_envs   = n_envs
        self.n_obs    = max(n_obs, 1)
        self.ptr      = 0

        self.obs        = torch.zeros(n_steps, n_agents, *obs_shape, dtype=torch.uint8)
        self.next_obs   = torch.zeros(n_steps, n_agents, *obs_shape, dtype=torch.uint8)
        self.actions    = torch.zeros(n_steps, n_agents, dtype=torch.int64)
        self.rewards    = torch.zeros(n_steps, n_agents)
        self.rewards_ext = torch.zeros(n_steps, n_agents)
        self.dones      = torch.zeros(n_steps, n_agents, dtype=torch.bool)
        self.roles      = torch.zeros(n_steps, n_agents, dtype=torch.int64)
        self.log_probs  = torch.zeros(n_steps, n_agents)
        self.values     = torch.zeros(n_steps, n_agents)
        self.rnn_states = torch.zeros(n_steps, n_agents, 2, memory_units)
        self.beliefs         = torch.zeros(n_steps, n_envs, n_roles)
        # one belief per observer
        self.per_obs_beliefs = torch.zeros(n_steps, n_envs, self.n_obs, n_roles)
        # belief after this step's update (belief critic trains on these)
        self.per_obs_next_beliefs = torch.zeros(n_steps, n_envs, self.n_obs, n_roles)
        self.lstm_h_red = torch.zeros(n_steps, n_envs, memory_units)
        self.bbm_rewards = torch.zeros(n_steps, n_agents)

    def store(self, obs, actions, rewards_ext, dones, roles, log_probs, values,
              rnn_states, beliefs, lstm_h_red, bbm_rewards=None,
              per_obs_beliefs=None, per_obs_next_beliefs=None, next_obs=None):
        t = self.ptr
        self.obs[t]          = torch.from_numpy(obs)
        self.next_obs[t]     = torch.from_numpy(next_obs if next_obs is not None else obs)
        self.actions[t]      = torch.from_numpy(actions)
        self.rewards_ext[t]  = torch.from_numpy(rewards_ext.astype(np.float32))
        self.rewards[t]      = self.rewards_ext[t].clone()
        self.dones[t]        = torch.from_numpy(dones)
        self.roles[t]        = torch.from_numpy(roles.astype(np.int64))
        self.log_probs[t]    = log_probs
        self.values[t]       = values
        self.rnn_states[t]   = rnn_states
        self.beliefs[t]      = beliefs
        if per_obs_beliefs is not None:
            self.per_obs_beliefs[t] = per_obs_beliefs
        else:
            self.per_obs_beliefs[t] = beliefs.unsqueeze(1).expand(
                -1, self.n_obs, -1)
        if per_obs_next_beliefs is not None:
            self.per_obs_next_beliefs[t] = per_obs_next_beliefs
        else:
            self.per_obs_next_beliefs[t] = self.per_obs_beliefs[t]
        self.lstm_h_red[t]   = lstm_h_red
        if bbm_rewards is not None:
            self.bbm_rewards[t] = torch.from_numpy(bbm_rewards)
        self.ptr += 1

    def compute_gae(self, last_values, gamma=0.99, lam=0.95):
        T, B       = self.n_steps, self.n_agents
        advantages = torch.zeros(T, B)
        gae        = torch.zeros(B)
        next_val   = last_values

        for t in reversed(range(T)):
            mask          = 1.0 - self.dones[t].float()
            delta         = self.rewards[t] + gamma * next_val * mask - self.values[t]
            gae           = delta + gamma * lam * mask * gae
            advantages[t] = gae
            next_val      = self.values[t]

        return advantages, advantages + self.values


def build_env_agent_map(roles_flat, n_envs, agents_per_env):
    red_idx      = np.full(n_envs, -1, dtype=np.int64)
    observer_idx = [[] for _ in range(n_envs)]
    for env_i in range(n_envs):
        base = env_i * agents_per_env
        obs_list = []
        for j in range(agents_per_env):
            role = roles_flat[base + j]
            if role == 0 and red_idx[env_i] < 0:
                red_idx[env_i] = base + j
            else:
                obs_list.append((role, base + j))
        # observers sorted by role
        obs_list.sort(key=lambda x: x[0])
        observer_idx[env_i] = [idx for role, idx in obs_list]
    return red_idx, observer_idx


def reset_done_env_beliefs(beliefs, dones, n_envs, agents_per_env, per_obs_beliefs=None):
    """Reset beliefs to uniform prior for completed episodes."""
    expected = n_envs * agents_per_env
    if dones.shape[0] != expected:
        raise ValueError(f"Expected dones with length {expected}, got {dones.shape[0]}")

    done_env = torch.from_numpy(dones.reshape(n_envs, agents_per_env).any(axis=1)).to(beliefs.device)
    if done_env.any():
        beliefs = beliefs.clone()
        beliefs[done_env] = 1.0 / beliefs.shape[-1]
        if per_obs_beliefs is not None:
            per_obs_beliefs = per_obs_beliefs.clone()
            per_obs_beliefs[done_env] = 1.0 / per_obs_beliefs.shape[-1]
    return beliefs, done_env, per_obs_beliefs
