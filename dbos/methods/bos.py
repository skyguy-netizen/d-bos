# D-BOS Belief-Based Opponent Shaping: exact gradients through k-step Bayesian belief updates

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
try:
    import wandb
except ImportError:
    wandb = None
from torch.optim import Adam


from envs.rtg.models import PolicyModel
from envs import make_env
from envs.rtg.utils import sample_action_from_logp
from dbos.belief import BeliefTracker
from dbos.rollout import RolloutBuffer, build_env_agent_map, reset_done_env_beliefs

class BeliefCritic(nn.Module):
    def __init__(self, obs_dim, n_roles, n_obs=1, hidden_dim=256, depth=2):
        super().__init__()
        # V(obs, B), B = flattened [n_obs, n_roles] belief matrix
        belief_dim = n_obs * n_roles
        layers = []
        in_dim = obs_dim + belief_dim
        for _ in range(max(1, int(depth))):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, obs, b):
        # b: [E, n_obs * n_roles] — joint belief matrix, already flattened
        return self.net(torch.cat([obs, b], dim=-1))


class ObserverObsEstimator(nn.Module):

    def __init__(self, obs_dim, n_roles, n_obs, hidden_dim=128):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_roles = int(n_roles)
        self.n_obs = int(max(n_obs, 1))
        in_dim = self.obs_dim + self.n_roles + self.n_obs
        self.lstm = nn.LSTM(in_dim, hidden_dim, batch_first=False)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.obs_dim),
            nn.Sigmoid(),
        )

    def forward(self, red_obs_seq, obs_role_seq, obs_slot_seq):
        T, E, n_obs, _ = red_obs_seq.shape
        role_oh = F.one_hot(obs_role_seq.clamp(0, self.n_roles - 1), self.n_roles).float()
        slot_oh_base = F.one_hot(obs_slot_seq.clamp(0, self.n_obs - 1), self.n_obs).float()
        slot_oh = slot_oh_base.unsqueeze(0).expand(T, E, n_obs, self.n_obs)
        x = torch.cat([red_obs_seq, role_oh, slot_oh], dim=-1)
        x = x.reshape(T, E * n_obs, -1)
        y, _ = self.lstm(x)
        pred = self.head(y).reshape(T, E, n_obs, self.obs_dim)
        return pred

class BOSTrainer:

    def __init__(self, config: dict):
        self.config    = config
        self.device    = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.scenario  = str(config.get("scenario", "rescue_training")).lower()

        self.n_steps        = config.get("n_steps", 128)
        self.n_epochs_ppo   = config.get("n_epochs_ppo", 4)
        self.n_minibatches  = config.get("n_minibatches", 4)
        self.gamma          = config.get("gamma", 0.99)
        self.gae_lambda     = config.get("gae_lambda", 0.95)
        self.clip_eps       = config.get("clip_eps", 0.2)
        self.ent_coef       = config.get("ent_coef", 0.01)
        self.val_coef       = config.get("val_coef", 0.5)
        self.max_grad_norm  = config.get("max_grad_norm", 0.5)

        self.k             = config.get("k", 10)
        self.lambda_dec    = config.get("lambda_dec", 0.1)
        self.warmup_epochs = config.get("warmup_epochs", 50)

        self.bos_belief_mode = str(config.get("bos_belief_mode", "role")).lower()
        if self.bos_belief_mode == "hidden_role":
            self.bos_belief_mode = "role"
        # BBM flag
        self.use_bbm_intrinsic   = bool(config.get("use_bbm_intrinsic", False))
        self._bbm_reward_var     = 1.0  # running variance for normalization
        self.z_true        = config.get("z_true", 0)
        self.dec_clip_eps  = config.get("dec_clip_eps", 0.2)
        if self.dec_clip_eps is not None and self.dec_clip_eps <= 0:
            self.dec_clip_eps = None
        self.dec_epochs     = max(1, int(config.get("dec_epochs", 2)))
        self.dec_head_only  = bool(config.get("dec_head_only", True))
        self.bos_update_all_role_heads = bool(config.get("bos_update_all_role_heads", False))
        self.dec_max_grad_norm = config.get("dec_max_grad_norm", self.max_grad_norm)
        self.bos_grad_frac = float(config.get("bos_grad_frac", config.get("dec_grad_frac", 0.0)) or 0.0)
        # BOS grad is capped relative to PPO, not amplified (unless set)
        self.bos_grad_allow_amplify = bool(config.get("bos_grad_allow_amplify", False))
        self.bos_grad_max_scale = float(config.get("bos_grad_max_scale", float("inf")))
        self._last_ppo_grad_norm = 1.0
        self.role_nll_coef  = config.get("role_nll_coef", 0.1)  # aux task weight for role prediction head
        self.bos_critic_hidden_dim = int(config.get("bos_critic_hidden_dim", 256))
        self.bos_critic_depth = int(config.get("bos_critic_depth", 2))
        self.bos_critic_lr = float(config.get("bos_critic_lr", 1e-3))
        self.bos_critic_updates = max(1, int(config.get("bos_critic_updates", 1)))
        self.use_belief_value_critic = bool(config.get("use_belief_value_critic", False))
        self.use_advantage_alignment = bool(config.get("use_advantage_alignment", False))
        self.aa_weight = float(config.get("aa_weight", 1.0))
        self.aa_warmup = bool(config.get("aa_warmup", False))
        self.aa_team_of_role = config.get("aa_team_of_role", None)
        # LOLA-DiCE flag (from LOLA-DiCE repo, head rows only)
        self.use_lola_dice = bool(config.get("use_lola_dice", False))
        self.lola_inner_lr = float(config.get("lola_inner_lr", 1.0))
        self.lola_weight = float(config.get("lola_weight", 1.0))
        # cap LOLA grad at this fraction of PPO head grad norm
        self.lola_grad_frac = float(config.get("lola_grad_frac", 0.0) or 0.0)
        self.lola_max_grad_norm = float(config.get("lola_max_grad_norm", self.max_grad_norm))
        # belief temperature / floor (keeps beliefs off the simplex boundary)
        self.bos_belief_temperature = max(1e-6, float(config.get("bos_belief_temperature", 1.0)))
        self.bos_belief_floor = max(0.0, float(config.get("bos_belief_floor", 0.0)))
        self.bos_likelihood_obs_mode = str(config.get("bos_likelihood_obs_mode", "canonical")).lower()
        self.bos_obs_estimator_hidden = int(config.get("bos_obs_estimator_hidden", 128))
        self.bos_obs_estimator_lr = float(config.get("bos_obs_estimator_lr", 1e-3))
        self.bos_obs_estimator_updates = max(1, int(config.get("bos_obs_estimator_updates", 1)))
        # autograd (direct -V backprop) | coeff (surrogate on likelihoods)
        self.bos_correction_mode = str(config.get("bos_correction_mode", "autograd")).lower()
        self.bos_coeff_normalize = bool(config.get("bos_coeff_normalize", False))
        self.bos_coeff_clip = float(config.get("bos_coeff_clip", 0.0) or 0.0)
        self.bos_min_entropy_frac = float(config.get("bos_min_entropy_frac", 0.0) or 0.0)
        # coeff mode: only shaper's z=0 path
        self.bos_shaper_role_only = bool(config.get("bos_shaper_role_only", False))
        self.bos_shaper_adapter = bool(config.get("bos_shaper_adapter", False))
        self.bos_shaper_adapter_hidden = int(config.get("bos_shaper_adapter_hidden", 128))

        self.n_train_epochs = config.get("n_train_epochs", 300)
        self.log_interval   = config.get("log_interval", 10)
        self.save_dir       = config.get("save_dir", None)
        self.save_interval  = config.get("save_interval", 50)
        self.use_wandb      = config.get("use_wandb", False)

        seed = config.get("seed", 42)
        self.method_name = str(config.get("method_name", "bos"))
        if self.method_name == "bos":
            self.run_tag = f"bos_k{self.k}_s{seed}"
        else:
            self.run_tag = f"{self.method_name}_s{seed}"

        self.env             = make_env(config.get("scenario", "rescue_training"),
                                        parallel_envs=config.get("n_envs", 64))
        self.n_envs          = config.get("n_envs", 64)
        self.n_agents        = self.env.total_agents
        self.agents_per_env  = self.n_agents // self.n_envs
        self.obs_shape       = self.env.observation_space.shape
        self.n_actions       = self.env.action_space.n
        self.memory_units    = config.get("hidden_dim", 128)

        # reset first so get_roles() is valid
        obs               = self.env.reset()
        self._current_obs = obs
        self.init_roles   = self.env.get_roles()
        self.n_roles      = max(int(self.env.max_roles), int(self.init_roles.max()) + 1)

        # belief relay only: observers' own learned model of the shaper
        self._has_observer_table = (
            hasattr(self.env, "set_observer_channel_log_liks")
            and hasattr(self.env, "get_observer_channel_decode")
        )
        if self._has_observer_table:
            decode = np.asarray(self.env.get_observer_channel_decode())  # [2, n_actions]
            self._observer_decode = torch.from_numpy(decode).long().to(self.device)
            n_channel_symbols = int(decode.max()) + 1
            self.observer_channel_table = torch.zeros(
                2, self.n_roles, n_channel_symbols, requires_grad=True, device=self.device)
            self.observer_table_optimizer = Adam(
                [self.observer_channel_table], lr=config.get("observer_table_lr", 0.05))
            self._observer_table_buffer = []
        else:
            self.observer_channel_table = None

        self.policy = PolicyModel(
            env=self.env, device=self.device,
            memory_units=self.memory_units, out_features=self.memory_units,
            lstm_mode=config.get("lstm_mode", "cat"), roles=self.n_roles, model="MLP",
            predict_roles=True,  # auxiliary head to compute b_true_paper (matches BBM Fig 4)
            shaper_adapter=self.bos_shaper_adapter,
            shaper_adapter_hidden=self.bos_shaper_adapter_hidden,
        ).to(self.device)

        self.optimizer       = Adam(self.policy.parameters(),
                                    lr=config.get("policy_lr", 2.5e-4), eps=1e-5)

        init_checkpoint = config.get("init_checkpoint", None)
        if init_checkpoint:
            ckpt = torch.load(init_checkpoint, map_location=self.device)
            self.policy.load_state_dict(ckpt["policy"])
            if config.get("init_checkpoint_load_optimizer", False):
                self.optimizer.load_state_dict(ckpt["optimizer"])
            print(f"Loaded initial policy weights from {init_checkpoint} "
                  f"(epoch {ckpt.get('epoch', '?')}); starting a fresh {self.method_name} run from this point.")

        self.belief_tracker  = BeliefTracker(n_roles=self.n_roles)
        # BeliefCritic is created after observer_idx is built (n_obs_per_env needed).
        self.belief_critic = None
        self.critic_optimizer = None
        self.obs_estimator = None
        self.obs_estimator_optimizer = None

        self.red_idx, self.observer_idx = build_env_agent_map(
            self.init_roles, self.n_envs, self.agents_per_env)
        fixed_red_fn = getattr(self.env, "get_red_agent_indices", None)
        self.has_fixed_focal_agent = callable(fixed_red_fn)
        if self.has_fixed_focal_agent:
            self.red_idx = np.asarray(fixed_red_fn(), dtype=np.int64).reshape(self.n_envs)
            self.observer_idx = []
            for env_i in range(self.n_envs):
                base = env_i * self.agents_per_env
                red_agent = int(self.red_idx[env_i])
                self.observer_idx.append([
                    base + j for j in range(self.agents_per_env) if (base + j) != red_agent
                ])
        if "kuhn_poker" in self.scenario:
            self.red_idx = np.arange(self.n_envs, dtype=np.int64) * self.agents_per_env
            self.observer_idx = [[int(base + 1)] for base in self.red_idx]
        if "avalon" in self.scenario:
            pass

        self.listener_agent_flat = None
        self.decoy_agents_flat = None
        self.speaker_agent_flat = None
        if self.bos_belief_mode in ("slot", "speaker"):
            listener_idx = np.full(self.n_envs, -1, dtype=np.int64)
            decoy_idx = np.full((self.n_envs, 2), -1, dtype=np.int64)
            speaker_idx = np.full((self.n_envs, self.n_roles), -1, dtype=np.int64)
            for e in range(self.n_envs):
                base = e * self.agents_per_env
                local_decoys = []
                for j in range(self.agents_per_env):
                    role = int(self.init_roles[base + j])
                    if role == 1 and listener_idx[e] < 0:
                        listener_idx[e] = base + j
                    elif role == 2:
                        local_decoys.append(base + j)
                    if self.bos_belief_mode == "speaker" and j > 0 and (j - 1) < self.n_roles:
                        speaker_idx[e, j - 1] = base + j
                if listener_idx[e] < 0:
                    raise ValueError(f"bos_belief_mode='{self.bos_belief_mode}' requires a role-1 listener agent per env")
                if self.bos_belief_mode == "slot" and len(local_decoys) < 1:
                    raise ValueError("bos_belief_mode='slot' requires at least one role-2 decoy per env")
                if len(local_decoys) == 1:
                    local_decoys = local_decoys * 2
                if self.bos_belief_mode == "slot":
                    decoy_idx[e, 0] = local_decoys[0]
                    decoy_idx[e, 1] = local_decoys[1]
                elif (speaker_idx[e] < 0).any():
                    raise ValueError("bos_belief_mode='speaker' requires three physical speakers per env")

            self.listener_agent_flat = torch.from_numpy(listener_idx)
            if self.bos_belief_mode == "slot":
                self.decoy_agents_flat = torch.from_numpy(decoy_idx)
            else:
                self.speaker_agent_flat = torch.from_numpy(speaker_idx)

            self.observer_idx = [[int(listener_idx[e])] for e in range(self.n_envs)]

        self.red_agent_flat = torch.from_numpy(self.red_idx)
        n_obs               = len(self.observer_idx[0]) if self.observer_idx else 0
        self.n_obs_per_env  = n_obs

        # BeliefCritic takes the joint belief matrix B ∈ R^{n_obs × n_roles}.
        if (self.lambda_dec > 0 and not self.use_bbm_intrinsic) or self.use_belief_value_critic:
            self.belief_critic = BeliefCritic(
                np.prod(self.obs_shape), self.n_roles,
                n_obs=max(self.n_obs_per_env, 1),
                hidden_dim=self.bos_critic_hidden_dim,
                depth=self.bos_critic_depth,
            ).to(self.device)
            self.critic_optimizer = torch.optim.Adam(
                self.belief_critic.parameters(), lr=self.bos_critic_lr)
            if self.bos_likelihood_obs_mode in ("estimated", "esti", "learned"):
                self.obs_estimator = ObserverObsEstimator(
                    int(np.prod(self.obs_shape)), self.n_roles, max(self.n_obs_per_env, 1),
                    hidden_dim=self.bos_obs_estimator_hidden,
                ).to(self.device)
                self.obs_estimator_optimizer = torch.optim.Adam(
                    self.obs_estimator.parameters(), lr=self.bos_obs_estimator_lr)
        self.obs_agent_flat = torch.tensor(
            [self.observer_idx[e] for e in range(self.n_envs)], dtype=torch.int64)
        self.observer_role_ids = torch.tensor(
            [[int(self.init_roles[agent_i]) for agent_i in self.observer_idx[e]]
             for e in range(self.n_envs)], dtype=torch.int64)

        # reference observer for belief logging
        blue_role = 1
        self.blue_obs_local_idx = []  # per-env index into observer list
        if "kuhn_poker" in self.scenario:
            self.blue_obs_local_idx = [0 for _ in range(self.n_envs)]
        elif "avalon" in self.scenario:
            # avalon: track a resistance observer
            for e in range(self.n_envs):
                chosen = None
                for local_i, agent_i in enumerate(self.observer_idx[e]):
                    if int(self.init_roles[agent_i]) >= 2:
                        chosen = local_i
                        break
                self.blue_obs_local_idx.append(chosen)
        else:
            for e in range(self.n_envs):
                for local_i, agent_i in enumerate(self.observer_idx[e]):
                    if self.init_roles[agent_i] == blue_role:
                        self.blue_obs_local_idx.append(local_i)
                        break
                else:
                    self.blue_obs_local_idx.append(None)  # no blue in this env

        self._per_obs_beliefs = None

    def _initial_per_observer_beliefs(self):
        """Construct episode-start priors in the trainer's observer ordering."""
        n_obs = max(self.n_obs_per_env, 1)
        beliefs = torch.full(
            (self.n_envs, n_obs, self.n_roles),
            1.0 / self.n_roles,
            dtype=torch.float32,
        )
        initial_belief_fn = getattr(self.env, "get_initial_observer_beliefs", None)
        if callable(initial_belief_fn):
            supplied = torch.as_tensor(initial_belief_fn(), dtype=torch.float32)
            if supplied.shape == beliefs.shape:
                beliefs = supplied

        if "avalon" in self.scenario and "blind" not in self.scenario and self.n_obs_per_env > 0:
            known_ally = self.observer_role_ids == 1
            beliefs[known_ally] = 0.0
            beliefs[..., 0][known_ally] = 1.0
        return beliefs

    def _freeze_known_observer_beliefs(self, beliefs: torch.Tensor) -> torch.Tensor:
        """Keep Avalon teammates' known focal-role posterior fixed."""
        if "avalon" not in self.scenario or "blind" in self.scenario or self.n_obs_per_env == 0:
            return beliefs
        known_ally = (self.observer_role_ids == 1).to(beliefs.device).unsqueeze(-1)
        fixed = self._initial_per_observer_beliefs().to(beliefs.device)
        return torch.where(known_ally, fixed, beliefs)

    def _avalon_belief_metrics(self, per_obs_beliefs: torch.Tensor):
        """Aggregate focal-role posterior by known ally and resistance observer."""
        if "avalon" not in self.scenario or self.n_obs_per_env == 0:
            return {}
        beliefs = per_obs_beliefs.float()
        dev = beliefs.device
        true_roles = torch.as_tensor(self._z_true_vec_np(), dtype=torch.long, device=dev)
        true_prob = beliefs.gather(
            -1,
            true_roles.view(self.n_envs, 1, 1).expand(-1, self.n_obs_per_env, 1),
        ).squeeze(-1)
        observer_roles = self.observer_role_ids.to(dev)

        def mean_for(mask):
            return float(true_prob[mask].mean().item()) if mask.any() else float("nan")

        metrics = {
            "ally_b_true": mean_for(observer_roles == 1),
            "resistance_b_true": mean_for(observer_roles >= 2),
        }
        for role in range(2, self.n_roles):
            metrics[f"resistance_role{role}_b_true"] = mean_for(observer_roles == role)
        return metrics

    def _regularize_belief(self, b: torch.Tensor) -> torch.Tensor:
        """Normalize and optionally keep beliefs away from simplex boundaries."""
        b = torch.nan_to_num(b, nan=1.0 / max(int(b.shape[-1]), 1), posinf=1.0, neginf=0.0)
        b = b.clamp_min(0.0)
        b = b / (b.sum(dim=-1, keepdim=True) + 1e-8)
        if self.bos_belief_floor > 0.0:
            R = b.shape[-1]
            floor = min(self.bos_belief_floor, 0.49 / max(R, 1))
            b = (1.0 - floor * R) * b + floor
            b = b / (b.sum(dim=-1, keepdim=True) + 1e-8)
        return b

    @torch.no_grad()
    def _belief_value(self, obs, per_obs_beliefs, red_agent_flat=None):
        """Detached belief-conditioned baseline for one focal agent per env."""
        if self.belief_critic is None:
            raise RuntimeError("Belief value requested without a BeliefCritic")
        red_agent_flat = self.red_agent_flat if red_agent_flat is None else red_agent_flat
        red_idx = red_agent_flat.to(torch.long)
        obs_t = torch.as_tensor(obs)[red_idx].to(self.device).float().reshape(self.n_envs, -1)
        belief_t = torch.as_tensor(per_obs_beliefs).to(self.device)
        n_observers = max(self.n_obs_per_env, 1)
        if belief_t.ndim == 2:
            belief_t = belief_t.unsqueeze(1).expand(-1, n_observers, -1)
        belief_t = belief_t.reshape(self.n_envs, n_observers * self.n_roles)
        return self.belief_critic(obs_t, belief_t).squeeze(-1)

    def _advantage_alignment(self, advantages: torch.Tensor, dones: torch.Tensor, weight=None, roles=None):
        # from advantage alignment repo (aa_terms); episode reset + teammate mask are ours
        if weight is None:
            weight = self.aa_weight
        if self.n_steps < 2:
            return advantages, torch.zeros_like(advantages)
        adv = advantages.reshape(self.n_steps, self.n_envs, self.agents_per_env)
        terminal = dones.reshape(self.n_steps, self.n_envs, self.agents_per_env).to(adv.device)

        team_mask = None
        if self.aa_team_of_role is not None and roles is not None:
            ape = self.agents_per_env
            team_lut = torch.tensor(self.aa_team_of_role, device=adv.device, dtype=torch.long)
            roles_grid = roles.reshape(self.n_steps, self.n_envs, ape).to(adv.device).long()
            roles_grid = roles_grid.clamp(0, team_lut.numel() - 1)
            team_grid = team_lut[roles_grid]  # [T, E, ape]
            same_team = (team_grid.unsqueeze(-1) == team_grid.unsqueeze(-2)).float()  # [T, E, ape, ape]
            eye = torch.eye(ape, device=adv.device).view(1, 1, ape, ape)
            team_mask = (same_team - eye).clamp_min(0.0)

        term =torch.zeros_like(adv)
        prefix = torch.zeros_like(adv[0])
        prefix_len = torch.zeros_like(adv[0])
        for t in range(1, self.n_steps):
            start = terminal[t - 1]
            prefix = torch.where(start, torch.zeros_like(prefix), prefix + adv[t - 1])
            prefix_len = torch.where(start, torch.zeros_like(prefix_len), prefix_len + 1.0)
            if team_mask is not None:
                other_current = torch.einsum('eij,ej->ei', team_mask[t], adv[t])
            else:
                other_current = adv[t].sum(dim=-1, keepdim=True) - adv[t]
            term[t] = prefix * other_current / prefix_len.clamp_min(1.0)
        return (adv + weight * term).reshape_as(advantages), term.reshape_as(advantages)

    def _lola_joint_taken_logp(self, buf, roles_full=None):
        T = self.n_steps
        B = self.n_agents
        dev = self.device

        obs_flat = buf.obs[:T].reshape(T * B, *buf.obs.shape[2:]).to(dev)
        rnn_flat = buf.rnn_states[:T].reshape(T * B, *buf.rnn_states.shape[2:]).to(dev)
        roles_grid = (buf.roles[:T] if roles_full is None else roles_full).to(dev)
        roles_flat = roles_grid.reshape(T * B)
        acts_flat = buf.actions[:T].reshape(T * B).to(dev)

        result, _ = self.policy.forward_sequence(
            obs_flat.unsqueeze(0), rnn_flat, roles=roles_flat.unsqueeze(0))
        logp_all = result['log_policy'][0]  # [T*B, n_actions]
        logp_taken = logp_all.gather(-1, acts_flat.unsqueeze(-1)).squeeze(-1).reshape(T, B)
        return logp_taken, roles_grid

    def _lola_episode_terms(self, logp_taken, dones, mask=None):
        T, B = logp_taken.shape
        ape = self.agents_per_env
        n_envs = B // ape
        dev = logp_taken.device

        contrib = logp_taken if mask is None else logp_taken * mask.to(logp_taken.dtype)
        step = contrib.reshape(T, n_envs, ape).sum(dim=-1)
        dones_env = dones.reshape(T, n_envs, ape).to(dev).any(dim=-1)

        cum = []
        discount = torch.empty(T, n_envs, device=dev)
        running = torch.zeros(n_envs, device=dev)
        elapsed = torch.zeros(n_envs, device=dev)
        for t in range(T):
            if t > 0:
                start = dones_env[t - 1]
                running = torch.where(start, torch.zeros_like(running), running)
                elapsed = torch.where(start, torch.zeros_like(elapsed), elapsed + 1.0)
            running = running + step[t]
            cum.append(running)
            discount[t] = self.gamma ** elapsed
        return step, torch.stack(cum), discount

    def _lola_agent_masks(self, roles_grid):
        T, B = roles_grid.shape
        if self.has_fixed_focal_agent:
            agent_idx = torch.arange(B, device=roles_grid.device)
            focal = self.red_agent_flat.to(roles_grid.device)[agent_idx // self.agents_per_env]
            shaper = (agent_idx == focal).unsqueeze(0).expand(T, B)
        else:
            shaper = roles_grid == 0
        shaper_roles = sorted(set(roles_grid[shaper].unique().tolist()) | {0})
        return shaper.float(), (~shaper).float(), shaper_roles

    def _lola_dice_loss(self, buf):
        # LOLA-DiCE surrogate from LOLA-DiCE repo (rpg.py); IS on same rollout, head rows only
        if self.n_steps < 2 or self.n_roles < 2:
            return None

        T = self.n_steps
        B = self.n_agents
        dev = self.device
        ape = self.agents_per_env

        logp_taken, roles_grid = self._lola_joint_taken_logp(buf)
        dones = buf.dones[:T].to(dev)
        rewards = buf.rewards[:T].to(dev)
        values = buf.values[:T].to(dev)

        def per_agent(x):
            return x.repeat_interleave(ape, dim=1)

        def box(x):
            return torch.exp((x - x.detach()).clamp(-10, 10))

        step_old, cum_old, discount = self._lola_episode_terms(logp_taken, dones)
        discount = per_agent(discount)

        shaper_mask, opp_mask, shaper_roles = self._lola_agent_masks(roles_grid)
        opp_surrogate = (
            discount * rewards * per_agent(box(cum_old))
            + discount * values * (1.0 - per_agent(box(step_old)))
        )
        opp_loss = -(opp_surrogate * opp_mask).sum() / opp_mask.sum().clamp_min(1.0)

        n_act = self.n_actions
        head_w = self.policy.policy_head.weight
        head_b = self.policy.policy_head.bias
        grad_w, grad_b = torch.autograd.grad(
            opp_loss, [head_w, head_b], create_graph=True, allow_unused=True, retain_graph=True)

        row_mask_w = torch.ones_like(head_w)
        for role in shaper_roles:
            row_mask_w[role * n_act:(role + 1) * n_act] = 0.0
        new_w = head_w if grad_w is None else head_w - self.lola_inner_lr * grad_w * row_mask_w
        new_b = head_b
        if head_b is not None and grad_b is not None:
            row_mask_b = torch.ones_like(head_b)
            for role in shaper_roles:
                row_mask_b[role * n_act:(role + 1) * n_act] = 0.0
            new_b = head_b - self.lola_inner_lr * grad_b * row_mask_b

        obs_flat = buf.obs[:T].reshape(T * B, *buf.obs.shape[2:]).to(dev)
        rnn_flat = buf.rnn_states[:T].reshape(T * B, *buf.rnn_states.shape[2:]).to(dev)
        acts_flat = buf.actions[:T].reshape(T * B).to(dev)
        encoder_output, _ = self.policy._forward_sequence(
            obs_flat.unsqueeze(0).to(self.policy.device), rnn_flat, None)
        policy_outputs_new = F.linear(encoder_output, new_w, new_b).reshape(
            1, T * B, self.n_roles, self.n_actions)
        if self.policy.shaper_adapter is not None:
            shaper_delta = self.policy.shaper_policy_head(self.policy.shaper_adapter(encoder_output))
            policy_outputs_new = policy_outputs_new.clone()
            policy_outputs_new[:, :, 0, :] = policy_outputs_new[:, :, 0, :] + shaper_delta
        log_policy_new = torch.log_softmax(policy_outputs_new, dim=-1)[0]

        from envs.rtg.models import extract_roles
        logp_all_new = extract_roles(log_policy_new.unsqueeze(0), roles_grid.reshape(1, T * B))[0]
        logp_taken_new = logp_all_new.gather(-1, acts_flat.unsqueeze(-1)).squeeze(-1).reshape(T, B)

        _, cum_opp_old, _ = self._lola_episode_terms(logp_taken, dones, mask=opp_mask)
        step_opp_new, cum_opp_new, _ = self._lola_episode_terms(logp_taken_new, dones, mask=opp_mask)
        weight = torch.exp((cum_opp_new - cum_opp_old.detach()).clamp(-10, 10))

        shaper_surrogate = (
            discount * rewards * per_agent(weight)
            - discount * values * per_agent(weight.detach() * (step_opp_new - step_opp_new.detach()))
        )
        lola_loss = -(shaper_surrogate * shaper_mask).sum() / shaper_mask.sum().clamp_min(1.0)
        return self.lola_weight * lola_loss

    def _focal_batch_mask(self, batch_agent_idx, batch_roles):
        if not self.has_fixed_focal_agent:
            return (batch_roles == 0).detach().cpu()
        env_idx = batch_agent_idx // self.agents_per_env
        focal_idx = self.red_agent_flat[env_idx].to(batch_agent_idx.device)
        return batch_agent_idx == focal_idx

    def _temper_log_liks(self, log_liks: torch.Tensor) -> torch.Tensor:
        log_liks = torch.nan_to_num(log_liks, nan=-20.0, posinf=20.0, neginf=-20.0)
        log_liks = log_liks.clamp(-20.0, 20.0)
        if self.bos_belief_temperature == 1.0:
            return log_liks
        return log_liks / self.bos_belief_temperature

    def _z_true_vec_np(self):
        """Shape [n_envs]. Latent hypothesis index per sub-env when env exposes it."""
        fn = getattr(self.env, "get_z_true_per_env", None)
        if callable(fn):
            z = np.asarray(fn(), dtype=np.int64).reshape(self.n_envs)
            return z
        return np.full(self.n_envs, int(self.z_true), dtype=np.int64)

    def _z_imp_vec_np(self):
        zi = (self._z_true_vec_np() + 1) % self.n_roles
        return zi.astype(np.int64)

    def _role0_agents_from_roles(self, roles_flat):
        """Return the current role-0/shaper agent index for each env."""
        red_idx, _ = build_env_agent_map(roles_flat, self.n_envs, self.agents_per_env)
        return torch.from_numpy(red_idx)

    def _speaker_agents_from_z(self, z_vec_np):
        """Return physical speaker agent indices for speaker-identity hypotheses."""
        if self.speaker_agent_flat is None:
            return self.red_agent_flat
        idx = torch.from_numpy(np.asarray(z_vec_np, dtype=np.int64))
        env_idx = torch.arange(self.n_envs)
        return self.speaker_agent_flat[env_idx, idx]

    def _listener_messages_from_obs(self, listener_obs_u8):
        """Decode three discrete listener-heard messages from uint8 observations."""
        msg_n = int(getattr(self.env, "speaker_message_n", 3) or 3)
        hd = listener_obs_u8[:, 3:6].float() / 255.0
        return torch.clamp((hd * float(msg_n - 1)).round(), 0, msg_n - 1).long()

    def _belief_uses_next_obs(self):
        """Signaling observations contain messages produced by the previous action."""
        return self.bos_belief_mode in ("slot", "speaker")

    def _speaker_identity_log_liks(self, speaker_role_log_policy, messages, detach_honest=False):
        E, R = self.n_envs, self.n_roles
        dev = speaker_role_log_policy.device

        msg_n = int(getattr(self.env, "speaker_message_n", 3) or 3)
        lp_h = speaker_role_log_policy[:, :, 0, :msg_n]
        lp_d = speaker_role_log_policy[:, :, 2, :msg_n].detach()
        if detach_honest:
            lp_h = lp_h.detach()

        pi_h = lp_h.exp()
        pi_d = lp_d.exp()
        pi_h = pi_h / (pi_h.sum(-1, keepdim=True) + 1e-8)
        pi_d = pi_d / (pi_d.sum(-1, keepdim=True) + 1e-8)

        p_corrupt = float(getattr(self.env, "signal_corrupt_prob", 0.0) or 0.0)
        p_u = 1.0 / float(msg_n)
        idx_e = torch.arange(E, device=dev)[:, None]
        idx_s = torch.arange(R, device=dev)[None, :]
        p_h_m = (1.0 - p_corrupt) * pi_h[idx_e, idx_s, messages] + p_corrupt * p_u
        p_d_m = (1.0 - p_corrupt) * pi_d[idx_e, idx_s, messages] + p_corrupt * p_u
        p_h_m = p_h_m.clamp_min(1e-8)
        p_d_m = p_d_m.clamp_min(1e-8)

        log_liks = torch.zeros(E, R, device=dev)
        for z in range(R):
            ll = torch.log(p_h_m[:, z])
            for s in range(R):
                if s != z:
                    ll = ll + torch.log(p_d_m[:, s])
            log_liks[:, z] = ll
        return log_liks

    def _train_observer_table(self):
        z_all = np.concatenate([z for z, _ in self._observer_table_buffer])
        sig_all = np.concatenate([s for _, s in self._observer_table_buffer])
        self._observer_table_buffer = []

        z_t = torch.from_numpy(z_all).long().to(self.device)
        sig_t = torch.from_numpy(sig_all).long().to(self.device)
        a_ally = self._observer_decode[0][sig_t]
        a_adv = self._observer_decode[1][sig_t]

        log_probs = F.log_softmax(self.observer_channel_table, dim=-1)  # [2, R, n_ch]
        ally_ll = log_probs[0][z_t, a_ally]
        adv_ll = log_probs[1][z_t, a_adv]
        loss = -(ally_ll.mean() + adv_ll.mean())

        self.observer_table_optimizer.zero_grad()
        loss.backward()
        self.observer_table_optimizer.step()

        with torch.no_grad():
            pushed = F.log_softmax(self.observer_channel_table, dim=-1).cpu().numpy()
        self.env.set_observer_channel_log_liks(pushed[0], pushed[1])

    @torch.no_grad()
    def collect_rollout(self, rnn_states, beliefs):
        buf  = RolloutBuffer(
            n_steps=self.n_steps, n_agents=self.n_agents, n_envs=self.n_envs,
            obs_shape=self.obs_shape, memory_units=self.memory_units, n_roles=self.n_roles,
            n_obs=self.n_obs_per_env,
        )
        obs = self._current_obs
        E, R = self.n_envs, self.n_roles
        # beliefs reset only at episode end
        initial_beliefs = self._initial_per_observer_beliefs()
        if self._per_obs_beliefs is None or self._per_obs_beliefs.shape != initial_beliefs.shape:
            self._per_obs_beliefs = initial_beliefs
        per_obs_beliefs = self._per_obs_beliefs.clone()
        beliefs = per_obs_beliefs.mean(dim=1).to(beliefs.device)
        ep_red_wins, ep_blue_wins, seen = [], [], set()
        ep_red_scores, ep_blue_scores = [], []   # per-episode team scores (BBM Fig 5 equivalent)
        ep_coop_success, ep_coop_return = [], []  # signaling: terminal cooperative metrics
        ep_coin_red_returns, ep_coin_blue_returns = [], []
        b_true_vals, b_imp_vals = [], []          # avg over all observers
        b_true_blue_vals = []                     # blue-only belief (matches BBM Fig 4)
        b_true_prior_vals, b_imp_prior_vals = [], []
        oracle_b_true_vals, oracle_b_true_obs0_vals, oracle_b_true_obs1_vals = [], [], []
        oracle_belief_l1_vals = []
        avalon_ally_b_true_vals, avalon_resistance_b_true_vals = [], []
        avalon_resistance_role_vals = {role: [] for role in range(2, self.n_roles)}
        avalon_shaper_team_approvals, avalon_other_team_approvals = [], []
        avalon_focal_sabotages, avalon_focal_quest_actions = [], []

        for _ in range(self.n_steps):
            roles = self.env.get_roles()
            zt_np = self._z_true_vec_np()
            zi_np = self._z_imp_vec_np()
            if self.bos_belief_mode == "speaker":
                red_agent_flat_step = self._speaker_agents_from_z(zt_np)
            else:
                red_agent_flat_step = self.red_agent_flat
            obs_t   = torch.from_numpy(obs).unsqueeze(0)
            roles_t = torch.from_numpy(roles).unsqueeze(0).to(torch.int64)
            result, new_rnn = self.policy.forward_sequence(obs_t, rnn_states, roles=roles_t)

            log_policy      = result['log_policy'][0]
            role_log_policy = result['role_log_policy'][0]
            role_ext_value  = result['role_ext_value'][0]

            role_idx = torch.from_numpy(roles.astype(np.int64))
            values   = role_ext_value[range(self.n_agents), role_idx]
            if self.use_belief_value_critic:
                belief_values = self._belief_value(obs, per_obs_beliefs, red_agent_flat_step)
                values = values.clone()
                values[red_agent_flat_step.to(values.device)] = belief_values.to(values.device)

            actions_np = sample_action_from_logp(log_policy.cpu().numpy())
            log_probs  = log_policy[range(self.n_agents),
                                    torch.from_numpy(actions_np)].cpu()

            if self.observer_channel_table is not None:
                shaper_signal = actions_np[red_agent_flat_step.numpy()]  # [E]
                self._observer_table_buffer.append((zt_np.copy(), shaper_signal.copy()))

            if "avalon5" in self.scenario:
                # vote: action 1 = reject
                flat_obs = obs.reshape(self.n_agents, -1)
                for e in range(E):
                    base = e * self.agents_per_env
                    phase = int(flat_obs[base, 61])
                    team_mask = int(flat_obs[base, 65])
                    shaper_local = int(red_agent_flat_step[e]) - base
                    if phase == 255 and team_mask & (1 << shaper_local):
                        avalon_focal_quest_actions.append(1.0)
                        avalon_focal_sabotages.append(float(actions_np[int(red_agent_flat_step[e])] == 1))
                    if phase != 128:
                        continue
                    approvals = [
                        float(actions_np[base + local_i] != 1)
                        for local_i in range(self.agents_per_env)
                        if int(roles[base + local_i]) >= 2
                    ]
                    if not approvals:
                        continue
                    if team_mask & (1 << shaper_local):
                        avalon_shaper_team_approvals.extend(approvals)
                    else:
                        avalon_other_team_approvals.extend(approvals)

            # Priors live on `beliefs.device`; posteriors from _belief_step are often CPU — index separately.
            idx_b = torch.arange(E, device=beliefs.device)
            zt_b = torch.from_numpy(zt_np).long().to(beliefs.device)
            zi_b = torch.from_numpy(zi_np).long().to(beliefs.device)
            b_true_prior_vals.append(float(beliefs[idx_b, zt_b].mean()))
            b_imp_prior_vals.append(float(beliefs[idx_b, zi_b].mean()))
            belief_prior = beliefs

            set_likelihood_fn = getattr(self.env, "set_focal_role_log_probs", None)
            if callable(set_likelihood_fn):
                focal_log_probs = role_log_policy[red_agent_flat_step.to(self.device)]
                set_likelihood_fn(focal_log_probs.detach().cpu().numpy())

            obs_new, rewards_np, dones_np, infos = self.env.step(actions_np)

            # signaling: update from next obs
            belief_obs = obs_new if self._belief_uses_next_obs() else obs
            observer_log_liks = self._rollout_observer_log_liks(
                obs, roles, rnn_states, role_log_policy, actions_np, red_agent_flat_step
            )
            new_beliefs, new_beliefs_per_obs = self._belief_step(
                belief_prior, role_log_policy, actions_np, belief_obs,
                per_obs_prior=per_obs_beliefs,
                observer_log_liks=observer_log_liks)

            oracle_fn = getattr(self.env, "get_last_oracle_observer_beliefs", None)
            if callable(oracle_fn):
                oracle_beliefs = torch.as_tensor(oracle_fn(), dtype=torch.float32)
                if oracle_beliefs.ndim == 3 and oracle_beliefs.shape[0] == E:
                    oracle_n_obs = min(oracle_beliefs.shape[1], new_beliefs_per_obs.shape[1])
                    if oracle_n_obs > 0:
                        env_idx = torch.arange(E)
                        zt_oracle = torch.from_numpy(zt_np).long()
                        oracle_true = oracle_beliefs[:, :oracle_n_obs][
                            env_idx[:, None], torch.arange(oracle_n_obs)[None, :], zt_oracle[:, None]
                        ]
                        oracle_b_true_vals.append(float(oracle_true.mean()))
                        oracle_b_true_obs0_vals.append(float(oracle_true[:, 0].mean()))
                        if oracle_n_obs > 1:
                            oracle_b_true_obs1_vals.append(float(oracle_true[:, 1].mean()))
                        surrogate = new_beliefs_per_obs[:, :oracle_n_obs].float()
                        oracle_belief_l1_vals.append(float((surrogate - oracle_beliefs[:, :oracle_n_obs]).abs().mean()))

            idx_nb = torch.arange(E, device=new_beliefs.device)
            zt_nb = torch.from_numpy(zt_np).long().to(new_beliefs.device)
            zi_nb = torch.from_numpy(zi_np).long().to(new_beliefs.device)

            # BBM per-step bonus: log(b_prior[z_true]) - log(b_posterior[z_true]) for red agents only
            bbm_step = None
            if self.use_bbm_intrinsic:
                b_prior_zt = beliefs[idx_b, zt_b]
                b_post_zt  = new_beliefs[idx_nb, zt_nb].to(beliefs.device)
                bonus_env  = (torch.log(b_prior_zt + 1e-8) - torch.log(b_post_zt + 1e-8)).cpu().numpy()
                bbm_step   = np.zeros(self.n_agents, dtype=np.float32)
                for _e in range(self.n_envs):
                    bbm_step[int(red_agent_flat_step[_e])] = bonus_env[_e]

            b_true_vals.append(float(new_beliefs[idx_nb, zt_nb].mean()))
            b_imp_vals.append(float(new_beliefs[idx_nb, zi_nb].mean()))
            # Blue-only belief: matches BBM Fig 4 (blue team's belief in red's true role)
            blue_b_true = [
                float(new_beliefs_per_obs[e, idx, zt_np[e]])
                for e, idx in enumerate(self.blue_obs_local_idx) if idx is not None
            ]
            if blue_b_true:
                b_true_blue_vals.append(float(np.mean(blue_b_true)))
            avalon_metrics = self._avalon_belief_metrics(new_beliefs_per_obs)
            if avalon_metrics:
                avalon_ally_b_true_vals.append(avalon_metrics["ally_b_true"])
                avalon_resistance_b_true_vals.append(avalon_metrics["resistance_b_true"])
                for role in avalon_resistance_role_vals:
                    avalon_resistance_role_vals[role].append(
                        avalon_metrics[f"resistance_role{role}_b_true"]
                    )

            buf.store(
                obs=obs, actions=actions_np, rewards_ext=rewards_np.astype(np.float32),
                dones=dones_np, roles=roles, log_probs=log_probs.detach(),
                values=values.detach().cpu(), rnn_states=rnn_states.cpu(),
                beliefs=belief_prior, lstm_h_red=rnn_states[red_agent_flat_step, 0, :].cpu(),
                bbm_rewards=bbm_step, per_obs_beliefs=per_obs_beliefs,
                per_obs_next_beliefs=new_beliefs_per_obs, next_obs=obs_new,
            )

            for agent_i, info in enumerate(infos):
                if 'outcome' not in info:
                    continue
                outcome = info['outcome']
                if outcome == 'none':
                    continue
                game_i = agent_i // self.agents_per_env
                if game_i in seen:
                    continue
                seen.add(game_i)
                # RTG:   'general_killed'=red wins, 'general_rescued'/'blue_win'=blue wins
                # Trust: 'exploited'=red wins, 'cooperated'=blue wins
                # Signaling: 'cooperated' / 'signaling_timeout' — team success → blue_win slot
                ep_red_wins.append(1.0 if outcome in ('general_killed', 'exploited') else 0.0)
                ep_blue_wins.append(1.0 if outcome in ('general_rescued', 'blue_win', 'cooperated') else 0.0)
                # game_scores: [rounds, n_teams] cumulative scores; 0=red, 1=green, 2=blue
                gs = info.get('game_scores')
                if gs is not None:
                    gs = np.asarray(gs)
                    ep_red_scores.append(float(gs[:, 0].sum()))
                    ep_blue_scores.append(float(gs[:, 2].sum()))
                if 'coop_success' in info:
                    ep_coop_success.append(float(info['coop_success']))
                if 'episode_return' in info:
                    ep_coop_return.append(float(info['episode_return']))
                if 'coin_red_return' in info:
                    ep_coin_red_returns.append(float(info['coin_red_return']))
                if 'coin_blue_return' in info:
                    ep_coin_blue_returns.append(float(info['coin_blue_return']))
            seen.clear()

            rnn_states = new_rnn.detach()
            rnn_states[torch.from_numpy(dones_np)] = 0.0
            new_beliefs, _, new_beliefs_per_obs = reset_done_env_beliefs(
                new_beliefs, dones_np, self.n_envs, self.agents_per_env,
                per_obs_beliefs=new_beliefs_per_obs)
            reset_priors = self._initial_per_observer_beliefs()
            done_env = torch.from_numpy(dones_np.reshape(self.n_envs, self.agents_per_env).any(axis=1))
            if done_env.any():
                new_beliefs_per_obs[done_env] = reset_priors[done_env]
                new_beliefs[done_env] = reset_priors[done_env].mean(dim=1).to(new_beliefs.device)
            per_obs_beliefs = new_beliefs_per_obs if new_beliefs_per_obs is not None else per_obs_beliefs
            beliefs, obs = new_beliefs, obs_new

        self._current_obs = obs
        self._per_obs_beliefs = per_obs_beliefs.detach().cpu()
        self._last_rollout_diagnostics = {
            "avalon_ally_b_true": float(np.nanmean(avalon_ally_b_true_vals)) if avalon_ally_b_true_vals else float("nan"),
            "avalon_resistance_b_true": float(np.nanmean(avalon_resistance_b_true_vals)) if avalon_resistance_b_true_vals else float("nan"),
            "avalon_resistance_approve_shaper_team": float(np.mean(avalon_shaper_team_approvals)) if avalon_shaper_team_approvals else float("nan"),
            "avalon_resistance_approve_other_team": float(np.mean(avalon_other_team_approvals)) if avalon_other_team_approvals else float("nan"),
            "avalon_focal_spy_sabotage_rate": (
                float(np.sum(avalon_focal_sabotages) / np.sum(avalon_focal_quest_actions))
                if avalon_focal_quest_actions else float("nan")
            ),
        }
        for role, values in avalon_resistance_role_vals.items():
            self._last_rollout_diagnostics[f"avalon_resistance_role{role}_b_true"] = \
                float(np.nanmean(values)) if values else float("nan")

        if self.observer_channel_table is not None and self._observer_table_buffer:
            with torch.enable_grad():
                self._train_observer_table()

        return (
            buf,
            rnn_states,
            beliefs,
            ep_red_wins,
            ep_blue_wins,
            ep_red_scores,
            ep_blue_scores,
            b_true_vals,
            b_true_blue_vals,
            b_imp_vals,
            b_true_prior_vals,
            b_imp_prior_vals,
            ep_coop_success,
            ep_coop_return,
            ep_coin_red_returns,
            ep_coin_blue_returns,
            oracle_b_true_vals,
            oracle_b_true_obs0_vals,
            oracle_b_true_obs1_vals,
            oracle_belief_l1_vals,
        )

    def _observer_selected_role_log_liks(self, buf, detach=False):
        # log p(a_red | obs_red, z) [T, E, n_obs, R]
        T, E, R = self.n_steps, self.n_envs, self.n_roles
        if self.n_obs_per_env == 0:
            return torch.zeros(T, E, 0, R, device=self.device)

        dev = self.device
        red_agents_flat = self.red_agent_flat  # [E]
        obs_agents_flat = self.obs_agent_flat  # [E, n_obs], CPU index tensor for rollout buffer
        n_obs = self.n_obs_per_env
        red_roles = torch.zeros(1, E, dtype=torch.int64, device=dev)
        obs_mode = self.bos_likelihood_obs_mode
        pred_obs_u8 = None
        if obs_mode in ("estimated", "esti", "learned"):
            pred_obs_u8 = self._predict_observer_obs_u8(buf)
            if pred_obs_u8 is None:
                obs_mode = "canonical"
        all_log_liks = []

        for t in range(T):
            a_t = buf.actions[t, red_agents_flat].to(dev)                   # [E]
            if obs_mode in ("estimated", "esti", "learned"):
                obs_t = pred_obs_u8[t].reshape(E * n_obs, *self.obs_shape).unsqueeze(0).to(dev)
                red_rnn_t = buf.rnn_states[t, red_agents_flat].to(dev)
                rnn_t = red_rnn_t.repeat_interleave(n_obs, dim=0)            # [E*n_obs, 2, H]
                roles_t = torch.zeros(1, E * n_obs, dtype=torch.int64, device=dev)
                result, _ = self.policy.forward_sequence(obs_t, rnn_t, roles=roles_t)
                rlp = result['role_log_policy'][0].reshape(E, n_obs, R, self.n_actions)
                a_rep = a_t.view(E, 1, 1, 1).expand(E, n_obs, R, 1)
                step_log_liks = rlp.gather(-1, a_rep).squeeze(-1)            # [E, n_obs, R]
            elif obs_mode in ("observer", "observer_proxy"):
                # observer mode: use observer's obs
                obs_flat = obs_agents_flat.reshape(E * n_obs)
                obs_t = buf.obs[t, obs_flat].unsqueeze(0).to(dev)            # [1, E*n_obs, *obs]
                red_rnn_t = buf.rnn_states[t, red_agents_flat].to(dev)
                rnn_t = red_rnn_t.repeat_interleave(n_obs, dim=0)            # [E*n_obs, 2, H]
                roles_t = torch.zeros(1, E * n_obs, dtype=torch.int64, device=dev)
                result, _ = self.policy.forward_sequence(obs_t, rnn_t, roles=roles_t)
                rlp = result['role_log_policy'][0].reshape(E, n_obs, R, self.n_actions)
                a_rep = a_t.view(E, 1, 1, 1).expand(E, n_obs, R, 1)
                step_log_liks = rlp.gather(-1, a_rep).squeeze(-1)            # [E, n_obs, R]
            else:
                red_obs_t = buf.obs[t, red_agents_flat].unsqueeze(0).to(dev) # [1, E, *obs]
                red_rnn_t = buf.rnn_states[t, red_agents_flat].to(dev)       # [E, 2, H]
                result, _ = self.policy.forward_sequence(red_obs_t, red_rnn_t, roles=red_roles)
                red_rlp = result['role_log_policy'][0]                       # [E, R, A]

                channel_liks = self._channel_decode_log_liks(red_rlp, a_t)
                if channel_liks is not None:
                    step_log_liks = channel_liks
                else:
                    step_log_liks = red_rlp.gather(
                        -1, a_t.view(E, 1, 1).expand(E, R, 1)
                    ).squeeze(-1).unsqueeze(1)                                  # [E, 1, R]
                    step_log_liks = self._apply_observer_visibility_weights(
                        step_log_liks, buf.obs[t], a_t)
            if detach:
                step_log_liks = step_log_liks.detach()
            all_log_liks.append(step_log_liks)

        return torch.stack(all_log_liks, dim=0)  # [T, E, n_obs, R]

    @torch.no_grad()
    def _rollout_observer_log_liks(self, obs, roles, rnn_states, role_log_policy,
                                   actions_np, red_agent_flat):
        E, R, n_obs = self.n_envs, self.n_roles, self.n_obs_per_env
        dev = self.device
        red_agent_flat = red_agent_flat.to(torch.long)
        red_actions = torch.as_tensor(actions_np, device=dev)[red_agent_flat]
        obs_mode = self.bos_likelihood_obs_mode

        if obs_mode in ("estimated", "esti", "learned") and self.obs_estimator is not None:
            red_obs = torch.as_tensor(obs, device=dev)[red_agent_flat]
            red_obs = red_obs.float().reshape(E, -1).div(255.0)
            red_obs = red_obs.unsqueeze(1).expand(E, n_obs, -1)
            observer_roles = torch.as_tensor(
                roles[self.obs_agent_flat.numpy()], dtype=torch.long, device=dev
            ).reshape(E, n_obs)
            observer_slots = torch.arange(n_obs, device=dev).view(1, n_obs).expand(E, n_obs)
            pred_obs = self.obs_estimator(
                red_obs.unsqueeze(0), observer_roles.unsqueeze(0), observer_slots
            ).squeeze(0)
            pred_obs = torch.nan_to_num(pred_obs, nan=0.5, posinf=1.0, neginf=0.0)
            pred_obs = (pred_obs.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
            pred_obs = pred_obs.reshape(E * n_obs, *self.obs_shape).unsqueeze(0)
            focal_rnn = rnn_states[red_agent_flat].to(dev).repeat_interleave(n_obs, dim=0)
            hypothesis_roles = torch.zeros(1, E * n_obs, dtype=torch.long, device=dev)
            result, _ = self.policy.forward_sequence(pred_obs, focal_rnn, roles=hypothesis_roles)
            rlp = result["role_log_policy"][0].reshape(E, n_obs, R, self.n_actions)
            return rlp.gather(
                -1, red_actions.view(E, 1, 1, 1).expand(E, n_obs, R, 1)
            ).squeeze(-1)

        if obs_mode in ("observer", "observer_proxy"):
            observer_flat = self.obs_agent_flat.reshape(E * n_obs)
            observer_obs = torch.as_tensor(obs)[observer_flat].unsqueeze(0).to(dev)
            focal_rnn = rnn_states[red_agent_flat].to(dev).repeat_interleave(n_obs, dim=0)
            hypothesis_roles = torch.zeros(1, E * n_obs, dtype=torch.long, device=dev)
            result, _ = self.policy.forward_sequence(observer_obs, focal_rnn, roles=hypothesis_roles)
            rlp = result["role_log_policy"][0].reshape(E, n_obs, R, self.n_actions)
            return rlp.gather(
                -1, red_actions.view(E, 1, 1, 1).expand(E, n_obs, R, 1)
            ).squeeze(-1)

        focal_rlp = role_log_policy[red_agent_flat.to(role_log_policy.device)]
        channel_liks = self._channel_decode_log_liks(focal_rlp, red_actions)
        if channel_liks is not None:
            return channel_liks
        canonical = focal_rlp.gather(
            -1, red_actions.view(E, 1, 1).expand(E, R, 1)
        ).squeeze(-1).unsqueeze(1).expand(E, n_obs, R)
        return self._apply_observer_visibility_weights(canonical, obs, red_actions)

    def _obs_estimator_inputs_targets(self, buf):
        T, E, n_obs = self.n_steps, self.n_envs, self.n_obs_per_env
        dev = self.device
        if n_obs == 0:
            return None
        red_obs = buf.obs[:, self.red_agent_flat].to(dev).float().view(T, E, -1) / 255.0
        red_obs = red_obs.unsqueeze(2).expand(T, E, n_obs, red_obs.shape[-1])
        obs_agents = self.obs_agent_flat
        obs_flat = obs_agents.reshape(E * n_obs)
        obs_roles = buf.roles[:, obs_flat].to(dev).reshape(T, E, n_obs).long()
        obs_slot = torch.arange(n_obs, device=dev).view(1, n_obs).expand(E, n_obs)
        target = buf.obs[:, obs_flat].to(dev).float().reshape(T, E, n_obs, -1) / 255.0
        return red_obs, obs_roles, obs_slot, target

    def _train_obs_estimator(self, buf):
        if self.obs_estimator is None or self.obs_estimator_optimizer is None:
            return {}
        data = self._obs_estimator_inputs_targets(buf)
        if data is None:
            return {}
        red_obs, obs_roles, obs_slot, target = data
        losses = []
        for _ in range(self.bos_obs_estimator_updates):
            pred = self.obs_estimator(red_obs, obs_roles, obs_slot)
            loss = F.mse_loss(pred, target)
            if not torch.isfinite(loss):
                self.obs_estimator_optimizer.zero_grad(set_to_none=True)
                continue
            self.obs_estimator_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.obs_estimator.parameters(), self.max_grad_norm)
            self.obs_estimator_optimizer.step()
            losses.append(float(loss.detach().item()))
        with torch.no_grad():
            pred = self.obs_estimator(red_obs, obs_roles, obs_slot)
            pred = torch.nan_to_num(pred, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            mae = (pred - target).abs().mean()
        return {
            "bos_obs_estimator_loss": float(np.mean(losses)) if losses else 0.0,
            "bos_obs_estimator_mae": float(mae.item()),
        }

    @torch.no_grad()
    def _predict_observer_obs_u8(self, buf):
        if self.obs_estimator is None:
            return None
        data = self._obs_estimator_inputs_targets(buf)
        if data is None:
            return None
        red_obs, obs_roles, obs_slot, _target = data
        pred = self.obs_estimator(red_obs, obs_roles, obs_slot)
        pred = torch.nan_to_num(pred, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        return torch.clamp((pred * 255.0).round(), 0, 255).to(torch.uint8).cpu()

    def belief_planning_loss(self, buf, ref_log_liks=None, clip_eps=None, return_aux=False):
        T, E, R, k = self.n_steps, self.n_envs, self.n_roles, self.k
        dev = self.device

        if self.n_obs_per_env == 0:
            zero = torch.zeros(1, device=dev)
            if return_aux:
                return zero, {"dec_clip_frac": 0.0, "dec_approx_kl": 0.0}
            return zero

        if self.bos_belief_mode == "slot":
            curr_log_liks = self._slot_selected_log_liks(buf, detach=False)
        elif self.bos_belief_mode == "speaker":
            curr_log_liks = self._speaker_selected_log_liks(buf, detach=False)
        else:
            curr_log_liks = self._observer_selected_role_log_liks(buf, detach=False)
        curr_log_liks = torch.nan_to_num(curr_log_liks, nan=-20.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        n_obs = self.n_obs_per_env
        # V at t+k
        n_windows = min(curr_log_liks.shape[0] - k + 1, T - k)
        if n_windows <= 0:
            zero = torch.zeros(1, device=dev)
            if return_aux:
                return zero, {"dec_clip_frac": 0.0, "dec_approx_kl": 0.0}
            return zero
        clip_enabled = ref_log_liks is not None and clip_eps is not None
        if clip_enabled:
            ref_log_liks = ref_log_liks.to(dev)

        total_loss = torch.zeros(1, device=dev)
        uniform = torch.full((E, R), 1.0 / R, device=dev)
        coeff_mode = self.bos_correction_mode == "coeff"
        coeff_abs_vals = []
        coeff_rms_vals = []
        entropy_frac_vals = []
        coeff_gate_vals = []

        clip_hits = torch.zeros(1, device=dev)
        approx_kl = torch.zeros(1, device=dev)
        clip_count = 0

        for t in range(n_windows):
            b_j = buf.per_obs_beliefs[t, :, :n_obs, :R].to(dev).clone()  # [E, n_obs, R]
            if b_j.shape[-1] < R:
                b_j = uniform.unsqueeze(1).expand(E, n_obs, R).clone()
            b_j = self._regularize_belief(b_j)
            raw_steps = []

            for i in range(k):
                raw_step_log_liks = curr_log_liks[t + i]  # [E, n_obs, R]
                raw_steps.append(raw_step_log_liks)
                step_log_liks = raw_step_log_liks
                if clip_enabled:
                    step_ref = ref_log_liks[t + i]
                    log_ratio = step_log_liks - step_ref
                    ratio = log_ratio.exp()
                    clipped_ratio = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
                    step_log_liks = step_ref + torch.log(clipped_ratio + 1e-8)

                    clip_hits = clip_hits + (ratio - clipped_ratio).abs().gt(1e-8).float().mean()
                    approx_kl = approx_kl + ((ratio - 1.0) - log_ratio).mean()
                    clip_count += 1

                step_log_liks = self._temper_log_liks(step_log_liks)
                log_unnorm = step_log_liks + torch.log(b_j.clamp(min=1e-8))
                b_j = self._regularize_belief(torch.softmax(log_unnorm, dim=-1))  # [E, n_obs, R]
                b_j = self._freeze_known_observer_beliefs(b_j)

            obs_idx = t + k
            if self.bos_belief_mode == "speaker":
                red_k = self._role0_agents_from_roles(buf.roles[obs_idx].cpu().numpy())
            else:
                red_k = self.red_agent_flat
            obs_k = buf.obs[obs_idx, red_k].to(dev).view(E, -1)   # [E, obs_dim]

            # B = [b_obs0 | ... | b_obs_n]
            B_flat = b_j.reshape(E, n_obs * R)           # [E, n_obs * R]
            v_joint = self.belief_critic(obs_k, B_flat)  # [E, 1]
            window_loss = -v_joint.mean()
            if not torch.isfinite(window_loss):
                continue

            if coeff_mode:
                # coeff mode: detached dV/dlogp coeffs
                coeffs = torch.autograd.grad(
                    window_loss,
                    raw_steps,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )

                entropy = -(b_j.clamp_min(1e-8) * b_j.clamp_min(1e-8).log()).sum(dim=-1)
                entropy_frac = entropy / max(float(np.log(R)), 1e-8)  # [E, n_obs]
                entropy_frac_vals.append(float(entropy_frac.detach().mean().item()))
                if self.bos_min_entropy_frac > 0.0:
                    gate = (entropy_frac >= self.bos_min_entropy_frac).float().unsqueeze(-1)
                else:
                    gate = torch.ones(E, n_obs, 1, device=dev)
                coeff_gate_vals.append(float(gate.detach().mean().item()))

                coeff_loss = torch.zeros(1, device=dev)
                for raw_step, coeff in zip(raw_steps, coeffs):
                    if coeff is None:
                        continue
                    coeff = coeff.detach() * gate
                    if self.bos_shaper_role_only:
                        # only z=0 path
                        mask = torch.zeros_like(coeff)
                        mask[..., 0:1] = 1.0
                        coeff = coeff * mask
                    if self.bos_coeff_normalize:
                        coeff = coeff / (coeff.pow(2).mean().sqrt() + 1e-8)
                    if self.bos_coeff_clip > 0.0:
                        coeff = coeff.clamp(-self.bos_coeff_clip, self.bos_coeff_clip)
                    coeff = torch.nan_to_num(coeff, nan=0.0, posinf=0.0, neginf=0.0)
                    coeff_abs_vals.append(float(coeff.abs().mean().item()))
                    coeff_rms_vals.append(float(coeff.pow(2).mean().sqrt().item()))
                    coeff_loss = coeff_loss + (coeff * raw_step).sum() / max(1, E * n_obs * R)
                total_loss = total_loss + coeff_loss / max(1, len(raw_steps))
            else:
                total_loss = total_loss + window_loss

        loss = total_loss / n_windows
        if not torch.isfinite(loss):
            loss = torch.zeros(1, device=dev)
        if not return_aux:
            return loss

        coeff_aux = {
            "bos_coeff_abs": float(np.mean(coeff_abs_vals)) if coeff_abs_vals else 0.0,
            "bos_coeff_rms": float(np.mean(coeff_rms_vals)) if coeff_rms_vals else 0.0,
            "bos_belief_entropy_frac": float(np.mean(entropy_frac_vals)) if entropy_frac_vals else 0.0,
            "bos_coeff_gate_frac": float(np.mean(coeff_gate_vals)) if coeff_gate_vals else 0.0,
        }
        if clip_count == 0:
            return loss, {"dec_clip_frac": 0.0, "dec_approx_kl": 0.0, **coeff_aux}

        return loss, {
            "dec_clip_frac": float((clip_hits / clip_count).item()),
            "dec_approx_kl": float((approx_kl / clip_count).item()),
            **coeff_aux,
        }

    def _mask_deception_gradients(self):
        """Restrict BOS update to role-0 policy rows and optionally to the policy head only."""
        n_act = self.n_actions
        head_w = self.policy.policy_head.weight
        head_b = self.policy.policy_head.bias

        head_role0_grad = 0.0
        head_other_grad = 0.0
        nonhead_grad_pre = 0.0
        adapter_params = set()
        for module_name in ("shaper_adapter", "shaper_policy_head"):
            module = getattr(self.policy, module_name, None)
            if module is not None:
                adapter_params.update(module.parameters())

        if head_w.grad is not None:
            head_role0_grad += float(head_w.grad[:n_act].norm().item())
            head_other_grad += float(head_w.grad[n_act:].norm().item())
            if not self.bos_update_all_role_heads:
                head_w.grad[n_act:] = 0.0

        if head_b is not None and head_b.grad is not None:
            head_role0_grad += float(head_b.grad[:n_act].norm().item())
            head_other_grad += float(head_b.grad[n_act:].norm().item())
            if not self.bos_update_all_role_heads:
                head_b.grad[n_act:] = 0.0

        nonhead_sq = 0.0
        for p in self.policy.parameters():
            if p is head_w or p is head_b:
                continue
            if p.grad is not None:
                g = float(p.grad.norm().item())
                nonhead_sq += g * g
                if self.dec_head_only and p not in adapter_params:
                    p.grad.zero_()
        nonhead_grad_pre = float(np.sqrt(nonhead_sq))

        return {
            "head_role0_grad": head_role0_grad,
            "head_other_grad": head_other_grad,
            "nonhead_grad_pre": nonhead_grad_pre,
        }

    @staticmethod
    def _grad_norm(parameters):
        sq = 0.0
        for p in parameters:
            if p.grad is not None:
                g = float(p.grad.norm().item())
                sq += g * g
        return float(np.sqrt(sq))

    @staticmethod
    def _clone_grads(parameters):
        return [None if p.grad is None else p.grad.detach().clone() for p in parameters]

    @staticmethod
    def _set_grads(parameters, grads):
        for p, g in zip(parameters, grads):
            if g is None:
                p.grad = None
            else:
                p.grad = g.detach().clone()

    @staticmethod
    def _add_grads(lhs, rhs):
        out = []
        for g_l, g_r in zip(lhs, rhs):
            if g_l is None and g_r is None:
                out.append(None)
            elif g_l is None:
                out.append(g_r)
            elif g_r is None:
                out.append(g_l)
            else:
                out.append(g_l + g_r)
        return out

    def _channel_decode_log_liks(self, red_rlp, red_actions):
        # belief relay: per-observer channel log-liks
        decode_fn = getattr(self.env, "get_observer_channel_decode", None)
        if not callable(decode_fn):
            return None
        decode = np.asarray(decode_fn())
        if decode.ndim != 2:
            return None
        n_obs_d, n_actions_d = decode.shape
        if n_obs_d != self.n_obs_per_env or n_actions_d != red_rlp.shape[-1]:
            return None

        dev = red_rlp.device
        E, R, A = red_rlp.shape
        C = int(decode.max()) + 1
        decode_t = torch.as_tensor(decode, dtype=torch.long, device=dev)  # [n_obs, A]
        probs = red_rlp.exp()  # [E, R, A]
        red_actions = red_actions.to(torch.long).to(dev)

        out = torch.empty(E, n_obs_d, R, device=dev, dtype=red_rlp.dtype)
        for o in range(n_obs_d):
            idx = decode_t[o]  # [A] -> symbol id per joint action
            marg = torch.zeros(E, R, C, device=dev, dtype=red_rlp.dtype)
            marg.index_add_(-1, idx, probs)
            marg = marg.clamp_min(1e-12)
            log_marg = marg.log()  # [E, R, C]
            sym = decode_t[o][red_actions].view(E, 1, 1).expand(E, R, 1)
            out[:, o, :] = log_marg.gather(-1, sym).squeeze(-1)
        return out

    def _apply_observer_visibility_weights(self, log_liks, obs_np, red_acts=None):
        """Scale role evidence per observer for simple multi-observer proxy envs."""
        weight_fn = getattr(self.env, "get_observer_likelihood_weights", None)
        if callable(weight_fn):
            weights = torch.as_tensor(weight_fn(), dtype=log_liks.dtype, device=log_liks.device)
            if weights.shape == log_liks.shape[:2]:
                centered = log_liks - log_liks.mean(dim=-1, keepdim=True)
                return log_liks.mean(dim=-1, keepdim=True) + centered * weights.unsqueeze(-1)
        if self.n_obs_per_env <= 1 or "coin_game_multi" not in self.scenario or obs_np is None:
            return log_liks

        obs = torch.as_tensor(obs_np, device=self.device)
        if obs.dtype == torch.uint8:
            obs = obs.float() / 255.0
        else:
            obs = obs.float()
            if obs.max().detach().item() > 1.0:
                obs = obs / 255.0

        if obs.ndim != 2 or obs.shape[-1] < 5:
            return log_liks

        obs_idx = self.obs_agent_flat.to(self.device)
        obs_rows = obs[obs_idx]  # [E, n_obs, obs_dim]
        own_xy = obs_rows[..., 1:3]
        red_xy = obs_rows[..., 3:5]
        dist = torch.linalg.norm(own_xy - red_xy, dim=-1, keepdim=True)
        visibility = 0.25 + 0.75 * (1.0 - dist / np.sqrt(2.0)).clamp(0.0, 1.0)

        centered = log_liks - log_liks.mean(dim=-1, keepdim=True)
        return log_liks.mean(dim=-1, keepdim=True) + centered * visibility

    def _belief_step(self, beliefs, role_log_policy, actions_np, obs_np=None,
                     per_obs_prior=None, observer_log_liks=None):

        if self.bos_belief_mode not in ("slot", "speaker"):
            E, R, n_obs = self.n_envs, self.n_roles, self.n_obs_per_env
            if observer_log_liks is None:
                red_acts = torch.from_numpy(actions_np[self.red_agent_flat.numpy()]).to(self.device)
                red_rlp = role_log_policy[self.red_agent_flat.to(self.device)]
                channel_liks = self._channel_decode_log_liks(red_rlp, red_acts)
                if channel_liks is not None:
                    log_liks = channel_liks
                else:
                    log_liks = red_rlp.gather(
                        -1, red_acts.view(E, 1, 1).expand(E, R, 1)
                    ).squeeze(-1).unsqueeze(1).expand(E, n_obs, R)
                    log_liks = self._apply_observer_visibility_weights(log_liks, obs_np, red_acts)
            else:
                log_liks = observer_log_liks.to(self.device)
            if per_obs_prior is not None:
                prior = per_obs_prior.to(self.device).clamp(min=1e-8)
            else:
                prior = beliefs.to(self.device).clamp(min=1e-8).unsqueeze(1).expand(E, n_obs, R)
            log_liks = self._temper_log_liks(log_liks)
            per_obs = self._regularize_belief(
                torch.softmax(log_liks + torch.log(prior), dim=-1)
            )
            per_obs = self._freeze_known_observer_beliefs(per_obs).cpu()  # [E, n_obs, R]
            return per_obs.mean(1), per_obs

        if self.bos_belief_mode == "speaker":
            if self.listener_agent_flat is None or self.speaker_agent_flat is None:
                raise RuntimeError("bos_belief_mode='speaker' requires listener/speaker indices")
            E, R = self.n_envs, self.n_roles
            dev = self.device
            if obs_np is None:
                raise ValueError("bos_belief_mode='speaker' requires obs_np for listener messages")

            listener_obs_u8 = torch.from_numpy(obs_np[self.listener_agent_flat.numpy()]).to(dev)
            m_speakers = self._listener_messages_from_obs(listener_obs_u8)

            speaker_idx = self.speaker_agent_flat.to(dev)  # [E, R]
            speaker_rlp = role_log_policy[speaker_idx.reshape(-1)].reshape(E, R, self.n_roles, self.n_actions)
            log_liks = self._speaker_identity_log_liks(speaker_rlp, m_speakers)

            log_prior = torch.log(beliefs.to(dev).clamp(min=1e-8))
            log_liks = self._temper_log_liks(log_liks)
            post = self._regularize_belief(torch.softmax(log_liks + log_prior, dim=-1))
            per_obs = post.unsqueeze(1).cpu()
            return post.detach().cpu(), per_obs

        # --- slot mode (multi-speaker): hypotheses are which *display slot* is honest ---
        if self.listener_agent_flat is None or self.decoy_agents_flat is None:
            raise RuntimeError("bos_belief_mode='slot' requires listener/decoy indices to be initialized")

        E, R = self.n_envs, self.n_roles
        dev = self.device

        if obs_np is None:
            raise ValueError("bos_belief_mode='slot' requires obs_np for listener messages")
        listener_obs_u8 = torch.from_numpy(obs_np[self.listener_agent_flat.numpy()]).to(dev)  # [E, 8]
        hd_u8 = listener_obs_u8[:, 3:6].float() / 255.0
        m_slots = torch.clamp((hd_u8 * 2.0).round(), 0, 2).long()  # [E, 3]

        # Honest speaker signal distribution (renormalized over actions 0..2).
        honest_rlp = role_log_policy[self.red_agent_flat.to(dev), 0, :3]  # [E, 3] log-probs
        pi_h = honest_rlp.exp()
        pi_h = pi_h / (pi_h.sum(-1, keepdim=True) + 1e-8)

        # Decoy signal distribution: average the (up to) two decoys, renormalized over actions 0..2.
        decoy_idx = self.decoy_agents_flat.to(dev).reshape(-1)  # [2E]
        decoy_rlp = role_log_policy[decoy_idx, 2, :3]  # [2E, 3] log-probs
        pi_d = decoy_rlp.exp()
        pi_d = pi_d / (pi_d.sum(-1, keepdim=True) + 1e-8)
        pi_d = pi_d.reshape(E, -1, 3).mean(1)  # [E, 3]

        # Channel corruption model (drop/noise is ignored here; corruption is modeled as uniform random).
        p_corrupt = float(getattr(self.env, "signal_corrupt_prob", 0.0) or 0.0)
        p_u = 1.0 / 3.0

        # P(m|speaker) = (1-p)pi[m] + p/3
        idx_e = torch.arange(E, device=dev)
        p_h_m = (1.0 - p_corrupt) * pi_h[idx_e[:, None], m_slots] + p_corrupt * p_u  # [E, 3]
        p_d_m = (1.0 - p_corrupt) * pi_d[idx_e[:, None], m_slots] + p_corrupt * p_u  # [E, 3]
        p_h_m = p_h_m.clamp_min(1e-8)
        p_d_m = p_d_m.clamp_min(1e-8)

        # Likelihood for each hypothesis z: slot z is honest, others are decoys.
        log_liks = torch.zeros(E, R, device=dev)
        for z in range(R):
            # log P(m_z | honest) + sum_{d!=z} log P(m_d | decoy)
            ll = torch.log(p_h_m[:, z])
            for d in range(R):
                if d != z:
                    ll = ll + torch.log(p_d_m[:, d])
            log_liks[:, z] = ll

        log_prior = torch.log(beliefs.to(dev).clamp(min=1e-8))
        log_liks = self._temper_log_liks(log_liks)
        post = self._regularize_belief(torch.softmax(log_liks + log_prior, dim=-1))
        per_obs = post.unsqueeze(1).cpu()  # [E, 1, R]
        return post.detach().cpu(), per_obs

    def _slot_selected_log_liks(self, buf, detach=False):
        if self.listener_agent_flat is None or self.decoy_agents_flat is None:
            raise RuntimeError("bos_belief_mode='slot' requires listener/decoy indices")

        T, E, R = self.n_steps, self.n_envs, self.n_roles
        dev = self.device
        p_corrupt = float(getattr(self.env, "signal_corrupt_prob", 0.0) or 0.0)
        p_u = 1.0 / 3.0

        out = []
        idx_e = torch.arange(E, device=dev)

        # Roles for forward passes
        honest_roles = torch.zeros(1, E, dtype=torch.int64, device=dev)
        decoy_roles = torch.full((1, 2 * E), 2, dtype=torch.int64, device=dev)

        for t in range(T - 1):
            # Listener heard messages caused by the speaker actions at time t.
            listener_obs_u8 = buf.obs[t + 1, self.listener_agent_flat].to(dev).float() / 255.0  # [E, 8]
            m_slots = torch.clamp((listener_obs_u8[:, 3:6] * 2.0).round(), 0, 2).long()  # [E, 3]

            # Honest policy over signals (0..2)
            honest_obs = buf.obs[t, self.red_agent_flat].unsqueeze(0).to(dev)  # [1, E, *obs]
            honest_rnn = buf.rnn_states[t, self.red_agent_flat].to(dev)        # [E, 2, H]
            res_h, _ = self.policy.forward_sequence(honest_obs, honest_rnn, roles=honest_roles)
            lp_h = res_h["log_policy"][0][:, :3]  # [E, 3]
            if detach:
                lp_h = lp_h.detach()
            pi_h = lp_h.exp()
            pi_h = pi_h / (pi_h.sum(-1, keepdim=True) + 1e-8)

            # Decoy policy over signals (0..2) — detached
            decoy_idx = self.decoy_agents_flat.reshape(-1)
            decoy_obs = buf.obs[t, decoy_idx].unsqueeze(0).to(dev)  # [1, 2E, *obs]
            decoy_rnn = buf.rnn_states[t, decoy_idx].to(dev)         # [2E, 2, H]
            res_d, _ = self.policy.forward_sequence(decoy_obs, decoy_rnn, roles=decoy_roles)
            lp_d = res_d["log_policy"][0][:, :3].detach()            # [2E, 3]
            pi_d = lp_d.exp()
            pi_d = pi_d / (pi_d.sum(-1, keepdim=True) + 1e-8)
            pi_d = pi_d.reshape(E, 2, 3).mean(1)                      # [E, 3]

            p_h_m = (1.0 - p_corrupt) * pi_h[idx_e[:, None], m_slots] + p_corrupt * p_u  # [E, 3]
            p_d_m = (1.0 - p_corrupt) * pi_d[idx_e[:, None], m_slots] + p_corrupt * p_u  # [E, 3]
            p_h_m = p_h_m.clamp_min(1e-8)
            p_d_m = p_d_m.clamp_min(1e-8)

            log_liks = torch.zeros(E, R, device=dev)
            for z in range(R):
                ll = torch.log(p_h_m[:, z])
                for d in range(R):
                    if d != z:
                        ll = ll + torch.log(p_d_m[:, d])
                log_liks[:, z] = ll
            out.append(log_liks.unsqueeze(1))  # [E, 1, R]

        return torch.stack(out, dim=0)  # [T-1, E, 1, R]

    def _speaker_selected_log_liks(self, buf, detach=False):
        """Log-likelihoods for physical honest-speaker hypotheses. [T-1, E, 1, R]."""
        if self.listener_agent_flat is None or self.speaker_agent_flat is None:
            raise RuntimeError("bos_belief_mode='speaker' requires listener/speaker indices")

        T, E, R = self.n_steps, self.n_envs, self.n_roles
        dev = self.device
        speaker_idx = self.speaker_agent_flat.reshape(-1)

        out = []
        for t in range(T - 1):
            listener_obs_u8 = buf.obs[t + 1, self.listener_agent_flat].to(dev)
            m_speakers = self._listener_messages_from_obs(listener_obs_u8)

            speaker_obs = buf.obs[t, speaker_idx].unsqueeze(0).to(dev)
            speaker_rnn = buf.rnn_states[t, speaker_idx].to(dev)
            res, _ = self.policy.forward_sequence(speaker_obs, speaker_rnn)
            speaker_rlp = res["role_log_policy"][0].reshape(E, R, self.n_roles, self.n_actions)

            log_liks = self._speaker_identity_log_liks(
                speaker_rlp, m_speakers, detach_honest=detach)
            out.append(log_liks.unsqueeze(1))

        return torch.stack(out, dim=0)

    def ppo_update(self, buf, last_values, epoch):
        ramp      = min(1.0, epoch / max(self.warmup_epochs, 1))
        lambda_eff = self.lambda_dec * ramp

        if self.use_bbm_intrinsic and lambda_eff > 0:
            raw = buf.bbm_rewards  # [T, n_agents]
            # running variance update (exponential moving average)
            self._bbm_reward_var = 0.99 * self._bbm_reward_var + 0.01 * float(raw.var().item() + 1e-8)
            norm_scale = self._bbm_reward_var ** 0.5
            buf.rewards = buf.rewards_ext.clone() + lambda_eff * raw / (norm_scale + 1e-8)

        advantages, returns = buf.compute_gae(last_values, self.gamma, self.gae_lambda)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        aa_term = torch.zeros_like(advantages)
        if self.use_advantage_alignment:
            aa_effective_weight = self.aa_weight * ramp if self.aa_warmup else None
            advantages, aa_term = self._advantage_alignment(
                advantages, buf.dones, weight=aa_effective_weight, roles=buf.roles)

        T  = self.n_steps
        B  = self.n_agents
        mb = T * B // self.n_minibatches

        pg_losses, vf_losses, ent_losses = [], [], []
        role_nll_losses, b_true_paper_vals = [], []
        dec_losses, dec_clip_fracs, dec_kls = [], [], []
        dec_head_role0_grads, dec_head_other_grads, dec_nonhead_grad_pre = [], [], []
        ppo_grad_norms = []
        dec_grad_pre_clip, dec_grad_post_clip, dec_adaptive_scales, dec_grad_final = [], [], [], []
        critic_losses, critic_evs, critic_corrs = [], [], []
        bos_coeff_abs_vals, bos_coeff_rms_vals = [], []
        bos_entropy_frac_vals, bos_coeff_gate_vals = [], []
        lola_losses = []
        lola_grad_norms = []
        lola_ppo_head_norms = []
        # no minibatch division here (RTG divides)
        bos_injection_scale = 1.0

        bos_active = (
            lambda_eff > 0
            and not self.use_bbm_intrinsic
            and self.belief_critic is not None
            and self.n_obs_per_env > 0
        )
        # Belief-ValuePPO also trains the estimator
        train_obs_estimator = bos_active or self.use_belief_value_critic
        obs_est_stats = self._train_obs_estimator(buf) if train_obs_estimator else {}
        ref_log_liks = None
        if bos_active and self.dec_clip_eps is not None:
            with torch.no_grad():
                if self.bos_belief_mode == "slot":
                    ref_log_liks = self._slot_selected_log_liks(buf, detach=True)
                elif self.bos_belief_mode == "speaker":
                    ref_log_liks = self._speaker_selected_log_liks(buf, detach=True)
                else:
                    ref_log_liks = self._observer_selected_role_log_liks(buf, detach=True)

        for _ in range(self.n_epochs_ppo):
            idx = torch.randperm(T * B)
            for start in range(0, T * B, mb):
                sl      = idx[start:start + mb]
                t_idx, b_idx = sl // B, sl % B

                obs_mb   = buf.obs[t_idx, b_idx].to(self.device)
                acts_mb  = buf.actions[t_idx, b_idx].to(self.device)
                lp_old   = buf.log_probs[t_idx, b_idx].to(self.device)
                adv_mb   = advantages[t_idx, b_idx].to(self.device)
                ret_mb   = returns[t_idx, b_idx].to(self.device)
                rnn_mb   = buf.rnn_states[t_idx, b_idx].to(self.device)
                roles_mb = buf.roles[t_idx, b_idx].to(self.device)

                result, _ = self.policy.forward_sequence(
                    obs_mb.unsqueeze(0), rnn_mb, roles=roles_mb.unsqueeze(0))
                lp_new = result['log_policy'][0]
                vals   = result['role_ext_value'][0][range(mb), roles_mb]

                ratio    = (lp_new[range(mb), acts_mb] - lp_old).exp()
                pg_loss  = torch.max(-adv_mb * ratio,
                                     -adv_mb * ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps)).mean()
                if self.use_belief_value_critic:
                    # focal baseline from BeliefCritic
                    focal_mask = self._focal_batch_mask(b_idx, roles_mb).to(self.device)
                    non_focal = ~focal_mask
                    vf_loss = F.mse_loss(vals[non_focal], ret_mb[non_focal]) \
                        if non_focal.any() else torch.zeros((), device=self.device)
                else:
                    vf_loss = F.mse_loss(vals, ret_mb)
                ent_loss = -lp_new.exp().mul(lp_new).sum(-1).mean()

                # critic trained on every t
                if self.belief_critic is not None:
                    focal_mask_cpu = self._focal_batch_mask(b_idx, roles_mb)
                    if focal_mask_cpu.any():
                        value_t_idx = t_idx[focal_mask_cpu]
                        value_b_idx = b_idx[focal_mask_cpu]
                        env_idx_r0 = value_b_idx // self.agents_per_env
                        target_r0 = returns[value_t_idx, value_b_idx].to(self.device)
                        if value_t_idx is not None:
                            n_obs_c = max(self.n_obs_per_env, 1)
                            b_joint = buf.per_obs_beliefs[value_t_idx, env_idx_r0].to(self.device)
                            b_joint = b_joint.reshape(b_joint.size(0), n_obs_c * self.n_roles)
                            obs_flat_r0 = buf.obs[value_t_idx, value_b_idx].to(self.device).float()
                            obs_flat_r0 = obs_flat_r0.view(value_t_idx.numel(), -1)
                            for _critic_step in range(self.bos_critic_updates):
                                v_pred = self.belief_critic(obs_flat_r0, b_joint).squeeze(-1)
                                critic_loss = F.mse_loss(v_pred, target_r0)
                                self.critic_optimizer.zero_grad()
                                critic_loss.backward()
                                self.critic_optimizer.step()
                            with torch.no_grad():
                                v_eval = self.belief_critic(obs_flat_r0, b_joint).squeeze(-1)
                                critic_losses.append(float(F.mse_loss(v_eval, target_r0).item()))
                                target_var = torch.var(target_r0, unbiased=False)
                                if target_var > 1e-8:
                                    ev = 1.0 - torch.var(target_r0 - v_eval, unbiased=False) / (target_var + 1e-8)
                                    critic_evs.append(float(ev.item()))
                                if target_r0.numel() > 1:
                                    vc = v_eval - v_eval.mean()
                                    tc = target_r0 - target_r0.mean()
                                    denom = torch.sqrt((vc.square().mean() * tc.square().mean()).clamp_min(1e-8))
                                    critic_corrs.append(float((vc * tc).mean().div(denom).item()))

                # role prediction aux head (BBM)
                role_nll_loss = torch.tensor(0.0, device=self.device)
                if 'policy_role_prediction' in result:
                    rp = result['policy_role_prediction'][0]  # [mb, n_players, n_roles]
                    g_idx = b_idx // self.agents_per_env  # [mb]
                    offsets = torch.arange(self.agents_per_env)  # [n_players]
                    all_agent_idx = g_idx.unsqueeze(1) * self.agents_per_env + offsets.unsqueeze(0)  # [mb, n_players]
                    player_roles_mb = buf.roles[t_idx.unsqueeze(1).expand_as(all_agent_idx), all_agent_idx].to(self.device)  # [mb, n_players]
                    per_elem_nll = F.nll_loss(rp.transpose(1, 2), player_roles_mb, reduction='none')  # [mb, n_players]
                    role_nll_loss = per_elem_nll.mean()
                    role_nll_losses.append(role_nll_loss.item())
                    blue_mask = (roles_mb >= 2)
                    if blue_mask.any():
                        blue_preds_of_agent0 = rp[blue_mask, 0] # [N_blue, n_roles]
                        target_role_agent0   = player_roles_mb[blue_mask, 0] # [N_blue]
                        
                        blue_nll_agent0 = F.nll_loss(blue_preds_of_agent0, target_role_agent0, reduction='mean')
                        b_true_paper_vals.append(float(torch.exp(-blue_nll_agent0).item()))

                pg_losses.append(pg_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())

                policy_params = list(self.policy.parameters())
                ppo_objective = (
                    pg_loss + self.val_coef * vf_loss - self.ent_coef * ent_loss
                    + self.role_nll_coef * role_nll_loss
                )

                self.optimizer.zero_grad()
                ppo_objective.backward()
                ppo_grad_norm = self._grad_norm(policy_params)
                ppo_grad_norms.append(float(ppo_grad_norm))
                ppo_grads = self._clone_grads(policy_params)
                # PPO grad norm on shaper head rows (LOLA cap)
                ppo_shaper_head_norm = 0.0
                if self.use_lola_dice:
                    n_act = self.n_actions
                    for p, g in zip(policy_params, ppo_grads):
                        if g is not None and (p is self.policy.policy_head.weight
                                              or p is self.policy.policy_head.bias):
                            ppo_shaper_head_norm += float(g[:n_act].norm().item()) ** 2
                    ppo_shaper_head_norm = ppo_shaper_head_norm ** 0.5
                total_grads = ppo_grads

                if bos_active:
                    self.optimizer.zero_grad()
                    bp_loss, aux = self.belief_planning_loss(
                        buf,
                        ref_log_liks=ref_log_liks,
                        clip_eps=self.dec_clip_eps,
                        return_aux=True,
                    )
                    if not torch.isfinite(bp_loss):
                        self.optimizer.zero_grad(set_to_none=True)
                        bp_loss = torch.zeros(1, device=self.device)
                        aux = {"dec_clip_frac": 0.0, "dec_approx_kl": 0.0}
                    else:
                        (lambda_eff * bp_loss).backward()

                    grad_stats = self._mask_deception_gradients()
                    shaping_params = [p for p in policy_params if p.grad is not None]
                    pre_clip_norm = self._grad_norm(shaping_params)
                    if not np.isfinite(pre_clip_norm):
                        for p in shaping_params:
                            p.grad = None
                        shaping_params = []
                        pre_clip_norm = 0.0
                    dec_grad_pre_clip.append(pre_clip_norm)
                    if shaping_params:
                        nn.utils.clip_grad_norm_(shaping_params, self.dec_max_grad_norm)
                    post_clip_norm = self._grad_norm(shaping_params)
                    if not np.isfinite(post_clip_norm):
                        for p in shaping_params:
                            p.grad = None
                        shaping_params = []
                        post_clip_norm = 0.0

                    adaptive_scale = 1.0
                    final_norm = post_clip_norm
                    if self.bos_grad_frac > 0.0 and post_clip_norm > 1e-8:
                        # ramp amplify target with warmup
                        target_norm = self.bos_grad_frac * ramp * max(float(ppo_grad_norm), 1e-8)
                        adaptive_scale = target_norm / post_clip_norm
                        if not self.bos_grad_allow_amplify:
                            adaptive_scale = min(1.0, adaptive_scale)
                        adaptive_scale = min(adaptive_scale, self.bos_grad_max_scale)
                        for p in shaping_params:
                            p.grad.mul_(adaptive_scale)
                        final_norm = post_clip_norm * adaptive_scale

                    bos_grads = self._clone_grads(policy_params)
                    total_grads = self._add_grads(ppo_grads, bos_grads)

                    dec_grad_post_clip.append(post_clip_norm)
                    dec_adaptive_scales.append(adaptive_scale)
                    dec_grad_final.append(final_norm)
                    dec_losses.append(bp_loss.item())
                    dec_clip_fracs.append(aux["dec_clip_frac"])
                    dec_kls.append(aux["dec_approx_kl"])
                    bos_coeff_abs_vals.append(aux.get("bos_coeff_abs", 0.0))
                    bos_coeff_rms_vals.append(aux.get("bos_coeff_rms", 0.0))
                    bos_entropy_frac_vals.append(aux.get("bos_belief_entropy_frac", 0.0))
                    bos_coeff_gate_vals.append(aux.get("bos_coeff_gate_frac", 0.0))
                    dec_head_role0_grads.append(grad_stats["head_role0_grad"])
                    dec_head_other_grads.append(grad_stats["head_other_grad"])
                    dec_nonhead_grad_pre.append(grad_stats["nonhead_grad_pre"])

                if self.use_lola_dice:
                    self.optimizer.zero_grad()
                    lola_loss = self._lola_dice_loss(buf)
                    if lola_loss is not None and torch.isfinite(lola_loss):
                        lola_loss.backward()
                        n_act = self.n_actions
                        head_w = self.policy.policy_head.weight
                        head_b = self.policy.policy_head.bias
                        if head_w.grad is not None:
                            head_w.grad[n_act:] = 0.0  # role-0 rows only
                        if head_b is not None and head_b.grad is not None:
                            head_b.grad[n_act:] = 0.0
                        for p in policy_params:
                            if p is head_w or p is head_b:
                                continue
                            if p.grad is not None:
                                p.grad.zero_()
                        lola_params = [p for p in policy_params if p.grad is not None]
                        if lola_params:
                            nn.utils.clip_grad_norm_(lola_params, self.lola_max_grad_norm)
                            if self.lola_grad_frac > 0.0:
                                lola_norm = self._grad_norm(lola_params)
                                if lola_norm > 1e-8:
                                    target_norm = self.lola_grad_frac * max(ppo_shaper_head_norm, 1e-8)
                                    scale = min(1.0, target_norm / lola_norm)
                                    for p in lola_params:
                                        p.grad.mul_(scale)
                            lola_grad_norms.append(self._grad_norm(lola_params))
                            lola_ppo_head_norms.append(ppo_shaper_head_norm)
                        lola_grads = self._clone_grads(policy_params)
                        total_grads = self._add_grads(total_grads, lola_grads)
                        lola_losses.append(float(lola_loss.item()))

                self.optimizer.zero_grad()
                self._set_grads(policy_params, total_grads)
                nn.utils.clip_grad_norm_(policy_params, self.max_grad_norm)
                self.optimizer.step()

        if ppo_grad_norms:
            self._last_ppo_grad_norm = (
                0.95 * self._last_ppo_grad_norm + 0.05 * float(np.mean(ppo_grad_norms))
            )

        head_grad_norm = float(np.mean(dec_head_role0_grads)) if dec_head_role0_grads else 0.0
        enc_grad_norm = 0.0 if self.dec_head_only else float(np.mean(dec_nonhead_grad_pre)) if dec_nonhead_grad_pre else 0.0

        bp_loss_raw = float(np.mean(dec_losses)) if dec_losses else 0.0
        bp_loss_pos = -bp_loss_raw
        bos_critic_loss = float(np.mean(critic_losses)) if critic_losses else 0.0
        bos_critic_ev = float(np.mean(critic_evs)) if critic_evs else 0.0
        bos_critic_corr = float(np.mean(critic_corrs)) if critic_corrs else 0.0

        bbm_bonus_mean = float(buf.bbm_rewards[buf.bbm_rewards != 0].mean().item()) \
            if self.use_bbm_intrinsic and buf.bbm_rewards.any() else 0.0

        return {
            "pol_loss":      float(np.mean(pg_losses)),
            "vf_loss":       float(np.mean(vf_losses)),
            "ent_loss":      float(np.mean(ent_losses)),
            "b_true_paper":  float(np.mean(b_true_paper_vals)) if b_true_paper_vals else float('nan'),
            "role_nll":      float(np.mean(role_nll_losses)) if role_nll_losses else float('nan'),
            "bp_loss":       bp_loss_raw,
            "bp_loss_raw":   bp_loss_raw,
            "bp_loss_pos":   bp_loss_pos,
            "lambda_eff":    float(lambda_eff),
            "ramp":          ramp,
            "dec_clip_frac": float(np.mean(dec_clip_fracs)) if dec_clip_fracs else 0.0,
            "dec_approx_kl": float(np.mean(dec_kls)) if dec_kls else 0.0,
            "dec_head_role0_grad": float(np.mean(dec_head_role0_grads)) if dec_head_role0_grads else 0.0,
            "dec_head_other_grad": float(np.mean(dec_head_other_grads)) if dec_head_other_grads else 0.0,
            "dec_nonhead_grad_pre": float(np.mean(dec_nonhead_grad_pre)) if dec_nonhead_grad_pre else 0.0,
            "dec_head_only": float(self.dec_head_only),
            "dec_epochs": float(self.dec_epochs),
            "enc_grad_norm": enc_grad_norm,
            "head_grad_norm": head_grad_norm,
            "ppo_grad_norm": float(np.mean(ppo_grad_norms)) if ppo_grad_norms else 0.0,
            "ppo_grad_norm_ema": float(self._last_ppo_grad_norm),
            "bos_grad_frac": float(self.bos_grad_frac),
            "bos_grad_allow_amplify": float(self.bos_grad_allow_amplify),
            "bos_grad_max_scale": float(self.bos_grad_max_scale),
            "bos_injection_scale": float(bos_injection_scale),
            "bos_belief_temperature": float(self.bos_belief_temperature),
            "bos_belief_floor": float(self.bos_belief_floor),
            "bos_correction_mode_coeff": float(self.bos_correction_mode == "coeff"),
            "bos_shaper_adapter": float(self.bos_shaper_adapter),
            "bos_update_all_role_heads": float(self.bos_update_all_role_heads),
            "bos_likelihood_obs_mode_estimated": float(self.bos_likelihood_obs_mode in ("estimated", "esti", "learned")),
            "bos_obs_estimator_loss": obs_est_stats.get("bos_obs_estimator_loss", 0.0),
            "bos_obs_estimator_mae": obs_est_stats.get("bos_obs_estimator_mae", 0.0),
            "bos_coeff_abs": float(np.mean(bos_coeff_abs_vals)) if bos_coeff_abs_vals else 0.0,
            "bos_coeff_rms": float(np.mean(bos_coeff_rms_vals)) if bos_coeff_rms_vals else 0.0,
            "bos_belief_entropy_frac": float(np.mean(bos_entropy_frac_vals)) if bos_entropy_frac_vals else 0.0,
            "bos_coeff_gate_frac": float(np.mean(bos_coeff_gate_vals)) if bos_coeff_gate_vals else 0.0,
            "dec_grad_norm_pre_clip": float(np.mean(dec_grad_pre_clip)) if dec_grad_pre_clip else 0.0,
            "dec_grad_norm_post_clip": float(np.mean(dec_grad_post_clip)) if dec_grad_post_clip else 0.0,
            "dec_adaptive_scale": float(np.mean(dec_adaptive_scales)) if dec_adaptive_scales else 1.0,
            "dec_grad_norm_final": float(np.mean(dec_grad_final)) if dec_grad_final else 0.0,
            "bos_critic_loss": bos_critic_loss,
            "bos_critic_ev": bos_critic_ev,
            "bos_critic_corr": bos_critic_corr,
            "bos_critic_hidden_dim": float(self.bos_critic_hidden_dim),
            "bos_critic_depth": float(self.bos_critic_depth),
            "bos_critic_updates": float(self.bos_critic_updates),
            "advantage_alignment": float(self.use_advantage_alignment),
            "aa_weight": float(self.aa_weight),
            "aa_term_abs": float(aa_term.abs().mean().item()),
            "lola_dice": float(self.use_lola_dice),
            "lola_weight": float(self.lola_weight),
            "lola_loss": float(np.mean(lola_losses)) if lola_losses else 0.0,
            "lola_grad_norm_final": float(np.mean(lola_grad_norms)) if lola_grad_norms else 0.0,
            "lola_ppo_head_grad_norm": float(np.mean(lola_ppo_head_norms)) if lola_ppo_head_norms else 0.0,
            "bbm_bonus_mean": bbm_bonus_mean,
        }

    @torch.no_grad()
    def evaluate(self, n_episodes=100, max_steps=8000):
        self.policy.eval()
        E = self.n_envs

        beliefs    = self.belief_tracker.init_belief(E)
        rnn_states = torch.zeros(self.n_agents, 2, self.memory_units)
        obs        = self.env.reset()
        outcomes, belief_stats, env_b_true = [], [], [[] for _ in range(E)]
        eval_red_scores, eval_blue_scores = [], []

        for _ in range(max_steps):
            roles = self.env.get_roles()
            zt_np = self._z_true_vec_np()
            obs_t   = torch.from_numpy(obs).unsqueeze(0)
            roles_t = torch.from_numpy(roles).unsqueeze(0).to(torch.int64)
            result, new_rnn = self.policy.forward_sequence(obs_t, rnn_states, roles=roles_t)

            actions_np  = sample_action_from_logp(result['log_policy'][0].cpu().numpy())
            rnn_states = new_rnn.detach()
            obs_new, _, dones_np, infos = self.env.step(actions_np)

            belief_obs = obs_new if self._belief_uses_next_obs() else obs
            new_beliefs, _ = self._belief_step(
                beliefs, result['role_log_policy'][0], actions_np, belief_obs)

            for e in range(E):
                env_b_true[e].append(float(new_beliefs[e, zt_np[e]]))

            rnn_states[torch.from_numpy(dones_np)] = 0.0
            beliefs, _, _ = reset_done_env_beliefs(
                new_beliefs, dones_np, self.n_envs, self.agents_per_env)
            obs = obs_new

            seen = set()
            for agent_i, info in enumerate(infos):
                if 'outcome' not in info:
                    continue
                outcome = info['outcome']
                if outcome == 'none':
                    continue
                game_i = agent_i // self.agents_per_env
                if game_i in seen or game_i >= E:
                    continue
                seen.add(game_i)
                row = {
                    'red_win':  1.0 if outcome in ('general_killed', 'exploited') else 0.0,
                    'blue_win': 1.0 if outcome in ('general_rescued', 'blue_win', 'cooperated') else 0.0,
                }
                if 'coop_success' in info:
                    row['coop_success'] = float(info['coop_success'])
                gs = info.get('game_scores')
                if gs is not None:
                    gs = np.asarray(gs)
                    eval_red_scores.append(float(gs[:, 0].sum()))
                    eval_blue_scores.append(float(gs[:, 2].sum()))
                outcomes.append(row)
                traj = np.array(env_b_true[game_i])
                if len(traj) >= 2:
                    belief_stats.append({
                        "b_true_at_k":  float(traj[min(self.k,    len(traj)-1)]),
                        "b_true_at_2k": float(traj[min(2*self.k,  len(traj)-1)]),
                        "b_true_final": float(traj[-1]),
                        "belief_var":   float(np.var(np.diff(traj))),
                    })
                env_b_true[game_i] = []
                beliefs[game_i]    = self.belief_tracker.init_belief(1)[0]

            if len(outcomes) >= n_episodes:
                break

        self._current_obs = self.env.reset()
        self.policy.train()

        metrics = {}
        if outcomes:
            out = outcomes[:n_episodes]
            metrics["red_win_rate"]  = float(np.mean([o['red_win']  for o in out]))
            metrics["blue_win_rate"] = float(np.mean([o['blue_win'] for o in out]))
            if eval_red_scores:
                metrics["red_score"] = float(np.mean(eval_red_scores[:n_episodes]))
                metrics["red_scores_raw"] = [float(s) for s in eval_red_scores[:n_episodes]]
            if eval_blue_scores:
                metrics["blue_score"] = float(np.mean(eval_blue_scores[:n_episodes]))
                metrics["blue_scores_raw"] = [float(s) for s in eval_blue_scores[:n_episodes]]
            coop_eval = [o['coop_success'] for o in out if 'coop_success' in o]
            if coop_eval:
                metrics["coop_success_rate"] = float(np.mean(coop_eval))
        if belief_stats:
            for key in ["b_true_at_k", "b_true_at_2k", "b_true_final", "belief_var"]:
                metrics[key] = float(np.mean([s[key] for s in belief_stats]))
        return metrics

    def _save(self, epoch, final=False):
        os.makedirs(self.save_dir, exist_ok=True)
        tag  = "final" if final else f"ep{epoch:04d}"
        path = os.path.join(self.save_dir, f"{self.scenario}_{self.run_tag}_{tag}.pt")
        torch.save({"epoch": epoch, "policy": self.policy.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "config": self.config}, path)
        print(f"  [ckpt] {path}")

    def train(self):
        if self.use_wandb:
            if wandb is None:
                print("Warning: wandb is not installed, skipping logging. Please run `pip install wandb`.")
                self.use_wandb = False
            else:
                method_name = self.method_name

                run_name_parts = [self.scenario, method_name]
                if method_name == 'bos' and self.lambda_dec <= 0:
                    run_name_parts.append("lambda0")
                elif method_name == 'bos':
                    run_name_parts.append(f"k{self.k}")
                run_name_parts.append(f"s{self.config.get('seed')}")
                run_name = "_".join(run_name_parts)

                # Create a clean config for wandb, ensuring all key params are present.
                wandb_config = self.config.copy()
                wandb_config['method'] = method_name
                wandb_config['k'] = self.k  # Ensure k is always logged
                wandb_config['env'] = self.scenario
                wandb_config['algo'] = method_name
                wandb_config['seed'] = int(self.config.get('seed', 42))
                extra_tags = list(self.config.get('wandb_tags', []))
                tags = [
                    self.scenario,
                    f"seed_{wandb_config['seed']}",
                    method_name,
                    f"k_{self.k}",
                    f"env:{self.scenario}",
                    f"algo:{method_name}",
                    f"k:{self.k}",
                ] + extra_tags

                wandb.init(
                    project=os.environ.get("WANDB_PROJECT", "temporal_deception"),
                    config=wandb_config,
                    name=run_name,
                    tags=list(dict.fromkeys(tags)),
                )

        rnn_states   = torch.zeros(self.n_agents, 2, self.memory_units)
        beliefs      = self.belief_tracker.init_belief(self.n_envs).to(self.device)
        eval_interval = self.config.get("eval_interval", 50)
        global_step  = 0
        ema_alpha    = 0.1  # smoothing factor for logged b_true/b_imp EMAs
        b_true_ema   = None
        b_imp_ema    = None

        if self.method_name == "advantage_alignment":
            print(f"AdvantageAlignment | beta={self.aa_weight} epochs={self.n_train_epochs}")
        elif self.method_name == "belief_value_ppo":
            print(f"BeliefValuePPO | epochs={self.n_train_epochs}")
        else:
            print(f"{self.method_name.upper()} | k={self.k} lambda={self.lambda_dec} "
                  f"warmup={self.warmup_epochs} grad_frac={self.bos_grad_frac} "
                  f"epochs={self.n_train_epochs}")

        for epoch in range(1, self.n_train_epochs + 1):
            (
                buf,
                rnn_states,
                beliefs,
                ep_red_wins,
                ep_blue_wins,
                ep_red_scores,
                ep_blue_scores,
                b_true_vals,
                b_true_blue_vals,
                b_imp_vals,
                b_true_prior_vals,
                b_imp_prior_vals,
                ep_coop_success,
                ep_coop_return,
                ep_coin_red_returns,
                ep_coin_blue_returns,
                oracle_b_true_vals,
                oracle_b_true_obs0_vals,
                oracle_b_true_obs1_vals,
                oracle_belief_l1_vals,
            ) = \
                self.collect_rollout(rnn_states, beliefs)
            global_step += self.n_steps * self.n_envs

            with torch.no_grad():
                obs_t  = torch.from_numpy(self._current_obs).unsqueeze(0)
                current_roles = self.env.get_roles()
                rol_t  = torch.from_numpy(current_roles).unsqueeze(0).to(torch.int64)
                res, _ = self.policy.forward_sequence(obs_t, rnn_states, roles=rol_t)
                ri     = torch.from_numpy(current_roles.astype(np.int64))
                last_v = res['role_ext_value'][0][range(self.n_agents), ri]
                if self.use_belief_value_critic:
                    belief_last_v = self._belief_value(self._current_obs, beliefs)
                    last_v = last_v.clone()
                    last_v[self.red_agent_flat.to(last_v.device)] = belief_last_v.to(last_v.device)

            upd = self.ppo_update(buf, last_v.cpu(), epoch)

            if epoch % self.log_interval == 0:
                red_win  = float(np.mean(ep_red_wins))  if ep_red_wins  else float('nan')
                blue_win = float(np.mean(ep_blue_wins)) if ep_blue_wins else float('nan')
                red_score  = float(np.mean(ep_red_scores))  if ep_red_scores  else float('nan')
                blue_score = float(np.mean(ep_blue_scores)) if ep_blue_scores else float('nan')
                coop_succ = float(np.mean(ep_coop_success)) if ep_coop_success else float('nan')
                coop_ret  = float(np.mean(ep_coop_return)) if ep_coop_return else float('nan')
                coin_red_return = float(np.mean(ep_coin_red_returns)) if ep_coin_red_returns else float('nan')
                coin_blue_return = float(np.mean(ep_coin_blue_returns)) if ep_coin_blue_returns else float('nan')
                b_true      = float(np.mean(b_true_vals))
                b_true_blue = float(np.mean(b_true_blue_vals)) if b_true_blue_vals else float('nan')
                b_imp       = float(np.mean(b_imp_vals))
                b_true_prior = float(np.mean(b_true_prior_vals))
                b_imp_prior = float(np.mean(b_imp_prior_vals))
                b_true_jump = b_true - b_true_prior
                b_imp_jump = b_imp - b_imp_prior
                b_true_t0_prior = float(b_true_prior_vals[0]) if b_true_prior_vals else float('nan')
                b_true_t0_post = float(b_true_vals[0]) if b_true_vals else float('nan')
                oracle_b_true = float(np.mean(oracle_b_true_vals)) if oracle_b_true_vals else float('nan')
                oracle_b_true_obs0 = float(np.mean(oracle_b_true_obs0_vals)) if oracle_b_true_obs0_vals else float('nan')
                oracle_b_true_obs1 = float(np.mean(oracle_b_true_obs1_vals)) if oracle_b_true_obs1_vals else float('nan')
                oracle_belief_l1 = float(np.mean(oracle_belief_l1_vals)) if oracle_belief_l1_vals else float('nan')
                # EMA-smoothed belief metrics to reduce per-epoch noise from episode resets
                b_true_ema = b_true if b_true_ema is None else (1 - ema_alpha) * b_true_ema + ema_alpha * b_true
                b_imp_ema  = b_imp  if b_imp_ema  is None else (1 - ema_alpha) * b_imp_ema  + ema_alpha * b_imp

                extra = ""
                if ep_coop_success:
                    extra = f" coop_succ={coop_succ:.2f} coop_ret={coop_ret:.2f}"
                print(f"ep {epoch:4d} | red_win={red_win:.2f} blue_win={blue_win:.2f} "
                      f"red_score={red_score:.2f} blue_score={blue_score:.2f}{extra} "
                      f"b_true={b_true:.3f} bp_loss={upd['bp_loss_pos']:.4f} "
                      f"lambda_eff={upd['lambda_eff']:.3f} ramp={upd['ramp']:.2f}"
                      + (f" aa_term={upd['aa_term_abs']:.3f}" if self.use_advantage_alignment else "")
                      + (f" lola_loss={upd['lola_loss']:.4f}" if self.use_lola_dice else ""))

                if self.use_wandb:
                    is_signaling = self.scenario.startswith("multi_speaker") or self.scenario.startswith("signaling")
                    log_row = {
                        # train/ metrics
                        # Figure 4: avg probability observer assigns to red's TRUE role (want ↓)
                        "train/b_true":         b_true,        # avg over all observers
                        "train/b_true_blue":    b_true_blue,   # blue-only (matches BBM Fig 4 exactly)
                        "train/b_true_ema":     b_true_ema,    # EMA-smoothed — main curve to watch
                        "train/b_true_paper":   upd.get("b_true_paper", float('nan')),  # exp(-blue_role_nll), same as BBM Fig 4
                        "train/oracle_b_true":   oracle_b_true,
                        "train/oracle_b_true_observer_1": oracle_b_true_obs0,
                        "train/oracle_b_true_observer_2": oracle_b_true_obs1,
                        "train/oracle_belief_l1": oracle_belief_l1,
                        # Figure 5: team scores (cumulative game reward per episode)
                        "train/red_score":      red_score,
                        "train/blue_score":     blue_score,
                        # Figure 5 alt: win rates
                        "train/red_win_rate":   red_win,
                        "train/blue_win_rate":  blue_win,
                        # Figure 6: raw deception bonus (positive = more deceptive)
                        "train/dec_bonus":      upd['bp_loss_pos'],
                        "train/bbm_bonus":      upd['bbm_bonus_mean'],  # non-zero only in BBM-MLP mode
                        # BOS gradient diagnostics (duplicated under train/ for easy W&B panels).
                        "train/bos_lambda_eff":       upd['lambda_eff'],
                        "train/bos_bp_loss_raw":      upd['bp_loss_raw'],
                        "train/bos_head_grad_norm":   upd['head_grad_norm'],
                        "train/bos_role0_grad_norm":  upd['dec_head_role0_grad'],
                        "train/bos_other_grad_norm":  upd['dec_head_other_grad'],
                        "train/bos_nonhead_grad_pre": upd['dec_nonhead_grad_pre'],
                        "train/bos_enc_grad_norm":    upd['enc_grad_norm'],
                        "train/bos_dec_clip_frac":    upd['dec_clip_frac'],
                        "train/bos_dec_approx_kl":    upd['dec_approx_kl'],
                        "train/bos_grad_frac":         upd['bos_grad_frac'],
                        "train/bos_grad_allow_amplify": upd['bos_grad_allow_amplify'],
                        "train/bos_grad_max_scale":    upd['bos_grad_max_scale'],
                        "train/bos_injection_scale":   upd['bos_injection_scale'],
                        "train/bos_belief_temperature": upd['bos_belief_temperature'],
                        "train/bos_belief_floor":       upd['bos_belief_floor'],
                        "train/bos_correction_mode_coeff": upd['bos_correction_mode_coeff'],
                        "train/bos_shaper_adapter":   upd['bos_shaper_adapter'],
                        "train/bos_likelihood_obs_mode_estimated": upd['bos_likelihood_obs_mode_estimated'],
                        "train/bos_obs_estimator_loss": upd['bos_obs_estimator_loss'],
                        "train/bos_obs_estimator_mae":  upd['bos_obs_estimator_mae'],
                        "train/bos_coeff_abs":         upd['bos_coeff_abs'],
                        "train/bos_coeff_rms":         upd['bos_coeff_rms'],
                        "train/bos_belief_entropy_frac": upd['bos_belief_entropy_frac'],
                        "train/bos_coeff_gate_frac":   upd['bos_coeff_gate_frac'],
                        "train/bos_adaptive_scale":    upd['dec_adaptive_scale'],
                        "train/bos_grad_norm_pre_clip": upd['dec_grad_norm_pre_clip'],
                        "train/bos_grad_norm_post_clip": upd['dec_grad_norm_post_clip'],
                        "train/bos_grad_norm_final":   upd['dec_grad_norm_final'],
                        "train/bos_critic_loss":       upd['bos_critic_loss'],
                        "train/bos_critic_ev":         upd['bos_critic_ev'],
                        "train/bos_critic_corr":       upd['bos_critic_corr'],
                        "train/bos_critic_updates":    upd['bos_critic_updates'],
                        "train/ppo_grad_norm":         upd['ppo_grad_norm'],
                        "train/ppo_grad_norm_ema":     upd['ppo_grad_norm_ema'],

                        # ── misc/ : diagnostics, not for paper figures ──
                        "misc/b_imp":             b_imp,
                        "misc/b_imp_ema":         b_imp_ema,
                        "misc/b_true_prior":      b_true_prior,
                        "misc/b_imp_prior":       b_imp_prior,
                        "misc/b_true_jump":       b_true_jump,
                        "misc/b_imp_jump":        b_imp_jump,
                        "misc/b_true_t0_prior":   b_true_t0_prior,
                        "misc/b_true_t0_post":    b_true_t0_post,
                        "misc/bp_loss_raw":       upd['bp_loss_raw'],
                        "misc/lambda_eff":        upd['lambda_eff'],
                        "misc/ramp":              upd['ramp'],
                        "misc/policy_loss":       upd['pol_loss'],
                        "misc/value_loss":        upd['vf_loss'],
                        "misc/entropy":           -upd['ent_loss'],
                        "misc/dec_clip_frac":     upd['dec_clip_frac'],
                        "misc/dec_approx_kl":     upd['dec_approx_kl'],
                        "misc/bos_critic_hidden_dim": upd['bos_critic_hidden_dim'],
                        "misc/bos_critic_depth":      upd['bos_critic_depth'],
                        "misc/head_role0_grad":   upd['dec_head_role0_grad'],
                        "misc/head_other_grad":   upd['dec_head_other_grad'],
                        "misc/nonhead_grad_pre":  upd['dec_nonhead_grad_pre'],
                        "misc/head_grad_norm":    upd['head_grad_norm'],
                        "misc/enc_grad_norm":     upd['enc_grad_norm'],
                    }
                    if is_signaling:
                        # Signaling aliases: red slot is honest/listener objective; blue slot is decoy objective.
                        log_row["train/honest_score"] = red_score
                        log_row["train/decoy_score"] = blue_score
                    if "avalon" in self.scenario:
                        rollout_diag = getattr(self, "_last_rollout_diagnostics", {})
                        ally_b_true = rollout_diag.get("avalon_ally_b_true", float("nan"))
                        resistance_b_true = rollout_diag.get("avalon_resistance_b_true", float("nan"))
                        log_row["train/avalon_spy_score"] = red_score
                        log_row["train/avalon_resistance_score"] = blue_score
                        log_row["train/avalon_spy_win_rate"] = red_win
                        log_row["train/avalon_resistance_win_rate"] = blue_win
                        log_row["train/avalon_ally_surrogate_b_true"] = ally_b_true
                        log_row["train/avalon_resistance_surrogate_b_true"] = resistance_b_true
                        log_row["train/avalon_ally_resistance_surrogate_gap"] = ally_b_true - resistance_b_true
                        log_row["train/avalon_resistance_approve_shaper_team"] = rollout_diag.get(
                            "avalon_resistance_approve_shaper_team", float("nan")
                        )
                        log_row["train/avalon_resistance_approve_other_team"] = rollout_diag.get(
                            "avalon_resistance_approve_other_team", float("nan")
                        )
                        log_row["train/avalon_resistance_approve_shaper_team_gap"] = (
                            log_row["train/avalon_resistance_approve_shaper_team"]
                            - log_row["train/avalon_resistance_approve_other_team"]
                        )
                        log_row["train/avalon_focal_spy_sabotage_rate"] = rollout_diag.get(
                            "avalon_focal_spy_sabotage_rate", float("nan")
                        )
                        for role in range(2, self.n_roles):
                            log_row[f"train/avalon_resistance_role{role}_surrogate_b_true"] = rollout_diag.get(
                                f"avalon_resistance_role{role}_b_true", float("nan")
                            )
                    if "coin_game" in self.scenario:
                        log_row["train/coin_red_return"] = coin_red_return
                        log_row["train/coin_blue_return"] = coin_blue_return
                        log_row["train/coin_observer_b_true"] = b_true_blue
                    if ep_coop_success:
                        log_row["train/coop_success"] = coop_succ
                        log_row["train/coop_return"] = coop_ret
                        if is_signaling:
                            log_row["train/listener_exit_success"] = coop_succ
                            log_row["train/honest_return"] = coop_ret
                    if self.method_name != "bos":
                        for key in list(log_row):
                            if key.startswith("train/bos_"):
                                log_row.pop(key)
                    if self.use_belief_value_critic:
                        log_row["train/obs_estimator_loss"] = upd["bos_obs_estimator_loss"]
                        log_row["train/obs_estimator_mae"] = upd["bos_obs_estimator_mae"]
                        log_row["train/belief_critic_ev"] = upd["bos_critic_ev"]
                    if self.use_advantage_alignment:
                        log_row["train/aa_weight"] = upd["aa_weight"]
                        log_row["train/aa_term_abs"] = upd["aa_term_abs"]
                    if self.use_lola_dice:
                        log_row["train/lola_weight"] = upd["lola_weight"]
                        log_row["train/lola_loss"] = upd["lola_loss"]
                        log_row["train/lola_grad_norm_final"] = upd["lola_grad_norm_final"]
                        log_row["train/lola_ppo_head_grad_norm"] = upd["lola_ppo_head_grad_norm"]
                        log_row["train/lola_grad_frac"] = self.lola_grad_frac
                    wandb.log(log_row, step=global_step)

            if epoch % eval_interval == 0:
                metrics = self.evaluate(n_episodes=self.config.get("eval_episodes", 100))
                ev_extra = ""
                if "coop_success_rate" in metrics:
                    ev_extra = f" coop_succ={metrics['coop_success_rate']:.2f}"
                print(f"         eval | red_win={metrics.get('red_win_rate',0):.2f} "
                      f"b_true_final={metrics.get('b_true_final', float('nan')):.3f}{ev_extra}")
                if self.use_wandb:
                    evlog = {
                        "eval/b_true_final":  metrics.get("b_true_final",  float('nan')),
                        "eval/b_true_at_k":   metrics.get("b_true_at_k",   float('nan')),
                        "eval/red_win_rate":  metrics.get("red_win_rate",  float('nan')),
                        "eval/blue_win_rate": metrics.get("blue_win_rate", float('nan')),
                        "eval/red_score":     metrics.get("red_score",     float('nan')),
                        "eval/blue_score":    metrics.get("blue_score",    float('nan')),
                        "misc/eval_belief_var": metrics.get("belief_var",  float('nan')),
                    }
                    is_signaling = self.scenario.startswith("multi_speaker") or self.scenario.startswith("signaling")
                    if "coop_success_rate" in metrics:
                        evlog["eval/coop_success"] = metrics["coop_success_rate"]
                        if is_signaling:
                            evlog["eval/listener_exit_success"] = metrics["coop_success_rate"]
                    if "avalon" in self.scenario:
                        evlog["eval/avalon_spy_win_rate"] = metrics.get("red_win_rate", float('nan'))
                        evlog["eval/avalon_resistance_win_rate"] = metrics.get("blue_win_rate", float('nan'))
                        evlog["eval/avalon_b_true_final"] = metrics.get("b_true_final", float('nan'))
                        evlog["eval/avalon_spy_score"] = metrics.get("red_score", float('nan'))
                        evlog["eval/avalon_resistance_score"] = metrics.get("blue_score", float('nan'))
                    if "coin_game" in self.scenario:
                        evlog["eval/coin_b_true_final"] = metrics.get("b_true_final", float('nan'))
                        evlog["eval/coin_red_return"] = metrics.get("red_score", float('nan'))
                        evlog["eval/coin_blue_return"] = metrics.get("blue_score", float('nan'))
                    wandb.log(evlog, step=global_step)

            if self.save_dir and epoch % self.save_interval == 0:
                self._save(epoch)

        if self.save_dir:
            self._save(self.n_train_epochs, final=True)
