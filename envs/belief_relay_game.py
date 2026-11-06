# belief relay game: shaper (seat 0) sends one joint action; ally and adversary each see their own sub-symbol
# reward comes only from the ally's investment / adversary's raid outcome, never from beliefs


import numpy as np


class Space:
    def __init__(self, n=None, shape=None, dtype=np.uint8):
        self.n = n
        self.shape = shape or (n,)
        self.dtype = dtype


def bayes_update(prior, log_lik, eps=1e-8):
    prior = np.asarray(prior, dtype=np.float64)
    log_lik = np.asarray(log_lik, dtype=np.float64)
    log_post = np.log(prior + eps) + log_lik
    log_post -= log_post.max(axis=-1, keepdims=True)
    post = np.exp(log_post)
    post = post / (post.sum(axis=-1, keepdims=True) + eps)
    return post.astype(np.float32)


class BeliefRelayGame:

    n_agents_per_env = 3
    n_roles = 4            # shaper hidden types z in {0,1,2,3}
    n_guilds = 2            # guild(z) = z // 2  ->  guild A={0,1}, guild B={2,3}
    n_channel_symbols = 3   # alphabet size of EACH observer's own channel
    n_actions = n_channel_symbols * n_channel_symbols  # joint action = 9 (shaper)

    INVEST_PAYOFF = 1.0
    RAID_PAYOFF = 1.0

    def __init__(self, num_envs=64, round_len=3, n_blocks=5):
        self.num_envs = int(num_envs)
        self.round_len = int(round_len)          # k: rounds per block
        self.n_blocks = int(n_blocks)
        self.max_steps = self.round_len * self.n_blocks
        self.total_agents = self.num_envs * self.n_agents_per_env
        self.max_players = self.n_agents_per_env
        self.max_roles = self.n_roles

        # obs: t, round, seat, z (shaper), channel symbol + belief (ally/adv)
        self.obs_dim = 2 + self.n_agents_per_env + self.n_roles + self.n_channel_symbols + self.n_roles
        self.observation_space = Space(shape=(self.obs_dim,), dtype=np.uint8)
        self.action_space = Space(n=self.n_actions)

        # joint action -> ally symbol (a // 3), adversary symbol (a % 3)
        joint = np.arange(self.n_actions)
        self.ally_decode = (joint // self.n_channel_symbols).astype(np.int64)   # a_ally = a // 3
        self.adv_decode = (joint % self.n_channel_symbols).astype(np.int64)     # a_adv  = a % 3
        self.channel_decode = np.stack([self.ally_decode, self.adv_decode], axis=0)  # [2, 9]

        self.steps = np.zeros(self.num_envs, dtype=np.int32)
        self.z_true = np.zeros(self.num_envs, dtype=np.int64)
        self.last_signal = np.full(self.num_envs, -1, dtype=np.int64)  # -1 = none yet this block
        self.ally_belief = np.full((self.num_envs, self.n_roles), 1.0 / self.n_roles, dtype=np.float32)
        self.adv_belief = np.full((self.num_envs, self.n_roles), 1.0 / self.n_roles, dtype=np.float32)
        self.last_ally_belief = self.ally_belief.copy()
        self.last_adv_belief = self.adv_belief.copy()
        self.episode_returns = np.zeros((self.num_envs, self.n_agents_per_env), dtype=np.float32)
        self.focal_role_log_probs = np.full(
            (self.num_envs, self.n_roles, self.n_actions), -np.log(self.n_actions), dtype=np.float32)

        # observers' learned log P(symbol | z), set by the trainer each epoch
        self.observer_channel_log_liks = np.full(
            (2, self.n_roles, self.n_channel_symbols),
            -np.log(self.n_channel_symbols), dtype=np.float32)

    def _new_block(self, env_i):
        """Start a fresh k-round signaling block: resample z, reset beliefs."""
        self.z_true[env_i] = np.random.randint(self.n_roles)
        self.last_signal[env_i] = -1
        self.ally_belief[env_i] = 1.0 / self.n_roles
        self.adv_belief[env_i] = 1.0 / self.n_roles

    def _reset_env(self, env_i):
        self.steps[env_i] = 0
        self.episode_returns[env_i] = 0.0
        self._new_block(env_i)
        self.last_ally_belief[env_i] = self.ally_belief[env_i]
        self.last_adv_belief[env_i] = self.adv_belief[env_i]

    def reset(self):
        for env_i in range(self.num_envs):
            self._reset_env(env_i)
        return self._get_obs()

    def get_roles(self):
        # Seat identity per physical agent: 0=shaper, 1=ally, 2=adversary.
        return np.tile(np.arange(self.n_agents_per_env), self.num_envs).astype(np.int64)

    def get_red_agent_indices(self):
        return np.arange(self.num_envs, dtype=np.int64) * self.n_agents_per_env

    def get_z_true_per_env(self):
        return self.z_true.astype(np.int64, copy=True)

    def get_oracle_observer_beliefs(self):
        """Current [ally, adversary] posterior, shape [E, 2, R]."""
        return np.stack([self.ally_belief, self.adv_belief], axis=1).copy()

    def get_last_oracle_observer_beliefs(self):
        """Posterior right after the most recent public signal, pre-reset."""
        return np.stack([self.last_ally_belief, self.last_adv_belief], axis=1).copy()

    def get_initial_observer_beliefs(self):
        return np.full((self.num_envs, 2, self.n_roles), 1.0 / self.n_roles, dtype=np.float32)

    def get_observer_channel_decode(self):
        return self.channel_decode.copy()

    def set_focal_role_log_probs(self, role_log_probs):
        values = np.asarray(role_log_probs, dtype=np.float32)
        expected = (self.num_envs, self.n_roles, self.n_actions)
        if values.shape != expected:
            raise ValueError(f"Expected focal role log-probabilities {expected}, got {values.shape}")
        self.focal_role_log_probs = values.copy()

    def set_observer_channel_log_liks(self, ally_log_liks, adv_log_liks):
        ally = np.asarray(ally_log_liks, dtype=np.float32)
        adv = np.asarray(adv_log_liks, dtype=np.float32)
        expected = (self.n_roles, self.n_channel_symbols)
        if ally.shape != expected or adv.shape != expected:
            raise ValueError(f"Expected observer channel log-liks {expected}, got {ally.shape}/{adv.shape}")
        self.observer_channel_log_liks[0] = ally
        self.observer_channel_log_liks[1] = adv

    def _get_obs(self):
        obs = np.zeros((self.total_agents, self.obs_dim), dtype=np.float32)
        z_slice = slice(5, 5 + self.n_roles)                                  # [5:9]
        ch_slice = slice(5 + self.n_roles, 5 + self.n_roles + self.n_channel_symbols)  # [9:12]
        belief_slice = slice(ch_slice.stop, ch_slice.stop + self.n_roles)     # [12:16]
        for env_i in range(self.num_envs):
            local_t = self.steps[env_i] % self.round_len
            t_norm = self.steps[env_i] / float(max(self.max_steps, 1))
            round_norm = local_t / float(max(self.round_len, 1))
            has_signal = self.last_signal[env_i] >= 0
            a_ally = int(self.ally_decode[self.last_signal[env_i]]) if has_signal else -1
            a_adv = int(self.adv_decode[self.last_signal[env_i]]) if has_signal else -1
            for agent_i in range(self.n_agents_per_env):
                flat_i = env_i * self.n_agents_per_env + agent_i
                row = obs[flat_i]
                row[0] = t_norm
                row[1] = round_norm
                row[2 + agent_i] = 1.0                      # own seat one-hot
                if agent_i == 0:                             # shaper: private type only
                    row[z_slice.start + self.z_true[env_i]] = 1.0
                elif agent_i == 1:                           # ally: own channel + own belief
                    if a_ally >= 0:
                        row[ch_slice.start + a_ally] = 1.0
                    row[belief_slice] = self.ally_belief[env_i]
                elif agent_i == 2:                           # adversary: own channel + own belief
                    if a_adv >= 0:
                        row[ch_slice.start + a_adv] = 1.0
                    row[belief_slice] = self.adv_belief[env_i]
        return np.rint(obs * 255.0).clip(0, 255).astype(np.uint8)

    def _channel_marginal_log_lik(self, log_probs_row, decode_row):
        """Marginalize log p(a_joint | z) [R, 9] over a decode map [9] -> [R, 3]."""
        probs = np.exp(log_probs_row.astype(np.float64))  # [R, 9]
        marg = np.zeros((self.n_roles, self.n_channel_symbols), dtype=np.float64)
        for a in range(self.n_actions):
            marg[:, decode_row[a]] += probs[:, a]
        marg = np.clip(marg, 1e-12, None)
        return np.log(marg)

    def _update_beliefs(self, env_i, signal):
        a_ally = int(self.ally_decode[signal])
        a_adv = int(self.adv_decode[signal])
        ally_lik = self.observer_channel_log_liks[0, :, a_ally]
        adv_lik = self.observer_channel_log_liks[1, :, a_adv]
        self.ally_belief[env_i] = bayes_update(self.ally_belief[env_i], ally_lik)
        self.adv_belief[env_i] = bayes_update(self.adv_belief[env_i], adv_lik)

    def step(self, actions_flat):
        actions = np.asarray(actions_flat, dtype=np.int64).reshape(
            self.num_envs, self.n_agents_per_env)
        actions = np.clip(actions, 0, self.n_actions - 1)
        rewards = np.zeros_like(actions, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = [{} for _ in range(self.total_agents)]

        for env_i in range(self.num_envs):
            signal = int(actions[env_i, 0])
            self._update_beliefs(env_i, signal)
            self.last_signal[env_i] = signal
            self.last_ally_belief[env_i] = self.ally_belief[env_i]
            self.last_adv_belief[env_i] = self.adv_belief[env_i]

            local_t = self.steps[env_i] % self.round_len
            block_end = (local_t == self.round_len - 1)
            if block_end:
                # reward: ally invest / adversary raid on the right guild
                guild = int(self.z_true[env_i]) // (self.n_roles // self.n_guilds)
                venture_ally = int(actions[env_i, 1]) % self.n_guilds
                venture_adv = int(actions[env_i, 2]) % self.n_guilds
                invest_hit = (venture_ally == guild)
                raid_hit = (venture_adv == guild)

                if invest_hit:
                    rewards[env_i, 1] += self.INVEST_PAYOFF   # ally's own share
                    rewards[env_i, 0] += self.INVEST_PAYOFF   # shaper benefits from correct investment
                if raid_hit:
                    rewards[env_i, 2] += self.RAID_PAYOFF     # adversary's own share
                    rewards[env_i, 0] -= self.RAID_PAYOFF     # shaper loses resources to successful raid

            self.episode_returns[env_i] += rewards[env_i]
            self.steps[env_i] += 1

            if block_end and not (self.steps[env_i] >= self.max_steps):
                # Start next block within the same episode.
                self._new_block(env_i)

            if self.steps[env_i] >= self.max_steps:
                dones[env_i] = True
                shaper_return = float(self.episode_returns[env_i, 0])
                observer_return = float(self.episode_returns[env_i, 1:].sum())
                info = {
                    "outcome": "completed",
                    "game_scores": np.array([[shaper_return, 0.0, observer_return]], dtype=np.float32),
                    "episode_return": shaper_return,
                    "ally_belief_true_final": float(self.ally_belief[env_i, self.z_true[env_i]]),
                    "adv_belief_true_final": float(self.adv_belief[env_i, self.z_true[env_i]]),
                }
                base = env_i * self.n_agents_per_env
                for agent_i in range(self.n_agents_per_env):
                    infos[base + agent_i] = info
                self._reset_env(env_i)

        return self._get_obs(), rewards.reshape(-1), np.repeat(dones, self.n_agents_per_env), infos


class WrapperEnv(BeliefRelayGame):
    def __init__(self, parallel_envs, round_len=3, n_blocks=5):
        super().__init__(num_envs=parallel_envs, round_len=round_len, n_blocks=n_blocks)

        class DummyPlayer:
            def __init__(self, idx):
                self.index = idx
                self.is_dead = False

        class DummyGame:
            def __init__(self):
                self.n_players = 3
                self.round_outcome = ""
                self.round_team_scores = (0.0, 0.0, 0.0)

        self.players = [DummyPlayer(i) for i in range(self.total_agents)]
        self.games = [DummyGame() for _ in range(self.num_envs)]
