import numpy as np


class Space:
    """A minimal mock of gymnasium.spaces.Space."""

    def __init__(self, n=None, shape=None, dtype=np.uint8):
        self.n = n
        self.shape = shape or (n,)
        self.dtype = dtype


class HiddenRoleCoinGameMulti:
    # coin game: agent 0 = red shaper, rest = observers (role 2)

    OBSERVER_ROLE = 2

    def __init__(self, num_envs=32, n_blue=3, grid_size=5, max_steps=30, full_obs=False):
        self.num_envs = int(num_envs)
        self.n_blue = int(n_blue)
        self.n_agents_per_env = 1 + self.n_blue
        self.total_agents = self.num_envs * self.n_agents_per_env
        self.max_players = self.n_agents_per_env
        # Red's two hidden types plus the observer role.
        self.max_roles = 3
        self.grid_size = int(grid_size)
        self.max_steps = int(max_steps)
        self.full_obs = bool(full_obs)

        obs_dim = (5 + 2 * self.n_blue + 3) if self.full_obs else 10
        self.observation_space = Space(shape=(obs_dim,), dtype=np.uint8)
        self.action_space = Space(n=4)

        self.steps = np.zeros(self.num_envs, dtype=np.int32)
        self.pos = np.zeros((self.num_envs, self.n_agents_per_env, 2), dtype=np.int32)
        self.coin_pos = np.zeros((self.num_envs, 2), dtype=np.int32)
        self.coin_color = np.zeros(self.num_envs, dtype=np.int32)  # 0 red, 1 blue
        self.red_roles = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_returns = np.zeros((self.num_envs, self.n_agents_per_env), dtype=np.float32)

    def _spawn_coin(self, e):
        self.coin_pos[e] = np.random.randint(0, self.grid_size, size=2)
        self.coin_color[e] = np.random.randint(0, 2)

    def _reset_env(self, e):
        self.steps[e] = 0
        for a in range(self.n_agents_per_env):
            self.pos[e, a] = np.random.randint(0, self.grid_size, size=2)
        self._spawn_coin(e)
        self.red_roles[e] = np.random.randint(0, 2)
        self.episode_returns[e] = 0.0

    def reset(self):
        for e in range(self.num_envs):
            self._reset_env(e)
        return self._get_obs()

    def get_roles(self):
        roles = np.zeros(self.total_agents, dtype=np.int32)
        for e in range(self.num_envs):
            base = e * self.n_agents_per_env
            roles[base] = self.red_roles[e]
            roles[base + 1 : base + self.n_agents_per_env] = self.OBSERVER_ROLE
        return roles

    def get_z_true_per_env(self):
        return self.red_roles.astype(np.int64)

    def get_initial_observer_beliefs(self):
        """Uniform prior over red's two types; none on the observer role."""
        beliefs = np.zeros((self.num_envs, self.n_blue, self.max_roles), dtype=np.float32)
        beliefs[..., :2] = 0.5
        return beliefs

    def get_red_agent_indices(self):
        return np.arange(self.num_envs, dtype=np.int64) * self.n_agents_per_env

    def _get_obs(self):
        obs_dim = self.observation_space.shape[0]
        obs = np.zeros((self.total_agents, obs_dim), dtype=np.float32)
        for e in range(self.num_envs):
            t = self.steps[e] / float(max(self.max_steps, 1))
            rx, ry = self.pos[e, 0] / float(self.grid_size)
            cx, cy = self.coin_pos[e] / float(self.grid_size)
            cc = float(self.coin_color[e])
            if self.full_obs:
                blue_pos = (self.pos[e, 1:1 + self.n_blue] / float(self.grid_size)).reshape(-1)
                for a in range(self.n_agents_per_env):
                    ax, ay = self.pos[e, a] / float(self.grid_size)
                    base = e * self.n_agents_per_env + a
                    obs[base] = [t, ax, ay, rx, ry, *blue_pos, cx, cy, cc]
            else:
                bx, by = self.pos[e, 1] / float(self.grid_size)
                nblue = float(self.n_blue) / 5.0
                for a in range(self.n_agents_per_env):
                    ax, ay = self.pos[e, a] / float(self.grid_size)
                    base = e * self.n_agents_per_env + a
                    obs[base] = [t, ax, ay, rx, ry, bx, by, cx, cy, (cc + nblue) / 2.0]
        return (obs * 255).astype(np.uint8)

    def step(self, actions_flat):
        actions = np.asarray(actions_flat, dtype=np.int64).reshape(self.num_envs, self.n_agents_per_env)
        rewards = np.zeros((self.num_envs, self.n_agents_per_env), dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = [{} for _ in range(self.total_agents)]

        for e in range(self.num_envs):
            for a in range(self.n_agents_per_env):
                act = int(actions[e, a])
                if act == 0:
                    self.pos[e, a, 1] -= 1
                elif act == 1:
                    self.pos[e, a, 1] += 1
                elif act == 2:
                    self.pos[e, a, 0] -= 1
                elif act == 3:
                    self.pos[e, a, 0] += 1
                self.pos[e, a] %= self.grid_size

            collector = None
            for a in range(self.n_agents_per_env):
                if np.array_equal(self.pos[e, a], self.coin_pos[e]):
                    collector = a
                    break

            if collector is not None:
                rewards[e, collector] += 1.0
                if collector == 0 and self.coin_color[e] == 1:
                    # Red took blue coin: hurt all observers equally.
                    rewards[e, 1:] -= 2.0 / float(self.n_blue)
                    # Altruist red is penalized for taking blue coin.
                    if self.red_roles[e] == 1:
                        rewards[e, 0] -= 1.0
                elif collector > 0 and self.coin_color[e] == 0:
                    rewards[e, 0] -= 2.0
                self._spawn_coin(e)

            self.episode_returns[e] += rewards[e]
            self.steps[e] += 1
            if self.steps[e] >= self.max_steps:
                dones[e] = True
                red_score = float(self.episode_returns[e, 0])
                blue_score = float(self.episode_returns[e, 1:].sum())
                gs = np.array([[red_score, 0.0, blue_score]], dtype=np.float32)
                term = {
                    "outcome": "completed",
                    "game_scores": gs,
                    "episode_return": red_score,
                    "coin_red_return": red_score,
                    "coin_blue_return": blue_score,
                }
                base = e * self.n_agents_per_env
                for a in range(self.n_agents_per_env):
                    infos[base + a] = term
                self._reset_env(e)

        return self._get_obs(), rewards.reshape(-1), np.repeat(dones, self.n_agents_per_env), infos


class WrapperEnv(HiddenRoleCoinGameMulti):
    def __init__(self, parallel_envs):
        super().__init__(num_envs=parallel_envs)

        class DummyPlayer:
            def __init__(self, idx):
                self.index = idx
                self.is_dead = False

        class DummyGame:
            def __init__(self, n_players):
                self.n_players = n_players
                self.round_outcome = ""
                self.round_team_scores = (0.0, 0.0, 0.0)

        self.players = [DummyPlayer(i) for i in range(self.total_agents)]
        self.games = [DummyGame(self.n_agents_per_env) for _ in range(self.num_envs)]


class WrapperEnvFullObs(HiddenRoleCoinGameMulti):
    # observers see all blue positions

    def __init__(self, parallel_envs):
        super().__init__(num_envs=parallel_envs, full_obs=True)

        class DummyPlayer:
            def __init__(self, idx):
                self.index = idx
                self.is_dead = False

        class DummyGame:
            def __init__(self, n_players):
                self.n_players = n_players
                self.round_outcome = ""
                self.round_team_scores = (0.0, 0.0, 0.0)

        self.players = [DummyPlayer(i) for i in range(self.total_agents)]
        self.games = [DummyGame(self.n_agents_per_env) for _ in range(self.num_envs)]


class WrapperEnv3(HiddenRoleCoinGameMulti):
    """Three-agent variant: one hidden-role red shaper and two blue observers."""

    def __init__(self, parallel_envs):
        super().__init__(num_envs=parallel_envs, n_blue=2)

        class DummyPlayer:
            def __init__(self, idx):
                self.index = idx
                self.is_dead = False

        class DummyGame:
            def __init__(self, n_players):
                self.n_players = n_players
                self.round_outcome = ""
                self.round_team_scores = (0.0, 0.0, 0.0)

        self.players = [DummyPlayer(i) for i in range(self.total_agents)]
        self.games = [DummyGame(self.n_agents_per_env) for _ in range(self.num_envs)]
