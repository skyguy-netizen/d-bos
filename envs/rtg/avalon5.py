# 5-player Resistance: 2 spies, 3 resistance
from __future__ import annotations

from itertools import combinations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .marl_env import MultiAgentEnv

N_PLAYERS = 5
N_ROUNDS = 5
# Standard 5-player Resistance mission sizes (same as common published tables).
MISSION_TEAM_SIZES = [2, 3, 2, 3, 3]
MAX_CONSEC_REJECTS = 5
MAX_STEPS = 400

REWARD_PER_QUEST = 1.0
REWARD_GAME_WIN = 10.0

PHASE_PROPOSE = 0
PHASE_VOTE = 1
PHASE_QUEST = 2

# History: played, team_mask (5 bits), 5× vote byte, outcome, sabotages, pad×2 = 12
HIST_BLOCK = 12
HIST_OFFSET = 0
CURR_OFFSET = N_ROUNDS * HIST_BLOCK  # 60

OBS_DIM = 128
# C(5,2)=C(5,3)=10 — all mission sizes in the table use ≤10 combinations.
ACTION_DIM = 10


def _spy_pair_for_slot(slot: int) -> tuple[int, int]:
    pairs = list(combinations(range(N_PLAYERS), 2))
    return pairs[slot % len(pairs)]


class ResistancePlayer:
    __slots__ = ("index", "local", "team", "x", "y", "is_dead", "is_alive")

    def __init__(self, local_idx: int, team: int):
        self.local = local_idx
        self.index = local_idx
        self.team = team
        self.x = self.y = 0
        self.is_dead = False
        self.is_alive = True

    def in_vision(self, x, y, distance=999):
        return True

    @property
    def is_spy(self) -> bool:
        return self.team in (0, 1)


class Resistance5Env(MultiAgentEnv):
    """Single-table five-player Resistance game."""

    observation_space = spaces.Box(low=0, high=255, shape=(OBS_DIM, 1, 1), dtype=np.uint8)
    action_space = spaces.Discrete(ACTION_DIM)

    supports_env_noops = False

    def __init__(
        self,
        name: str = "resistance5",
        spy_pair_slot: int = 0,
        max_steps: int = MAX_STEPS,
        reject_limit: int = MAX_CONSEC_REJECTS,
        reward_per_quest: float = REWARD_PER_QUEST,
        reward_game_win: float = REWARD_GAME_WIN,
        blind: bool = False,
    ):
        super().__init__(n_players=N_PLAYERS)
        self.name = name
        self.blind = blind
        self.spy_pair_slot = int(spy_pair_slot)
        self.max_steps = int(max_steps)
        self.reject_limit = int(reject_limit)
        self.reward_per_quest = float(reward_per_quest)
        self.reward_game_win = float(reward_game_win)
        sp1, sp2 = _spy_pair_for_slot(spy_pair_slot)
        self._spy_set = frozenset({sp1, sp2})

        teams = []
        sorted_spies = sorted(self._spy_set)
        resist_sorted = sorted(p for p in range(N_PLAYERS) if p not in self._spy_set)
        for p in range(N_PLAYERS):
            if p == sorted_spies[0]:
                teams.append(0)
            elif p == sorted_spies[1]:
                teams.append(1)
            else:
                teams.append(2 + resist_sorted.index(p))
        self.players = [ResistancePlayer(i, teams[i]) for i in range(N_PLAYERS)]

        self.round_timer = 0
        self.round_outcome = ""
        self.round_team_scores = np.zeros(3, dtype=np.float64)
        self._episode_team_gs = np.zeros(3, dtype=np.float64)
        self._rng = np.random.default_rng()
        self._combos = {k: list(combinations(range(N_PLAYERS), k)) for k in set(MISSION_TEAM_SIZES)}
        self._reset_state()

    @property
    def n_agents(self) -> int:
        return N_PLAYERS

    def _reset_state(self):
        self.phase = PHASE_PROPOSE
        self.round_idx = 0
        self.leader_idx = 0
        self.spy_mission_wins = 0   # failed missions
        self.res_mission_wins = 0   # successful missions
        self.consec_rejects = 0
        self.proposed_team: list[int] = []
        self.step_count = 0
        self.round_timer = 0
        self.done = False
        self.outcome = None
        self.round_outcome = ""
        self.round_team_scores[:] = 0
        self._episode_team_gs[:] = 0
        self.round_history: list[dict] = []
        self.current_votes = [0] * N_PLAYERS

    def _mission_size(self) -> int:
        return MISSION_TEAM_SIZES[self.round_idx]

    def _build_history_block(self, r: int) -> np.ndarray:
        blk = np.zeros(HIST_BLOCK, dtype=np.uint8)
        if r >= len(self.round_history):
            return blk
        h = self.round_history[r]
        blk[0] = 255
        mask = 0
        for j in h["team"]:
            mask |= 1 << j
        blk[1] = min(mask, 255)
        for j in range(N_PLAYERS):
            blk[2 + j] = int(h["votes"][j] * 255)
        if h["outcome"] == 1:
            blk[7] = 128
        elif h["outcome"] == 2:
            blk[7] = 255
        blk[8] = min(int(h["sabotages"]) * 80, 255)
        return blk

    def _obs_for(self, local_idx: int) -> np.ndarray:
        obs = np.zeros(OBS_DIM, dtype=np.uint8)
        for r in range(N_ROUNDS):
            lo = HIST_OFFSET + r * HIST_BLOCK
            obs[lo : lo + HIST_BLOCK] = self._build_history_block(r)

        obs[CURR_OFFSET + 0] = int(min(self.round_idx * 51, 255))
        obs[CURR_OFFSET + 1] = {PHASE_PROPOSE: 0, PHASE_VOTE: 128, PHASE_QUEST: 255}[self.phase]
        obs[CURR_OFFSET + 2] = int(min(self.leader_idx * 51, 255))
        obs[CURR_OFFSET + 3] = 255 if local_idx == self.leader_idx else 0
        obs[CURR_OFFSET + 4] = 255 if local_idx in self.proposed_team else 0
        mask = 0
        for j in self.proposed_team:
            mask |= 1 << j
        obs[CURR_OFFSET + 5] = min(mask, 255)
        obs[CURR_OFFSET + 6] = min(self.consec_rejects * 50, 255)
        obs[CURR_OFFSET + 7] = int(min(self.spy_mission_wins * 85, 255))
        obs[CURR_OFFSET + 8] = int(min(self.res_mission_wins * 85, 255))
        obs[CURR_OFFSET + 9] = 255 if self.spy_mission_wins >= 3 else 0
        obs[CURR_OFFSET + 10] = 255 if self.res_mission_wins >= 3 else 0
        obs[CURR_OFFSET + 11] = min(self._mission_size() * 50, 255)

        priv = CURR_OFFSET + 16
        obs[priv + 0] = 255 if self.players[local_idx].is_spy else 0
        obs[priv + 1] = min(self.step_count * 2, 255)
        obs[priv + 2] = 255 if (
            self.players[local_idx].is_spy and local_idx in self.proposed_team
        ) else 0
        if self.players[local_idx].is_spy and not getattr(self, "blind", False):
            spy_mask = 0
            for j in range(N_PLAYERS):
                if j != local_idx and self.players[j].is_spy:
                    spy_mask |= (1 << j)
            obs[priv + 3] = spy_mask

        return obs.reshape(OBS_DIM, 1, 1)

    def _all_obs(self) -> np.ndarray:
        return np.stack([self._obs_for(i) for i in range(N_PLAYERS)])

    def _propose(self, actions: list[int]):
        k = self._mission_size()
        combos = self._combos[k]
        a = int(actions[self.leader_idx]) % len(combos)
        self.proposed_team = list(combos[a])
        self.current_votes = [0] * N_PLAYERS
        self.phase = PHASE_VOTE
        return np.zeros(N_PLAYERS), [{} for _ in range(N_PLAYERS)]

    def _vote(self, actions: list[int]):
        votes = [1 if int(a) == 1 else 0 for a in actions]
        self.current_votes = votes
        n_reject = sum(votes)
        # Strict majority approve among 5: at least 3 approves.
        if n_reject <= 2:
            self.consec_rejects = 0
            self.phase = PHASE_QUEST
        else:
            self.consec_rejects += 1
            if self.consec_rejects >= self.reject_limit:
                return self._end_game(spies_win=True)
            self.leader_idx = (self.leader_idx + 1) % N_PLAYERS
            self.phase = PHASE_PROPOSE
            self.proposed_team = []
        return np.zeros(N_PLAYERS), [{} for _ in range(N_PLAYERS)]

    def _quest(self, actions: list[int]):
        sabotages = 0
        for j in self.proposed_team:
            act = int(actions[j])
            if self.players[j].is_spy:
                sabotages += int(act == 1)
            else:
                # Resistance must play success; ignore sabotage attempt.
                pass
        spy_won = sabotages > 0

        rewards = np.zeros(N_PLAYERS, dtype=np.float64)
        for p in self.players:
            if p.is_spy:
                rewards[p.local] += self.reward_per_quest if spy_won else -self.reward_per_quest
            else:
                rewards[p.local] += -self.reward_per_quest if spy_won else self.reward_per_quest

        for p in self.players:
            if p.is_spy:
                self._episode_team_gs[0] += rewards[p.local]
            else:
                self._episode_team_gs[2] += rewards[p.local]

        self.round_history.append(
            {
                "team": list(self.proposed_team),
                "votes": list(self.current_votes),
                "outcome": 1 if spy_won else 2,
                "sabotages": sabotages,
            }
        )

        if spy_won:
            self.spy_mission_wins += 1
        else:
            self.res_mission_wins += 1

        if self.spy_mission_wins >= 3:
            bonus, infos = self._end_game(spies_win=True)
            return rewards + bonus, infos
        if self.res_mission_wins >= 3:
            bonus, infos = self._end_game(spies_win=False)
            return rewards + bonus, infos

        self.round_idx += 1
        self.leader_idx = (self.leader_idx + 1) % N_PLAYERS
        self.phase = PHASE_PROPOSE
        self.current_votes = [0] * N_PLAYERS
        return rewards, [{} for _ in range(N_PLAYERS)]

    def _end_game(self, spies_win: bool):
        self.done = True
        self.outcome = "spy_wins" if spies_win else "resistance_wins"
        spy_r = self.reward_game_win if spies_win else -self.reward_game_win
        res_r = -self.reward_game_win if spies_win else self.reward_game_win

        rewards = np.zeros(N_PLAYERS, dtype=np.float64)
        gts = np.zeros((1, 3), dtype=np.float64)
        gts[0] = self._episode_team_gs.copy()
        for p in self.players:
            if p.is_spy:
                rewards[p.local] += spy_r
                gts[0, 0] += spy_r
            else:
                rewards[p.local] += res_r
                gts[0, 2] += res_r

        rtg_outcome = "general_killed" if spies_win else "general_rescued"
        self.round_outcome = rtg_outcome
        self.round_team_scores[:] = gts.reshape(-1)
        infos = [{"outcome": rtg_outcome, "game_scores": gts.copy()} for _ in range(N_PLAYERS)]
        return rewards, infos

    def seed(self, seed=None):
        if seed is not None:
            s = int(seed) % (2**32 - 1)
            self._rng = np.random.default_rng(s)
            np.random.seed(s)
        return seed

    def close(self):
        pass

    def reset(self) -> np.ndarray:
        self._reset_state()
        return self._all_obs()

    def step(self, actions: list[int]):
        if self.done:
            self._reset_state()
        assert len(actions) == N_PLAYERS

        self.step_count += 1
        self.round_timer = self.step_count

        if self.step_count >= self.max_steps:
            rewards, infos = self._end_game(spies_win=False)
            return self._all_obs(), rewards, np.ones(N_PLAYERS, dtype=bool), infos

        if self.phase == PHASE_PROPOSE:
            rewards, infos = self._propose(actions)
        elif self.phase == PHASE_VOTE:
            rewards, infos = self._vote(actions)
        else:
            rewards, infos = self._quest(actions)

        dones = np.full(N_PLAYERS, self.done, dtype=bool)
        return self._all_obs(), rewards, dones, infos
