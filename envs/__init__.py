import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

ENV_REGISTRY = {
    'coin_game_multi3': 'coin_game_multi3',
    'belief_relay': 'belief_relay',
    'belief_relay_k1': 'belief_relay_k1',
    'belief_relay_k3': 'belief_relay_k3',
    'belief_relay_k5': 'belief_relay_k5',
    'belief_relay_k10': 'belief_relay_k10',

    'rtg': 'rescue_training',
    'rtg_scenarios': 'rescue_training',

    'avalon5': 'avalon5',
    'avalon5_hard': 'avalon5_hard',
    'avalon5_blind': 'avalon5_blind',

}


def make_env(scenario: str, parallel_envs: int, **kwargs):
    if scenario in ENV_REGISTRY:
        scenario = ENV_REGISTRY[scenario]

    if scenario in ['rescue', 'rescue_training'] or scenario.startswith('rtg') or scenario.startswith('avalon5'):
        from envs.rtg.support import make_env as make_rtg_env
        return make_rtg_env(scenario, parallel_envs, **kwargs)

    if scenario == 'coin_game_multi':
        from envs.coin_game_multi import WrapperEnv
        return WrapperEnv(parallel_envs)
    if scenario == 'coin_game_multi_fullobs':
        from envs.coin_game_multi import WrapperEnvFullObs
        return WrapperEnvFullObs(parallel_envs)
    if scenario == 'coin_game_multi3':
        from envs.coin_game_multi import WrapperEnv3
        return WrapperEnv3(parallel_envs)
    if scenario.startswith('belief_relay'):
        from envs.belief_relay_game import WrapperEnv
        round_len = 3
        if '_k' in scenario:
            try:
                round_len = int(scenario.rsplit('_k', 1)[1])
            except ValueError:
                round_len = 3
        return WrapperEnv(parallel_envs, round_len=round_len)

    raise ValueError(f"Unknown scenario: '{scenario}'. Check envs/__init__.py")


__all__ = ['make_env', 'ENV_REGISTRY']
