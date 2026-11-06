from dbos.methods.bos import BOSTrainer


class PPOTrainer(BOSTrainer):
    """PPO baseline: pure extrinsic reward (lambda_dec=0, no BBM)."""

    def __init__(self, config: dict):
        config = {
            **config,
            "method_name": "ppo",
            "lambda_dec": 0.0,
            "use_bbm_intrinsic": False,
        }
        super().__init__(config)


class BeliefValuePPOTrainer(BOSTrainer):
    """PPO control with a detached belief-conditioned focal value baseline."""

    def __init__(self, config: dict):
        config = {
            **config,
            "method_name": "belief_value_ppo",
            "lambda_dec": 0.0,
            "use_bbm_intrinsic": False,
            "use_belief_value_critic": True,
        }
        super().__init__(config)


class AdvantageAlignmentTrainer(BOSTrainer):
    """PPO with the integrated all-other-agents Advantage Alignment term."""

    def __init__(self, config: dict):
        config = {
            **config,
            "method_name": "advantage_alignment",
            "lambda_dec": 0.0,
            "use_bbm_intrinsic": False,
            "use_advantage_alignment": True,
        }
        super().__init__(config)


class LOLADiceTrainer(BOSTrainer):
    """PPO with a one-step LOLA opponent lookahead via the DiCE magic-box estimator."""

    def __init__(self, config: dict):
        config = {
            **config,
            "method_name": "lola_dice",
            "lambda_dec": 0.0,
            "use_bbm_intrinsic": False,
            "use_lola_dice": True,
        }
        super().__init__(config)


class BBMTrainer(BOSTrainer):
    """BBM baseline: per-step intrinsic bonus (lambda_dec>0, use_bbm_intrinsic=True)."""

    def __init__(self, config: dict):
        config = {**config, "method_name": "bbm", "use_bbm_intrinsic": True}
        if "lambda_dec" not in config:
            config["lambda_dec"] = 0.5
        super().__init__(config)
