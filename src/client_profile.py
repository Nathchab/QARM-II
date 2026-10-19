from copy import deepcopy

from src.data import RECOMMENDED, RECOMMENDED_CRYPTO
from src.gamma_policy import DEFAULT_RESEARCH_MAPPING, gamma_binding_explanation
from src.financial_capacity import (FINANCIAL_FIELDS, POLICY_VERSION, allocation_limits,
                                    assess_capacity, review_flags)
from src.questionnaire_routing import portfolio_routes
from src.questionnaire_scoring import dimension_labels, finite_number, illustrative_risk_matches, score_answers

_AUTOMATIC_GAMMA = object()

class ClientProfile:
    """
    Legacy construction combines:
      - risk willingness: gamma from the Holt-Laury questionnaire
      - risk capacity: factual questions -> minimum gamma + RECOMMENDED portfolio constraints
    Final gamma = the more conservative (higher) of the two.
    The client can then change the recommended products and constraints (page "Your choices").
    from_questionnaire automatically maps RT using provisional research anchors.
    Explicit baseline calls remain available for legacy compatibility.
    """

    # Answer -> capacity points (more points = can afford more risk)
    AGE = {"Under 30": 3, "30-45": 2, "46-60": 1, "Over 60": 0}
    HORIZON = {"Less than 3 years": 0, "3-7 years": 1, "7-15 years": 2, "More than 15 years": 3}
    NEEDS_MONEY = {"Yes": 0, "No": 2}
    INCOME = {"No regular income": 0, "Fairly stable": 1, "Very stable": 2}
    MAX_LOSS = {"-5%": 0, "-10%": 1, "-20%": 2, "-30% or more": 3}

    # Constraints linked to some answers
    MAX_EQUITY_HORIZON = {"Less than 3 years": 0.3, "3-7 years": 0.6, "7-15 years": 0.8, "More than 15 years": 1.0}
    MAX_EQUITY_LOSS = {"-5%": 0.2, "-10%": 0.4, "-20%": 0.7, "-30% or more": 1.0}

    MAX_WEIGHT_PER_ASSET = 0.4
    MAX_CRYPTO = 0.10  # maximum in crypto for a client who wants crypto
    MAX_OTHER = 0.20   # maximum in gold and commodities (volatile, no income)

    # Largest acceptable one-year loss, as a number (used by the time machine)
    MAX_LOSS_VALUE = {"-5%": -0.05, "-10%": -0.10, "-20%": -0.20, "-30% or more": -0.30}

    def __init__(self, gamma_willingness, age, horizon, needs_money, income, max_loss, crypto):
        if not finite_number(gamma_willingness, 0) or gamma_willingness == 0:
            raise ValueError("Optimization gamma must be finite and strictly positive.")
        self.gamma_willingness = gamma_willingness
        self.max_loss = self.MAX_LOSS_VALUE[max_loss]

        # Risk capacity score (0 to 13)
        self.capacity_score = (self.AGE[age] + self.HORIZON[horizon] + self.NEEDS_MONEY[needs_money]
                               + self.INCOME[income] + self.MAX_LOSS[max_loss])
        self.gamma_capacity = self.gamma_from_capacity(self.capacity_score)

        # Final gamma: the more conservative of willingness and capacity
        self.gamma = max(self.gamma_willingness, self.gamma_capacity)
        self.label = self.profile_label(self.gamma)

        # Our recommendation (the client can change it)
        self.recommended_constraints = {
            "max_equity": min(self.MAX_EQUITY_HORIZON[horizon], self.MAX_EQUITY_LOSS[max_loss]),
            "min_bonds": 0.3 if needs_money == "Yes" else 0.0,
            "max_weight_per_asset": self.MAX_WEIGHT_PER_ASSET,
            "max_other": self.MAX_OTHER,
            "max_crypto": self.MAX_CRYPTO if crypto == "Yes" else 0.0,
        }
        self.recommended_assets = RECOMMENDED + (RECOMMENDED_CRYPTO if crypto == "Yes" else [])

        self.optimization_gamma = self.gamma
        self.optimization_gamma_input = gamma_willingness
        self.gamma_provisional = False
        self.gamma_source = "Legacy willingness gamma with existing financial-capacity floor"
        self.capacity_answers = dict(age=age, horizon=horizon, needs_money=needs_money,
                                     income=income, max_loss=max_loss, crypto=crypto)

    @classmethod
    def from_questionnaire(cls, answers, capacity_answers, optimization_gamma=_AUTOMATIC_GAMMA,
                           *, gamma_mapping=DEFAULT_RESEARCH_MAPPING):
        """Accept the new four-field schema or unchanged legacy capacity inputs.

        Scores and raw horizon/withdrawal answers remain independent. HZ, EX, KN,
        CX and CV never adjust gamma or the unchanged capacity-point calculation.
        gamma_mapping configures provisional research anchors, not investor edits.
        """
        if isinstance(capacity_answers, dict) and any(key in capacity_answers for key in ("emergency_savings", "commitments")):
            if optimization_gamma is not _AUTOMATIC_GAMMA:
                raise ValueError("The four-question research policy uses automatic willingness gamma, not an explicit baseline.")
            return cls.from_financial_questionnaire(answers, capacity_answers, gamma_mapping=gamma_mapping)
        result = score_answers(answers)
        if not isinstance(capacity_answers, dict):
            raise ValueError("Financial-capacity answers must be a dictionary.")
        expected = {"age": cls.AGE, "horizon": cls.HORIZON, "needs_money": cls.NEEDS_MONEY,
                    "income": cls.INCOME, "max_loss": cls.MAX_LOSS}
        for key, choices in expected.items():
            value = capacity_answers.get(key)
            if not isinstance(value, str) or value not in choices:
                raise ValueError(f"Complete the legacy financial-capacity question: {key}.")
        crypto = "Yes" if "h" in answers["QF1"] else "No"
        automatic = optimization_gamma is _AUTOMATIC_GAMMA
        if automatic:
            optimization_gamma = gamma_mapping.willingness_gamma(result.scores["RT"])
        profile = cls(optimization_gamma, **{key: capacity_answers[key] for key in expected}, crypto=crypto)
        cls._attach_questionnaire(profile, answers, result)
        profile.gamma_provisional = True
        profile.gamma_automatic = automatic
        profile.gamma_source = ("Provisional research exponential RT mapping + legacy capacity floor" if automatic
                                else "Legacy explicit provisional baseline + legacy capacity floor; no RT mapping")
        profile.gamma_mapping_assumptions = gamma_mapping if automatic else None
        profile.gamma_binding_explanation = gamma_binding_explanation(profile.gamma_willingness,
                                                                     profile.gamma_capacity)
        profile.mandatory_constraints = deepcopy(profile.recommended_constraints) if automatic else {}
        profile.suitability_status = "research_prototype_not_validated"
        return profile

    @classmethod
    def from_financial_questionnaire(cls, answers, capacity_answers, *, gamma_mapping=DEFAULT_RESEARCH_MAPPING):
        """V2 research policy. The legacy constructor/factory remain numerically unchanged."""
        result = score_answers(answers)
        assessment = assess_capacity(capacity_answers)
        limits, sources = allocation_limits(assessment.floor, answers["QD1"], answers["QD2"])
        # Do not synthesize old horizon or acceptable-loss answers to call __init__.
        profile = cls.__new__(cls)
        profile.capacity_policy_version = POLICY_VERSION
        profile.capacity_answers = {key: deepcopy(capacity_answers[key]) for key in FINANCIAL_FIELDS}
        profile.capacity_score = assessment.score
        profile.capacity_score_max = 9
        profile.capacity_subscores = assessment.points
        profile.gamma_capacity_base = assessment.base_floor
        profile.capacity_safeguards = assessment.safeguards
        profile.gamma_capacity = assessment.floor
        profile.gamma_willingness = gamma_mapping.willingness_gamma(result.scores["RT"])
        profile.gamma = max(profile.gamma_willingness, profile.gamma_capacity)
        profile.optimization_gamma = profile.gamma
        profile.optimization_gamma_input = profile.gamma_willingness
        profile.label = cls.profile_label(profile.gamma)
        crypto = "h" in answers["QF1"]
        profile.recommended_constraints = dict(limits, max_weight_per_asset=cls.MAX_WEIGHT_PER_ASSET,
                                              max_other=cls.MAX_OTHER, max_crypto=cls.MAX_CRYPTO if crypto else 0.)
        profile.mandatory_constraints = deepcopy(profile.recommended_constraints)
        profile.recommended_assets = RECOMMENDED + (RECOMMENDED_CRYPTO if crypto else [])
        profile.allocation_limit_sources = sources
        profile.financial_review_flags = review_flags(profile.capacity_answers, answers["QD1"], answers["QD2"])
        profile.max_loss = None  # QE2 is willingness, not a replacement financial loss limit.
        profile.gamma_automatic = True
        profile.gamma_provisional = True
        profile.gamma_source = "Provisional exponential RT mapping + approved four-question research capacity floor"
        profile.gamma_mapping_assumptions = gamma_mapping
        profile.gamma_binding_explanation = gamma_binding_explanation(profile.gamma_willingness, profile.gamma_capacity)
        profile.suitability_status = "research_prototype_not_validated"
        cls._attach_questionnaire(profile, answers, result)
        return profile

    @staticmethod
    def _attach_questionnaire(profile, answers, result):
        """Common original score/routing metadata, without changing score methodology."""
        profile.questionnaire_version = "QARM-II-stage-1"
        profile.questionnaire_answers = deepcopy(answers)
        profile.scores = result.scores
        profile.score_subscores = result.subscores
        profile.applicable_questions = result.applicable
        for dimension, value in result.scores.items():
            setattr(profile, dimension.lower(), value)
        profile.dimension_labels = dimension_labels(result.scores)
        profile.risk_classification = profile.dimension_labels["RT"]
        profile.illustrative_risk_matches = illustrative_risk_matches(result.scores)
        profile.portfolio_routes = portfolio_routes(answers, result.scores)
        profile.optimization_label = profile.label
        profile.label = profile.risk_classification
        profile.raw_horizon_answers = {code: deepcopy(answers[code]) for code in ("QD1", "QD2")}
        # A conservative prototype policy boundary. Optional requests are stored
        # separately and can only tighten these limits in the automatic app flow.
        profile.requested_portfolio_settings = {
            code: deepcopy(answers[code]) for code in result.applicable
            if code.startswith(("QF", "QG", "QH", "QI", "QJ")) and code in answers
        }
        profile.requested_portfolio_settings.update({
            key: deepcopy(value) for key, value in answers.items()
            if "_" in key and key.split("_", 1)[0] in result.applicable
        })
        profile.portfolio_settings_applied = False

    def policy_snapshot(self):
        """Archive factual policy inputs/limits, never reinterpret an old score."""
        fields = ("capacity_answers", "capacity_score", "gamma_capacity", "gamma_willingness", "gamma",
                  "recommended_constraints", "mandatory_constraints", "recommended_assets", "max_loss",
                  "questionnaire_answers", "scores")
        result = {key: deepcopy(getattr(self, key)) for key in fields if hasattr(self, key)}
        result["capacity_policy_version"] = getattr(self, "capacity_policy_version", "legacy-capacity-0-13")
        return result

    @staticmethod
    def gamma_from_capacity(score):
        """Low capacity -> the client must be treated as at least this risk averse."""
        if score <= 4:
            return 12.0
        if score <= 8:
            return 6.0
        return 2.0

    @staticmethod
    def profile_label(gamma):
        if gamma <= 4:
            return "Aggressive"
        if gamma <= 9:
            return "Balanced"
        return "Conservative"
