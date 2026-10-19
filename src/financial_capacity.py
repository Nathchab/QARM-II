"""Approved provisional four-question capacity/allocation policy; no utility calibration."""
from dataclasses import dataclass

from src.questionnaire_scoring import finite_number

POLICY_VERSION = "QARM-capacity-allocation-v2-research"
AGE_POINTS = {"Under 30": 3, "30-45": 2, "46-60": 1, "Over 60": 0}
INCOME_POINTS = {"No regular income": 0, "Fairly stable": 1, "Very stable": 2}
SAVINGS_POINTS = {"Less than 3 months": 0, "3-6 months": 1, "More than 6 months": 2}
COMMITMENT_POINTS = {"None significant": 2, "Manageable": 1,
                     "Significant obligations or financial difficulties": 0}
FINANCIAL_FIELDS = {
    "age": ("Your age", AGE_POINTS),
    "income": ("How stable is your income?", INCOME_POINTS),
    "emergency_savings": ("How many months of essential expenses could your emergency savings cover outside this investment account?", SAVINGS_POINTS),
    "commitments": ("How would you describe your current financial commitments?", COMMITMENT_POINTS),
}
CAPACITY_LIMITS = {12.: (.2, .6), 6.: (.4, .3), 2.: (.8, 0.)}
HORIZON_LIMITS = {"a": (.2, .6), "b": (.3, .5), "c": (.6, .3), "d": (.6, 0.), "e": (.8, 0.)}
WITHDRAWAL_LIMITS = {"a": (.2, .6), "b": (.4, .5), "c": (.6, .3), "d": (.8, 0.), "e": (1., 0.)}
REVIEW_MESSAGES = {
    "low_external_reserve": "Emergency savings cover less than three months of essential expenses.",
    "significant_commitments_or_difficulties": "Significant commitments or possible financial difficulties require further financial review.",
    "unquantified_near_term_need": "A substantial capital need within three years remains unquantified; liquidity needs require review.",
    "unquantified_unexpected_withdrawal": "A possible unexpected withdrawal remains unquantified; the bond allocation does not guarantee that it can be funded.",
}
POLICY_NOTICE = (
    "Research/prototype output, not a validated suitability recommendation. Capacity points and "
    "allocation/volatility limits are provisional research policies, not a measured monetary loss budget. "
    "The RT-to-gamma assumptions remain illustrative; gamma and bonds do not guarantee losses or liquidity."
)


@dataclass(frozen=True)
class CapacityAssessment:
    points: dict
    score: int
    base_floor: float
    floor: float
    safeguards: tuple


def assess_capacity(answers):
    if not isinstance(answers, dict):
        raise ValueError("Complete the four financial-capacity questions.")
    points = {}
    for key, (label, choices) in FINANCIAL_FIELDS.items():
        value = answers.get(key)
        if not isinstance(value, str) or value not in choices:
            raise ValueError(f"Complete the financial-capacity question: {label}")
        points[key] = choices[value]
    score = sum(points.values())
    base_floor = 12. if score <= 3 else 6. if score <= 6 else 2.
    safeguards = tuple(key for key in ("emergency_savings", "commitments") if points[key] == 0)
    return CapacityAssessment(points, score, base_floor, max(base_floor, 6.) if safeguards else base_floor, safeguards)


def allocation_limits(floor, qd1, qd2):
    if (not finite_number(floor) or floor not in CAPACITY_LIMITS
            or not isinstance(qd1, str) or qd1 not in HORIZON_LIMITS
            or not isinstance(qd2, str) or qd2 not in WITHDRAWAL_LIMITS):
        raise ValueError("A capacity floor and both original QD1/QD2 answers are required.")
    sources = dict(capacity=CAPACITY_LIMITS[floor], QD1=HORIZON_LIMITS[qd1], QD2=WITHDRAWAL_LIMITS[qd2])
    limits = dict(max_equity=min(v[0] for v in sources.values()), min_bonds=max(v[1] for v in sources.values()))
    return limits, sources


def review_flags(answers, qd1, qd2):
    flags = []
    if answers["emergency_savings"] == "Less than 3 months":
        flags.append("low_external_reserve")
    if answers["commitments"] == "Significant obligations or financial difficulties":
        flags.append("significant_commitments_or_difficulties")
    if qd1 in ("a", "b"):
        flags.append("unquantified_near_term_need")
    if qd2 != "e":
        flags.append("unquantified_unexpected_withdrawal")
    return tuple(flags)


def volatility_allowance(rt, floor, qd1, qd2, flags=()):
    if not finite_number(rt, 5, 97.5):
        raise ValueError("Original RT must be within its attainable 5–97.5 range.")
    enhanced = floor == 2. and qd1 in ("d", "e") and qd2 == "e" and not flags
    coordinate = max(0., min(1., (rt - 75.) / 22.5)) if enhanced else 0.
    return .05 + .05 * coordinate
