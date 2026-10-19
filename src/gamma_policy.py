"""Provisional research assumptions, separate from questionnaire scoring.

The default anchors are the rounded illustrative exponential endpoints in
analysis/gamma_calibration/REPORT.md. They are not validated preferences, CRRA
estimates, loss limits, or a complete Alternative B suitability policy.
"""
from dataclasses import asdict, dataclass
from math import exp, isclose, log

from src.questionnaire_scoring import finite_number


@dataclass(frozen=True)
class ResearchGammaMapping:
    rt_min: float = 5.0
    rt_max: float = 97.5
    gamma_at_rt_min: float = 5.56716
    gamma_at_rt_max: float = 1.16315

    def __post_init__(self):
        if (not all(finite_number(v) for v in asdict(self).values())
                or not 0 <= self.rt_min < self.rt_max <= 100
                or not 0 < self.gamma_at_rt_max < self.gamma_at_rt_min):
            raise ValueError("Research gamma anchors must be finite, with increasing RT in [0,100] "
                             "and strictly decreasing positive gamma.")

    def willingness_gamma(self, rt):
        if not finite_number(rt, self.rt_min, self.rt_max):
            raise ValueError(f"RT must be finite and within [{self.rt_min}, {self.rt_max}]; no extrapolation.")
        x = (rt - self.rt_min) / (self.rt_max - self.rt_min)
        return exp((1 - x) * log(self.gamma_at_rt_min) + x * log(self.gamma_at_rt_max))


DEFAULT_RESEARCH_MAPPING = ResearchGammaMapping()
PROTOTYPE_NOTICE = (
    "Research/prototype output, not a validated suitability recommendation. Automatic gamma uses "
    "provisional illustrative assumptions, not empirically validated preferences. The legacy capacity "
    "floor is retained; gamma is not a loss guarantee. Alternative B financial loss-budget and "
    "cross-method outcome screening, and Alternative C liquidity reserves, are not implemented."
)
REQUIRED_FINANCIAL_INPUTS = (
    "Account capital, investable net wealth and asset values, ownership and valuation dates",
    "Debts and liabilities: amounts, maturities, servicing costs and currencies",
    "Protected expenses and goals: amounts, priorities, dates, currencies and minimum funding needs",
    "Income amounts, essential expenditure, stability and resilience to income interruption",
    "External liquid resources and existing emergency reserves, availability and encumbrances",
    "Planned and contingent withdrawals: amounts, dates and currencies, beyond raw QD1/QD2 categories",
    "Other accounts/resources committed to goals, to avoid double counting loss headroom or reserves",
)


def gamma_binding_explanation(willingness, floor):
    if isclose(willingness, floor, rel_tol=1e-10, abs_tol=1e-10):
        return "Willingness gamma and the legacy capacity floor both bind (equal values)."
    if floor > willingness:
        return "The legacy capacity floor binds and overrides willingness gamma; execution uses the higher floor."
    return "Willingness gamma determines execution; the legacy capacity floor is not binding."


def effective_constraints(mandatory, optional=None):
    """Optional choices may tighten, but cannot relax, mandatory prototype limits.

    This is allocation-policy enforcement, not a financial suitability assessment.
    Neither input is mutated; optional requests remain available for explanation.
    """
    result = dict(mandatory)
    for key, limit in mandatory.items():
        requested = (optional or {}).get(key, limit)
        if not finite_number(requested, 0, 1):
            raise ValueError(f"Optional constraint {key} must be finite and within [0,1].")
        result[key] = max(limit, requested) if key == "min_bonds" else min(limit, requested)
    return result
