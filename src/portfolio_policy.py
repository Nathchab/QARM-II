"""Academic portfolio validation around unchanged optimizers.

The covariance/reference helpers are retained for historical research scripts
only. Application assessments do not use a relative volatility pass/fail screen.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from src.black_litterman import posterior_returns
from src.data import ASSETS
from src.financial_capacity import POLICY_VERSION, REVIEW_MESSAGES, volatility_allowance
from src.optimizer import mean_variance, resampled, risk_parity
from src.portfolio_diagnostics import portfolio_diagnostics
from src.portfolios import METHODS, WINDOW
from src.questionnaire_scoring import finite_number

SCENARIO_NAMES = ("base", "half_return_spreads", "one_half_return_spreads", "stress_covariance", "weak_equity_returns")
SCENARIO_DIRECTORY = Path(__file__).resolve().parents[1] / "analysis" / "gamma_calibration"
CONSTRAINT_KEYS = ("max_equity", "min_bonds", "max_other", "max_crypto", "max_weight_per_asset")
STATUS_LABELS = {
    "normal": "Normal",
    "information": "Information",
    "warning": "Warning",
}
_METHOD_REFERENCE = object()


@dataclass
class Feasibility:
    feasible: bool
    problems: list = field(default_factory=list)
    kind: str = "feasible"


@dataclass
class AllocationPlan:
    tickers: list
    constraints: dict
    feasible: bool
    illustrative: bool = False
    problems: list = field(default_factory=list)
    assumptions: list = field(default_factory=list)
    data_issue: bool = False


@dataclass
class PortfolioAssessment:
    weights: pd.Series | None
    status: str
    messages: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    scenarios: pd.DataFrame = field(default_factory=pd.DataFrame)
    allowance: float | None = None  # Historical compatibility; never used by the app.
    information: list = field(default_factory=list)
    profile_warnings: list = field(default_factory=list)
    illustrative: bool = False

    @property
    def label(self):
        return STATUS_LABELS[self.status]


def allocation_feasibility(tickers, c):
    """LP certificate before calling any method; explain jointly infeasible limits."""
    if any(not finite_number(c.get(key), 0, 1) for key in CONSTRAINT_KEYS):
        return Feasibility(False, ["Allocation limits must be finite values between 0% and 100%."], "invalid")
    if len(set(tickers)) != len(tickers) or any(t not in ASSETS for t in tickers):
        return Feasibility(False, ["Product identifiers must be known and unique."], "invalid")
    if not tickers:
        return Feasibility(False, ["No products are selected."], "infeasible")
    classes = np.array([ASSETS[t][1] for t in tickers])
    rules = np.array([classes == "Equity", -(classes == "Bond").astype(float),
                      classes == "Other", classes == "Crypto"], dtype=float)
    result = linprog(np.zeros(len(tickers)), A_ub=rules,
                     b_ub=[c["max_equity"], -c["min_bonds"], c["max_other"], c["max_crypto"]],
                     A_eq=np.ones((1, len(tickers))), b_eq=[1.],
                     bounds=[(0., c["max_weight_per_asset"])] * len(tickers), method="highs")
    if result.success:
        return Feasibility(True)
    if result.status != 2:
        return Feasibility(False, [f"Allocation feasibility could not be established: {result.message}"], "solver_failure")
    problems = []
    product_room = len(tickers) * c["max_weight_per_asset"]
    if product_room < 1 - 1e-9:
        problems.append(f"The {len(tickers)} selected products at {c['max_weight_per_asset']:.0%} each can hold "
                        f"only {product_room:.0%}, below the required 100% investment.")
    bond_room = float((classes == "Bond").sum()) * c["max_weight_per_asset"]
    if bond_room < c["min_bonds"] - 1e-9:
        problems.append(f"Selected bonds can hold at most {bond_room:.0%}, below the {c['min_bonds']:.0%} bond minimum.")
    rooms = {kind: float((classes == kind).sum()) * c["max_weight_per_asset"] for kind in ("Equity", "Bond", "Other", "Crypto")}
    room = min(rooms["Equity"], c["max_equity"]) + rooms["Bond"] + min(rooms["Other"], c["max_other"]) + min(rooms["Crypto"], c["max_crypto"])
    if room < 1 - 1e-9:
        problems.append(f"The asset-class and product ceilings together allow only {room:.0%}, below 100%.")
    return Feasibility(False, problems or ["The bond minimum, class ceilings and product bounds cannot be satisfied together."], "infeasible")


def plan_allocation(tickers, requested, mandatory, available, defaults):
    """Try disclosed fallback assumptions, never weaken mandatory restrictions.

    A fallback is always illustrative even if it happens to pass all numerical
    checks. The investor's stored requests are not rewritten.
    """
    original = allocation_feasibility(tickers, requested)
    if original.feasible:
        return AllocationPlan(list(tickers), dict(requested), True)
    if original.kind != "infeasible":
        return AllocationPlan(list(tickers), dict(requested), False, problems=original.problems)
    broader = list(dict.fromkeys(list(tickers) + [t for t in defaults if t in available]))
    strategies = []
    if requested != mandatory:
        strategies.append((list(tickers), mandatory, ["Tighter optional allocation requests are not applied in this illustration; mandatory profile limits remain enforced."]))
    if broader != list(tickers):
        added = [t for t in broader if t not in tickers]
        assumption = "This illustration adds available default products: " + ", ".join(added) + "."
        strategies.append((broader, requested, [assumption]))
        if requested != mandatory:
            strategies.append((broader, mandatory, [assumption, "Tighter optional allocation requests are not applied; mandatory profile limits remain enforced."]))
    for products, limits, assumptions in strategies:
        if allocation_feasibility(products, limits).feasible:
            return AllocationPlan(products, dict(limits), True, True, original.problems, assumptions)
    return AllocationPlan(list(tickers), dict(requested), False, problems=original.problems + [
        "No feasible illustration was found using available default products while preserving mandatory limits."])


def load_covariance_scenarios(tickers):
    """Frozen, disclosed synthetic research matrices, not live-market forecasts.

    Unsupported products are never assigned another product's risk. The caller
    still evaluates estimated covariance but cannot claim a complete panel pass.
    """
    scenarios, issues = {}, []
    for name in SCENARIO_NAMES:
        try:
            matrix = pd.read_csv(SCENARIO_DIRECTORY / f"covariance_{name}.csv", index_col=0)
            missing = [t for t in tickers if t not in matrix.index or t not in matrix.columns]
            if missing:
                issues.append("The research covariance panel does not cover: " + ", ".join(missing) + ". The full volatility screen is incomplete.")
                continue
            scenarios[name] = matrix.loc[tickers, tickers]
        except (OSError, ValueError, KeyError) as error:
            issues.append(f"Research covariance scenario {name} could not be loaded: {error}")
    return scenarios, list(dict.fromkeys(issues))


def profile_review_messages(profile):
    """Material financial/liquidity facts, kept separate from normal adjustments."""
    return [REVIEW_MESSAGES[flag] for flag in getattr(profile, "financial_review_flags", ())]


def profile_information_messages(profile):
    messages = []
    if getattr(profile, "capacity_policy_version", None) != POLICY_VERSION:
        messages.append("Your saved legacy capacity policy remains active; migration requires an explicit choice.")
    if profile.gamma_capacity > profile.gamma_willingness + 1e-10:
        baseline = "willingness gamma" if getattr(profile, "gamma_automatic", False) else "explicit gamma baseline"
        messages.append(f"Capacity floor applied: final gamma {profile.gamma:g}, "
                        f"{baseline} {profile.gamma_willingness:.5f}. This is a normal allocation-policy adjustment.")
    return list(dict.fromkeys(messages))


def profile_allowance(profile):
    if getattr(profile, "capacity_policy_version", None) != POLICY_VERSION:
        return .05
    return volatility_allowance(profile.rt, profile.gamma_capacity,
                                profile.raw_horizon_answers["QD1"], profile.raw_horizon_answers["QD2"],
                                profile.financial_review_flags)


def volatility_screen(weights, reference, covariances, allowance):
    """Historical research only; never called by application assessments."""
    rows = []
    for name, matrix in covariances.items():
        matrix = matrix.loc[weights.index, weights.index]
        values = matrix.to_numpy()
        if (not np.isfinite(values).all() or not np.allclose(values, values.T, atol=1e-10)
                or np.linalg.eigvalsh(values).min() < -1e-10):
            raise ValueError(f"Covariance scenario {name} is not finite, symmetric and positive semidefinite.")
        candidate = float(np.sqrt(max(0., weights @ matrix @ weights)))
        ref = float(np.sqrt(max(0., reference @ matrix @ reference)))
        cap = (1 + allowance) * ref
        ratio = candidate / cap if cap > 0 else (1. if candidate == 0 else float("inf"))
        rows.append(dict(Scenario=name, Volatility=candidate, Ceiling=cap,
                         Difference=candidate - cap, Ratio=ratio, Pass=ratio <= 1 + 1e-7))
    return pd.DataFrame(rows)


def _physical_weights(weights, tickers):
    if not isinstance(weights, pd.Series) or not weights.index.is_unique or set(weights.index) != set(tickers):
        raise ValueError("The optimizer returned mismatched product weights.")
    weights = weights.reindex(tickers)
    if (not np.isfinite(weights).all() or abs(weights.sum() - 1.) > 1e-6
            or weights.min() < -1e-8 or weights.max() > 1 + 1e-8):
        raise ValueError("The optimizer did not return reliable finite, fully invested, long-only weights. No normalization was applied.")
    return weights


def calculate_methods(mu, cov, rf, gamma, c, views, n_sim=100):
    """Isolate failures so one method cannot hide the other three results."""
    views = {t: v for t, v in views.items() if t in mu.index}
    calls = {
        "Mean-variance": lambda: mean_variance(mu, cov, gamma, c),
        "Resampled MV": lambda: resampled(mu, cov, gamma, c, n_sim),
        "Black-Litterman": lambda: mean_variance(posterior_returns(cov, rf, views), cov, gamma, c),
        "Risk parity (ERC)": lambda: risk_parity(cov, c),
    }
    results, errors = {}, {}
    for method, calculate in calls.items():
        try:
            results[method] = calculate()
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            errors[method] = str(error)
    return results, errors


def assess_portfolios(mu, cov, rf, profile, c, views=None, *, raw_portfolios=None,
                      covariance_scenarios=None, scenario_issues=(), illustrative=False,
                      assumptions=(), n_sim=100, reference_weights=_METHOD_REFERENCE,
                      data_issues=()):
    """Validate data, physical weights and allocation limits, without an MV screen.

    Old reference/scenario keyword arguments are accepted for compatibility but
    do not control status. Volatility remains an estimated portfolio diagnostic.
    """
    profile_warnings = profile_review_messages(profile) + list(data_issues)
    information = profile_information_messages(profile)
    try:
        aligned = cov.loc[mu.index, mu.index].to_numpy()
        valid_estimates = (mu.index.is_unique and finite_number(rf)
                           and np.isfinite(mu).all() and np.isfinite(aligned).all()
                           and aligned.shape == (len(mu), len(mu))
                           and np.allclose(aligned, aligned.T, atol=1e-10)
                           and np.linalg.eigvalsh(aligned).min() >= -1e-10)
    except (KeyError, ValueError, TypeError, np.linalg.LinAlgError):
        valid_estimates = False
    if not valid_estimates:
        return {m: PortfolioAssessment(None, "warning", ["Estimated returns, risk-free rate or covariance are invalid; no reliable calculation is available."],
                                      information=information, profile_warnings=profile_warnings)
                for m in METHODS}
    feasibility = allocation_feasibility(list(mu.index), c)
    if not feasibility.feasible:
        return {m: PortfolioAssessment(None, "warning", feasibility.problems,
                                      information=information, profile_warnings=profile_warnings) for m in METHODS}
    if raw_portfolios is None:
        raw_portfolios, errors = calculate_methods(mu, cov, rf, profile.gamma, c, views or {}, n_sim)
    else:
        errors = {}
    valid, invalid = {}, dict(errors)
    for method, weights in raw_portfolios.items():
        try:
            valid[method] = _physical_weights(weights, mu.index)
        except ValueError as error:
            invalid[method] = str(error)
    mandatory = getattr(profile, "mandatory_constraints", {}) or c
    results = {}
    for method in METHODS:
        if method not in valid:
            results[method] = PortfolioAssessment(None, "warning", [invalid.get(method, "The method did not return a reliable calculation.")],
                                                  information=information, profile_warnings=profile_warnings)
            continue
        w = valid[method]
        diagnostics = portfolio_diagnostics(w, mu, cov, c)
        violations = list(dict.fromkeys(diagnostics["Allocation limit violations"]
                                       + portfolio_diagnostics(w, mu, cov, mandatory)["Allocation limit violations"]))
        diagnostics["Allocation limit violations"] = violations
        messages = list(assumptions) if illustrative else []
        info = list(information)
        binding = [key for key in diagnostics["Binding allocation limits"]
                   if key != "max_crypto" or any(ASSETS[t][1] == "Crypto" for t in mu.index)]
        binding = [key for key in binding if key != "min_bonds" or c["min_bonds"] > 0]
        if binding:
            info.append("Allocation limits restrict exposure: " + ", ".join(binding) + ".")
        if violations:
            messages.append("The calculated weights violate allocation limits: " + ", ".join(violations) + ".")
        # Common investor-level financial flags are displayed once, separately
        # from each method's mathematical/data validity.
        status = "warning" if data_issues or illustrative or violations else "information" if info else "normal"
        results[method] = PortfolioAssessment(None if violations else w, status, list(dict.fromkeys(messages)), diagnostics,
                                              information=info, profile_warnings=profile_warnings, illustrative=illustrative)
    return results


def assess_backtest_weights(returns, histories, profile, c, *, illustrative=False, assumptions=()):
    """Allocation/data audit; estimation uses observations BEFORE each rebalance.

    It never changes trades, costs, gamma, targets or the original backtest.
    No MV reference or volatility compliance calculation is performed.
    """
    by_date = {}
    for method, history in histories.items():
        if method not in METHODS:
            continue
        for date, weights in history.iterrows():
            by_date.setdefault(date, {})[method] = weights
    rows = []
    for date, methods in by_date.items():
        past = returns.loc[returns.index < date].dropna().tail(WINDOW)
        mu, cov = past.mean() * 12, past.cov() * 12
        # Check actual cost-aware holdings without changing trades or objectives.
        assessments = assess_portfolios(mu, cov, 0., profile, c, raw_portfolios=methods,
                                        illustrative=illustrative, assumptions=assumptions)
        for method in methods:
            result = assessments[method]
            rows.append(dict(Date=date, Method=method, Status=result.label, Code=result.status,
                             Volatility=result.diagnostics.get("Volatility", np.nan),
                             **{"Allocation violations": ", ".join(result.diagnostics.get("Allocation limit violations", [])),
                                "Explanation": " ".join(result.messages)}))
    return pd.DataFrame(rows)
