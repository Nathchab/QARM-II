"""
Small helpers shared by the pages of the app (with Streamlit cache, so pages load fast).
"""
import streamlit as st
import numpy as np
import pandas as pd

from src.data import available_assets, load_prices, monthly_returns, risk_free
from src.gamma_policy import effective_constraints
from src.portfolio_policy import assess_portfolios, plan_allocation, profile_review_messages
from src.optimizer import check_constraints
from src.portfolios import BENCHMARKS, benchmark_weights, build_portfolios, estimate

ACADEMIC_NOTICE = ("Academic portfolio optimization project. Gamma calibration and capacity rules are "
                   "illustrative research assumptions; estimates are not guarantees or investment advice.")


def profile_draft_changed(profile):
    if not hasattr(profile, "questionnaire_answers"):
        return False
    return (st.session_state.get("questionnaire_answers", profile.questionnaire_answers) != profile.questionnaire_answers
            or any(st.session_state.get("capacity_answers", {}).get(key, value) != value
                   for key, value in profile.capacity_answers.items() if key != "crypto"))


def get_profile():
    """Stop the page if the client has not filled in the questionnaire yet."""
    if "profile" not in st.session_state:
        st.warning("Please fill in your investor profile first.")
        st.page_link("app_pages/profile.py", label="Go to your profile", icon="🧑")
        st.stop()
    profile = st.session_state["profile"]
    if getattr(profile, "gamma_provisional", False):
        st.caption(ACADEMIC_NOTICE)
        if profile_draft_changed(profile):
            st.page_link("app_pages/profile.py", label="Review and save your profile", icon="🧑")
    return profile


def get_choices(allow_infeasible=False, profile=None):
    """Products and constraints chosen by the client (our recommendation by default)."""
    profile = get_profile() if profile is None else profile
    tickers = st.session_state.get("assets", profile.recommended_assets)
    tickers = [t for t in tickers if t in available_assets()]
    c = st.session_state.get("constraints", profile.recommended_constraints)
    if getattr(profile, "gamma_automatic", False):
        c = effective_constraints(profile.mandatory_constraints, c)
    problems = check_constraints(tickers, c)
    if problems and not allow_infeasible:
        for p in problems:
            st.error(p)
        st.page_link("app_pages/choices.py", label="Change your products and constraints", icon="🧩")
        st.stop()
    return tickers, c


def get_portfolio_context(profile, currency, tickers, c):
    """Keep valid calculations visible; disclose fallback assumptions separately."""
    returns, rf, prices = get_data(currency)
    available = [t for t in available_assets() if t in returns and returns[t].notna().sum() >= 2]
    mandatory = getattr(profile, "mandatory_constraints", {}) or c
    plan = plan_allocation(tickers, c, mandatory, available, profile.recommended_assets)
    if not plan.feasible:
        return plan, {}, None, None, returns, rf, prices
    if rf.empty or not np.isfinite(rf.iloc[-1]):
        plan.feasible = False
        plan.data_issue = True
        plan.problems.append("The risk-free rate is missing or invalid; a reliable complete portfolio comparison cannot be calculated.")
        return plan, {}, None, None, returns, rf, prices
    usable = returns[plan.tickers].dropna()
    if len(usable) < 2:
        plan.feasible = False
        plan.data_issue = True
        plan.problems.append("There are fewer than two overlapping return observations; no reliable covariance or simulation can be calculated.")
        return plan, {}, None, None, returns, rf, prices
    mu, cov = estimate(usable)
    if not np.isfinite(mu).all() or not np.isfinite(cov).all().all():
        plan.feasible = False
        plan.data_issue = True
        plan.problems.append("Return/risk estimates are not finite. No portfolio weights have been fabricated.")
        return plan, {}, mu, cov, returns, rf, prices
    try:
        raw = get_portfolios(currency, plan.tickers, profile.gamma, plan.constraints, get_views())
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        raw = None  # Recalculate independently so one failing method cannot hide siblings.
    results = assess_portfolios(mu, cov, rf.iloc[-1] * 12, profile, plan.constraints, get_views(),
                                raw_portfolios=raw, illustrative=plan.illustrative, assumptions=plan.assumptions,
                                data_issues=(["Estimates use fewer than 60 overlapping months of returns."] if len(usable) < 60 else []))
    return plan, results, mu, cov, returns, rf, prices


def show_allocation_plan(plan):
    if not plan.feasible:
        st.error("No reliable simulation is available for the requested inputs.")
    elif plan.illustrative:
        st.caption("Illustrative fallback: the requested allocation is infeasible. The disclosed calculation preserves mandatory limits.")
    for problem in plan.problems:
        st.write(problem)
    for assumption in plan.assumptions:
        st.caption(assumption)


def show_profile_notice(profile, plan=None, assessments=None, extra_warnings=(), *, warning_slot=None):
    """One consolidated warning, retaining profile facts and calculation issues."""
    messages = profile_review_messages(profile)
    if profile_draft_changed(profile):
        messages.append("Your questionnaire draft has changed. Calculations use your last saved profile.")
    if plan is not None and (not plan.feasible or plan.illustrative):
        messages.append(("Critical market data are missing or invalid. " if plan.data_issue else "The requested allocation is infeasible. ") +
                        ("A clearly labelled illustrative fallback is shown." if plan.feasible else "No reliable simulation is available."))
        messages.extend(plan.problems)
    failures = {}
    for name, result in (assessments or {}).items():
        messages.extend(result.profile_warnings)
        if result.weights is None:
            for message in result.messages:
                failures.setdefault(message, []).append(name)
    messages.extend(f"{', '.join(names)}: {message}" for message, names in failures.items())
    messages.extend(extra_warnings)
    messages = list(dict.fromkeys(messages))
    slot = st.empty() if warning_slot is None else warning_slot
    if messages:
        slot.warning("Financial, liquidity or calculation issues:\n\n" + "\n\n".join(messages))
    return slot


def show_portfolio_assessment(result):
    if result.weights is None:
        st.error("No reliable simulation is available for this method.")
    elif result.status == "normal":
        st.success("Normal: valid portfolio weights and allocation limits satisfied.")
    elif result.status == "information":
        st.info("Capacity and allocation limits are applied. The calculated portfolio satisfies allocation constraints.")
    else:
        st.caption("Warning: see the consolidated profile notice above. " +
                   ("This is an illustrative fallback." if result.illustrative else "Calculated portfolio remains available."))


def portfolio_status_table(results, *, advanced=False):
    rows = [{"Method": method, "Method status": result.label,
                          "Calculation": "Illustrative fallback" if result.illustrative else "Available" if result.weights is not None else "Unavailable",
                          "Allocation violations": ", ".join(result.diagnostics.get("Allocation limit violations", [])),
                          "Explanation": " ".join(result.messages + result.information)} for method, result in results.items()]
    frame = pd.DataFrame(rows)
    return frame if advanced else frame[["Method", "Method status", "Calculation"]]


def get_currency():
    return st.session_state.get("currency", "USD")


def get_views():
    """Black-Litterman views of the client: {ticker: (expected return, confidence)}."""
    return st.session_state.get("views", {})


def get_benchmark(tickers, c):
    """Name and weights of the benchmark chosen on the backtest page (profile benchmark by default)."""
    name = st.session_state.get("benchmark", BENCHMARKS[0])
    return name, benchmark_weights(name, tickers, c)


@st.cache_data
def get_data(currency):
    """Monthly returns of all products, monthly risk-free rate, daily prices."""
    prices = load_prices(currency)
    return monthly_returns(prices), risk_free(currency), prices


@st.cache_data
def get_portfolios(currency, tickers, gamma, c, views):
    """Today's four portfolios for this client."""
    returns, rf, prices = get_data(currency)
    mu, cov = estimate(returns[tickers])
    return build_portfolios(mu, cov, rf.iloc[-1] * 12, gamma, c, views)
