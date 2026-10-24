"""
Small helpers shared by the pages of the app (with Streamlit cache, so pages load fast).
"""
import streamlit as st

from src.data import available_assets, load_prices, monthly_returns, risk_free
from src.optimizer import check_constraints
from src.data import ASSETS
from src.portfolios import BENCHMARKS, RECOMMENDED_METHOD, benchmark_weights, build_portfolios_safe, estimate


def tighten(profile_limits, chosen):
    """The client can make the profile limits stricter, never looser."""
    return {key: max(limit, chosen.get(key, limit)) if key == "min_bonds" else min(limit, chosen.get(key, limit))
            for key, limit in profile_limits.items()}


def get_profile():
    """Stop the page if the client has not filled in the questionnaire yet."""
    if "profile" not in st.session_state:
        st.warning("Please fill in your investor profile first.")
        st.page_link("app_pages/profile.py", label="Go to your profile", icon="🧑")
        st.stop()
    return st.session_state["profile"]


def show_profile_messages(profile, notes=False):
    """Material financial facts (one warning box) and, on the profile page, neutral explanations."""
    if profile.warnings:
        st.warning("**Points to review in your situation**\n\n" + "\n".join(f"- {w}" for w in profile.warnings))
    for note in profile.notes if notes else []:
        st.info(note)


def get_choices():
    """Products and constraints chosen by the client (our recommendation by default)."""
    profile = get_profile()
    tickers = st.session_state.get("assets", profile.recommended_assets)
    tickers = [t for t in tickers if t in available_assets()]
    c = tighten(profile.recommended_constraints, st.session_state.get("constraints", {}))
    problems = check_constraints(tickers, c)
    if problems:
        for p in problems:
            st.error(p)
        st.page_link("app_pages/choices.py", label="Change your products and constraints", icon="🧩")
        st.stop()
    return tickers, c


def get_currency():
    return st.session_state.get("currency", "USD")


def get_views():
    """Black-Litterman views of the client: tuple of (asset, other or None, q, confidence).
    Only experienced investors who asked for views (QH4) have views; everybody else uses the equilibrium."""
    profile = st.session_state.get("profile")
    if profile is None or not profile.wants_views:
        return ()
    return tuple(st.session_state.get("views", ()))


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
    """Today's four portfolios for this client: (weights by method, errors by method)."""
    returns, rf, prices = get_data(currency)
    mu, cov = estimate(returns[list(tickers)])
    return build_portfolios_safe(mu, cov, rf.iloc[-1] * 12, gamma, c, views)


def get_valid_portfolios(currency, tickers, gamma, c, views):
    """Portfolios that could be calculated; stop the page if none could."""
    portfolios, errors = get_portfolios(currency, tickers, gamma, c, views)
    for method, error in errors.items():
        st.warning(f"{method} could not be calculated with these inputs ({error}).")
    if not portfolios:
        st.error("No portfolio could be calculated. Change your products or constraints.")
        st.stop()
    return portfolios


def recommendation_summary(profile, currency, tickers, c):
    """One sentence: what the recommended method (Black-Litterman with the market equilibrium) gives
    for this client's gamma and limits. None if it cannot be calculated."""
    try:
        portfolios, _ = get_portfolios(currency, tuple(tickers), profile.gamma, c, ())
        returns, rf, prices = get_data(currency)
        mu, cov = estimate(returns[list(tickers)])
    except Exception:  # missing data: the pages show their own messages
        return None
    w = portfolios.get(RECOMMENDED_METHOD)
    if w is None:
        return None
    exposure = lambda kind: sum(x for t, x in w.items() if ASSETS[t][1] == kind)
    vol = float((w @ cov @ w) ** 0.5)
    return (f"With your γ = {profile.gamma:.2f} and your limits, our recommended portfolio has about "
            f"{exposure('Equity'):.0%} in equities, {exposure('Bond'):.0%} in bonds and "
            f"{exposure('Other') + exposure('Crypto'):.0%} in other assets, with an estimated volatility of {vol:.1%} per year.")
