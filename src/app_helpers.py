"""
Small helpers shared by the pages of the app (with Streamlit cache, so pages load fast).
"""
import streamlit as st

from src.data import available_assets, load_prices, monthly_returns, risk_free
from src.optimizer import check_constraints
from src.portfolios import BENCHMARKS, benchmark_weights, build_portfolios_safe, estimate


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
    """Black-Litterman views of the client: {ticker: (expected return, confidence between 0 and 1)}."""
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
    """Today's four portfolios for this client: (weights by method, errors by method)."""
    returns, rf, prices = get_data(currency)
    mu, cov = estimate(returns[tickers])
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
