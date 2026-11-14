"""
Small helpers shared by the pages of the app (with Streamlit cache, so pages load fast).
"""
import streamlit as st

from src.black_litterman import posterior_returns, restrict_views
from src.data import available_assets, load_prices, monthly_returns, risk_free
from src.optimizer import check_constraints, class_exposures
from src.portfolios import (BENCHMARKS, LABELS, bad_year_loss, benchmark_weights, build_portfolios, estimate,
                            gamma_for_loss_limit, recommended_method)


def tighten(profile_limits, chosen):
    """The client can make the profile limits stricter, never looser."""
    return {key: max(limit, chosen.get(key, limit)) if key.startswith("min_") else min(limit, chosen.get(key, limit))
            for key, limit in profile_limits.items()}


def next_button(page, label):
    """Big button to go to another page."""
    if st.button(label, type="primary", icon="➡️", width="stretch", key=f"go_{page}"):
        st.switch_page(page)


def get_profile():
    """Stop the page if the client has not filled in the questionnaire yet."""
    if "profile" not in st.session_state:
        st.warning("Please fill in your investor profile first.")
        next_button("app_pages/profile.py", "Go to the questionnaire")
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
        next_button("app_pages/choices.py", "Change your products and limits")
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


def method_label(method):
    """Name of a method for the client."""
    return LABELS.get(method, method)


def client_method():
    """The client's portfolio: the method chosen on the portfolio page, by default our recommendation."""
    return st.session_state.get("method", recommended_method(get_views()))


YOURS, BENCHMARK_COLOR, OTHERS = "#e4572e", "#222222", "#b8b8b8"


def highlight_lines(fig, yours, benchmark):
    """Plotly line chart: the client's portfolio thick and coloured, the benchmark dashed, other methods grey."""
    for trace in fig.data:
        if trace.name == yours:
            trace.update(line=dict(color=YOURS, width=4), legendrank=1)
        elif trace.name == benchmark:
            trace.update(line=dict(color=BENCHMARK_COLOR, width=2.5, dash="dash"), legendrank=2)
        else:
            trace.update(line=dict(color=OTHERS, width=1.5), opacity=0.8, legendrank=3)
    return fig


def recommended_equity_share(profile, tickers, c):
    """Share of equities in the recommended portfolio (used by the default benchmark)."""
    views = get_views()
    portfolios, errors, gamma = get_portfolios(get_currency(), tuple(tickers), profile.gamma, profile.max_loss, c, views)
    best = recommended_method(views)
    return class_exposures(portfolios[best])["Equity"] if best in portfolios else c["max_equity"]


def get_benchmark(profile, tickers, c):
    """Name and weights of the benchmark chosen on the backtest page (same equity share by default)."""
    name = st.session_state.get("benchmark", BENCHMARKS[0])
    return name, benchmark_weights(name, tickers, recommended_equity_share(profile, tickers, c))


@st.cache_data
def get_data(currency):
    """Monthly returns of all products, monthly risk-free rate, daily prices."""
    prices = load_prices(currency)
    return monthly_returns(prices), risk_free(currency), prices


@st.cache_data(max_entries=100)  # keep at most 100 results in memory (the website runs all the time)
def get_portfolios(currency, tickers, gamma, max_loss, c, views):
    """Today's portfolios for this client: (weights by method, errors by method, risk aversion used).
    The risk aversion is the client's gamma, raised if needed to respect his loss limit (QE2)."""
    returns, rf, prices = get_data(currency)
    mu, cov = estimate(returns[list(tickers)])
    rf_now = rf.iloc[-1] * 12
    try:
        gamma = gamma_for_loss_limit(posterior_returns(cov, rf_now, restrict_views(views, mu.index)), cov, gamma, c,
                                     max_loss)
    except (ValueError, RuntimeError):
        pass  # the optimisation fails: build_portfolios below reports the error
    portfolios, errors = build_portfolios(mu, cov, rf_now, gamma, c, views)
    return portfolios, errors, gamma


def get_valid_portfolios(profile, tickers, c):
    """(portfolios that could be calculated, risk aversion used); stop the page if none could."""
    portfolios, errors, gamma = get_portfolios(get_currency(), tuple(tickers), profile.gamma, profile.max_loss, c,
                                               get_views())
    for method, error in errors.items():
        st.warning(f"{method_label(method)} could not be calculated with these inputs ({error}).")
    if not portfolios:
        st.error("No portfolio could be calculated. Change your products or constraints.")
        st.stop()
    return portfolios, gamma


def recommendation_summary(profile, currency, tickers, c):
    """One sentence: what our recommendation (mean-variance with the market equilibrium returns) gives
    for this client's gamma and limits. None if it cannot be calculated."""
    try:
        portfolios, errors, gamma = get_portfolios(currency, tuple(tickers), profile.gamma, profile.max_loss, c, ())
        returns, rf, prices = get_data(currency)
        mu, cov = estimate(returns[list(tickers)])
        w = portfolios["Equilibrium"]
    except Exception:  # missing data or failed optimisation: no sentence (the pages show their own messages)
        return None
    exposure = class_exposures(w)
    vol = float((w @ cov @ w) ** 0.5)
    loss = bad_year_loss(w, posterior_returns(cov, rf.iloc[-1] * 12, ()), cov)
    note = loss_limit_note(profile, gamma, loss, currency)
    return (f"Our recommended portfolio for you has about {exposure['Equity']:.0%} in equities, "
            f"{exposure['Bond']:.0%} in bonds and {exposure['Other'] + exposure['Crypto']:.0%} in other assets, "
            f"with an estimated volatility of {vol:.1%} per year and a loss of about {-loss:.1%} in a bad year "
            "(5% chance)." + (" " + note if note else ""))


def loss_limit_note(profile, gamma, loss, currency):
    """Explains how the loss limit of QE2 was applied to the recommended portfolio (None if nothing to say).
    loss = bad-year loss of the recommended portfolio."""
    if profile.max_loss is None:
        return None
    if loss < profile.max_loss - 1e-4:  # even the safest portfolio inside the limits loses more
        advice = ("In CHF every ETF carries the USD/CHF currency risk, even the short-term Treasuries (SHY): "
                  "investing in USD, or keeping this money in a CHF savings account, would be safer."
                  if currency == "CHF" else "Adding short-term Treasuries (SHY) would lower the risk.")
        return (f"Even the safest portfolio within your limits could lose about {-loss:.0%} in a bad year, more "
                f"than the {-profile.max_loss:.0%} you can tolerate (QE2): we recommend this safest portfolio "
                f"(minimum risk). {advice}")
    if gamma > profile.gamma + 1e-6:
        return (f"Your answers give γ = {profile.gamma:.2f}, but with it the portfolio could lose more than "
                f"{-profile.max_loss:.0%} in a bad year. We use γ = {gamma:.2f}: the closest risk aversion that "
                "respects the loss you can tolerate (as 'targeting a volatility' in lab LN1).")
    return None
