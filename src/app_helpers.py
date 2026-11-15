"""
Small helpers shared by the pages of the app (with Streamlit cache, so pages load fast).
"""
import streamlit as st

from src.black_litterman import posterior_returns, prior_returns, restrict_views
from src.client_profile import ClientProfile
from src.data import SHORT_TERM, available_assets, load_prices, monthly_returns, risk_free
from src.optimizer import check_constraints, class_exposures, portfolio_stats
from src.portfolios import (BENCHMARKS, LABELS, bad_year_loss, benchmark_weights, build_portfolios, estimate,
                            gamma_for_loss_limit, recommended_method)


def tighten(profile_limits, chosen):
    """The client can make the profile limits stricter, never looser (the region stays the one of the profile)."""
    return {key: limit if isinstance(limit, str) else max(limit, chosen.get(key, limit)) if key.startswith("min_")
            else min(limit, chosen.get(key, limit)) for key, limit in profile_limits.items()}


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
    return refresh_profile()


def reset_choices():
    """Forget the products, limits and method chosen by the client: back to our recommendation.
    The widgets of the products page get new keys (choices_version), otherwise the browser keeps showing
    the old values."""
    for key in ["assets", "constraints", "method"] + [k for k in st.session_state if k.startswith("pick_")]:
        st.session_state.pop(key, None)
    st.session_state["choices_version"] = st.session_state.get("choices_version", 0) + 1


def refresh_profile():
    """The client's profile, recomputed if he changed the currency (bonds and cash depend on it)."""
    profile = st.session_state["profile"]
    if profile.currency != get_currency():
        profile = st.session_state["profile"] = ClientProfile(profile.answers, get_currency())
        reset_choices()
    return profile


def explain(profile, text):
    """Explanation of a chart or table: always visible for beginners, one click away for experienced investors."""
    if profile.experienced:
        with st.expander("💡 What does this show?"):
            st.write(text)
    else:
        st.info("💡 **What does this show?** " + text)


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


def advice(profile, tickers, c, views=None):
    """get_portfolios for this client, with his own views by default."""
    return get_portfolios(get_currency(), tuple(tickers), profile.gamma, profile.max_loss, c,
                          get_views() if views is None else views)


def recommended(profile, tickers, c, views=None):
    """The method we recommend to this client (rules in recommended_method)."""
    portfolios, errors, gamma, forecast, cov = advice(profile, tickers, c, views)
    return recommended_method(profile, portfolios, forecast, cov)


def client_method(profile, tickers, c):
    """The client's portfolio: the method chosen on the portfolio page, by default our recommendation."""
    return st.session_state.get("method") or recommended(profile, tickers, c)


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


def get_benchmark(profile, tickers, c):
    """Name and weights of the benchmark chosen on the backtest page. By default: the S&P 500 and bonds in the
    client's currency, with the same share of equities as his recommended portfolio."""
    name = st.session_state.get("benchmark", BENCHMARKS[0])
    portfolios, best = advice(profile, tickers, c)[0], recommended(profile, tickers, c)
    equity_share = class_exposures(portfolios[best])["Equity"] if best in portfolios else c["max_equity"]
    return name, benchmark_weights(name, tickers, equity_share, get_currency())


@st.cache_data
def get_data(currency):
    """Monthly returns of all products, monthly risk-free rate, daily prices."""
    prices = load_prices(currency)
    return monthly_returns(prices), risk_free(currency), prices


@st.cache_data(max_entries=100)  # keep at most 100 results in memory (the website runs all the time)
def get_portfolios(currency, tickers, gamma, max_loss, c, views):
    """Today's advice: (weights by method, errors by method, risk aversion used, forecast, covariance matrix).
    forecast = the market equilibrium returns (+ the client's views), the same for every method.
    The risk aversion is the client's gamma, raised if needed to respect his loss limit (QE2 or capacity)."""
    returns, rf, prices = get_data(currency)
    mu, cov, market_cov = estimate(returns, tickers)
    pi = prior_returns(market_cov, rf.iloc[-1] * 12)
    forecast = posterior_returns(pi, cov, restrict_views(views, mu.index))
    try:
        gamma = gamma_for_loss_limit(forecast, cov, gamma, c, max_loss)
    except (ValueError, RuntimeError):
        pass  # the optimisation fails: build_portfolios below reports the error
    portfolios, errors = build_portfolios(mu, cov, pi, gamma, c, views)
    return portfolios, errors, gamma, forecast, cov


def get_valid_portfolios(profile, tickers, c):
    """(portfolios that could be calculated, risk aversion used, forecast of the returns, covariance matrix);
    stop the page if no portfolio could be calculated."""
    portfolios, errors, gamma, forecast, cov = advice(profile, tickers, c)
    for method, error in errors.items():
        st.warning(f"{method_label(method)} could not be calculated with these inputs ({error}).")
    if not portfolios:
        st.error("No portfolio could be calculated. Change your products or constraints.")
        st.stop()
    return portfolios, gamma, forecast, cov


def recommendation_summary(profile, tickers, c):
    """One sentence: what our recommendation gives for this client's gamma and limits (the client has not
    entered views yet). None if it cannot be calculated."""
    try:
        portfolios, errors, gamma, forecast, cov = advice(profile, tickers, c, views=())
        best = recommended_method(profile, portfolios, forecast, cov)
        w = portfolios[best]
    except Exception:  # missing data or failed optimisation: no sentence (the pages show their own messages)
        return None
    exposure = class_exposures(w)
    loss = bad_year_loss(w, forecast, cov)
    note = loss_limit_note(profile, gamma, loss, get_currency(), best)
    return (f"Our recommended portfolio for you ({LABELS[best]}) has about {exposure['Equity']:.0%} in equities, "
            f"{exposure['Bond']:.0%} in bonds and {exposure['Other'] + exposure['Crypto']:.0%} in other assets, "
            f"with an estimated volatility of {portfolio_stats(w, forecast, cov)[1]:.1%} per year and "
            + (f"a loss of about {-loss:.1%} in a bad year (5% chance)." if loss < 0 else "no loss even in a bad year.")
            + (" " + note if note else ""))


def loss_limit_note(profile, gamma, loss, currency, method):
    """Explains how the client's loss limit was applied to the recommended portfolio (None if nothing to say).
    loss = bad-year loss of the recommended portfolio `method`."""
    if profile.max_loss is None:
        return None
    if loss < profile.max_loss - 1e-4:  # even the safest portfolio inside the limits loses more
        advice = (f"Adding short-term government bonds ({SHORT_TERM[currency]}) or removing risky products "
                  "would lower the risk.")
        return (f"Even the safest portfolio within your limits could lose about {-loss:.0%} in a bad year, more "
                f"than the {-profile.max_loss:.0%} {profile.loss_reason}: we recommend this safest portfolio "
                f"(minimum risk). {advice}")
    if gamma > profile.gamma + 1e-6 and method != "Minimum variance":  # minimum variance does not use gamma
        return (f"Your answers give γ = {profile.gamma:.2f}, but with it the portfolio could lose more than "
                f"{-profile.max_loss:.0%} in a bad year, the loss {profile.loss_reason}. We use γ = {gamma:.2f}: "
                "the closest risk aversion that respects this limit.")
    return None
