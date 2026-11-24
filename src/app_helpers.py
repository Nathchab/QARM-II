"""
Small helpers shared by the pages of the app (with Streamlit cache, so pages load fast).
"""
import streamlit as st

from src.black_litterman import posterior_returns, prior_returns, restrict_views
from src.client_profile import ClientProfile
from src.data import SHORT_TERM, available_assets, load_prices, monthly_returns, risk_free
from src.optimizer import check_constraints, class_exposures, frontier, portfolio_stats
from src.portfolios import (BENCHMARKS, LABELS, bad_year_loss, benchmark_weights, build_portfolios, estimate,
                            gamma_for_limits, recommended_method)


def tighten(profile_limits, chosen):
    """The client can make the profile limits stricter, never looser (the region stays the one of the profile)."""
    return {key: limit if isinstance(limit, str) else max(limit, chosen.get(key, limit)) if key.startswith("min_")
            else min(limit, chosen.get(key, limit)) for key, limit in profile_limits.items()}


def next_button(page, label):
    """Button to another page (used when a page cannot be shown yet)."""
    if st.button(label, type="primary", icon=":material/arrow_forward:", icon_position="right", key=f"go_{page}"):
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


def show_profile_messages(profile, notes=False):
    """Material financial facts (one warning box) and, on the profile page, neutral explanations."""
    if profile.warnings:
        st.warning("**Points to review in your situation**\n\n" + "\n".join(f"- {w}" for w in profile.warnings),
                   icon=":material/warning:")
    if notes and profile.notes:
        st.markdown("**How your answers shape the portfolio**\n\n" + "\n".join(f"- {n}" for n in profile.notes))


def get_choices():
    """Products and constraints chosen by the client (our recommendation by default)."""
    profile = get_profile()
    tickers = st.session_state.get("assets", profile.recommended_assets)
    available = available_assets()
    tickers = [t for t in tickers if t in available]
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
    """The client's portfolio: the method he chose on the portfolio page if it still exists (Black-Litterman
    disappears when he removes his views), otherwise our recommendation."""
    chosen = st.session_state.get("method")
    return chosen if chosen in advice(profile, tickers, c)[0] else recommended(profile, tickers, c)


def get_benchmark(profile, tickers, c):
    """Name and weights of the benchmark chosen on the backtest page. By default: the S&P 500 and bonds in the
    client's currency, with the same share of equities as his recommended portfolio."""
    name = st.session_state.get("benchmark", BENCHMARKS[0])
    portfolios, best = advice(profile, tickers, c)[0], recommended(profile, tickers, c)
    equity_share = class_exposures(portfolios[best])["Equity"] if best in portfolios else c["max_equity"]
    return name, benchmark_weights(name, tickers, equity_share, get_currency())


@st.cache_data(ttl="1d")  # like the prices: new data every day
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
        gamma = gamma_for_limits(forecast, cov, gamma, c, max_loss)
    except (ValueError, RuntimeError):
        pass  # the optimisation fails: build_portfolios below reports the error
    portfolios, errors = build_portfolios(mu, cov, pi, gamma, c, views)
    return portfolios, errors, gamma, forecast, cov


@st.cache_data(max_entries=100)
def get_frontier(forecast, cov, c):
    """Efficient frontier (40 optimisations): kept in memory, so the portfolio page reacts at once to the other widgets."""
    return frontier(forecast, cov, c)


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
    """One sentence: what our recommendation gives for this client's gamma, limits and views (the same portfolio
    as on the portfolio page). None if it cannot be calculated."""
    try:
        portfolios, errors, gamma, forecast, cov = advice(profile, tickers, c)
        best = recommended_method(profile, portfolios, forecast, cov)
        w = portfolios[best]
    except Exception:  # missing data or failed optimisation: no sentence (the pages show their own messages)
        return None
    exposure = class_exposures(w)
    loss = bad_year_loss(w, forecast, cov)
    note = limits_note(profile, gamma, loss, get_currency(), best, c)
    return (f"{LABELS[best]}: about {exposure['Equity']:.0%} in equities, "
            f"{exposure['Bond']:.0%} in bonds and {exposure['Other'] + exposure['Crypto']:.0%} in other assets, "
            f"with an estimated volatility of {portfolio_stats(w, forecast, cov)[1]:.1%} per year and "
            + (f"a loss of about {-loss:.1%} in a bad year (5% chance)." if loss < 0 else "no loss even in a bad year.")
            + (" " + note if note else ""))


def limits_note(profile, gamma, loss, currency, method, c):
    """Explains how the client's limits were applied to the recommended portfolio (None if nothing to say).
    loss = bad-year loss of the recommended portfolio `method`; gamma = the risk aversion we used."""
    if profile.max_loss is not None and loss < profile.max_loss - 1e-4:  # even the safest portfolio loses more
        advice = (f"Adding short-term government bonds ({SHORT_TERM[currency]}) or removing risky products "
                  "would lower the risk.")
        return (f"Even the safest portfolio within your limits could lose about {-loss:.0%} in a bad year, more "
                f"than the {-profile.max_loss:.0%} {profile.loss_reason}: we recommend this safest portfolio "
                f"(minimum risk). {advice}")
    if gamma > profile.gamma + 1e-6 and method not in ("Minimum variance", "Risk parity (ERC)"):  # no gamma there
        limits = ([f"a loss of more than {-profile.max_loss:.0%} in a bad year"] if profile.max_loss is not None
                  else []) + ([f"more than {c['max_equity']:.0%} in risky assets"] if c["max_equity"] < 1 else [])
        if not limits:
            return None
        return (f"Your answers give γ = {profile.gamma:.2f}, but with it the portfolio would go beyond your limits ("
                + " or ".join(limits) + f"). We use γ = {gamma:.2f}: the closest risk aversion that respects them, so "
                "the portfolio stays balanced instead of filling your limits with the riskiest products.")
    return None
