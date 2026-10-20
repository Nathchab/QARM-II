"""Page 3: the optimal portfolio of the client, with 4 methods."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.app_helpers import (get_benchmark, get_choices, get_currency, get_data, get_profile,
                             get_valid_portfolios, get_views, show_profile_messages)
from src.black_litterman import CONFIDENCE_LEVELS, market_weights, posterior_returns, prior_returns
from src.data import ASSETS
from src.optimizer import beta, frontier, portfolio_stats, risk_contributions, tracking_error
from src.portfolio_diagnostics import portfolio_diagnostics
from src.portfolios import METHODS, WINDOW, estimate

CLASS_COLORS = {"Equity": "#1f77b4", "Bond": "#2ca02c", "Other": "#ff7f0e", "Crypto": "#9467bd"}

profile = get_profile()
tickers, c = get_choices()
currency = get_currency()
benchmark_name, benchmark = get_benchmark(tickers, c)

st.title("Your portfolio")
returns, rf, prices = get_data(currency)
mu, cov = estimate(returns[tickers])
rf_now = rf.iloc[-1] * 12
st.caption(f"Profile: {profile.label} (γ = {profile.gamma:.2f}) · Capacity: {profile.capacity_label} · "
           f"{len(tickers)} products · Currency: {currency} · Estimated on the last {WINDOW} months of data")
show_profile_messages(profile)

# Monthly returns of the last 60 months, to compare with the benchmark (beta, tracking error)
past = returns[tickers].dropna().iloc[-WINDOW:]
benchmark_returns = returns.loc[past.index, benchmark.index].fillna(0) @ benchmark

method = st.radio("Choose a method", METHODS, horizontal=True,
                  index=METHODS.index(st.session_state.get("method", "Mean-variance")))
st.session_state["method"] = method

# ---------- Black-Litterman: the client gives his views (QH5) and his confidence (QH6) ----------
if method == "Black-Litterman":
    st.subheader("Your views on the markets")
    prior = prior_returns(cov, rf_now)
    st.write("Without views, Black-Litterman uses the **market equilibrium** returns: the returns that make "
             "the world market portfolio optimal. Then you can add your own views.")
    st.dataframe(pd.DataFrame({
        "Product": [ASSETS[t][0] for t in mu.index],
        "Market weight": market_weights(mu.index),
        "Equilibrium return": prior,
        f"Historical return ({WINDOW} months)": mu,
    }).style.format({"Market weight": "{:.0%}", "Equilibrium return": "{:.1%}",
                     f"Historical return ({WINDOW} months)": "{:.1%}"}))

    old_views = get_views()
    chosen = st.multiselect("On which products do you have a view?", list(mu.index),
                            default=[t for t in old_views if t in mu.index])
    levels = list(CONFIDENCE_LEVELS)
    views = {}
    for t in chosen:
        q_old, conf_old = old_views.get(t, (round(prior[t] * 100) / 100, CONFIDENCE_LEVELS["Medium"]))
        level_old = min(levels, key=lambda name: abs(CONFIDENCE_LEVELS[name] - conf_old))
        col1, col2 = st.columns(2)
        q = col1.slider(f"Expected return of {t} ({ASSETS[t][0]}), % per year", -30, 50, int(round(q_old * 100)),
                        key=f"view_{t}")
        level = col2.select_slider("Your confidence in this view", levels, value=level_old, key=f"conf_{t}")
        views[t] = (q / 100, CONFIDENCE_LEVELS[level])
    st.session_state["views"] = views
    st.caption("Confidence sets the uncertainty of the view: Ω = τσ²(1 − c)/c, with c = 20%, 50%, 80% or 95%. "
               "Medium gives the view the same uncertainty as the market equilibrium.")

portfolios = get_valid_portfolios(currency, tickers, profile.gamma, c, get_views())
if method not in portfolios:
    method = next(iter(portfolios))
    st.info(f"Showing {method} instead.")
weights = portfolios[method]

# Expected returns used by this method (Black-Litterman has its own)
mu_method = (posterior_returns(cov, rf_now, {t: v for t, v in get_views().items() if t in mu.index})
             if method == "Black-Litterman" else mu)

# ---------- Key numbers ----------
ret, vol = portfolio_stats(weights, mu_method, cov)
diagnostics = portfolio_diagnostics(weights, mu_method, cov, c)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Expected return", f"{ret:.1%}")
col2.metric("Volatility", f"{vol:.1%}")
col3.metric("Sharpe ratio", f"{(ret - rf_now) / vol:.2f}" if vol > 0 else "–")
col4.metric("Beta vs benchmark", f"{beta(past @ weights, benchmark_returns):.2f}",
            help=f"Benchmark: {benchmark_name} (you can change it on the backtest page)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("In equities", f"{diagnostics['Equity exposure']:.0%}", help=f"Your maximum: {c['max_equity']:.0%}")
col2.metric("In bonds", f"{diagnostics['Bond exposure']:.0%}", help=f"Your minimum: {c['min_bonds']:.0%}")
col3.metric("Largest product", f"{diagnostics['Largest holding']:.0%}")
col4.metric("Effective number of products", f"{diagnostics['Effective holdings']:.1f}", help="1 / Σ wᵢ²")
binding = [b for b in diagnostics["Binding allocation limits"] if not b.startswith("max_weight")
           and not (b == "max_crypto" and c["max_crypto"] == 0) and not (b == "min_bonds" and c["min_bonds"] == 0)]
if binding and method != "Risk parity (ERC)":
    names = {"max_equity": "maximum in equities", "min_bonds": "minimum in bonds",
             "max_other": "maximum in gold and commodities", "max_crypto": "maximum in crypto"}
    st.caption("Your limits are binding: " + ", ".join(names[b] for b in binding) + ". Without them, the "
               "optimiser would choose a different mix for your γ.")

# ---------- Weights and risk contributions ----------
table = pd.DataFrame({
    "Product": [ASSETS[t][0] for t in weights.index],
    "Class": [ASSETS[t][1] for t in weights.index],
    "Weight": weights,
    "Risk contribution": risk_contributions(weights, cov),
})
col1, col2 = st.columns(2)
col1.plotly_chart(px.bar(table, x=table.index, y="Weight", color="Class", title="Weights",
                         color_discrete_map=CLASS_COLORS,
                         text_auto=".0%").update_layout(yaxis_tickformat=".0%", xaxis_title=None))
col2.plotly_chart(px.bar(table, x=table.index, y="Risk contribution", color="Class",
                         color_discrete_map=CLASS_COLORS,
                         title="Where does the risk come from?",
                         text_auto=".0%").update_layout(yaxis_tickformat=".0%", xaxis_title=None))

# ---------- Efficient frontier ----------
try:
    front = frontier(mu_method, cov, c)
except (ValueError, RuntimeError, np.linalg.LinAlgError):
    front = pd.DataFrame()
    st.caption("The efficient frontier could not be calculated with these inputs.")
fig = go.Figure()
if not front.empty:
    fig.add_scatter(x=front["Volatility"], y=front["Return"], mode="lines", name="Efficient frontier (your constraints)")
fig.add_scatter(x=np.sqrt(np.diag(cov)), y=mu_method, mode="markers+text", text=list(mu.index),
                textposition="top center", name="Single products")
fig.add_scatter(x=[vol], y=[ret], mode="markers", marker=dict(size=16, symbol="star"), name="Your portfolio")
fig.update_layout(title="Efficient frontier", xaxis_title="Volatility", yaxis_title="Expected return",
                  xaxis_tickformat=".0%", yaxis_tickformat=".0%")
st.plotly_chart(fig)
if method == "Risk parity (ERC)":
    st.caption("Risk parity does not use expected returns nor γ, so it is usually below the frontier: "
               "it gives up some expected return to spread the risk.")

# ---------- Compare the 4 methods ----------
st.subheader("Compare the methods")
compare = pd.DataFrame(portfolios)
compare.index = [f"{t} ({ASSETS[t][0]})" for t in compare.index]
stats = pd.DataFrame({m: {"Expected return (historical μ)": portfolio_stats(w, mu, cov)[0],
                          "Volatility": portfolio_stats(w, mu, cov)[1],
                          "Beta vs benchmark": beta(past @ w, benchmark_returns),
                          "Tracking error vs benchmark": tracking_error(past @ w, benchmark_returns)}
                      for m, w in portfolios.items()})
st.dataframe(pd.concat([compare, stats]).style.format("{:.1%}").format(
    "{:.2f}", subset=pd.IndexSlice[["Beta vs benchmark"], :]))
st.caption(f"Benchmark: {benchmark_name}. Beta and tracking error (lab LN2) on the last {WINDOW} months. "
           "Expected returns use the same historical estimates for every method, so they can be compared.")

# ---------- Composition map (lab LN3): sensitivity to gamma ----------
if not front.empty and method != "Risk parity (ERC)":
    st.subheader("What if your γ was different?")
    st.write("Composition map (as in lab LN3): the optimal mean-variance weights for every γ, with your "
             "constraints and the expected returns of the method. The dashed line is your γ.")
    area = px.area(front, x="Gamma", y=list(mu.index), log_x=True)
    area.add_vline(x=profile.gamma, line_dash="dash")
    area.update_layout(yaxis_title="Weight", yaxis_tickformat=".0%", xaxis_title="γ (risk aversion)")
    st.plotly_chart(area)
