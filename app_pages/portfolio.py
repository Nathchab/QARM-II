"""Page 3: the optimal portfolio of the client, with 4 methods."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.app_helpers import (get_benchmark, get_choices, get_currency, get_portfolio_context,
                             get_profile, get_views, portfolio_status_table,
                             show_allocation_plan, show_portfolio_assessment, show_profile_notice)
from src.black_litterman import market_weights, posterior_returns, prior_returns
from src.data import ASSETS
from src.optimizer import beta, frontier, portfolio_stats, risk_contributions, tracking_error
from src.portfolio_diagnostics import portfolio_diagnostics
from src.portfolios import METHODS, WINDOW, estimate

CLASS_COLORS = {"Equity": "#1f77b4", "Bond": "#2ca02c", "Other": "#ff7f0e", "Crypto": "#9467bd"}

profile = get_profile()
requested_tickers, requested_c = get_choices(allow_infeasible=True, profile=profile)
currency = get_currency()

st.title("Your portfolio")
plan, assessments, mu, cov, returns, rf, prices = get_portfolio_context(profile, currency, requested_tickers, requested_c)
show_allocation_plan(plan)
if not plan.feasible:
    show_profile_notice(profile, plan, assessments)
    st.page_link("app_pages/choices.py", label="Review products and constraints")
    st.stop()
tickers, c = plan.tickers, plan.constraints
benchmark_name, benchmark = get_benchmark(tickers, c)
rf_now = rf.iloc[-1] * 12
st.caption(f"Risk Tolerance: {profile.label} · {len(tickers)} products · Currency: {currency} · "
           f"Estimated on the last {WINDOW} months of data")

# Monthly returns of the last 60 months, to compare with the benchmark (beta, tracking error)
past = returns[tickers].dropna().iloc[-WINDOW:]
benchmark_returns = returns.loc[past.index, benchmark.index].fillna(0) @ benchmark

method = st.radio("Choose a method", METHODS, horizontal=True,
                  index=METHODS.index(st.session_state.get("method", "Mean-variance")))
st.session_state["method"] = method
if method == "Risk parity (ERC)":
    st.caption("Risk Parity does not use gamma in its objective.")

# ---------- Black-Litterman: the client gives his views ----------
if method == "Black-Litterman":
    before_views = dict(get_views())
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
    views = {}
    for t in chosen:
        q_old, conf_old = old_views.get(t, (round(prior[t] * 100) / 100, 0.5))
        col1, col2 = st.columns(2)
        q = col1.slider(f"Expected return of {t} ({ASSETS[t][0]}), % per year", -30, 50, int(round(q_old * 100)),
                        key=f"view_{t}")
        conf = col2.slider(f"Your confidence in this view, %", 5, 95, int(conf_old * 100), key=f"conf_{t}")
        views[t] = (q / 100, conf / 100)
    st.session_state["views"] = views
    if views != before_views:
        plan, assessments, mu, cov, returns, rf, prices = get_portfolio_context(profile, currency, requested_tickers, requested_c)

profile_notice = show_profile_notice(profile, plan, assessments)
st.dataframe(portfolio_status_table(assessments), hide_index=True)
st.caption("Method status describes calculation and allocation validity. Financial/liquidity information above applies to the investor profile.")
portfolios = {name: result.weights for name, result in assessments.items() if result.weights is not None}
if assessments[method].weights is None:
    show_portfolio_assessment(assessments[method])
    if not portfolios:
        st.stop()
    requested_method = method
    method = next(iter(portfolios))
    st.info(f"Illustrative alternative: {requested_method} has no reliable calculation. Showing the calculated {method} portfolio with the same mandatory limits.")
show_portfolio_assessment(assessments[method])
weights = portfolios[method]

# Expected returns used by this method (Black-Litterman has its own)
mu_method = posterior_returns(cov, rf_now, {t: v for t, v in get_views().items() if t in mu.index}) if method == "Black-Litterman" else mu

# ---------- Key numbers ----------
ret, vol = portfolio_stats(weights, mu_method, cov)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Expected return", f"{ret:.1%}")
col2.metric("Volatility", f"{vol:.1%}")
col3.metric("Sharpe ratio", f"{(ret - rf_now) / vol:.2f}" if vol > 0 else "Undefined")
col4.metric("Beta vs benchmark", f"{beta(past @ weights, benchmark_returns):.2f}",
            help=f"Benchmark: {benchmark_name} (you can change it on the backtest page)")
diagnostics = portfolio_diagnostics(weights, mu_method, cov, c)
cols = st.columns(4)
for col, key in zip(cols, ("Equity exposure", "Bond exposure", "Largest holding", "HHI")):
    col.metric(key, f"{diagnostics[key]:.1%}" if key != "HHI" else f"{diagnostics[key]:.4f}")
st.caption(f"Effective holdings (1/HHI): {diagnostics['Effective holdings']:.2f}. "
           "HHI is the sum of squared capital weights; a higher value means greater concentration. "
           "Return and volatility are annual estimates, not guarantees.")
if diagnostics["Allocation limit violations"]:
    st.error("Observed allocation limit violations: " + ", ".join(diagnostics["Allocation limit violations"]))

# ---------- Weights and risk contributions ----------
table = pd.DataFrame({
    "Product": [ASSETS[t][0] for t in weights.index],
    "Class": [ASSETS[t][1] for t in weights.index],
    "Weight": weights,
    "Risk contribution": risk_contributions(weights, cov),
})
with st.expander("Advanced details"):
    st.write(f"RT: {getattr(profile, 'rt', 'legacy')}; willingness gamma: {profile.gamma_willingness:.5f}; "
             f"capacity floor: {profile.gamma_capacity:g}; final gamma: {profile.gamma:.5f}.")
    st.write(getattr(profile, "gamma_binding_explanation", profile.gamma_source))
    st.caption("Risk Parity does not consume gamma. Other methods use final gamma.")
    st.write("Binding execution allocation limits: "
             + (", ".join(diagnostics["Binding allocation limits"]) or "none within numerical tolerance"))
    if getattr(profile, "gamma_automatic", False):
        mandatory_binding = portfolio_diagnostics(weights, mu_method, cov, profile.mandatory_constraints)
        st.write("Binding mandatory allocation limits: "
                 + (", ".join(mandatory_binding["Binding allocation limits"]) or "none within numerical tolerance"))
    st.dataframe(portfolio_status_table(assessments, advanced=True), hide_index=True)
    st.dataframe(table.style.format({"Weight": "{:.5%}", "Risk contribution": "{:.5%}"}))
    if method == "Risk parity (ERC)":
        shares = table["Risk contribution"].to_numpy()
        target = 1 / len(weights)
        st.write(f"Equal-risk target per asset: {target:.5%}; contribution sum: {shares.sum():.5%}; "
                 f"ERC objective (1000 × squared share deviations): {1000 * ((shares - target)**2).sum():.8f}.")
        st.caption("Binding allocation limits can prevent equal risk shares. "
                   "The existing optimizer interface returns weights, not a convergence certificate; "
                   "the focused solver audit is recorded in analysis/risk_parity_validation/REPORT.md.")
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
except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
    front = pd.DataFrame()
    show_profile_notice(profile, plan, assessments,
                        [f"The frontier could not be calculated ({error}); the portfolio and its diagnostics remain available."],
                        warning_slot=profile_notice)
fig = go.Figure()
if not front.empty:
    fig.add_scatter(x=front["Volatility"], y=front["Return"], mode="lines", name="Efficient frontier (calculation constraints)")
fig.add_scatter(x=np.sqrt(np.diag(cov)), y=mu_method, mode="markers+text", text=list(mu.index),
                textposition="top center", name="Single products")
fig.add_scatter(x=[vol], y=[ret], mode="markers", marker=dict(size=16, symbol="star"), name="Your portfolio")
fig.update_layout(title="Efficient frontier", xaxis_title="Volatility", yaxis_title="Expected return",
                  xaxis_tickformat=".0%", yaxis_tickformat=".0%")
st.plotly_chart(fig)
if method == "Risk parity (ERC)":
    st.caption("Risk parity does not use expected returns, so it is usually below the frontier: "
               "it gives up some expected return to spread the risk.")

# ---------- Compare the 4 methods ----------
st.subheader("Compare the methods")
compare = pd.DataFrame(portfolios)
compare.index = [f"{t} ({ASSETS[t][0]})" for t in compare.index]
stats = pd.DataFrame({m: {"Expected return (common historical estimates)": portfolio_stats(w, mu, cov)[0],
                          **{key: portfolio_diagnostics(w, mu, cov, c)[key]
                             for key in ("Volatility", "Equity exposure", "Bond exposure", "Largest holding", "HHI")},
                          "Beta vs benchmark": beta(past @ w, benchmark_returns),
                          "Tracking error vs benchmark": tracking_error(past @ w, benchmark_returns)}
                      for m, w in portfolios.items()})
st.dataframe(pd.concat([compare, stats]).style.format("{:.1%}").format(
    "{:.2f}", subset=pd.IndexSlice[["Beta vs benchmark"], :]).format(
    "{:.4f}", subset=pd.IndexSlice[["HHI"], :]))
st.caption(f"Benchmark: {benchmark_name}. Beta and tracking error (lab LN2) on the last {WINDOW} months.")
st.caption("Comparison returns use common historical estimates for all methods; Black-Litterman's "
           "selected-portfolio expected return above uses its posterior. Risk Parity: gamma N/A.")

# ---------- Composition map (lab LN3): sensitivity to gamma ----------
if method != "Risk parity (ERC)" and not front.empty:
    st.subheader("Mean-variance reference: what if gamma was different?")
    st.write("This composition map solves the mean-variance objective with these expected returns and "
             "constraints. It is a reference for sensitivity, not a resampled allocation path. "
             "The dashed line is execution gamma.")
    area = px.area(front, x="Gamma", y=list(mu.index), log_x=True)
    area.add_vline(x=profile.gamma, line_dash="dash")
    area.update_layout(yaxis_title="Weight", yaxis_tickformat=".0%", xaxis_title="γ (risk aversion)")
    st.plotly_chart(area)
