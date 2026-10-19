"""Page 4: out-of-sample backtest of the 4 methods against a benchmark chosen by the client."""
import pandas as pd
import plotly.express as px
import streamlit as st

from src.app_helpers import (get_choices, get_currency, get_data, get_portfolio_context,
                             get_profile, get_views, show_allocation_plan, show_portfolio_assessment, show_profile_notice)
from src.backtest import N_SIM, performance, run_backtest
from src.portfolios import BENCHMARKS, WINDOW, benchmark_weights
from src.portfolio_policy import assess_backtest_weights

profile = get_profile()
tickers, c = get_choices(allow_infeasible=True, profile=profile)
currency = get_currency()
plan, assessments, mu, cov, returns, rf, prices = get_portfolio_context(profile, currency, tickers, c)

st.title("Backtest: how would it have worked?")
show_allocation_plan(plan)
if not plan.feasible:
    show_profile_notice(profile, plan, assessments)
    st.stop()
tickers, c = plan.tickers, plan.constraints
st.caption("Historical simulations retain allocation checks at rebalancing dates. "
           "Volatility differences between methods are risk metrics, not pass/fail tests.")
st.write(f"At each rebalancing we re-estimate the inputs on the **past {WINDOW} months only** and "
         "re-optimise gamma-based methods with execution gamma and allocation constraints. "
         "Risk Parity does not use gamma. Between two rebalancings we hold the portfolio.")

REBALANCING = {"Every month": 1, "Every 3 months": 3, "Every 6 months": 6, "Every year": 12}
if len(returns[tickers].dropna()) <= WINDOW:
    show_profile_notice(profile, plan, assessments, ["There is insufficient overlapping history for a 60-month out-of-sample backtest. A current illustrative calculation is shown instead."])
    for name, result in assessments.items():
        if result.weights is not None:
            st.write(name)
            show_portfolio_assessment(result)
            st.dataframe(result.weights.rename("Weight").to_frame())
            break
    st.stop()
first = returns[tickers].dropna().index[WINDOW].date()  # we need 60 months of history first
last = returns.index[-1].date()

# ---------- Settings (saved, so they stay when you change page) ----------
s = st.session_state.get("bt_settings", {"period": (first, last), "rebalancing": "Every 3 months",
                                          "cost_bps": 10, "cost_aware": True, "max_turnover": 200})
period = (max(s["period"][0], first), min(s["period"][1], last))
with st.form("settings"):
    col1, col2 = st.columns(2)
    period = col1.slider("Backtest period", first, last, period, format="MMM YYYY")
    benchmark_name = col2.selectbox("Benchmark", BENCHMARKS,
                                    index=BENCHMARKS.index(st.session_state.get("benchmark", BENCHMARKS[0])))
    rebalancing = col1.selectbox("Rebalancing", list(REBALANCING), index=list(REBALANCING).index(s["rebalancing"]))
    cost_bps = col2.slider("Transaction cost (basis points of the amount traded)", 0, 100, s["cost_bps"],
                           help="10 bp = 0.10%. Typical for ETFs: 5 to 20 bp.")
    max_turnover = col1.slider("Maximum turnover per rebalancing (%)", 5, 200, s["max_turnover"], 5,
                               help="Turnover = sum of |new weight - old weight|. 200% = no limit (LN3).")
    cost_aware = col2.checkbox("Cost-aware optimisation (LN3)", s["cost_aware"],
                               help="Mean-variance and Black-Litterman include the costs in the optimisation, "
                                    "so they only trade when it is worth it.")
    st.form_submit_button("Run the backtest", type="primary")

st.session_state["benchmark"] = benchmark_name
st.session_state["bt_settings"] = {"period": period, "rebalancing": rebalancing, "cost_bps": cost_bps,
                                   "cost_aware": cost_aware, "max_turnover": max_turnover}


@st.cache_data(show_spinner=False)
def cached_backtest(currency, tickers, benchmark_name, gamma, c, views, period, rebalance, cost, cost_aware,
                    max_turnover):
    returns, rf, prices = get_data(currency)
    benchmark = benchmark_weights(benchmark_name, tickers, c)
    return run_backtest(returns, rf, tickers, benchmark, gamma, c, views, pd.Timestamp(period[0]),
                        pd.Timestamp(period[1]), rebalance, cost, cost_aware, max_turnover)


cost = cost_bps / 10_000
with st.spinner("Running the backtest (10 to 60 seconds the first time)..."):
    try:
        results, weights_history, turnover = cached_backtest(
            currency, tickers, benchmark_name, profile.gamma, c, get_views(), period,
            REBALANCING[rebalancing], cost, cost_aware, max_turnover / 100)
    except (ValueError, RuntimeError) as error:
        show_profile_notice(profile, plan, assessments, [f"A reliable historical simulation could not be calculated: {error}"])
        for name, result in assessments.items():
            if result.weights is not None:
                st.write(f"Current {name} calculation (not a backtest):")
                show_portfolio_assessment(result)
                st.dataframe(result.weights.rename("Weight").to_frame())
                break
        st.stop()

audit = assess_backtest_weights(returns[tickers], weights_history, profile, c,
                                illustrative=plan.illustrative, assumptions=plan.assumptions)
historical_issues = []
if not audit.empty:
    historical_issues = list(dict.fromkeys(audit.loc[audit.Explanation.ne(""), "Explanation"]))
show_profile_notice(profile, plan, assessments, historical_issues)
if not audit.empty:
    st.subheader("Allocation checks at rebalancing")
    st.dataframe(audit.groupby(["Method", "Status"]).size().rename("Rebalancing dates").reset_index(), hide_index=True)
    with st.expander("Rebalancing allocation and risk diagnostics"):
        st.caption("Allocation checks use actual rebalance holdings. Volatility estimates use only returns before each date; there is no MV-reference ceiling.")
        st.dataframe(audit, hide_index=True)

if len(results) < 12:
    st.error("The period is too short (or the benchmark has no data on it). Choose a longer period.")
    st.stop()
results = results.rename(columns={"Benchmark": f"Benchmark: {benchmark_name}"})
turnover = turnover.rename(columns={"Benchmark": f"Benchmark: {benchmark_name}"})

st.caption(f"From {results.index[0]:%B %Y} to {results.index[-1]:%B %Y}, in {currency}, after transaction "
           f"costs. Black-Litterman uses your current views at every date. Resampled MV uses {N_SIM} "
           f"simulations per rebalancing (100 on the portfolio page).")

# ---------- Growth of 100 ----------
growth = 100 * (1 + results).cumprod()
fig = px.line(growth, title=f"Value of 100 {currency} invested")
fig.update_layout(yaxis_title=currency, xaxis_title=None, legend_title=None)
st.plotly_chart(fig)

# ---------- Performance table ----------
benchmark_r = results.iloc[:, -1]
table = pd.DataFrame({name: performance(results[name], rf, benchmark_r, turnover[name], cost) for name in results})
st.dataframe(table.style.format("{:.1%}").format(
    "{:.2f}", subset=pd.IndexSlice[["Sharpe ratio", "Beta vs benchmark"], :]).format(
    "{:.2%}", subset=pd.IndexSlice[["Costs paid per year"], :]))

# ---------- Transaction costs (LN3) ----------
st.subheader("Trading and transaction costs")
st.write("Turnover at each rebalancing (the first one is the initial purchase). With cost-aware optimisation "
         "or a turnover limit, the portfolios trade less, so they pay less costs.")
fig = px.bar(turnover.iloc[1:], barmode="group", title="Turnover at each rebalancing")
fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Turnover", xaxis_title=None, legend_title=None)
st.plotly_chart(fig)

# ---------- Weights over time ----------
st.subheader("How did the weights change over time?")
method = st.selectbox("Method", list(weights_history)[:-1])
area = px.area(weights_history[method], title=f"Weights of {method} at each rebalancing")
area.update_layout(yaxis_tickformat=".0%", yaxis_title="Weight", xaxis_title=None, legend_title=None)
st.plotly_chart(area)
