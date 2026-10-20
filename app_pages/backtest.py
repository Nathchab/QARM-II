"""Page 4: out-of-sample backtest of the 4 methods against a benchmark chosen by the client."""
import pandas as pd
import plotly.express as px
import streamlit as st

from src.app_helpers import get_choices, get_currency, get_data, get_profile, get_views
from src.backtest import N_SIM, performance, run_backtest
from src.portfolios import BENCHMARKS, WINDOW, benchmark_weights

profile = get_profile()
tickers, c = get_choices()
currency = get_currency()
returns, rf, prices = get_data(currency)

st.title("Backtest: how would it have worked?")
st.write(f"At each rebalancing we re-estimate the inputs on the **past {WINDOW} months only** and "
         "re-optimise with your γ and your constraints (risk parity does not use γ). Between two "
         "rebalancings we hold the portfolio.")

history = returns[tickers].dropna()
if len(history) <= WINDOW + 12:
    st.error(f"Your products have only {len(history)} months of common history: we need {WINDOW} months to "
             "estimate the inputs plus at least one year to test. Remove the youngest products.")
    st.stop()

REBALANCING = {"Every month": 1, "Every 3 months": 3, "Every 6 months": 6, "Every year": 12}
first = history.index[WINDOW].date()  # we need 60 months of history first
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
        st.error(f"The backtest could not be calculated on this period ({error}). Try another period or other products.")
        st.stop()

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
