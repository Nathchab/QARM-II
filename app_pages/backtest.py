"""Page 5: out-of-sample backtest of the client's portfolio, the other methods and a benchmark."""
import pandas as pd
import plotly.express as px
import streamlit as st

from src.app_helpers import (client_method, get_benchmark, get_choices, get_currency, get_data, get_profile, get_views,
                             method_label)
from src.backtest import N_SIM, performance, run_backtest
from src.planning import MANAGEMENT_FEE
from src.portfolios import BENCHMARKS, WINDOW
from src.style import BENCHMARK_COLOR, HIGHLIGHT, OTHERS, YOURS, chart, explain, highlight_lines, nav_buttons, page_header

profile = get_profile()
tickers, c = get_choices()
currency = get_currency()
returns, rf, prices = get_data(currency)

page_header(5, "Backtest", f"How would the portfolio have done in the past? At each rebalancing we re-estimate the "
            f"inputs on the **past {WINDOW} months only** and re-optimise with your γ and your limits (risk parity "
            "does not use γ). Between two rebalancings we hold the portfolio.")

history = returns[tickers].dropna()
if len(history) <= WINDOW + 12:
    st.error(f"Your products have only {len(history)} months of common history: we need {WINDOW} months to "
             "estimate the inputs plus at least one year to test. Remove the youngest products.")
    st.stop()

REBALANCING = {"Every month": 1, "Every 3 months": 3, "Every 6 months": 6, "Every year": 12}
first = history.index[WINDOW].date()  # we need 60 months of history first
last = returns.index[-1].date()

# ---------- Settings (saved, so they stay when you change page) ----------
# Every widget has a key: Streamlit then keeps exactly what the client chose. Without a key, a widget whose default
# changes (here: the saved settings) is replaced by a new one and the client's last change is lost.
s = st.session_state.get("bt_settings", {"period": (first, last), "rebalancing": "Every 3 months",
                                          "cost_bps": 10, "cost_aware": True, "max_turnover": 200})
period = (max(s["period"][0], first), min(s["period"][1], last))
if period[0] >= period[1]:  # the saved period does not exist for these products: back to the full history
    period = (first, last)
with st.form("settings"):
    col1, col2 = st.columns(2)
    period = col1.slider("Backtest period", first, last, period, format="MMM YYYY", key="bt_period")
    benchmark_name = col2.selectbox("Benchmark", BENCHMARKS, key="bt_benchmark",
                                    index=BENCHMARKS.index(st.session_state.get("benchmark", BENCHMARKS[0])))
    rebalancing = col1.selectbox("Rebalancing", list(REBALANCING), index=list(REBALANCING).index(s["rebalancing"]),
                                 key="bt_rebalancing")
    cost_bps = col2.slider("Transaction cost (basis points of the amount traded)", 0, 100, s["cost_bps"],
                           key="bt_cost", help="10 bp = 0.10% of every amount bought or sold. Typical for ETFs: 5 to "
                                               "20 bp.")
    max_turnover = col1.slider("Maximum turnover per rebalancing (%)", 5, 200, s["max_turnover"], 5,
                               key="bt_turnover", help="Turnover = sum of |new weight - old weight|. 200% = no limit.")
    cost_aware = col2.checkbox("Cost-aware optimisation", s["cost_aware"], key="bt_cost_aware",
                               help="The mean-variance methods (market equilibrium, past returns, Black-Litterman) "
                                    "include the costs in the optimisation, so they only trade when it is worth it.")
    st.form_submit_button("Run the backtest", type="primary", icon=":material/play_arrow:")

st.session_state["benchmark"] = benchmark_name
st.session_state["bt_settings"] = {"period": period, "rebalancing": rebalancing, "cost_bps": cost_bps,
                                   "cost_aware": cost_aware, "max_turnover": max_turnover}


@st.cache_data(show_spinner=False, max_entries=100)
def cached_backtest(currency, tickers, benchmark, gamma, max_loss, c, views, period, rebalance, cost, cost_aware,
                    max_turnover):
    returns, rf, prices = get_data(currency)
    return run_backtest(returns, rf, tickers, benchmark, gamma, c, views, pd.Timestamp(period[0]),
                        pd.Timestamp(period[1]), rebalance, cost, cost_aware, max_turnover, max_loss)


cost = cost_bps / 10_000
with st.spinner("Running the backtest (10 to 60 seconds the first time)..."):
    try:
        results, weights_history, turnover, costs = cached_backtest(
            currency, tickers, get_benchmark(profile, tickers, c)[1], profile.gamma, profile.max_loss, c, get_views(),
            period, REBALANCING[rebalancing], cost, cost_aware, max_turnover / 100)
    except (ValueError, RuntimeError) as error:
        st.error(f"The backtest could not be calculated on this period ({error}). Try another period or other products.")
        st.stop()

if len(results) < 12:
    st.error("The period is too short (or the benchmark has no data on it). Choose a longer period.")
    st.stop()
# The client's portfolio first and clearly named, then the benchmark, then the other methods (for comparison)
mine = client_method(profile, tickers, c)
mine = mine if mine in results.columns else results.columns[0]
yours, bench = f"Your portfolio: {method_label(mine)}", f"Benchmark: {benchmark_name}"
names = {m: method_label(m) for m in results.columns} | {mine: yours, "Benchmark": bench}
order = [yours, bench] + [names[m] for m in results.columns if m not in (mine, "Benchmark")]
results, turnover, costs = (x.rename(columns=names)[order] for x in (results, turnover, costs))
weights_history = {names[m]: w for m, w in weights_history.items()}

st.caption(f"From {results.index[0]:%B %Y} to {results.index[-1]:%B %Y}, in {currency}. "
           + ("Black-Litterman uses your current views at every date. " if get_views() else "")
           + f"Resampled MV uses {N_SIM} simulations per rebalancing (100 on the portfolio page).")

# ---------- Your portfolio against the benchmark ----------
benchmark_r = results[bench]
table = pd.DataFrame({name: performance(results[name], rf, benchmark_r, turnover[name], costs[name],
                                        0.0 if name == bench else MANAGEMENT_FEE) for name in results})
st.subheader("Your portfolio against the benchmark")
st.caption(method_label(mine) + " · " + benchmark_name + " · all numbers after transaction costs and fee")
col1, col2, col3, col4 = st.columns(4)
for col, (row, label, fmt) in zip([col1, col2, col3, col4], [
        ("Annual return after costs and fee", "Annual return", "{:.1%}"), ("Volatility", "Volatility", "{:.1%}"),
        ("Max drawdown", "Max drawdown", "{:.1%}"), ("Sharpe ratio", "Sharpe ratio", "{:.2f}")]):
    gap = table.loc[row, yours] - table.loc[row, bench]
    col.metric(label, fmt.format(table.loc[row, yours]), f"{fmt.replace(':', ':+').format(gap)} vs benchmark",
               delta_color="inverse" if row == "Volatility" else "normal")

# ---------- Growth of 100 ----------
growth = 100 * (1 + results).cumprod()
fig = highlight_lines(px.line(growth), yours, bench)
fig.update_layout(yaxis_title=currency, xaxis_title=None)
chart(fig, f"Value of 100 {currency} invested, after transaction costs and fee")
st.caption("Blue: your portfolio. Dashed: the benchmark. Grey: the other methods, only for comparison.")
explain(profile, "What 100 invested at the start would have become, month by month, if we had used the robo-advisor "
                 "at that time: at each rebalancing it only knew the past, never the future (\"out of sample\"). "
                 "Falls in the line are losses that you would have lived through. Transaction costs and our fee are "
                 "already taken out.")

# ---------- Where does the return go? Before costs -> after costs and fee ----------
st.subheader("From the return of the markets to your return")
mine_row = table[yours]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Before costs and fee", f"{mine_row['Annual return before costs and fee']:.2%}",
            help="What the portfolio earned per year if trading were free and without our fee")
col2.metric("Transaction costs", f"−{mine_row['Transaction costs per year']:.2%}",
            help=f"{cost_bps} bp × the amount traded, on average per year (the initial purchase included)")
col3.metric("Our management fee", f"−{MANAGEMENT_FEE:.2%}", help="Taken every month: 1/12 of the yearly fee")
col4.metric("After costs and fee", f"{mine_row['Annual return after costs and fee']:.2%}",
            help="What you would have earned per year: the number used everywhere else on this page")
st.caption(f"Your portfolio traded on average {mine_row['Turnover per year']:.0%} of its value per year after the "
           f"initial purchase, so with a cost of {cost_bps} bp it paid about "
           f"{mine_row['Transaction costs per year']:.2%} per year in transaction costs. Costs and fee compound "
           "over time, so the four numbers do not add up exactly.")
explain(profile, "**When are the costs taken?** Transaction costs are taken in the month when the portfolio trades: "
                 "at the start (buying the whole portfolio) and at each rebalancing, equal to the cost × the amount "
                 f"bought and sold. Our fee of {MANAGEMENT_FEE:.2%} per year is taken every month "
                 f"({MANAGEMENT_FEE / 12:.3%}). The benchmark pays transaction costs when it rebalances, but not our "
                 "fee: you would not pay it if you bought the benchmark yourself.\n\n"
                 "**Why can a higher cost give a higher return?** With cost-aware optimisation the cost also changes "
                 "the decisions: a higher cost makes the optimiser trade less, so it holds a slightly different "
                 "portfolio. Its return before costs changes too, up or down. To see only the effect of paying the "
                 "costs, turn cost-aware optimisation off.")

# ---------- Performance table ----------
st.subheader("All portfolios")
st.dataframe(table.rename(columns={yours: "Your portfolio", bench: "Benchmark"}).style.format("{:.1%}").format(
    "{:.2f}", subset=pd.IndexSlice[["Sharpe ratio", "Beta vs benchmark"], :]).format(
    "{:.2%}", subset=pd.IndexSlice[["Transaction costs per year", "Management fee per year"], :])
    .set_properties(subset=["Your portfolio"], **HIGHLIGHT), height="content")
explain(profile, "Every number except the first one is after transaction costs and fee. **Annual return**: average "
                 "growth per year. **Max drawdown**: the worst fall from a previous peak (the worst moment to have "
                 "invested). **Turnover per year**: how much was traded per year after the initial purchase; more "
                 "trading means more transaction costs.")

# ---------- Transaction costs ----------
st.subheader("Trading over time")
st.write("Turnover at each rebalancing (the initial purchase is not shown). With cost-aware optimisation or a "
         "turnover limit, the portfolios trade less, so they pay less costs.")
fig = px.bar(turnover.iloc[1:], barmode="group", color_discrete_sequence=[YOURS, BENCHMARK_COLOR] + [OTHERS] * 6)
fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Turnover", xaxis_title=None)
chart(fig, "Turnover at each rebalancing")
explain(profile, "Each bar is the share of the portfolio traded at one rebalancing (100% = everything sold and "
                 f"bought again). Each bar costs its height × {cost_bps} bp: lower bars are cheaper for you.")

# ---------- Weights over time ----------
st.subheader("How did the weights change over time?")
method = st.selectbox("Portfolio", [m for m in order if m != bench], key="bt_weights_of")
area = px.area(weights_history[method])
area.update_layout(yaxis_tickformat=".0%", yaxis_title="Weight", xaxis_title=None)
chart(area, "Weights at each rebalancing")
explain(profile, "How the portfolio changed over time: each colour is one product. Stable colours mean a stable "
                 "portfolio, which is cheaper to manage and easier to follow.")

nav_buttons("app_pages/plan.py", "app_pages/stress_test.py", "Stress test")
