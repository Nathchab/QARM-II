"""Page 4: the client's plan – wealth projection, savings goal, order list and downloadable report."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.app_helpers import (client_method, get_choices, get_currency, get_data, get_profile, get_valid_portfolios,
                             method_label, recommended)
from src.optimizer import portfolio_stats
from src.planning import (FREQUENCIES, HORIZON_YEARS, MANAGEMENT_FEE, TRADING_COST, chance, order_list, paid_in,
                          quantity_text, simulate_wealth, worst_drawdowns)
from src.portfolios import METHODS
from src.report import client_report
from src.style import ACCENT, BENCHMARK_COLOR, chart, explain, nav_buttons, page_header

profile = get_profile()
tickers, c = get_choices()
currency = get_currency()
returns, rf, prices = get_data(currency)
portfolios, gamma, forecast, cov = get_valid_portfolios(profile, tickers, c)

page_header(4, "Your plan", "What could your portfolio become, and what do you buy today?")

# ---------- Inputs ----------
methods = [m for m in METHODS if m in portfolios]
saved = client_method(profile, tickers, c)
with st.container(border=True):
    col1, col2, col3 = st.columns(3)
inputs = st.session_state.get("plan_inputs", {})  # saved, so they stay when the client changes page
# Every widget has a key, so Streamlit keeps exactly what the client chose (see backtest.py)
rec = recommended(profile, tickers, c)
method = col1.selectbox("Portfolio", methods, index=methods.index(saved) if saved in methods else 0,
                        format_func=lambda m: method_label(m) + (" – recommended" if m == rec else ""),
                        key=f"plan_method_{rec}")
st.session_state["method"] = method if method != rec else None  # the same portfolio on every page
initial = col2.number_input(f"Initial amount ({currency})", min_value=1_000, value=inputs.get("initial", 100_000),
                            step=10_000, key="plan_initial")
contribution = col3.number_input(f"Contribution ({currency})", min_value=0, value=inputs.get("contribution", 500),
                                 step=100, help="Amount you add regularly (0 = no savings plan)",
                                 key="plan_contribution")
years = col1.slider("Horizon (years)", 1, 40, inputs.get("years", HORIZON_YEARS[profile.answers["QD1"]]),
                    help="Default: your answer to QD1 (when you need the money).", key="plan_years")
frequency = col3.selectbox("Contribution frequency", list(FREQUENCIES), key="plan_frequency",
                           index=list(FREQUENCIES).index(inputs.get("frequency", "Monthly")))
every, per = FREQUENCIES[frequency]
# Default goal: get back at least what you pay in (rounded up); a fixed 200,000 was out of reach for short horizons
default_goal = int(np.ceil(paid_in(initial, contribution, every, 12 * years) / 10_000) * 10_000)
goal = col2.number_input(f"Your goal ({currency})", min_value=1_000, value=inputs.get("goal", default_goal),
                         step=10_000, key="plan_goal", help="Default: what you pay in, rounded up (get your money back)")
# One transaction cost for the whole app: the backtest page uses (and can change) the same one
cost_bps = col1.slider("Transaction cost (bp of each purchase)", 0, 100,
                       st.session_state.get("cost_bps", round(TRADING_COST * 10_000)), key="plan_cost",
                       help="10 bp = 0.10% of every amount you invest. Typical for ETFs: 5 to 20 bp. The same cost is "
                            "used in the backtest.")
st.session_state["cost_bps"] = cost_bps
st.session_state["plan_inputs"] = {"initial": initial, "contribution": contribution, "frequency": frequency,
                                   "years": years, "goal": goal}

# ---------- Projection ----------
# Expected returns: our forecast (market equilibrium, + your views if you entered some) for every method. It is the
# most prudent estimate: we do not project the returns of the last 5 years into the future.
weights = portfolios[method]
exp_return, vol = portfolio_stats(weights, forecast, cov)
# Our fee is taken from the return every month; each purchase (the initial amount and every contribution) pays the
# transaction cost chosen above (default 10 bp). The ETF costs (TER) are already in the ETF prices.
cost = cost_bps / 10_000
wealth, growth = simulate_wealth(exp_return - MANAGEMENT_FEE, vol, years, initial, contribution, every, cost)
# The same futures without our fee, and without fee nor costs: what they cost the client (same random draws)
no_fee = simulate_wealth(exp_return, vol, years, initial, contribution, every, cost)[0][:, -1]
free = simulate_wealth(exp_return, vol, years, initial, contribution, every)[0][:, -1]

total_paid = paid_in(initial, contribution, every, 12 * years)
final = wealth[:, -1]
loss_limit = -profile.qe2_loss if profile.qe2_loss is not None else 0.30  # QE2 ("more than 30%" -> 30%)
plan = {"initial": initial, "contribution": contribution, "per": per, "years": years, "paid_in": total_paid,
        "goal": goal, "p5": np.percentile(final, 5), "p50": np.percentile(final, 50), "p95": np.percentile(final, 95),
        "prob_goal": (final >= goal).mean(), "loss_limit": loss_limit, "cost_bps": cost * 10_000,
        "fee_cost": np.median(no_fee - final), "trading_cost": np.median(free - no_fee),
        "prob_loss": (worst_drawdowns(growth) < -loss_limit).mean()}

st.subheader(f"Your wealth in {years} year{'s' if years > 1 else ''}")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total paid in", f"{total_paid:,.0f} {currency}")
col2.metric("Pessimistic (5%)", f"{plan['p5']:,.0f} {currency}", help="5% of the futures end lower")
col3.metric("Median", f"{plan['p50']:,.0f} {currency}", help="Half of the futures end higher, half lower")
col4.metric("Optimistic (95%)", f"{plan['p95']:,.0f} {currency}", help="5% of the futures end higher")

months = np.arange(wealth.shape[1]) / 12
band = np.percentile(wealth, [5, 50, 95], axis=0)
fig = go.Figure()
fig.add_scatter(x=months, y=band[2], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
fig.add_scatter(x=months, y=band[0], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(42,120,214,0.15)",
                name="90% of the simulated futures")
fig.add_scatter(x=months, y=band[1], mode="lines", name="Median", line=dict(color=ACCENT, width=3))
fig.add_scatter(x=months, y=paid_in(initial, contribution, every, np.arange(wealth.shape[1])), mode="lines",
                name="Paid in", line=dict(color=BENCHMARK_COLOR, dash="dot", width=2, shape="hv"))
fig.add_hline(y=goal, line_dash="dash", line_color="#5b6472", annotation_text="Your goal")
fig.update_layout(xaxis_title="Years", yaxis_title=currency, yaxis_tickformat=",.0f")
chart(fig, "5,000 simulated futures of your portfolio")
explain(profile, "We simulated 5,000 possible futures of your portfolio. The middle line is the typical result "
                 "(half of the futures do better, half worse). The coloured band contains 90% of the futures: the "
                 "bottom of the band is a bad but possible outcome, the top a very good one. The dotted line is what "
                 "you paid in: above it, you made money.")

col1, col2 = st.columns(2)
col1.metric("Probability of reaching your goal", chance(plan["prob_goal"]))
col2.metric(f"Probability of a fall of more than {loss_limit:.0%} at some point", chance(plan["prob_loss"]),
            help="From a previous peak of the portfolio. The limit is your answer to QE2.")
if plan["prob_goal"] < 0.5:
    st.caption("Your goal is reached in less than half of the futures: save more each month, wait longer, "
               "or lower the goal.")

# ---------- What fees and costs take over the horizon ----------
st.subheader(f"What fees and costs take over {years} year{'s' if years > 1 else ''}")
col1, col2, col3 = st.columns(3)
col1.metric(f"Our fee ({MANAGEMENT_FEE:.2%} per year)", f"−{plan['fee_cost']:,.0f} {currency}",
            help="Median difference between the final wealth with and without our fee. It grows with time, because "
                 "the fee is taken every month and the money it takes no longer earns a return.")
col2.metric(f"Transaction costs ({cost * 10_000:.0f} bp)", f"−{plan['trading_cost']:,.0f} {currency}",
            help="Paid on each purchase: the initial amount and every contribution. You can change the cost above.")
col3.metric("ETF costs (TER)", "in the prices",
            help="Each ETF takes its own yearly cost (TER) from its value every day, so its price is already after "
                 "it: the past returns and the backtest include it.")
st.caption(f"Already taken out of the futures above. Expected return {exp_return:.1%} per year (our forecast) "
           f"minus our fee of {MANAGEMENT_FEE:.2%}, volatility {vol:.1%}. The costs of rebalancing are not included: "
           "the portfolio trades only a few percent of its value per year, so they are very small (see the "
           "backtest). Monthly returns are drawn from a normal distribution and are independent from one month to "
           "the next. Amounts are not adjusted for inflation.")

# ---------- Order list ----------
st.subheader(f"Your orders for {initial:,.0f} {currency}")
# We buy with the amount net of the transaction costs, so that the cash left always pays them
orders, _ = order_list(weights, prices.iloc[-1], initial * (1 - cost))
cash = initial - orders["Amount"].sum() * (1 + cost)
st.dataframe(orders.style.format({"Target weight": "{:.1%}", "Last price": "{:,.2f}", "Quantity": quantity_text,
                                  "Amount": "{:,.0f}"}), hide_index=True)
if cash > 0.05 * initial:  # small amounts: some ETF shares cost more than their share of the money
    st.warning(f"{cash:,.0f} {currency} ({cash / initial:.0%} of your money) stay in cash, because some ETF shares cost "
               "more than their share of this amount. Invest a larger amount, or use a broker that sells fractions "
               "of shares.", icon=":material/info:")
st.caption(f"Prices of {prices.index[-1]:%d %B %Y}. ETFs are bought in whole shares, crypto in fractions. The "
           f"transaction costs of these orders are about {orders['Amount'].sum() * cost:,.0f} {currency}; "
           f"{cash:,.0f} {currency} stay in cash. Products traded in another currency (the .SW products trade in "
           "CHF, the others in USD) are converted at the last USD/CHF rate.")

# ---------- Report ----------
# The PDF is written only when the client clicks (not at every change of the inputs), and the page does not reload.
report = lambda: client_report(profile, c, method_label(method), weights, exp_return, vol, currency, plan, orders, cash)
st.download_button("Download my report (PDF)", report, file_name="robo_advisor_report.pdf", mime="application/pdf",
                   icon=":material/download:", on_click="ignore")
nav_buttons("app_pages/portfolio.py", "app_pages/backtest.py", "Backtest")
