"""Page 4: the client's plan – wealth projection, savings goal, order list and downloadable report."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.app_helpers import (client_method, explain, get_choices, get_currency, get_data, get_profile,
                             get_valid_portfolios, method_label, next_button)
from src.optimizer import portfolio_stats
from src.planning import HORIZON_YEARS, MANAGEMENT_FEE, order_list, simulate_wealth, worst_drawdowns
from src.portfolios import METHODS
from src.report import client_report

profile = get_profile()
tickers, c = get_choices()
currency = get_currency()
returns, rf, prices = get_data(currency)
portfolios, gamma, forecast, cov = get_valid_portfolios(profile, tickers, c)

st.title("4. Your plan")
st.write("What could your portfolio become, and what do you buy today?")

# ---------- Inputs ----------
methods = [m for m in METHODS if m in portfolios]
saved = client_method(profile, tickers, c)
col1, col2, col3 = st.columns(3)
method = col1.selectbox("Portfolio", methods, index=methods.index(saved), format_func=method_label)
initial = col2.number_input(f"Initial amount ({currency})", min_value=1_000, value=100_000, step=10_000)
monthly = col3.number_input(f"Monthly contribution ({currency})", min_value=0, value=500, step=100)
years = col1.slider("Horizon (years)", 1, 40, HORIZON_YEARS[profile.answers["QD1"]],
                    help="Default: your answer to QD1 (when you need the money).")
goal = col2.number_input(f"Your goal ({currency})", min_value=1_000, value=200_000, step=10_000)

# ---------- Projection ----------
# Expected returns: our forecast (market equilibrium, + your views if you entered some) for every method. It is the
# most prudent estimate: we do not project the returns of the last 5 years into the future.
weights = portfolios[method]
exp_return, vol = portfolio_stats(weights, forecast, cov)
wealth, growth = simulate_wealth(exp_return - MANAGEMENT_FEE, vol, years, initial, monthly)

paid_in = initial + monthly * 12 * years
final = wealth[:, -1]
loss_limit = -profile.qe2_loss if profile.qe2_loss is not None else 0.30  # QE2 ("more than 30%" -> 30%)
plan = {"initial": initial, "monthly": monthly, "years": years, "paid_in": paid_in, "goal": goal,
        "p5": np.percentile(final, 5), "p50": np.percentile(final, 50), "p95": np.percentile(final, 95),
        "prob_goal": (final >= goal).mean(), "loss_limit": loss_limit,
        "prob_loss": (worst_drawdowns(growth) < -loss_limit).mean()}

st.subheader(f"Your wealth in {years} years")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total paid in", f"{paid_in:,.0f}")
col2.metric("Pessimistic (5%)", f"{plan['p5']:,.0f}")
col3.metric("Median", f"{plan['p50']:,.0f}")
col4.metric("Optimistic (95%)", f"{plan['p95']:,.0f}")

months = np.arange(wealth.shape[1]) / 12
band = np.percentile(wealth, [5, 50, 95], axis=0)
fig = go.Figure()
fig.add_scatter(x=months, y=band[2], mode="lines", line=dict(width=0), showlegend=False)
fig.add_scatter(x=months, y=band[0], mode="lines", line=dict(width=0), fill="tonexty",
                name="90% of the simulated futures")
fig.add_scatter(x=months, y=band[1], mode="lines", name="Median")
fig.add_scatter(x=months, y=initial + monthly * 12 * months, mode="lines", line=dict(dash="dot"), name="Paid in")
fig.add_hline(y=goal, line_dash="dash", annotation_text="Your goal")
fig.update_layout(xaxis_title="Years", yaxis_title=currency, title="5,000 simulated futures")
st.plotly_chart(fig)
explain(profile, "We simulated 5,000 possible futures of your portfolio. The middle line is the typical result "
                 "(half of the futures do better, half worse). The coloured band contains 90% of the futures: the "
                 "bottom of the band is a bad but possible outcome, the top a very good one. The dotted line is what "
                 "you paid in: above it, you made money.")

col1, col2 = st.columns(2)
col1.metric("Probability of reaching your goal", f"{plan['prob_goal']:.0%}")
col2.metric(f"Probability of a fall of more than {loss_limit:.0%} at some point", f"{plan['prob_loss']:.0%}",
            help="From a previous peak of the portfolio. The limit is your answer to QE2.")
if plan["prob_goal"] < 0.5:
    st.info("Your goal is reached in less than half of the futures: save more each month, wait longer, "
            "or lower the goal.")
st.caption(f"Expected return {exp_return:.1%} per year (our forecast: market equilibrium), minus our fee of "
           f"{MANAGEMENT_FEE:.2%}, volatility {vol:.1%}. Monthly returns are drawn from a normal distribution and "
           "are independent from one month to the next. Amounts are not adjusted for inflation.")

# ---------- Order list ----------
st.subheader(f"Your orders for {initial:,.0f} {currency}")
orders, cash = order_list(weights, prices.iloc[-1], initial)
st.dataframe(orders.style.format({"Target weight": "{:.1%}", "Last price": "{:,.2f}", "Quantity": "{:,.4g}",
                                  "Amount": "{:,.0f}"}), hide_index=True)
st.caption(f"Prices of {prices.index[-1]:%d %B %Y}. ETFs are bought in whole shares, crypto in fractions; "
           f"{cash:,.0f} {currency} stay in cash. Products traded in another currency (the .SW products trade in "
           "CHF, the others in USD) are converted at the last USD/CHF rate.")

# ---------- Report ----------
report = client_report(profile, c, method_label(method), weights, exp_return, vol, currency, plan, orders, cash)
st.download_button("📄 Download my report (PDF)", report, file_name="robo_advisor_report.pdf", mime="application/pdf")
next_button("app_pages/backtest.py", "Next: how would it have worked in the past?")
