"""Page 5: time machine. Replay past crises with the client's portfolio."""
import pandas as pd
import plotly.express as px
import streamlit as st

from src.app_helpers import (get_benchmark, get_choices, get_currency, get_portfolio_context,
                             get_profile, show_allocation_plan, show_portfolio_assessment, show_profile_notice)
from src.portfolios import METHODS
from src.time_machine import CRISES, crisis_summary, replay
from src.questionnaire_definitions import QUESTIONS

profile = get_profile()
tickers, c = get_choices(allow_infeasible=True, profile=profile)
currency = get_currency()

st.title("⏳ Time machine")
st.write("What if you had bought your portfolio just before a crisis? "
         "We buy it on the day the market was at its peak and hold it.")

plan, assessments, mu, cov, returns, rf, prices = get_portfolio_context(profile, currency, tickers, c)
show_allocation_plan(plan)
show_profile_notice(profile, plan, assessments)
if not plan.feasible:
    st.stop()
tickers, c = plan.tickers, plan.constraints
benchmark_name, benchmark = get_benchmark(tickers, c)
portfolios = {name: result.weights for name, result in assessments.items() if result.weights is not None}
if not portfolios:
    for result in assessments.values():
        show_portfolio_assessment(result)
    st.stop()
portfolios["Benchmark"] = benchmark

col1, col2, col3 = st.columns(3)
crisis = col1.selectbox("Crisis", list(CRISES))
methods = [name for name in METHODS if name in portfolios]
selected = st.session_state.get("method", methods[0])
method = col2.selectbox("Your portfolio", methods, index=methods.index(selected) if selected in methods else 0)
show_portfolio_assessment(assessments[method])
st.caption("The status above describes today's calculated portfolio. The crisis replay illustrates historical losses; volatility is not a pass/fail criterion.")
if method == "Risk parity (ERC)":
    st.caption("Risk Parity does not use gamma; its risk is not controlled by gamma.")
amount = col3.number_input(f"Amount invested ({currency})", min_value=1000, value=100_000, step=10_000)

start, end, story = CRISES[crisis]
st.info(f"**{crisis}** ({pd.Timestamp(start):%b %Y} → {pd.Timestamp(end):%b %Y}): {story}")

# ---------- Value during the crisis and one year after ----------
show_until = pd.Timestamp(end) + pd.DateOffset(years=1)
paths = pd.DataFrame({name: amount * replay(prices, portfolios[name], start, show_until)
                      for name in [method, "Benchmark"]})
paths = paths.rename(columns={"Benchmark": f"Benchmark: {benchmark_name}"})
fig = px.line(paths, title=f"Your {amount:,.0f} {currency} during the crisis (and one year after)")
fig.add_vline(x=pd.Timestamp(end), line_dash="dash")
fig.update_layout(yaxis_title=currency, xaxis_title=None, legend_title=None)
st.plotly_chart(fig)

# ---------- Worst loss and recovery ----------
loss, months = crisis_summary(prices, portfolios[method], start, end)
col1, col2, col3 = st.columns(3)
col1.metric("Worst loss", f"{loss:.1%}")
col2.metric(f"Worst loss in {currency}", f"{loss * amount:,.0f}")
col3.metric("Time to get your money back", "not yet" if months is None else f"{months} months")

if profile.max_loss is None:
    response = profile.questionnaire_answers["QE2"]
    st.info("Your original QE2 willingness response: " + QUESTIONS["QE2"].options[response].text
            + ". This temporary-loss comfort answer is not a financial loss-bearing limit; no crisis pass/fail is inferred from it.")
elif loss < profile.max_loss:
    st.error(f"This portfolio lost {-loss:.1%}, more than the {-profile.max_loss:.0%} you told us you could accept. "
             "Would you have stayed invested? Maybe choose a more prudent profile or method.")
else:
    st.success(f"This portfolio lost {-loss:.1%}, within the {-profile.max_loss:.0%} you told us you could accept.")
st.caption("Historical replay and any legacy comfort-threshold comparison are descriptive. This is not an Alternative B "
           "financial loss-budget screen or validation of mandatory suitability requirements.")

missing = [t for t in portfolios[method].index if prices[t].loc[:start].isna().all()]
if missing:
    st.caption(f"{', '.join(missing)} did not exist yet: this part of the portfolio is kept as cash.")

# ---------- All crises, all portfolios ----------
st.subheader("All crises at a glance")
rows = {}
for name, w in portfolios.items():
    row = {}
    for crisis_name, (s, e, _) in CRISES.items():
        l, m = crisis_summary(prices, w, s, e)
        row[f"{crisis_name}: worst loss"] = f"{l:.1%}"
        row[f"{crisis_name}: months to recover"] = "not yet" if m is None else str(m)
    rows[name] = row
st.dataframe(pd.DataFrame(rows).T)
