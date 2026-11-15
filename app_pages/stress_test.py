"""Page 6: stress test. Replay past crises with the client's portfolio."""
import pandas as pd
import plotly.express as px
import streamlit as st

from src.app_helpers import (client_method, explain, get_benchmark, get_choices, get_currency, get_data, get_profile,
                             get_valid_portfolios, highlight_lines, method_label, next_button)
from src.portfolios import METHODS
from src.stress_test import CRISES, crisis_summary, replay

profile = get_profile()
tickers, c = get_choices()
currency = get_currency()
benchmark_name, benchmark = get_benchmark(profile, tickers, c)

st.title("6. Stress test")
st.write("What if you had bought your portfolio just before a crisis? "
         "We buy it on the day the market was at its peak and hold it.")

returns, rf, prices = get_data(currency)
portfolios = dict(get_valid_portfolios(profile, tickers, c)[0])
portfolios["Benchmark"] = benchmark

col1, col2, col3 = st.columns(3)
crisis = col1.selectbox("Crisis", list(CRISES))
methods = [m for m in METHODS if m in portfolios]
selected = client_method(profile, tickers, c)
method = col2.selectbox("Your portfolio", methods, index=methods.index(selected) if selected in methods else 0,
                        format_func=method_label)
amount = col3.number_input(f"Amount invested ({currency})", min_value=1000, value=100_000, step=10_000)

start, end, story = CRISES[crisis]
st.info(f"**{crisis}** ({pd.Timestamp(start):%b %Y} → {pd.Timestamp(end):%b %Y}): {story}")

# ---------- Value during the crisis and one year after ----------
show_until = pd.Timestamp(end) + pd.DateOffset(years=1)
yours, bench = f"★ Your portfolio: {method_label(method)}", f"Benchmark: {benchmark_name}"
label = lambda name: yours if name == method else bench if name == "Benchmark" else method_label(name)
names = [method, "Benchmark"] + [m for m in portfolios if m not in (method, "Benchmark")]  # yours first
paths = pd.DataFrame({label(name): amount * replay(prices, portfolios[name], start, show_until) for name in names})
fig = px.line(paths, title=f"Your {amount:,.0f} {currency} during the crisis (and one year after)")
highlight_lines(fig, yours, bench)
fig.add_vline(x=pd.Timestamp(end), line_dash="dot")
fig.update_layout(yaxis_title=currency, xaxis_title=None, legend_title=None)
st.plotly_chart(fig)
st.caption("Orange: your portfolio. Dashed: the benchmark. Grey: the other methods, only for comparison. "
           "The dotted line is the end of the crisis.")
explain(profile, "What your money would have become if you had invested at the worst moment, just before the "
                 "crisis, and kept the portfolio. The question is not only how much it fell, but whether you would "
                 "have stayed invested: selling at the bottom turns a temporary loss into a real one.")

# ---------- Worst loss and recovery ----------
loss, months = crisis_summary(prices, portfolios[method], start, end)
bench_loss, bench_months = crisis_summary(prices, benchmark, start, end)
st.subheader(yours)
col1, col2, col3 = st.columns(3)
col1.metric("Worst loss", f"{loss:.1%}", f"{loss - bench_loss:+.1%} vs benchmark ({bench_loss:.1%})")
col2.metric(f"Worst loss in {currency}", f"{loss * amount:,.0f}")
col3.metric("Time to get your money back", "not yet" if months is None else f"{months} months")

if profile.max_loss is None:
    st.info(f"This portfolio lost {-loss:.1%}. You told us (QE2) you could tolerate a temporary loss of more than 30%.")
elif loss < profile.max_loss:
    st.error(f"This portfolio lost {-loss:.1%}, more than the {-profile.max_loss:.0%} you told us (QE2) you could "
             "tolerate without abandoning the strategy. Would you have stayed invested? Maybe choose a more "
             "prudent method or tighten your limits.")
else:
    st.success(f"This portfolio lost {-loss:.1%}, within the {-profile.max_loss:.0%} you told us (QE2) you could tolerate.")

missing = [t for t in portfolios[method].index if prices[t].loc[:start].isna().all()]
if missing:
    st.caption(f"{', '.join(missing)} did not exist yet: this part of the portfolio is kept as cash.")

# ---------- All crises, all portfolios ----------
st.subheader("All crises at a glance")
rows = {}
for name in names:
    row = {}
    for crisis_name, (s, e, _) in CRISES.items():
        l, m = crisis_summary(prices, portfolios[name], s, e)
        row[f"{crisis_name}: worst loss"] = f"{l:.1%}"
        row[f"{crisis_name}: months to recover"] = "not yet" if m is None else str(m)
    rows[label(name)] = row
st.dataframe(pd.DataFrame(rows).T.style.set_properties(subset=pd.IndexSlice[[yours], :],
                                                       **{"background-color": "#fde4dc", "font-weight": "bold"}))
explain(profile, "**Worst loss**: the largest fall during the crisis. **Months to recover**: how long it took to get "
                 "back to the starting value ('not yet' = still below).")

next_button("app_pages/how_it_works.py", "Next: how does it work?")
