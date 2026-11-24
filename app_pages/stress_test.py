"""Page 6: stress test. Replay past crises with the client's portfolio."""
import pandas as pd
import plotly.express as px
import streamlit as st

from src.app_helpers import (client_method, get_benchmark, get_choices, get_currency, get_data, get_profile,
                             get_valid_portfolios, method_label, recommended)
from src.portfolios import METHODS
from src.stress_test import CRISES, crisis_summary, replay, too_young
from src.style import HIGHLIGHT, chart, explain, highlight_lines, nav_buttons, page_header

profile = get_profile()
tickers, c = get_choices()
currency = get_currency()
benchmark_name, benchmark = get_benchmark(profile, tickers, c)

page_header(6, "Stress test", "What if you had bought your portfolio just before a crisis? We buy it on the day the "
            "market was at its peak and hold it.")

returns, rf, prices = get_data(currency)
portfolios = dict(get_valid_portfolios(profile, tickers, c)[0])
portfolios["Benchmark"] = benchmark

with st.container(border=True):
    col1, col2, col3 = st.columns(3)
inputs = st.session_state.get("stress_inputs", {})  # saved, so they stay when the client changes page
crisis = col1.selectbox("Crisis", list(CRISES), index=list(CRISES).index(inputs.get("crisis", list(CRISES)[0])),
                        key="stress_crisis")
methods = [m for m in METHODS if m in portfolios]
selected, rec = client_method(profile, tickers, c), recommended(profile, tickers, c)
method = col2.selectbox("Your portfolio", methods, index=methods.index(selected) if selected in methods else 0,
                        format_func=lambda m: method_label(m) + (" – recommended" if m == rec else ""),
                        key=f"stress_method_{rec}")
st.session_state["method"] = method if method != rec else None  # the same portfolio on every page
amount = col3.number_input(f"Amount invested ({currency})", min_value=1000, value=inputs.get("amount", 100_000),
                           step=10_000, key="stress_amount")
st.session_state["stress_inputs"] = {"crisis": crisis, "amount": amount}

start, end, story = CRISES[crisis]
st.markdown(f"**{crisis}** ({pd.Timestamp(start):%b %Y} to {pd.Timestamp(end):%b %Y}). {story}")

# ---------- Value during the crisis and one year after ----------
show_until = pd.Timestamp(end) + pd.DateOffset(years=1)
yours, bench = f"Your portfolio: {method_label(method)}", f"Benchmark: {benchmark_name}"
label = lambda name: yours if name == method else bench if name == "Benchmark" else method_label(name)
names = [method, "Benchmark"] + [m for m in portfolios if m not in (method, "Benchmark")]  # yours first
paths = pd.DataFrame({label(name): amount * replay(prices, portfolios[name], start, show_until) for name in names})
fig = highlight_lines(px.line(paths), yours, bench)
fig.add_vline(x=pd.Timestamp(end), line_dash="dot", line_color="#5b6472")
fig.update_layout(yaxis_title=currency, xaxis_title=None, yaxis_tickformat=",.0f")
chart(fig, f"Your {amount:,.0f} {currency} during the crisis and one year after")
st.caption("Blue: your portfolio. Dashed: the benchmark. Grey: the other methods, only for comparison. "
           "The dotted line is the end of the crisis.")
explain(profile, "What your money would have become if you had invested at the worst moment, just before the "
                 "crisis, and kept the portfolio. The question is not only how much it fell, but whether you would "
                 "have stayed invested: selling at the bottom turns a temporary loss into a real one.")

# ---------- Worst loss and recovery ----------
loss, months = crisis_summary(prices, portfolios[method], start, end)
bench_loss = crisis_summary(prices, benchmark, start, end)[0]
st.subheader("Your portfolio in this crisis")
col1, col2, col3 = st.columns(3)
col1.metric("Worst loss", f"{loss:.1%}", f"{loss - bench_loss:+.1%} vs benchmark ({bench_loss:.1%})")
col2.metric(f"Worst loss in {currency}", f"{loss * amount:,.0f}")
col3.metric("Time to get your money back", "no loss" if loss >= -0.0005 else "not yet" if months is None
            else f"{months} months")

# Products that did not exist yet: an older fund on the same market replaces them, otherwise they are kept as cash
young = too_young(prices, portfolios[method], start)
replaced = [f"{t} by {proxy}" for t, proxy in young.items() if proxy]
cash = sum(portfolios[method][t] for t, proxy in young.items() if not proxy)
if replaced:
    st.caption("Did not exist yet, replaced by an older fund on the same market: " + ", ".join(replaced) + ".")
if cash > 0.25:  # the replay says little about this portfolio: no verdict
    st.warning(f"{cash:.0%} of this portfolio did not exist yet at the start of this crisis and is counted as cash: "
               "the real loss would have been different. Look at a more recent crisis.", icon=":material/warning:")
elif loss >= -0.0005:
    st.success("This portfolio did not lose money in this crisis.", icon=":material/check_circle:")
elif profile.qe2_loss is None:
    st.write(f"This portfolio lost {-loss:.1%}. You told us (QE2) you could tolerate a temporary loss of more than 30%.")
elif loss < profile.qe2_loss:
    st.warning(f"This portfolio lost {-loss:.1%}, more than the {-profile.qe2_loss:.0%} you told us (QE2) you could "
               "tolerate without abandoning the strategy. "
               + ("Our loss limit covers a bad year with a 5% chance; a crisis like this one is rarer and deeper. "
                  "Would you have stayed invested? If not, make your limits stricter on step 2."
                  if method == rec else
                  "Would you have stayed invested? Our recommended portfolio is built to respect your loss limit."),
               icon=":material/warning:")
else:
    st.success(f"This portfolio lost {-loss:.1%}, within the {-profile.qe2_loss:.0%} you told us (QE2) you could "
               "tolerate.", icon=":material/check_circle:")
if 0 < cash <= 0.25:
    st.caption(f"{cash:.0%} of this portfolio did not exist yet and is counted as cash.")

# ---------- All crises, all portfolios ----------
st.subheader("All crises at a glance")
rows = {}
for name in names:
    row = {}
    for crisis_name, (s, e, _) in CRISES.items():
        l, m = crisis_summary(prices, portfolios[name], s, e)
        row[crisis_name] = ("no loss" if l >= -0.0005 else f"{l:.1%}, " + ("not recovered yet" if m is None
                                                                     else f"back in {m} months"))
    rows[label(name)] = row
st.dataframe(pd.DataFrame(rows).T.style.set_properties(subset=pd.IndexSlice[[yours], :], **HIGHLIGHT), height="content")
explain(profile, "For each crisis: the **worst loss** (the largest fall from the day of purchase) and how many months "
                 "it took to get back to the amount invested.")

nav_buttons("app_pages/backtest.py", "app_pages/how_it_works.py", "How it works")
