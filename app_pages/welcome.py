"""Welcome page: what the robo-advisor does and how to use it."""
import streamlit as st

from src.data import ASSETS
from src.planning import MANAGEMENT_FEE

st.caption("ROBO-ADVISOR")
st.title("A diversified ETF portfolio that fits your profile")
st.markdown("We measure **how much risk you are willing to take** and **how much risk you can afford**, then build "
            "a portfolio of exchange-traded funds (ETFs) with the methods of modern portfolio theory. Every number "
            "is explained.")
col, _ = st.columns([1, 2])
if col.button("Start the questionnaire", type="primary", icon=":material/arrow_forward:", icon_position="right",
              width="stretch"):
    st.switch_page("app_pages/profile.py")
st.caption("About 5 minutes · 15 to 17 questions · portfolio in USD or CHF (menu on the left)")

st.divider()
st.subheader("How it works")
STEPS = [
    ("Profile", "Your attitude to risk, horizon, financial situation, the markets you want and your experience."),
    ("Products and limits", f"We pre-select ETFs (out of {len(ASSETS)}, Swiss products in CHF included) and limits "
                            "for your profile. You can change the products and make the limits stricter."),
    ("Portfolio", "The method that fits your profile, compared with the other methods, so you can see why."),
    ("Plan", "What your money could become with a regular savings plan, the orders to place and a PDF report."),
    ("Backtest", "How the portfolio would have done in the past, after transaction costs and our fee."),
    ("Stress test", "What would have happened if you had invested just before the 2008, 2020 or 2022 crisis."),
]
cells = st.columns(3, gap="large") + st.columns(3, gap="large")  # two rows of three
for number, (cell, (title, text)) in enumerate(zip(cells, STEPS), start=1):
    cell.markdown(f"**{number}. {title}**  \n{text}")

st.divider()
st.subheader("Our principles")
PRINCIPLES = [
    ("Your limits come first", "In a bad year (5% chance) the portfolio should not lose more than you can tolerate "
                               "and afford. Money you may need soon stays in short-term government bonds."),
    ("One prudent forecast", "We do not assume that the last 5 years will repeat: expected returns come from the "
                             "market equilibrium, what the market as a whole expects."),
    ("Diversified and cheap", f"Broad ETFs, at most 40% in one product and 15% in a narrow one (one sector or "
                              f"country). Our fee: {MANAGEMENT_FEE:.2%} per year."),
    ("Transparent", "Every method and number can be checked on the page How it works."),
]
cells = st.columns(2, gap="large") + st.columns(2, gap="large")
for cell, (title, text) in zip(cells, PRINCIPLES):
    cell.markdown(f"**{title}**  \n{text}")

st.divider()
st.caption("Academic project, QARM II, HEC Lausanne. Prices from Yahoo Finance. This is not investment advice.")
