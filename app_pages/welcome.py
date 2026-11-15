"""Welcome page: what the robo-advisor does and how to use it."""
import streamlit as st

from src.app_helpers import next_button
from src.planning import MANAGEMENT_FEE

st.title("Welcome to your robo-advisor")
st.subheader("A diversified ETF portfolio that fits you, in about 5 minutes")
st.write("You answer a short questionnaire. We measure **how much risk you are willing to take** and **how much "
         "risk you can afford**, then we build a portfolio of exchange-traded funds (ETFs) with the methods of modern "
         "portfolio theory, and we show you what it could become. Everything is explained: no black box.")

st.header("How it works")
STEPS = [
    ("🧑", "1. Your profile", "15 to 17 short questions (about 3 minutes): your attitude to risk, your horizon, your "
                             "financial situation, what and where you want to invest, and your experience."),
    ("🧩", "2. Products & limits", "We pre-select the ETFs (33 available, Swiss products in CHF included) and the "
                                  "limits that fit your profile. You can change the products and make the limits "
                                  "stricter."),
    ("📊", "3. Your portfolio", "The method that fits your profile (market equilibrium, minimum variance, risk "
                               "parity or Black-Litterman), compared with the other methods, so you can see why."),
    ("🎯", "4. Your plan", "What your money could become with a monthly savings plan, the chance of reaching "
                          "your goal, the orders to place and a PDF report to download."),
    ("📈", "5. Backtest", "How the portfolio would have done in the past, rebalanced with the information "
                         "available at the time, after transaction costs and our fee."),
    ("⏳", "6. Stress test", "What would have happened if you had invested just before the 2008, 2020 or 2022 "
                             "crisis, and whether you could have stayed invested."),
]
for row in (STEPS[:3], STEPS[3:]):
    for column, (icon, title, text) in zip(st.columns(3), row):
        with column.container(border=True):
            st.markdown(f"### {icon} {title}")
            st.write(text)

st.header("Our principles")
col1, col2 = st.columns(2)
col1.markdown("""
- **Your limits come first.** In a bad year (5% chance) the recommended portfolio should not lose more than
  what you told us you can tolerate (we warn you if no
  portfolio can promise that), and money you may need soon is kept in short-term Treasuries.
- **One prudent forecast.** We do not assume that the last 5 years will repeat: expected returns come from
  the market equilibrium (what the market as a whole expects).
""")
col2.markdown(f"""
- **Diversified and cheap.** Broad ETFs, at most 40% in one product and 15% in a narrow one (one sector or
  country). Our fee: {MANAGEMENT_FEE:.2%} per year.
- **Transparent.** Every number can be checked on the page *How it works*.
""")
st.info("Choose the currency of your portfolio (USD or CHF) in the menu on the left. "
        "You can start again from zero at any time with **Start over (new client)**.")

next_button("app_pages/profile.py", "Start: build my profile")
st.caption("Academic project (QARM II, HEC Lausanne). Prices from Yahoo Finance. This is not investment advice.")
