"""
Robo-advisor web app. Run it with:  python -m streamlit run app.py
This file only creates the menu and the sidebar; each page is in the folder app_pages/.
"""
import streamlit as st

from src.data import download_prices

st.set_page_config(page_title="Robo-advisor", page_icon="📈", layout="wide")

pages = st.navigation([
    st.Page("app_pages/profile.py", title="1. Your profile", icon="🧑", default=True),
    st.Page("app_pages/choices.py", title="2. Your products & constraints", icon="🧩"),
    st.Page("app_pages/portfolio.py", title="3. Your portfolio", icon="📊"),
    st.Page("app_pages/backtest.py", title="4. Backtest", icon="📈"),
    st.Page("app_pages/time_machine.py", title="5. Time machine", icon="⏳"),
    st.Page("app_pages/how_it_works.py", title="How it works", icon="📚"),
])

# ---------- Sidebar: currency and data ----------
st.sidebar.radio("Currency", ["USD", "CHF"], key="currency", horizontal=True,
                 help="CHF: prices converted with the USD/CHF rate, risk-free rate = SNB policy rate.")
try:
    last_date = download_prices().index[-1]
except Exception:
    st.error("We could not download the prices from Yahoo Finance. Check the internet connection and try again.")
    if st.button("Try again"):
        st.rerun()
    st.stop()
st.sidebar.caption(f"Prices until {last_date:%d %B %Y} (Yahoo Finance, updated every day)")
if st.sidebar.button("🔄 Update data now"):
    st.cache_data.clear()  # forget the prices in memory -> new download from Yahoo Finance
    st.rerun()

# ---------- Sidebar: new client ----------
if st.sidebar.button("🧹 Start over (new client)", help="Forget all the answers and choices of this session."):
    client_number = st.session_state.get("client_number", 0) + 1
    st.session_state.clear()  # forget the profile, the products, the views, the backtest settings...
    st.session_state["client_number"] = client_number  # new keys for the questions, so they come back empty
    st.switch_page("app_pages/profile.py")

pages.run()
