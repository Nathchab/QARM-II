"""
Robo-advisor web app. Run it with:  python -m streamlit run app.py
This file only creates the menu and the sidebar; each page is in the folder app_pages/.
"""
import streamlit as st

from src.data import download_prices

st.set_page_config(page_title="Robo-advisor", page_icon="📈", layout="wide")
# Bigger "Next" buttons (the primary buttons of every page)
st.markdown("<style>button[kind='primary'] {min-height: 3.5rem;} button[kind='primary'] p {font-size: 1.25rem; "
            "font-weight: 600;}</style>", unsafe_allow_html=True)

pages = st.navigation([
    st.Page("app_pages/welcome.py", title="Welcome", icon="👋", default=True),
    st.Page("app_pages/profile.py", title="1. Your profile", icon="🧑"),
    st.Page("app_pages/choices.py", title="2. Your products & constraints", icon="🧩"),
    st.Page("app_pages/portfolio.py", title="3. Your portfolio", icon="📊"),
    st.Page("app_pages/plan.py", title="4. Your plan", icon="🎯"),
    st.Page("app_pages/backtest.py", title="5. Backtest", icon="📈"),
    st.Page("app_pages/stress_test.py", title="6. Stress test", icon="⏳"),
    st.Page("app_pages/how_it_works.py", title="How it works", icon="📚"),
])

# ---------- Sidebar: currency and data ----------
# The currency is kept in st.session_state["currency"] (not in the widget itself): Streamlit can forget the value
# of a widget when the page changes, and the browser would then show another currency than the one we use.
CURRENCIES = ["USD", "CHF"]
st.session_state["currency"] = st.sidebar.radio(
    "Currency", CURRENCIES, index=CURRENCIES.index(st.session_state.get("currency", "USD")), horizontal=True,
    help="The bonds and the cash of your portfolio are in this currency. Prices are converted with the "
         "USD/CHF rate; the riskless rate in CHF is the SNB policy rate.")
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
    keep = {"client_number": st.session_state.get("client_number", 0) + 1,  # new keys for the questions: empty again
            "currency": st.session_state["currency"]}                       # the currency is a setting of the app
    st.session_state.clear()  # forget the profile, the products, the views, the backtest settings...
    st.session_state.update(keep)
    st.switch_page("app_pages/welcome.py")

pages.run()
