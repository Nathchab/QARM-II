"""
Robo-advisor web app. Run it with:  python -m streamlit run app.py
This file only creates the menu and the sidebar; each page is in the folder app_pages/.
"""
import streamlit as st

from src.data import download_prices
from src.style import apply_style

st.set_page_config(page_title="Robo-advisor", page_icon=":material/donut_small:", layout="wide")
apply_style()

pages = st.navigation({
    "": [st.Page("app_pages/welcome.py", title="Welcome", icon=":material/home:", default=True)],
    "Your portfolio": [
        st.Page("app_pages/profile.py", title="1. Profile", icon=":material/person:"),
        st.Page("app_pages/choices.py", title="2. Products and limits", icon=":material/tune:"),
        st.Page("app_pages/portfolio.py", title="3. Portfolio", icon=":material/donut_small:"),
        st.Page("app_pages/plan.py", title="4. Plan", icon=":material/flag:"),
        st.Page("app_pages/backtest.py", title="5. Backtest", icon=":material/show_chart:"),
        st.Page("app_pages/stress_test.py", title="6. Stress test", icon=":material/bolt:"),
    ],
    "Reference": [st.Page("app_pages/how_it_works.py", title="How it works", icon=":material/menu_book:")],
})

# ---------- Sidebar: currency and data ----------
# The currency is kept in st.session_state["currency"] (not in the widget itself): Streamlit can forget the value
# of a widget when the page changes, and the browser would then show another currency than the one we use.
CURRENCIES = ["USD", "CHF"]
st.session_state["currency"] = st.sidebar.radio(
    "Portfolio currency", CURRENCIES, index=CURRENCIES.index(st.session_state.get("currency", "USD")),
    horizontal=True, help="The bonds and the cash of your portfolio are in this currency. Prices are converted "
                          "with the USD/CHF rate; the riskless rate in CHF is the SNB policy rate.")
try:
    last_date = download_prices().index[-1]
except Exception:
    st.error("We could not download the prices from Yahoo Finance. Please try again in a minute.")
    if st.button("Try again", icon=":material/refresh:"):
        st.cache_data.clear()
        st.rerun()
    st.stop()
st.sidebar.caption(f"Prices until {last_date:%d %B %Y} · Yahoo Finance, refreshed daily")
if st.sidebar.button("Update prices", icon=":material/refresh:", width="stretch"):
    st.cache_data.clear()  # forget the prices in memory -> new download from Yahoo Finance
    st.rerun()

# ---------- Sidebar: new client ----------
if st.sidebar.button("New client", icon=":material/restart_alt:", width="stretch",
                     help="Forget all the answers and choices of this session."):
    keep = {"client_number": st.session_state.get("client_number", 0) + 1,  # new keys for the questions: empty again
            "currency": st.session_state["currency"]}                       # the currency is a setting of the app
    st.session_state.clear()  # forget the profile, the products, the views, the backtest settings...
    st.session_state.update(keep)
    st.switch_page("app_pages/welcome.py")

pages.run()
