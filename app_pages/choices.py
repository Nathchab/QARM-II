"""Page 2: the client picks the products and can tighten the constraints of his profile."""
import streamlit as st

from src.app_helpers import (get_currency, get_profile, recommendation_summary, reset_choices, show_profile_messages,
                             tighten)
from src.data import ASSETS, SHORT_TERM, available_assets, first_date
from src.optimizer import check_constraints, lacks_home_equity
from src.portfolios import WINDOW
from src.style import nav_buttons, page_header

profile = get_profile()
rec_assets = profile.recommended_assets
rec_c = profile.recommended_constraints

page_header(2, "Products and limits",
            "We pre-selected **our recommendation** for your profile. You can choose other products, and you can make "
            "the limits **stricter** than your profile, but not looser: they come from your financial situation, "
            "horizon and liquidity needs.")

if st.button("Back to our recommendation", icon=":material/undo:"):
    reset_choices()
    st.rerun()

version = st.session_state.get("choices_version", 0)  # changes at every reset: the widgets start again
saved_assets = st.session_state.get("assets", rec_assets)
saved_c = tighten(rec_c, st.session_state.get("constraints", {}))

# ---------- Products ----------
st.subheader("Products")
label = lambda t: f"{t} – {ASSETS[t][0]} (data since {first_date(t):%Y})"
tickers = []
for asset_class in ["Equity", "Bond", "Other", "Crypto"]:
    options = [t for t in available_assets() if ASSETS[t][1] == asset_class]
    if asset_class == "Crypto" and rec_c["max_crypto"] == 0:
        st.caption("Crypto: not allowed by your profile (maximum 0%).")
        continue
    tickers += st.multiselect(asset_class, options, default=[t for t in saved_assets if t in options],
                              format_func=label, key=f"pick_{asset_class}_{version}")

added = [t for t in tickers if t not in rec_assets]
removed = [t for t in rec_assets if t not in tickers]
if added or removed:
    st.caption(f"Compared with our recommendation: added {', '.join(added) or 'nothing'}, "
               f"removed {', '.join(removed) or 'nothing'}.")
if tickers:
    youngest = max(tickers, key=first_date)
    st.caption(f"The youngest product is {youngest} ({first_date(youngest):%B %Y}). We need {WINDOW} months "
               f"of history to estimate the inputs, so the backtest can start 5 years later.")


# ---------- Constraints ----------
def slider(column, label, key, low, high, value, step):
    """Slider in % between low and high (a fixed value when the profile leaves no choice)."""
    if low >= high:
        column.write(f"{label}: **{low:.0%}** (set by your profile)")
        return low
    pct = column.slider(label + " (%)", int(round(low * 100)), int(round(high * 100)),
                        int(round(value * 100)), step, key=f"pick_{key}_{version}")
    return pct / 100


st.subheader("Limits")
col1, col2 = st.columns(2)
c = {
    "max_equity": slider(col1, "Maximum in risky assets", "max_equity", 0, rec_c["max_equity"], saved_c["max_equity"],
                         5),
    "min_bonds": slider(col2, "Minimum in bonds", "min_bonds", rec_c["min_bonds"], 1, saved_c["min_bonds"], 5),
    "min_short_term": slider(col2, "Minimum in short-term government bonds", "min_short_term", rec_c["min_short_term"],
                             1, saved_c["min_short_term"], 5),
    "max_other": slider(col1, "Maximum in gold and commodities", "max_other", 0, rec_c["max_other"],
                        saved_c["max_other"], 5),
    "max_crypto": slider(col2, "Maximum in crypto", "max_crypto", 0, rec_c["max_crypto"], saved_c["max_crypto"], 1)
    if rec_c["max_crypto"] > 0 else 0.0,  # crypto not allowed: already said above, no fixed "0%" line
    "max_weight_per_asset": slider(col1, "Maximum per product", "max_weight", 0.05, rec_c["max_weight_per_asset"],
                                   saved_c["max_weight_per_asset"], 5),
}

c["min_home_equity"] = 0.0
if rec_c["min_home_equity"] > 0:  # region chosen in QF3
    c["min_home_equity"] = slider(col2, f"Minimum share of your equities in {rec_c['region']}", "min_home_equity",
                                  rec_c["min_home_equity"], 1, saved_c["min_home_equity"], 5)
c["region"] = rec_c["region"]

st.caption(f"Short-term government bonds ({SHORT_TERM[get_currency()]}, almost cash in {get_currency()}) have no "
           "maximum. Narrow products (one sector or one country, like REITs or emerging markets) are limited to 15% "
           "each. Products marked .SW trade in CHF on the Swiss exchange.")

# Save the choices for the next pages
st.session_state["assets"] = tickers
st.session_state["constraints"] = c

stricter = [label for key, label in [("max_equity", "risky assets"), ("min_bonds", "bonds"), ("min_short_term", "short-term"),
                                      ("max_other", "gold and commodities"), ("max_crypto", "crypto"),
                                      ("max_weight_per_asset", "maximum per product"),
                                      ("min_home_equity", "region")] if c[key] != rec_c[key]]
if stricter:
    st.caption("Stricter than your profile: " + ", ".join(stricter) + ".")
if lacks_home_equity(tickers, c):
    st.warning(f"None of your equity products is mainly invested in {c['region']}: with the rule 'at least "
               f"{c['min_home_equity']:.0%} of your equities in {c['region']}', your portfolio will hold no equities. "
               "Add an equity product of this region (or click 'Back to our recommendation').")
if any(ASSETS[t][1] == "Crypto" for t in tickers) and c["max_crypto"] == 0:
    st.info("You chose crypto products but your maximum in crypto is 0%: they will get no weight.")

show_profile_messages(profile)
problems = check_constraints(tickers, c)
for p in problems:
    st.error(p)
if not problems:
    st.success(f"{len(tickers)} products selected. Your limits can all be respected.", icon=":material/check_circle:")
    summary = recommendation_summary(profile, tickers, c)
    if summary:
        with st.container(border=True):
            st.markdown("**Our recommendation with these products and limits**")
            st.write(summary)
nav_buttons("app_pages/profile.py", None if problems else "app_pages/portfolio.py", "See your portfolio")
