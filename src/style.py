"""
Look of the app: colours, page header, navigation buttons, explanations and chart style.
The theme itself (font, accent colour) is in .streamlit/config.toml.
"""
import streamlit as st

ACCENT = "#1c5cab"
# Asset classes: always the same colour in every chart (checked for colour-blind readers, with labels on the charts)
CLASS_COLORS = {"Equity": "#2a78d6", "Bond": "#1baf7a", "Other": "#eda100", "Crypto": "#e87ba4"}
# Each method keeps its colour on the efficient frontier (the recommended one is drawn as a star in the accent colour)
METHOD_COLORS = {"Equilibrium": "#2a78d6", "Minimum variance": "#1baf7a", "Black-Litterman": "#6250d6",
                 "Mean-variance": "#eb6834", "Resampled MV": "#eda100", "Risk parity (ERC)": "#e87ba4"}
YOURS, BENCHMARK_COLOR, OTHERS = ACCENT, "#3b4250", "#c3c8d1"
HIGHLIGHT = {"background-color": "#e8f0fa", "font-weight": "600"}  # the client's portfolio in the tables
STEPS = 6


def apply_style():
    """A few touches the theme cannot set: less empty space at the top, steady button height, and a size that
    follows the screen. Streamlit sizes text and spacing in rem (a multiple of the root font size), so on large
    screens we raise the root font size: everything grows together instead of a small column in the middle."""
    st.markdown("""<style>
        .block-container {padding-top: 2.5rem; max-width: 75rem;}
        @media (min-width: 1800px) {html {font-size: 17.5px;}}
        @media (min-width: 2200px) {html {font-size: 20px;}}
        @media (min-width: 2800px) {html {font-size: 24px;}}
        @media (min-width: 3400px) {html {font-size: 28px;}}
        button[kind="primary"], button[kind="secondary"] {min-height: 2.75rem;}
        [data-testid="stMetric"] {background: #f4f6f9; border-radius: 0.4rem; padding: 0.75rem 1rem;}
        [data-testid="stMetricLabel"] p {color: #5b6472;}
        span[data-tag] {background-color: #e8f0fa !important; color: #1b2430 !important;}
        span[data-tag] svg {color: #5b6472 !important; fill: #5b6472 !important;}
        </style>""", unsafe_allow_html=True)


def page_header(step, title, intro=None):
    """Step number, title and one short sentence: the same header on every page of the process."""
    if step:
        st.caption(f"STEP {step} OF {STEPS}")
    st.title(title)
    if intro:
        st.write(intro)


def nav_buttons(previous=None, following=None, label="Continue"):
    """Back and Continue buttons at the bottom of a page."""
    st.write("")
    left, _, right = st.columns([1, 2, 1])
    if previous and left.button("Back", icon=":material/arrow_back:", width="stretch", key=f"back_{previous}"):
        st.switch_page(previous)
    if following and right.button(label, type="primary", icon=":material/arrow_forward:", icon_position="right",
                                  width="stretch", key=f"go_{following}"):
        st.switch_page(following)


def explain(profile, text):
    """How to read a chart or table: open for beginners, closed for experienced investors."""
    with st.expander("How to read this", icon=":material/help_outline:", expanded=not profile.experienced):
        st.markdown(text)


def chart(fig, title=None):
    """Show a Plotly chart with the common style: title above, legend on top, light grid."""
    if title:
        st.markdown(f"**{title}**")
    fig.update_layout(title=None, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
                      margin=dict(l=10, r=10, t=30, b=10), hovermode="closest")
    st.plotly_chart(fig)


def highlight_lines(fig, yours, benchmark):
    """Line chart: the client's portfolio thick and in the accent colour, the benchmark dashed, the others grey."""
    for trace in fig.data:
        if trace.name == yours:
            trace.update(line=dict(color=YOURS, width=3.5), legendrank=1)
        elif trace.name == benchmark:
            trace.update(line=dict(color=BENCHMARK_COLOR, width=2, dash="dash"), legendrank=2)
        else:
            trace.update(line=dict(color=OTHERS, width=1.5), legendrank=3)
    return fig
