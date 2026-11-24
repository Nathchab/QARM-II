"""
PDF report that the client downloads on the plan page (library fpdf2).
Same colours as the app (src/style.py). Every text that could be too long is cut to the width of its column,
so nothing overlaps.
"""
from datetime import date

from fpdf import FPDF

from src import style
from src.data import ASSETS
from src.optimizer import region_exposures
from src.planning import MANAGEMENT_FEE
from src.portfolios import BAD_YEAR

rgb = lambda color: tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))  # "#1c5cab" -> (28, 92, 171)
INK, ACCENT, MUTED, RULE, PANEL = (27, 36, 48), rgb(style.ACCENT), (91, 100, 114), (221, 226, 234), (244, 246, 249)
CLASS_COLORS = {name: rgb(color) for name, color in style.CLASS_COLORS.items()}
CLASS_NAMES = {"Equity": "Equities", "Bond": "Bonds", "Other": "Gold and commodities", "Crypto": "Crypto"}
LEFT, RIGHT, WIDTH = 18, 192, 174  # page margins (mm) and width of the text
# The built-in PDF fonts only know Latin-1 characters
REPLACE = {"–": "-", "—": "-", "’": "'", "γ": "gamma", "σ": "sigma", "≤": "<=", "≥": ">=", "→": "->", "★": "*"}


def clean(text):
    for old, new in REPLACE.items():
        text = str(text).replace(old, new)
    return text.encode("latin-1", "replace").decode("latin-1")


class Report(FPDF):
    def header(self):
        self.set_fill_color(*ACCENT)
        self.rect(0, 0, 210, 2.5, "F")  # thin accent line at the top of every page
        self.set_xy(LEFT, 10)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*MUTED)
        self.cell(WIDTH / 2, 5, "ROBO-ADVISOR")
        self.set_font("Helvetica", "", 9)
        self.cell(WIDTH / 2, 5, clean(self.subtitle), align="R")
        self.set_draw_color(*RULE)
        self.set_line_width(0.2)
        self.line(LEFT, 17, RIGHT, 17)
        self.set_y(24)

    def footer(self):
        self.set_y(-15)
        self.set_draw_color(*RULE)
        self.line(LEFT, self.get_y(), RIGHT, self.get_y())
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(*MUTED)
        self.cell(WIDTH - 20, 7, "Academic project, HEC Lausanne. Simulations, not guarantees. This is not investment "
                                 "advice.")
        self.cell(20, 7, f"Page {self.page_no()} of {{nb}}", align="R")

    def fit(self, text, width):
        """Cut a text so that it fits in `width` mm with the current font."""
        text = clean(text)
        if self.get_string_width(text) <= width:
            return text
        while text and self.get_string_width(text + "...") > width:
            text = text[:-1]
        return text.rstrip() + "..."

    def section(self, number, title, need=40):
        """Title of a section; a new page if less than `need` mm are left (a section is not cut in two)."""
        if self.get_y() + need > 280:
            self.add_page()
        self.ln(4)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*ACCENT)
        self.cell(9, 7, f"{number:02d}")
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*INK)
        self.cell(WIDTH - 9, 7, self.fit(title, WIDTH - 9), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*RULE)
        self.line(LEFT, self.get_y() + 1, RIGHT, self.get_y() + 1)
        self.ln(4)

    def paragraph(self, value, size=9.5, color=INK, style=""):
        self.set_font("Helvetica", style, size)
        self.set_text_color(*color)
        self.multi_cell(WIDTH, 5, clean(value), align="L", new_x="LMARGIN", new_y="NEXT")

    def key_values(self, rows, label_width=52):
        """Two columns: label (muted) and value (bold, on several lines if needed)."""
        for label, value in rows:
            y = self.get_y()
            self.set_xy(LEFT + label_width, y)
            self.set_font("Helvetica", "B", 9.5)
            self.set_text_color(*INK)
            self.multi_cell(WIDTH - label_width, 5.5, clean(value), align="L", new_x="LMARGIN", new_y="NEXT")
            value_bottom = self.get_y()
            self.set_xy(LEFT, y)
            self.set_font("Helvetica", "", 9.5)
            self.set_text_color(*MUTED)
            self.multi_cell(label_width - 3, 5.5, clean(label), align="L", new_x="LMARGIN", new_y="NEXT")
            self.set_y(max(self.get_y(), value_bottom) + 1)

    def figures(self, items, highlight=None):
        """A row of key figures: big number, small label below, on a light panel."""
        gap, height = 3, 19
        width = (WIDTH - gap * (len(items) - 1)) / len(items)
        y = self.get_y()
        for i, (label, value) in enumerate(items):
            x = LEFT + i * (width + gap)
            self.set_fill_color(*PANEL)
            self.rect(x, y, width, height, "F")
            if i == highlight:
                self.set_fill_color(*ACCENT)
                self.rect(x, y, 1.2, height, "F")
            self.set_xy(x + 4, y + 3.5)
            self.set_font("Helvetica", "B", 13)
            self.set_text_color(*INK)
            self.cell(width - 8, 7, self.fit(value, width - 8))
            self.set_xy(x + 4, y + 11)
            self.set_font("Helvetica", "", 7.5)
            self.set_text_color(*MUTED)
            self.cell(width - 8, 5, self.fit(label, width - 8))
        self.set_y(y + height + 5)

    def table(self, columns, rows):
        """columns: list of (title, width, align). Header in small muted capitals, thin lines between rows."""
        self.set_font("Helvetica", "B", 7.5)
        self.set_text_color(*MUTED)
        for title, width, align in columns:
            self.cell(width, 6, clean(title.upper()), align=align)
        self.ln()
        self.set_draw_color(*RULE)
        self.line(LEFT, self.get_y(), RIGHT, self.get_y())
        for row in rows:
            if self.get_y() > 268:
                self.add_page()
            self.set_text_color(*INK)
            for k, (value, (_, width, align)) in enumerate(zip(row, columns)):
                self.set_font("Helvetica", "B" if k == 0 else "", 9)
                self.cell(width, 6.5, self.fit(value, width - 2), align=align)
            self.ln()
            self.line(LEFT, self.get_y(), RIGHT, self.get_y())


def client_report(profile, c, method, weights, exp_return, vol, currency, plan, orders, cash):
    """The advice as a PDF (bytes): profile, portfolio, plan, orders and points to review."""
    pdf = Report()
    pdf.subtitle = f"{date.today():%d %B %Y}  |  Portfolio in {currency}"
    pdf.set_auto_page_break(True, margin=20)
    pdf.set_margins(LEFT, 15, 210 - RIGHT)
    pdf.add_page()

    # ---------- Title ----------
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*INK)
    pdf.cell(WIDTH, 11, "Your investment proposal", new_x="LMARGIN", new_y="NEXT")
    pdf.paragraph(f"{profile.label} risk tolerance, {profile.capacity_label.lower()} financial capacity. "
                  f"Recommended portfolio: {method}.", size=10.5, color=MUTED)

    # ---------- 1. Profile ----------
    pdf.section(1, "Your profile")
    limits = [f"equities at most {c['max_equity']:.0%}", f"bonds at least {c['min_bonds']:.0%}",
              f"gold and commodities at most {c['max_other']:.0%}", f"crypto at most {c['max_crypto']:.0%}",
              f"at most {c['max_weight_per_asset']:.0%} per product"]
    if c.get("min_short_term", 0) > 0:
        limits.insert(2, f"short-term government bonds at least {c['min_short_term']:.0%}")
    if c.get("min_home_equity", 0) > 0:
        limits.append(f"at least {c['min_home_equity']:.0%} of your equities in {c['region']}")
    pdf.key_values([
        ("Risk tolerance", f"{profile.rt:.0f}/100, {profile.label} (risk aversion gamma = {profile.gamma:.2f})"),
        ("Financial capacity", f"{profile.capacity_label}. {profile.capacity_explanation}"),
        ("Loss limit in a bad year", "none" if profile.max_loss is None
         else f"{-profile.max_loss:.0%}, the loss {profile.loss_reason}"),
        ("Your limits", "; ".join(limits)),
    ])

    # ---------- 2. Portfolio ----------
    pdf.section(2, "Your portfolio", need=60 + 7 * len(weights))
    pdf.figures([("Expected return per year", f"{exp_return:.1%}"), ("Volatility per year", f"{vol:.1%}"),
                 ("Loss in a bad year (5%)", f"{exp_return - BAD_YEAR * vol:.1%}"),
                 ("Our fee per year", f"{MANAGEMENT_FEE:.2%}")], highlight=0)
    held = weights[weights >= 0.0005].sort_values(ascending=False)
    x_ticker, x_name, x_bar, x_value = LEFT, LEFT + 26, LEFT + 108, RIGHT - 14  # columns of the allocation
    bar_width = x_value - x_bar - 3
    for ticker, w in held.items():  # one line per product: ticker, name, bar coloured by asset class, weight
        y = pdf.get_y()
        pdf.set_xy(x_ticker, y)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*INK)
        pdf.cell(x_name - x_ticker, 6.5, pdf.fit(ticker, x_name - x_ticker - 2))
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*MUTED)
        pdf.cell(x_bar - x_name, 6.5, pdf.fit(ASSETS[ticker][0], x_bar - x_name - 3))
        pdf.set_fill_color(*PANEL)
        pdf.rect(x_bar, y + 1.8, bar_width, 3, "F")
        pdf.set_fill_color(*CLASS_COLORS[ASSETS[ticker][1]])
        pdf.rect(x_bar, y + 1.8, max(w / held.max(), 0.01) * bar_width, 3, "F")
        pdf.set_xy(x_value, y)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*INK)
        pdf.cell(RIGHT - x_value, 6.5, f"{w:.1%}", align="R", new_x="LMARGIN", new_y="NEXT")
    # legend: a coloured square and the name of each asset class in the portfolio
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*MUTED)
    for asset_class in dict.fromkeys(ASSETS[t][1] for t in held.index):
        pdf.set_fill_color(*CLASS_COLORS[asset_class])
        pdf.rect(pdf.get_x(), pdf.get_y() + 1.6, 2.6, 2.6, "F")
        pdf.set_x(pdf.get_x() + 4)
        pdf.cell(pdf.get_string_width(CLASS_NAMES[asset_class]) + 7, 6, CLASS_NAMES[asset_class])
    pdf.ln(8)
    regions = {r: w for r, w in region_exposures(weights).items() if w >= 0.005}
    pdf.key_values([("Regions (inside the funds, approximate)",
                     "  |  ".join(f"{r} {w:.0%}" for r, w in sorted(regions.items(), key=lambda x: -x[1])))],
                   label_width=70)

    # ---------- 3. Plan ----------
    pdf.section(3, f"Your plan over {plan['years']} years", need=70)
    pdf.paragraph(f"Initial amount {plan['initial']:,.0f} {currency} and {plan['contribution']:,.0f} {currency} "
                  f"{plan['per']}: you pay in {plan['paid_in']:,.0f} {currency} in total. Wealth at the end, in 5,000 "
                  "simulated futures:")
    pdf.ln(3)
    pdf.figures([(f"Paid in ({currency})", f"{plan['paid_in']:,.0f}"),
                 (f"Pessimistic, 5% ({currency})", f"{plan['p5']:,.0f}"),
                 (f"Median ({currency})", f"{plan['p50']:,.0f}"),
                 (f"Optimistic, 95% ({currency})", f"{plan['p95']:,.0f}")], highlight=2)
    pdf.key_values([
        (f"Chance to reach {plan['goal']:,.0f} {currency}", f"{plan['prob_goal']:.0%}"),
        (f"Chance of a fall of more than {plan['loss_limit']:.0%}",
         f"{plan['prob_loss']:.0%} (at some point, from a previous peak)"),
        (f"Our fee over {plan['years']} years", f"about {plan['fee_cost']:,.0f} {currency} ({MANAGEMENT_FEE:.2%} "
                                                 "per year, already taken out above)"),
        ("Transaction costs", f"about {plan['trading_cost']:,.0f} {currency} ({plan['cost_bps']:.0f} bp on each "
                              "purchase); the ETF costs (TER) are already in the ETF prices"),
    ], label_width=70)

    # ---------- 4. Orders ----------
    pdf.section(4, "Orders to place", need=35 + 7 * len(orders))
    rows = []
    for _, r in orders.iterrows():
        ticker, name = r["Product"].split(" – ", 1)
        rows.append((ticker, name, f"{r['Target weight']:.1%}", f"{r['Quantity']:,.4g}", f"{r['Amount']:,.0f}"))
    pdf.table([("Ticker", 26, "L"), ("Product", 84, "L"), ("Weight", 20, "R"), ("Quantity", 20, "R"),
               (f"Amount ({currency})", 24, "R")], rows)
    pdf.ln(2)
    pdf.paragraph(f"Cash left: {cash:,.0f} {currency}. ETFs are bought in whole shares, crypto in fractions.",
                  size=8.5, color=MUTED)

    # ---------- 5. Points to review ----------
    pdf.section(5, "Points to review", need=30)
    for message in profile.warnings + profile.notes or ["Nothing special: your answers are consistent."]:
        y = pdf.get_y()
        pdf.set_fill_color(*ACCENT)
        pdf.rect(LEFT, y + 2, 1.5, 1.5, "F")
        pdf.set_xy(LEFT + 5, y)
        pdf.set_font("Helvetica", "", 9.5)
        pdf.set_text_color(*INK)
        pdf.multi_cell(WIDTH - 5, 5, clean(message), align="L", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1.5)
    return bytes(pdf.output())
