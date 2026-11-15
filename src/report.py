"""
PDF report that the client downloads on the plan page (library fpdf2).
"""
from datetime import date

from fpdf import FPDF

from src.data import ASSETS
from src.optimizer import region_exposures
from src.planning import MANAGEMENT_FEE
from src.portfolios import BAD_YEAR

NAVY, ORANGE, GREY, LIGHT = (24, 44, 84), (228, 87, 46), (110, 110, 110), (242, 244, 248)
CLASS_COLORS = {"Equity": (31, 119, 180), "Bond": (44, 160, 44), "Other": (255, 127, 14), "Crypto": (148, 103, 189)}
# The built-in PDF fonts only know Latin-1 characters
REPLACE = {"–": "-", "—": "-", "’": "'", "γ": "gamma", "σ": "sigma", "≤": "<=", "≥": ">=", "→": "->", "★": "*"}


def clean(text):
    for old, new in REPLACE.items():
        text = text.replace(old, new)
    return text.encode("latin-1", "replace").decode("latin-1")


class Report(FPDF):
    def header(self):
        self.set_fill_color(*NAVY)
        self.rect(0, 0, 210, 28, "F")
        self.set_xy(15, 8)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 18)
        self.cell(0, 8, "Your investment proposal")
        self.set_xy(15, 17)
        self.set_font("Helvetica", "", 10)
        self.cell(0, 6, clean(self.subtitle))
        self.set_y(36)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GREY)
        self.cell(0, 5, "Academic project (QARM II, HEC Lausanne). Simulations, not guarantees. This is not "
                        f"investment advice.   Page {self.page_no()}", align="C")

    def section(self, title, need=30):
        """Title of a section; new page if less than `need` mm are left (a section is not cut in two)."""
        if self.get_y() + need > 279:
            self.add_page()
        self.ln(3)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*NAVY)
        self.cell(0, 8, clean(title), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*ORANGE)
        self.set_line_width(0.6)
        self.line(15, self.get_y(), 195, self.get_y())
        self.ln(3)
        self.set_text_color(0, 0, 0)

    def text(self, value, size=10, style=""):
        self.set_font("Helvetica", style, size)
        self.multi_cell(0, 5, clean(value), align="L", new_x="LMARGIN", new_y="NEXT")

    def key_values(self, rows):
        """Two columns: label (grey) and value (bold)."""
        for label, value in rows:
            self.set_font("Helvetica", "", 10)
            self.set_text_color(*GREY)
            self.cell(62, 6, clean(label))
            self.set_font("Helvetica", "B", 10)
            self.set_text_color(0, 0, 0)
            self.multi_cell(0, 6, clean(value), align="L", new_x="LMARGIN", new_y="NEXT")

    def boxes(self, items):
        """A row of coloured boxes with a big number and a small label."""
        width = 180 / len(items)
        y = self.get_y()
        for i, (label, value) in enumerate(items):
            x = 15 + i * width
            self.set_fill_color(*LIGHT)
            self.rect(x + 1, y, width - 2, 20, "F")
            self.set_xy(x + 1, y + 3)
            self.set_font("Helvetica", "B", 13)
            self.set_text_color(*(ORANGE if i == 2 else NAVY))
            self.cell(width - 2, 7, clean(value), align="C")
            self.set_xy(x + 1, y + 11)
            self.set_font("Helvetica", "", 8)
            self.set_text_color(*GREY)
            self.cell(width - 2, 5, clean(label), align="C")
        self.set_text_color(0, 0, 0)
        self.set_y(y + 24)

    def table(self, columns, widths, rows, aligns):
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(*NAVY)
        self.set_text_color(255, 255, 255)
        for name, w, a in zip(columns, widths, aligns):
            self.cell(w, 7, clean(name), fill=True, align=a)
        self.ln()
        self.set_font("Helvetica", "", 9)
        self.set_text_color(0, 0, 0)
        for k, row in enumerate(rows):
            self.set_fill_color(*(LIGHT if k % 2 else (255, 255, 255)))
            for value, w, a in zip(row, widths, aligns):
                self.cell(w, 6, clean(str(value)), fill=True, align=a)
            self.ln()


def client_report(profile, c, method, weights, exp_return, vol, currency, plan, orders, cash):
    """The advice as a PDF (bytes): profile, portfolio, plan, orders and points to review."""
    pdf = Report()
    pdf.subtitle = f"Robo-advisor  |  {date.today():%d %B %Y}  |  Currency: {currency}"
    pdf.set_auto_page_break(True, margin=18)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()

    pdf.section("1. Your profile")
    pdf.key_values([
        ("Risk tolerance", f"{profile.rt:.0f}/100 - {profile.label} (risk aversion gamma = {profile.gamma:.2f})"),
        ("Financial capacity", f"{profile.capacity_label} ({profile.capacity_score}/9)"),
        ("Loss limit in a bad year", "none" if profile.max_loss is None
         else f"{-profile.max_loss:.0%}, the loss {profile.loss_reason}"),
        ("Your limits", f"equities max {c['max_equity']:.0%}, bonds min {c['min_bonds']:.0%}, short-term government "
                        f"bonds min {c.get('min_short_term', 0):.0%}, gold and commodities max {c['max_other']:.0%}, "
                        f"crypto max {c['max_crypto']:.0%}, max {c['max_weight_per_asset']:.0%} per product"
                        + (f", at least {c['min_home_equity']:.0%} of your equities in {c['region']}"
                           if c.get("min_home_equity", 0) > 0 else "")),
    ])

    pdf.section(f"2. Your portfolio: {method}")
    pdf.boxes([("Expected return per year", f"{exp_return:.1%}"), ("Volatility per year", f"{vol:.1%}"),
               ("Loss in a bad year (5%)", f"{exp_return - BAD_YEAR * vol:.1%}"),
               ("Our fee per year", f"{MANAGEMENT_FEE:.2%}")])
    held = weights[weights >= 0.0005].sort_values(ascending=False)
    for ticker, w in held.items():  # one horizontal bar per product, coloured by asset class
        y = pdf.get_y()
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(16, 6, ticker)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(70, 6, clean(ASSETS[ticker][0][:42]))
        pdf.set_fill_color(*CLASS_COLORS[ASSETS[ticker][1]])
        pdf.rect(101, y + 1.2, max(w, 0.003) * 80, 3.6, "F")
        pdf.set_xy(101 + w * 80 + 2, y)
        pdf.cell(16, 6, f"{w:.1%}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*GREY)
    pdf.cell(0, 6, "Blue: equities   Green: bonds   Orange: gold and commodities   Purple: crypto",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    regions = {r: w for r, w in region_exposures(weights).items() if w >= 0.005}
    pdf.text("Regions (looking inside the funds, approximate): "
             + ", ".join(f"{r} {w:.0%}" for r, w in sorted(regions.items(), key=lambda x: -x[1])), size=9)

    pdf.section(f"3. Your plan over {plan['years']} years")
    pdf.text(f"Initial amount {plan['initial']:,.0f} {currency} and {plan['monthly']:,.0f} {currency} per month: "
             f"you pay in {plan['paid_in']:,.0f} {currency} in total. Wealth at the end in 5,000 simulated futures:")
    pdf.ln(2)
    pdf.boxes([("Paid in", f"{plan['paid_in']:,.0f}"), ("Pessimistic (5%)", f"{plan['p5']:,.0f}"),
               ("Median", f"{plan['p50']:,.0f}"), ("Optimistic (95%)", f"{plan['p95']:,.0f}")])
    pdf.key_values([
        (f"Chance to reach {plan['goal']:,.0f}", f"{plan['prob_goal']:.0%}"),
        (f"Chance of a fall > {plan['loss_limit']:.0%}", f"{plan['prob_loss']:.0%} (at some point, from a previous peak)"),
    ])

    pdf.section("4. Orders to place", need=30 + 6 * len(orders))
    rows = [(r["Product"][:48], f"{r['Target weight']:.1%}", f"{r['Quantity']:,.4g}", f"{r['Amount']:,.0f}")
            for _, r in orders.iterrows()]
    pdf.table(["Product", "Weight", "Quantity", f"Amount ({currency})"], [105, 22, 25, 28], rows,
              ["L", "R", "R", "R"])
    pdf.text(f"Cash left: {cash:,.0f} {currency}", size=9, style="I")

    pdf.section("5. Points to review")
    for message in profile.warnings + profile.notes or ["Nothing special: your answers are consistent."]:
        pdf.text(f"- {message}", size=9)
    return bytes(pdf.output())
