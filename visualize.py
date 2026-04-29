"""
COVID-19 Bayesian Network — Visualization
Generates a multi-panel PNG report: bayesian_report.png
Run:  python visualize.py
"""

import matplotlib
matplotlib.use("Agg")           # no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
import numpy as np

# ── import inference engine from bayesian.py ──────────────────────────────
from bayesian import infer, contact_risk as _contact_risk_raw

# ═══════════════════════════════════════════════════════════════════════════
#  COLOUR PALETTE
# ═══════════════════════════════════════════════════════════════════════════
BG      = "#0f1117"
PANEL   = "#1a1d27"
ACCENT  = "#00c8ff"
RED     = "#ff4b4b"
YELLOW  = "#ffc107"
GREEN   = "#4caf7d"
TEXT    = "#e8eaf0"
DIM     = "#6b7280"
WHITE   = "#ffffff"

def risk_colour(p):
    if p >= 0.75: return RED
    if p >= 0.40: return YELLOW
    return GREEN

# ═══════════════════════════════════════════════════════════════════════════
#  SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════
SCENARIOS = [
    ("Baseline\n(no evidence)",          {}),
    ("Positive test\nonly",              {"test": True}),
    ("Positive test +\nfever + cough",   {"test": True, "fever": True, "cough": True}),
    ("Positive test,\nno symptoms",      {"test": True, "fever": False, "cough": False}),
    ("Close contact +\nfever, no test",  {"contact": True, "fever": True}),
    ("Symptoms,\nnegative test",         {"fever": True, "cough": True, "test": False}),
]

scenario_labels  = [s[0] for s in SCENARIOS]
scenario_probs   = [infer(s[1])["covid"] for s in SCENARIOS]

# Contact-network data
INDEX_EVIDENCE = {"test": True, "fever": True, "cough": True}
CONTACTS = [
    {"name": "Alice",   "fever": True,  "cough": False},
    {"name": "Bob",     "fever": False, "cough": False},
    {"name": "Charlie", "fever": True,  "cough": True},
    {"name": "Diana",   "test": False},
    {"name": "Eve"},
]

def _contact_p(ct_evidence, p_infected):
    ce_exp  = {**ct_evidence, "contact": True}
    ce_safe = {**ct_evidence, "contact": False}
    return (p_infected * infer(ce_exp)["covid"]
            + (1 - p_infected) * infer(ce_safe)["covid"])

p_index   = infer(INDEX_EVIDENCE)["covid"]
ct_names  = [c["name"] for c in CONTACTS]
ct_probs  = [_contact_p({k: v for k, v in c.items() if k != "name"}, p_index)
             for c in CONTACTS]

# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE LAYOUT
# ═══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 13), facecolor=BG)
fig.suptitle("COVID-19 Bayesian Network — Results Report",
             fontsize=20, fontweight="bold", color=WHITE, y=0.97)

gs = GridSpec(2, 3, figure=fig,
              hspace=0.52, wspace=0.38,
              left=0.06, right=0.97, top=0.91, bottom=0.07)

ax_bar   = fig.add_subplot(gs[0, :2])   # top-left wide: scenario bars
ax_dag   = fig.add_subplot(gs[0, 2])    # top-right:     network diagram
ax_ct    = fig.add_subplot(gs[1, :2])   # bottom-left:   contact risk
ax_info  = fig.add_subplot(gs[1, 2])    # bottom-right:  key takeaways

for ax in [ax_bar, ax_dag, ax_ct, ax_info]:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(DIM)

# ───────────────────────────────────────────────────────────────────────────
#  PANEL 1 — Scenario bar chart
# ───────────────────────────────────────────────────────────────────────────
colours = [risk_colour(p) for p in scenario_probs]
bars = ax_bar.barh(range(len(scenario_labels)), scenario_probs,
                   color=colours, height=0.55, zorder=3)

# percentage labels
for i, (bar, p) in enumerate(zip(bars, scenario_probs)):
    ax_bar.text(min(p + 0.015, 0.99), i, f"{p*100:.1f}%",
                va="center", ha="left", color=WHITE,
                fontsize=10, fontweight="bold")

ax_bar.set_xlim(0, 1.12)
ax_bar.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
ax_bar.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], color=DIM, fontsize=9)
ax_bar.set_yticks(range(len(scenario_labels)))
ax_bar.set_yticklabels(scenario_labels, color=TEXT, fontsize=9)
ax_bar.invert_yaxis()
ax_bar.axvline(0.75, color=RED,    lw=1, ls="--", alpha=0.5, zorder=2)
ax_bar.axvline(0.40, color=YELLOW, lw=1, ls="--", alpha=0.5, zorder=2)
ax_bar.set_title("P(COVID) Across Scenarios", color=ACCENT,
                 fontsize=12, fontweight="bold", pad=8)
ax_bar.tick_params(colors=DIM)
ax_bar.grid(axis="x", color=DIM, alpha=0.25, zorder=1)

legend_handles = [
    mpatches.Patch(color=RED,    label="High risk  (≥75%)"),
    mpatches.Patch(color=YELLOW, label="Medium risk (40–75%)"),
    mpatches.Patch(color=GREEN,  label="Low risk    (<40%)"),
]
ax_bar.legend(handles=legend_handles, loc="lower right",
              framealpha=0.2, labelcolor=TEXT, fontsize=8)

# ───────────────────────────────────────────────────────────────────────────
#  PANEL 2 — Bayesian network DAG diagram
# ───────────────────────────────────────────────────────────────────────────
ax_dag.set_xlim(0, 10)
ax_dag.set_ylim(0, 10)
ax_dag.axis("off")
ax_dag.set_title("Network Structure (DAG)", color=ACCENT,
                 fontsize=12, fontweight="bold", pad=8)

NODE_POS = {
    "Contact":  (2.5, 8.0),
    "COVID":    (5.0, 5.5),
    "Fever":    (2.0, 2.8),
    "Cough":    (5.0, 2.8),
    "Test":     (8.0, 2.8),
}
NODE_COLOUR = {
    "Contact": YELLOW,
    "COVID":   RED,
    "Fever":   ACCENT,
    "Cough":   ACCENT,
    "Test":    ACCENT,
}
EDGES = [("Contact", "COVID"), ("COVID", "Fever"),
         ("COVID", "Cough"),   ("COVID", "Test")]

# Draw edges
for src, dst in EDGES:
    x0, y0 = NODE_POS[src]
    x1, y1 = NODE_POS[dst]
    ax_dag.annotate("",
        xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=DIM, lw=1.4,
                        connectionstyle="arc3,rad=0.05"))

# Draw nodes
for name, (x, y) in NODE_POS.items():
    circle = plt.Circle((x, y), 0.9, color=NODE_COLOUR[name],
                         alpha=0.18, zorder=3)
    ax_dag.add_patch(circle)
    ax_dag.text(x, y, name, ha="center", va="center",
                color=NODE_COLOUR[name], fontsize=9, fontweight="bold", zorder=4)

# Annotations
ax_dag.text(5.0, 9.3, "Observable\nevidence", ha="center",
            color=ACCENT, fontsize=7, style="italic")
ax_dag.text(5.0, 4.4, "Hidden\nvariable", ha="center",
            color=RED, fontsize=7, style="italic")

# ───────────────────────────────────────────────────────────────────────────
#  PANEL 3 — Contact-network risk
# ───────────────────────────────────────────────────────────────────────────
ct_colours = [risk_colour(p) for p in ct_probs]
ct_bars = ax_ct.bar(range(len(ct_names)), ct_probs,
                    color=ct_colours, width=0.55, zorder=3)

for i, (bar, p) in enumerate(zip(ct_bars, ct_probs)):
    ax_ct.text(i, p + 0.02, f"{p*100:.1f}%",
               ha="center", va="bottom", color=WHITE,
               fontsize=10, fontweight="bold")

# Index person reference line
ax_ct.axhline(p_index, color=RED, lw=1.5, ls="--", alpha=0.7, zorder=2,
              label=f"Index person ({p_index*100:.1f}%)")

ax_ct.set_ylim(0, 1.15)
ax_ct.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
ax_ct.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], color=DIM, fontsize=9)
ax_ct.set_xticks(range(len(ct_names)))
ax_ct.set_xticklabels(ct_names, color=TEXT, fontsize=10)
ax_ct.tick_params(colors=DIM)
ax_ct.grid(axis="y", color=DIM, alpha=0.25, zorder=1)
ax_ct.set_title("Contact-Network Risk Assessment\n"
                "(Index person: test+ fever+ cough+)",
                color=ACCENT, fontsize=12, fontweight="bold", pad=8)
ax_ct.legend(framealpha=0.2, labelcolor=TEXT, fontsize=8)

# Evidence labels under bars
evidence_notes = ["fever=Y\ncough=N", "fever=N\ncough=N",
                  "fever=Y\ncough=Y", "test=N", "no info"]
for i, note in enumerate(evidence_notes):
    ax_ct.text(i, -0.10, note, ha="center", va="top",
               color=DIM, fontsize=7, transform=ax_ct.get_xaxis_transform())

# ───────────────────────────────────────────────────────────────────────────
#  PANEL 4 — Key takeaways text box
# ───────────────────────────────────────────────────────────────────────────
ax_info.axis("off")
ax_info.set_title("Key Takeaways", color=ACCENT,
                  fontsize=12, fontweight="bold", pad=8)

takeaways = [
    ("Positive test alone",   "~54% — not certain\ndue to false positives"),
    ("+ fever & cough",        "~99.5% — near certain"),
    ("No symptoms, test+",     "~9% — likely false positive"),
    ("Contact + fever",        "~87% — get tested!"),
    ("Symptoms, test−",        "~46% — test may have missed it"),
    ("Charlie (contacts)",     "Highest risk (98%) —\nfever + cough + exposure"),
    ("Diana (contacts)",       "Lowest risk (2.7%) —\nnegative test"),
]

y = 0.94
for label, detail in takeaways:
    ax_info.text(0.04, y, f"▸ {label}",
                 transform=ax_info.transAxes,
                 color=ACCENT, fontsize=8.5, fontweight="bold", va="top")
    ax_info.text(0.06, y - 0.045, detail,
                 transform=ax_info.transAxes,
                 color=TEXT, fontsize=8, va="top")
    y -= 0.135

# Footer
fig.text(0.5, 0.005,
         "CS 5002 Final Project  ·  Authors: Aditya Elayavalli, James Rota, Yevgeniy Karangel  ·  April 2026",
         ha="center", color=DIM, fontsize=8)

# ═══════════════════════════════════════════════════════════════════════════
#  SAVE
# ═══════════════════════════════════════════════════════════════════════════
OUTPUT = "bayesian_report.png"
plt.savefig(OUTPUT, dpi=160, bbox_inches="tight", facecolor=BG)
print(f" Saved → {OUTPUT}")
