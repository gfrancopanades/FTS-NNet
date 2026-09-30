#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure 1 — where crash-risk prediction operates, and the gap this paper fills.

Positions the literature on two axes an operator cares about: how far ahead a
prediction is issued, and how large the thing being predicted about is. The
vertical axis multiplies cell area by cell duration, so one number captures both
spatial and temporal granularity and "lower is finer".

The shaded band is the claim: a multi-day horizon AT operational granularity.

    Band bounds are declared, not drawn by eye. The horizontal span runs from
    half a day to one week: the window over which patrol rosters, ambulance
    staging and speed-management programmes are actually fixed, and beyond
    which a forecast stops informing a schedule and starts informing a budget.
    That upper edge is what separates this work (three days) from STZITD-GNN
    (a fortnight) -- two horizons that serve different decisions and should not
    be read as neighbours. The vertical span is the granularity at which a
    specific carriageway and hour can be actioned.

    The two nearest papers miss the band on different axes, which is the point
    of the figure: RiskOracle reaches operational granularity but only fifteen
    minutes ahead, and STZITD-GNN reaches a multi-day horizon but at a cell
    three times coarser and ten times further out. Neither combination is the
    one an agency needs.

Marker convention: filled markers are coordinates confirmed from the source
text; hollow markers are pending confirmation, as the caption states.
"""
from __future__ import annotations
from src.paths import (  # portable paths -- see src/paths.py
    PROJECT_ROOT_STR as _AP7_ROOT,
    EXPERIMENTS_ROOT_STR as _AP7_EXPERIMENTS,
    DATA_DIR_STR as _AP7_DATA,
    TABLES_DIR as _AP7_TABLES,
    FIGURES_DIR as _AP7_FIGS,
)
import os
import sys

sys.path.insert(0, _AP7_ROOT)
from src.evaluation.plot_style import apply, PALETTE

apply()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

OUT = os.environ.get("AP7_FIGURES_DIR",
                     str(_AP7_FIGS))

BAND_X = (720, 10080)        # minutes: 12 h -> 1 week
BAND_Y_MAX = 0.50            # km^2*h; see module docstring
BAND_Y_MIN = 0.005

# Every coordinate below is (cell area in km^2) x (cell duration in hours), and
# every paper is placed at its LONGEST stated horizon and FINEST stated cell.
# That rule is deliberately generous to the comparison: it gives each method its
# best shot at reaching the band, so an empty band is the strongest form of the
# claim rather than an artefact of unfavourable placement.
#
#   MSGNN      link-level, 5-min bins, 15/60-min horizons. Urban link taken as
#              0.5 km x 1 km nominal -> 0.5 km^2 x 5/60 h = 0.042. Horizon 60.
#   Yuan 2019  83 intersection approaches, 5-min bins, 5-10 min horizon.
#              Approach taken as 0.2 km x 1 km -> 0.2 x 5/60 = 0.017. Horizon 10.
#   STCL-Net   Manhattan (~59 km^2) on its finest 30x10 grid -> 0.20 km^2 per
#              cell, hourly -> 0.20 x 1 h = 0.20. Horizon hourly = 60 min.
#   DGCN       200 m links, 2-min bins, 20/30-min horizons -> 0.2 km^2 x 2/60 h
#              = 0.0067. Horizon 30. Forecasts TRAFFIC, not crashes (see style).
#   STZITD-GNN road-level segments, daily, 14 days ahead. Segment taken as
#              0.3 km^2 -> 0.3 x 24 h = 7.2. Horizon 20160 min.
#   DMD-STGNN  78 Denver neighbourhoods (~401 km^2) -> 5.1 km^2 each, daily,
#              1-7 days ahead -> 5.1 x 24 h = 123. Horizon 10080 min.
#   This work  1 km directional segment, 5-min bins, 72 h ahead
#              -> 1.0 km^2 x 5/60 h = 0.083. Horizon 4320 min.
#
# name, horizon (min), cell size (km^2*h), cluster, coordinates explicit in source
LITERATURE = [
    # --- real-time crash prediction: fine cells, minutes ahead ---------------
    ("LSTM RNN (Yuan et al. 2019)",   10,   0.0167, "rtcp",  False),
    ("MSGNN (Tran et al. 2023)", 60,  0.0417, "rtcp",  False),
    # --- traffic forecasting (not crash risk) -------------------------------
    ("DGCN (Li et al. 2021)",   30,   0.0067, "flow",  False),
    # --- citywide / multi-day risk maps: coarse cells, days ahead -----------
    ("STCL-Net (Bao et al. 2019)", 60,  0.20,  "grid",  False),
    ("STZITD-GNN (Gao et al. 2024)", 20160, 7.2, "grid", False),
    ("DMD-STGNN (Xu et al. 2026)", 10080, 123.0, "grid", False),
    # --- this work -----------------------------------------------------------
    ("GeoLSTM + MLP-focal", 4320, 0.0833, "ours",  True),
]
# EB screening has no temporal axis at all; drawn as a reference band, not a point.
STYLE = {
    "rtcp": dict(color=PALETTE["blue"],   marker="o",
                 name="Real-time crash prediction"),
    "grid": dict(color=PALETTE["orange"], marker="s",
                 name="Multi-day / area-aggregated crash risk"),
    "flow": dict(color=PALETTE["gray"],   marker="D",
                 name="Traffic forecasting (not crash risk)"),
    "ours": dict(color=PALETTE["aqua"],   marker="*",
                 name="This work"),
}
# label offsets in points, tuned to avoid collisions
# Seven real-time papers occupy one decade of x and one of y, so no offset
# places seven labels beside their markers without collision. Those are fanned
# into the empty mid-left region on thin leader lines at fixed data coordinates;
# the rest take simple point offsets from their marker.
# The two bottom-left markers sit a decade apart in x but their labels are long
# enough to overlap each other and DGCN's marker at any point offset, so they are
# fanned into the empty region between 1 h and 6 h on thin leader lines.
# Right-aligned at the band's left edge so neither label crosses the shaded
# region: a label lying inside the "empty" band reads as a marker inside it.
LEADERED = {                       # name -> (label x, label y) in data units
    "LSTM RNN (Yuan et al. 2019)": (700, 0.0135),
    "DGCN (Li et al. 2021)":            (700, 0.0058),
}
OFFSETS = {
    "MSGNN (Tran et al. 2023)":         (13,  10),
    "STCL-Net (Bao et al. 2019)":       (13,  10),
    "STZITD-GNN (Gao et al. 2024)":     (-56, -22),
    "DMD-STGNN (Xu et al. 2026)":       (-62,  14),
    "GeoLSTM + MLP-focal":              (-74, -34),
}


def main() -> int:
    fig, ax = plt.subplots(figsize=(9.0, 6.2))

    # --- the empty quadrant --------------------------------------------------
    ax.add_patch(Rectangle((BAND_X[0], BAND_Y_MIN),
                           BAND_X[1] - BAND_X[0], BAND_Y_MAX - BAND_Y_MIN,
                           facecolor=PALETTE["aqua"], alpha=0.09,
                           edgecolor=PALETTE["aqua"], linewidth=1.0,
                           linestyle=(0, (5, 3)), zorder=0))
    ax.text(BAND_X[0] * 1.10, BAND_Y_MAX * 0.62,
            "multi-day horizon at\noperational granularity",
            fontsize=9.5, color=PALETTE["aqua"], style="italic",
            va="center", ha="left", zorder=1, linespacing=1.4)

    # --- network screening: no temporal resolution ---------------------------
    ax.axhspan(1.0e3, 1.0e5, facecolor=PALETTE["gray"], alpha=0.06, zorder=0)
    ax.axhline(1.0e4, color=PALETTE["gray"], linewidth=1.2,
               linestyle=(0, (2, 2)), zorder=1)
    ax.text(6.5, 1.35e4, "EB network screening (Hauer 1997; AASHTO 2010) — "
                         "site ranking, no temporal resolution",
            fontsize=8.5, color=PALETTE["gray"], va="bottom", ha="left")

    # --- the literature ------------------------------------------------------
    for name, x, y, cluster, confirmed in LITERATURE:
        st = STYLE[cluster]
        big = cluster == "ours"
        ax.scatter(x, y, s=560 if big else 110, marker=st["marker"],
                   facecolor=st["color"] if confirmed else "white",
                   edgecolor=st["color"],
                   linewidth=1.9 if big else 1.4,
                   zorder=6 if big else 4)
        if name in LEADERED:
            lx, ly = LEADERED[name]
            ax.annotate(name, xy=(x, y), xytext=(lx, ly), textcoords="data",
                        fontsize=8.5, color=PALETTE["muted"],
                        va="center", ha="right", zorder=7,
                        arrowprops=dict(arrowstyle="-", color=PALETTE["axis"],
                                        linewidth=0.7, shrinkA=2, shrinkB=4))
        else:
            dx, dy = OFFSETS.get(name, (10, 6))
            ax.annotate(name, (x, y), textcoords="offset points", xytext=(dx, dy),
                        fontsize=9.5 if big else 8.5,
                        fontweight="semibold" if big else "normal",
                        color=PALETTE["ink"] if big else PALETTE["muted"], zorder=7)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(3.4, 1.6e5); ax.set_ylim(0.0035, 1.2e5)
    ax.set_xlabel("prediction horizon (minutes, log scale)")
    ax.set_ylabel("size of one prediction cell (km$^2\\cdot$h, log scale) — lower is finer")
    ax.set_title("Where crash-risk prediction operates", loc="left")

    ax.set_xticks([5, 15, 60, 360, 1440, 4320, 10080, 40320])
    ax.set_xticklabels(["5 min", "15 min", "1 h", "6 h", "1 day",
                        "3 days", "1 week", "4 weeks"])

    handles = [Line2D([], [], marker=STYLE[c]["marker"], linestyle="none",
                      markerfacecolor=STYLE[c]["color"] if c == "ours" else "white",
                      markeredgecolor=STYLE[c]["color"], markeredgewidth=1.5,
                      markersize=15 if c == "ours" else 9, label=STYLE[c]["name"])
               for c in ("rtcp", "grid", "flow", "ours")]
    handles.append(Line2D([], [], marker="^", linestyle="none", markerfacecolor="white",
                          markeredgecolor=PALETTE["gray"], markersize=9,
                          label="Network screening (EB / HSM)"))
    handles.append(Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
                          markeredgecolor=PALETTE["muted"], markersize=9,
                          label="hollow = cell size derived, not stated"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=3, fontsize=8.0, handletextpad=0.6, columnspacing=1.6)

    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"figure1_quadrant.{ext}"))
    plt.close(fig)

    inside = [n for n, x, y, c, _ in LITERATURE
              if BAND_X[0] <= x <= BAND_X[1] and BAND_Y_MIN <= y <= BAND_Y_MAX]
    print(f"[band] x={BAND_X} min, y=({BAND_Y_MIN}, {BAND_Y_MAX}) km^2*h")
    print(f"[band] markers inside: {inside}")
    print(f"[DONE] {OUT}/figure1_quadrant.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
