#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""One seaborn theme for every figure in the paper.

Each figure script used to set its own colours, fonts and grid, so the plates
did not read as one system. Every plotting module now calls :func:`apply` first
and draws from :data:`PALETTE`, which fixes typography, grid weight, spine
policy and the categorical hue order in a single place.

The categorical hues are assigned in fixed order and were validated for
colour-vision deficiency rather than chosen by eye: on the all-pairs test
(scatter and small-multiple forms put every pair on screen together, not just
adjacent ones) the worst normal-vision separation is dE 21.8 against a floor of
15, and the worst CVD separation dE 10.6 against a target of 8, in OKLab x100.
The aqua slot sits at 2.74:1 against the paper surface, below the 3:1 bar, so
any figure using it must carry visible direct labels rather than relying on the
legend alone -- which Figure 1 does.

Usage::

    from src.evaluation.plot_style import apply, PALETTE
    apply()
"""
from __future__ import annotations

PALETTE = {
    "blue":   "#2a78d6",   # slot 1 - real-time crash prediction
    "orange": "#eb6834",   # slot 2 - citywide / multi-day risk maps
    "aqua":   "#1baf7a",   # slot 3 - this work
    "gray":   "#52514e",   # neutral - network screening (re-stepped: the
                           # lighter #898781 fell to dE 14.5 against aqua,
                           # below the 15 floor, and those two carry the
                           # paper's central contrast)
    "muted":  "#898781",   # axis text
    "grid":   "#e1e0d9",
    "axis":   "#c3c2b7",
    "ink":    "#0b0b0b",
}
CATEGORICAL = [PALETTE[k] for k in ("blue", "orange", "aqua", "gray")]


def apply(context: str = "paper", font_scale: float = 1.0) -> None:
    """Install the shared theme. Call once, before any figure is created."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(context=context, style="whitegrid", font_scale=font_scale,
                  rc={
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.edgecolor":    PALETTE["axis"],
        "axes.linewidth":    0.8,
        "axes.labelcolor":   PALETTE["ink"],
        "axes.titlesize":    11,
        "axes.titleweight":  "semibold",
        "axes.labelsize":    10,
        "axes.grid":         True,
        "grid.color":        PALETTE["grid"],
        "grid.linewidth":    0.6,
        "grid.alpha":        0.9,
        "xtick.color":       PALETTE["muted"],
        "ytick.color":       PALETTE["muted"],
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.frameon":    False,
        "legend.fontsize":   9,
        "lines.linewidth":   2.0,
        "lines.markersize":  8,
        "font.family":       "sans-serif",
        "font.sans-serif":   ["DejaVu Sans", "Arial", "Helvetica"],
        "savefig.bbox":      "tight",
        "savefig.dpi":       300,
        "pdf.fonttype":      42,   # embed TrueType so the journal can edit text
        "ps.fonttype":       42,
    })
    sns.set_palette(CATEGORICAL)
    # grid behind the marks, and no top/right spines anywhere
    plt.rcParams["axes.axisbelow"] = True
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def despine(ax=None) -> None:
    """Drop top/right spines on an axis created before :func:`apply` ran."""
    import seaborn as sns
    sns.despine(ax=ax)
