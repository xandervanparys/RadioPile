#!/usr/bin/env python3
"""
Class distribution figure for the RadioPile v1.0 data descriptor.

Two-panel, full-width figure:
  - Top panel:    technology-view classes (10), under PRIMARY_LABEL_TYPE="technology"
  - Bottom panel: modulation-view classes, under PRIMARY_LABEL_TYPE="modulation"
                  (modulation_type "unknown"/"noise" filtered out, matching the
                  baseline classifier's training population)

Horizontal bars, log-x (window count), stacked by SOURCE dataset so each bar
shows both total class size (length) and per-source composition (segments).

All counts are hardcoded from the verified per-class/per-source tables
(tab:class_composition and tab:class_composition_mod) so the figure is
self-contained and cannot drift from the paper. No HDF5 / TF dependency.

Numbers verified against live data/aggregation/processed/aggregated.h5.

Regenerate:  python scripts/paper_figures/class_distribution.py [--out-dir DIR]
Output:      figures/class_distribution.{pdf,png}
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT_DIR = Path("figures")

# ----------------------------------------------------------------------
# Source color palette. One stable color per source dataset, shared across
# both panels so the legend is global.
# ----------------------------------------------------------------------
SOURCE_ORDER = [
    "NIST WiFi",
    "RML2018",
    "RML2016",
    "RadChar",
    "Sub-GHz",
    "LTE/WiFi/DVB-T",
    "ITS Band",
]
SOURCE_COLORS = {
    "NIST WiFi":     "#2c5f8a",  # deep blue  (the dominant real-world source)
    "RML2018":        "#c44e52",  # red        (large simulated modulation set)
    "RML2016":        "#dd8452",  # orange     (smaller simulated modulation set)
    "RadChar":        "#8172b3",  # purple     (radar; appears only in radar view, kept for legend completeness)
    "Sub-GHz":        "#55a868",  # green      (narrowband IoT)
    "LTE/WiFi/DVB-T": "#937860",  # brown      (broadband OFDM)
    "ITS Band":        "#da8bc3",  # pink       (V2X)
}

# ----------------------------------------------------------------------
# TECHNOLOGY VIEW  (tab:class_composition)
# Each class maps to a dict of {source: window_count}. LTE/WiFi/DVB-T 1 MHz
# and 10 MHz captures are merged into the single LTE/WiFi/DVB-T source here
# (the per-rate split is a table detail, not needed for the composition view).
# ----------------------------------------------------------------------
TECH = {
    "WiFi": {
        "NIST WiFi": 20_627_827,
        "Sub-GHz": 574_553,
        "LTE/WiFi/DVB-T": 96_800 + 19_800,
        "ITS Band": 15_000,
    },
    "LTE": {
        "LTE/WiFi/DVB-T": 132_000 + 24_978,
        "ITS Band": 15_000,
    },
    "DVB-T": {
        "LTE/WiFi/DVB-T": 130_766 + 11_000,
    },
    "IEEE 802.15.4": {"Sub-GHz": 102_103},
    "LoRa": {"Sub-GHz": 101_838},
    "Noise": {"Sub-GHz": 81_920, "ITS Band": 15_000},
    "Sigfox": {"Sub-GHz": 66_738},
    "5G NR": {"ITS Band": 15_000},
    "C-V2X PC5": {"ITS Band": 15_000},
    "ITS-G5": {"ITS Band": 15_000},
}

# ----------------------------------------------------------------------
# MODULATION VIEW  (tab:class_composition_mod)
# Named multi-source classes, then the two uniform RML-only groups expanded
# into individual classes (they render as flat shelves, visually showing the
# "tail is all RML" point).
# ----------------------------------------------------------------------
MOD = {
    "256QAM": {"NIST WiFi": 11_068_353, "RML2018": 106_496},
    "64QAM": {"NIST WiFi": 9_559_474, "Sub-GHz": 226_220,
              "RML2018": 106_496, "RML2016": 20_000},
    "BPSK": {"Sub-GHz": 415_071, "RML2018": 106_496, "RML2016": 20_000},
    "OQPSK": {"RML2018": 106_496, "Sub-GHz": 69_348},
    "16QAM": {"RML2018": 106_496, "RML2016": 20_000, "Sub-GHz": 8_090},
    "8PSK": {"RML2018": 106_496, "RML2016": 20_000},
    "QPSK": {"RML2018": 106_496, "RML2016": 20_000},
    "CSS": {"Sub-GHz": 101_838},
    "BFSK": {"Sub-GHz": 24_665},
}
# RML2018-only classes (17), 106,496 windows each
RML2018_ONLY = [
    "128APSK", "128QAM", "16APSK", "16PSK", "32APSK", "32PSK", "32QAM",
    "4ASK", "64APSK", "8ASK", "AM-DSB-SC", "AM-DSB-WC", "AM-SSB-SC",
    "AM-SSB-WC", "FM", "GMSK", "OOK",
]
for _c in RML2018_ONLY:
    MOD[_c] = {"RML2018": 106_496}
# RML2016-only classes (6), 20,000 windows each
RML2016_ONLY = ["AM-DSB", "AM-SSB", "CPFSK", "GFSK", "PAM4", "WBFM"]
for _c in RML2016_ONLY:
    MOD[_c] = {"RML2016": 20_000}


def class_total(d):
    return sum(d.values())


def draw_panel(ax, data, title):
    """Horizontal stacked bars, sorted by total descending, log-x."""
    classes = sorted(data.keys(), key=lambda c: class_total(data[c]))  # ascending -> largest on top
    y = np.arange(len(classes))
    # log-x needs a positive left edge; start stacks at 1 window.
    left0 = 1.0
    for i, cls in enumerate(classes):
        left = left0
        # stack sources in the global order for consistent coloring
        for src in SOURCE_ORDER:
            if src in data[cls]:
                w = data[cls][src]
                ax.barh(y[i], w, left=left, height=0.72,
                        color=SOURCE_COLORS[src], edgecolor="white", linewidth=0.3)
                left += w
    ax.set_yticks(y)
    ax.set_yticklabels(classes, fontsize=7)
    ax.set_xscale("log")
    ax.set_xlim(left=left0)
    ax.set_xlabel("Windows (log scale)", fontsize=8)
    ax.set_title(title, fontsize=9, loc="left", fontweight="bold")
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", which="major", color="0.85", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR,
                        help=f"output directory (default: {OUT_DIR})")
    out_dir = parser.parse_args().out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7.0, 8.2),
        gridspec_kw={"height_ratios": [len(TECH), len(MOD)]},
    )

    draw_panel(ax_top, TECH, "Technology view")
    draw_panel(ax_bot, MOD, "Modulation view")

    # Global legend: only sources that actually appear in either panel
    used = set()
    for d in (TECH, MOD):
        for cls in d.values():
            used.update(cls.keys())
    legend_sources = [s for s in SOURCE_ORDER if s in used]
    handles = [Patch(facecolor=SOURCE_COLORS[s], label=s) for s in legend_sources]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out_pdf = out_dir / "class_distribution.pdf"
    out_png = out_dir / "class_distribution.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"wrote {out_pdf} and {out_png}")
    print(f"technology classes: {len(TECH)} | modulation classes: {len(MOD)}")
    print(f"tech total windows: {sum(class_total(d) for d in TECH.values()):,}")
    print(f"mod total windows:  {sum(class_total(d) for d in MOD.values()):,}")


if __name__ == "__main__":
    main()
