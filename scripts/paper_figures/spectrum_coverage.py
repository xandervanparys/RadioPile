"""
RadioPile v1.0 -- Frequency band coverage figure.

Generates a clean IEEE-style figure showing the frequency coverage of the seven
source datasets, with synthetic/baseband datasets visually recessed to avoid
implying they span the full frequency axis.

Color palette matches the RadioPile pipeline architecture figure.

Regenerate:  python scripts/paper_figures/spectrum_coverage.py [--out-dir DIR]
Output:      figures/spectrum_coverage.{pdf,png}
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

OUT_DIR = Path("figures")

# ---------------------------------------------------------------------------
# Color palette (matched to pipeline_architecture.png)
# ---------------------------------------------------------------------------
# Body fills are light; outlines are dark
COLOR_TECH_FILL    = "#A9C0D9"   # steel blue   (matches 'Raw RF Datasets')
COLOR_TECH_EDGE    = "#3E5871"
COLOR_MOD_FILL     = "#C9B8D8"   # dusty purple (matches 'Processed Datasets')
COLOR_MOD_EDGE     = "#5A4770"
COLOR_RADAR_FILL   = "#E0A8A8"   # muted coral  (matches 'Aggregation & Label Routing')
COLOR_RADAR_EDGE   = "#8B3A3A"

# Task-type background bands (very subtle)
COLOR_BG_RADAR     = "#FAF0F0"
COLOR_BG_MOD       = "#F6F2FA"
COLOR_BG_TECH      = "#F2F6FB"

# Text
COLOR_TEXT         = "#222222"
COLOR_AXIS         = "#555555"
COLOR_GRID         = "#D8D8D8"

# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------
# Each real-capture band: (f_low_MHz, f_high_MHz, label)
# Synthetic datasets carry no band -- they occupy a full-width recessed strip.

DATASETS = [
    # name,        task_type,   synthetic, bands (MHz),                  band_labels
    ("RadChar",          "radar",   True,  [],                             []),
    ("RML2018.01A",      "mod",     True,  [],                             []),
    ("RML2016.10A",      "mod",     True,  [],                             []),
    ("ITS Band (V2X)",    "tech",    False, [(5850, 5925)],                 ["5.9 GHz ITS"]),
    ("NIST WiFi",        "tech",    False, [(2400, 2483), (5730, 5835)],   ["2.437 GHz", "5.825 GHz"]),
    ("LTE / WiFi / DVB-T","tech",   False, [(470, 790), (800, 862), (2400, 2483)],
                                                                           ["DVB-T", "LTE", "WiFi"]),
    ("Sub-GHz",          "tech",    False, [(863, 869)],                   ["863 to 869 MHz"]),
]

TASK_STYLE = {
    "tech":  (COLOR_TECH_FILL,  COLOR_TECH_EDGE,  COLOR_BG_TECH),
    "mod":   (COLOR_MOD_FILL,   COLOR_MOD_EDGE,   COLOR_BG_MOD),
    "radar": (COLOR_RADAR_FILL, COLOR_RADAR_EDGE, COLOR_BG_RADAR),
}

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR,
                        help=f"output directory (default: {OUT_DIR})")
    out_dir = parser.parse_args().out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------------
    # Figure setup
    # ---------------------------------------------------------------------------
    # Single-column IEEE figure is ~3.5 in wide. This will likely span a column or
    # slightly more; regenerate with different figsize for two-column layout.
    fig, ax = plt.subplots(figsize=(9.0, 4.2), dpi=200)

    # X axis: log scale, 100 MHz to 7 GHz
    F_LOW  = 100      # MHz
    F_HIGH = 6500     # MHz
    ax.set_xscale("log")
    ax.set_xlim(F_LOW, F_HIGH)

    # Y positions: one row per dataset (top to bottom order as in DATASETS)
    n = len(DATASETS)
    row_height = 1.0
    y_positions = np.arange(n) * row_height
    # DATASETS[0] (RadChar) goes at the top of the plot, so flip:
    y_positions = y_positions[::-1]

    ax.set_ylim(-0.5, n - 0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([d[0] for d in DATASETS], fontsize=9, color=COLOR_TEXT)

    # ---------------------------------------------------------------------------
    # Task-type background bands
    # ---------------------------------------------------------------------------
    # Group contiguous rows with the same task type and shade behind them.
    task_sequence = [d[1] for d in DATASETS]
    i = 0
    while i < n:
        j = i
        while j < n and task_sequence[j] == task_sequence[i]:
            j += 1
        # rows i..j-1 share a task
        y_top = y_positions[i] + 0.5
        y_bot = y_positions[j - 1] - 0.5
        bg_color = TASK_STYLE[task_sequence[i]][2]
        ax.axhspan(y_bot, y_top, facecolor=bg_color, edgecolor="none", zorder=0)
        i = j

    # ---------------------------------------------------------------------------
    # Draw bars
    # ---------------------------------------------------------------------------
    BAR_HALF_HEIGHT_REAL = 0.28
    BAR_HALF_HEIGHT_SYN  = 0.12   # thinner for synthetic datasets

    for idx, (name, task, synthetic, bands, band_labels) in enumerate(DATASETS):
        y = y_positions[idx]
        fill, edge, _ = TASK_STYLE[task]

        if synthetic:
            # Recessed full-width strip, low opacity, very thin, no strong edge.
            # This is deliberately visually quiet so it does not imply coverage of
            # the frequency axis.
            rect = Rectangle(
                (F_LOW, y - BAR_HALF_HEIGHT_SYN),
                F_HIGH - F_LOW,
                2 * BAR_HALF_HEIGHT_SYN,
                facecolor=fill,
                edgecolor=edge,
                linewidth=0.4,
                alpha=0.28,
                hatch="////",
                zorder=2,
            )
            # Apply hatching with the edge color at low alpha: set hatch color
            rect.set_hatch("////")
            ax.add_patch(rect)
            # Annotation: "no defined carrier (synthetic baseband)"
            ax.text(
                np.sqrt(F_LOW * F_HIGH),  # geometric center on log axis
                y,
                "no defined carrier (synthetic baseband)",
                ha="center", va="center",
                fontsize=7.5, style="italic", color=edge, alpha=0.9,
                zorder=3,
            )
        else:
            # Real-capture bands: one rectangle per band.
            # For bands close to each other on the log axis, stagger labels vertically
            # above the bar (not below, to avoid bleeding into the row below).
            for bi, ((f_lo, f_hi), label) in enumerate(zip(bands, band_labels)):
                # Ensure minimum visual width on log scale so narrow bands don't vanish
                log_width = np.log10(f_hi) - np.log10(f_lo)
                min_log_width = 0.025  # ensures visibility of very narrow bands
                if log_width < min_log_width:
                    center = np.sqrt(f_lo * f_hi)
                    half = 10 ** (min_log_width / 2)
                    f_lo_draw = center / half
                    f_hi_draw = center * half
                else:
                    f_lo_draw, f_hi_draw = f_lo, f_hi

                rect = Rectangle(
                    (f_lo_draw, y - BAR_HALF_HEIGHT_REAL),
                    f_hi_draw - f_lo_draw,
                    2 * BAR_HALF_HEIGHT_REAL,
                    facecolor=fill,
                    edgecolor=edge,
                    linewidth=0.8,
                    alpha=0.95,
                    zorder=3,
                )
                ax.add_patch(rect)

                # Labels always above. Stagger (higher) for adjacent bands.
                center = np.sqrt(f_lo_draw * f_hi_draw)
                close_to_prev = False
                if bi > 0:
                    prev_hi = bands[bi - 1][1]
                    if f_lo / prev_hi < 1.3:
                        close_to_prev = True
                base_offset = 0.06
                stagger_offset = 0.20
                y_text = y + BAR_HALF_HEIGHT_REAL + (stagger_offset if close_to_prev else base_offset)

                ax.text(
                    center, y_text, label,
                    ha="center", va="bottom",
                    fontsize=6.8, color=edge, alpha=0.95,
                    zorder=4,
                )

    # ---------------------------------------------------------------------------
    # X axis formatting
    # ---------------------------------------------------------------------------
    xticks_mhz = [100, 300, 500, 1000, 2400, 5800]
    xticks_labels = ["100 MHz", "300 MHz", "500 MHz", "1 GHz", "2.4 GHz", "5.8 GHz"]
    ax.set_xticks(xticks_mhz)
    ax.set_xticklabels(xticks_labels, fontsize=8.5, color=COLOR_AXIS)
    ax.minorticks_off()

    ax.set_xlabel("Frequency (log scale)", fontsize=9.5, color=COLOR_TEXT, labelpad=6)

    # Grid: only vertical, very subtle
    ax.xaxis.grid(True, which="major", color=COLOR_GRID, linewidth=0.5, linestyle="-", zorder=1)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)

    # Spines: keep it clean
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLOR_AXIS)
        ax.spines[spine].set_linewidth(0.8)

    ax.tick_params(axis="x", colors=COLOR_AXIS, length=3, width=0.7)
    ax.tick_params(axis="y", length=0, pad=6)

    # ---------------------------------------------------------------------------
    # Legend -- placed BELOW the plot, outside the axes
    # ---------------------------------------------------------------------------
    legend_handles = [
        mpatches.Patch(facecolor=COLOR_TECH_FILL,  edgecolor=COLOR_TECH_EDGE,
                       linewidth=0.8, label="Technology recognition"),
        mpatches.Patch(facecolor=COLOR_MOD_FILL,   edgecolor=COLOR_MOD_EDGE,
                       linewidth=0.8, label="Modulation classification"),
        mpatches.Patch(facecolor=COLOR_RADAR_FILL, edgecolor=COLOR_RADAR_EDGE,
                       linewidth=0.8, label="Radar classification"),
        mpatches.Patch(facecolor="#CCCCCC", edgecolor="#666666",
                       linewidth=0.4, alpha=0.4, hatch="////",
                       label="Synthetic / baseband (no carrier)"),
    ]

    leg = ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=4,
        frameon=False,
        fontsize=8.5,
        handlelength=1.6,
        handleheight=1.0,
        columnspacing=1.6,
    )
    for text in leg.get_texts():
        text.set_color(COLOR_TEXT)

    # ---------------------------------------------------------------------------
    # Task-type side labels (small, on the right margin)
    # ---------------------------------------------------------------------------
    task_group_labels = {"radar": "Radar", "mod": "Modulation", "tech": "Technology"}
    i = 0
    while i < n:
        j = i
        while j < n and task_sequence[j] == task_sequence[i]:
            j += 1
        y_mid = (y_positions[i] + y_positions[j - 1]) / 2
        _, edge, _ = TASK_STYLE[task_sequence[i]]
        ax.text(
            F_HIGH * 1.06, y_mid,
            task_group_labels[task_sequence[i]],
            ha="left", va="center",
            fontsize=8, color=edge, alpha=0.9,
            rotation=90,
            fontweight="semibold",
        )
        i = j

    # ---------------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------------
    plt.tight_layout()
    out_pdf = out_dir / "spectrum_coverage.pdf"
    out_png = out_dir / "spectrum_coverage.png"
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.1)
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0.1, dpi=300)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
