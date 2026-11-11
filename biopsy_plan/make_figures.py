"""
Purpose of this file:
    Draw the explanatory figures for the biopsy-plan README.
    All numbers in the figures come from compute_biopsy_plan().

    figures/biopsy_geometry.png   - annotated geometry of the plan
    figures/biopsy_scenarios.png  - the test scenarios side by side

Run with:  python3 make_figures.py
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from biopsy_plan import compute_biopsy_plan  # noqa: E402

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

TUMOR_FC = "#f2c38b"
TUMOR_EC = "#b5651d"
NEEDLE_C = "#4a6fa5"
CORE_C = "#d62728"
TRAJ_C = "#555555"
D_C = "#7b3294"


def dim_arrow(ax, x, y0, y1, label, color="k", side="left", fs=10):
    """Vertical double-headed dimension arrow at x from y0 to y1."""
    ax.annotate("", xy=(x, y0), xytext=(x, y1),
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.3))
    if side == "on":
        # label sits on the arrow, with a white box behind it
        ax.text(x, (y0 + y1) / 2, label, color=color, ha="center",
                va="center", fontsize=fs,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))
        return
    ha = "right" if side == "left" else "left"
    dx = -0.6 if side == "left" else 0.6
    ax.text(x + dx, (y0 + y1) / 2, label, color=color, ha=ha, va="center",
            fontsize=fs)


# Figure 1: annotated geometry

def figure_geometry():
    # Schematic 2D scene in the plane containing the needle axis
    # and the tumour centre. Needle points down (-y), as in Fig. 6.
    P = np.array([0.0, 30.0, 0.0])
    u = np.array([0.0, -1.0, 0.0])
    C = np.array([6.0, 0.0, 0.0])
    R = 10.0
    plan = compute_biopsy_plan(C, R, P, u)
    s_c = plan["closest_depth"]
    d = plan["targeting_error"]
    entry, exit_ = plan["entry_point"], plan["exit_point"]

    fig, ax = plt.subplots(figsize=(9, 9))

    # skin surface and needle above the tip
    xs = np.linspace(-22, 22, 200)
    ax.plot(xs, 30 + 0.004 * xs**2, color="#c49a6c", lw=2)
    ax.text(14, 32.5, "skin surface", color="#8c6239", fontsize=10)
    ax.plot([0, 0], [30, 44], color=NEEDLE_C, lw=6, solid_capstyle="butt")
    ax.text(1.2, 41, "biopsy needle\n(current pose)", color=NEEDLE_C, fontsize=10)

    # planned trajectory
    ax.plot([0, 0], [30, -16], color=TRAJ_C, ls="--", lw=1.2)
    ax.annotate("", xy=(0, 23), xytext=(0, 30),
                arrowprops=dict(arrowstyle="-|>", color="#ff7f0e", lw=2.5))
    ax.text(0.8, 25.5, "u = v_ct / |v_ct|", color="#ff7f0e", fontsize=10)

    # tumour
    ax.add_patch(Circle(C[:2], R, fc=TUMOR_FC, ec=TUMOR_EC, lw=2, alpha=0.9))
    ax.scatter(*C[:2], color=TUMOR_EC, zorder=5)
    ax.text(C[0] + 0.6, C[1] - 1.8, "C_tum", color=TUMOR_EC, fontsize=11,
            fontweight="bold")
    ax.text(C[0] + 5, C[1] - 7.5, "R_tum", color=TUMOR_EC, fontsize=10)
    ax.plot([C[0], C[0] + R * np.cos(-0.9)], [C[1], C[1] + R * np.sin(-0.9)],
            color=TUMOR_EC, lw=1)

    # core segment
    ax.plot([entry[0], exit_[0]], [entry[1], exit_[1]], color=CORE_C, lw=5,
            solid_capstyle="butt", zorder=4)
    for pt, name in ((entry, "entry"), (exit_, "exit")):
        ax.scatter(*pt[:2], color=CORE_C, s=45, zorder=6)
        ax.text(pt[0] - 0.8, pt[1], name, color=CORE_C, ha="right",
                va="center", fontsize=10)

    # tip
    ax.scatter(*P[:2], color="k", s=50, zorder=6)
    ax.text(-0.8, 30.8, "P_ct (tip)", ha="right", fontsize=11, fontweight="bold")

    # closest point and targeting error d
    foot = P + s_c * u
    ax.plot([foot[0], C[0]], [foot[1], C[1]], color=D_C, lw=2)
    ax.plot([0, 0.9, 0.9], [0.9, 0.9, 0], color=D_C, lw=1)  # right-angle mark
    ax.text(3, 1.0, "d", color=D_C, fontsize=13, fontweight="bold")
    ax.text(C[0] + 2.2, 8.5,
            "targeting error d\n= distance from C\n   to the needle axis",
            color=D_C, fontsize=9)

    # half chord h
    dim_arrow(ax, 1.5, entry[1], foot[1], "h", color=CORE_C, side="right")
    dim_arrow(ax, 1.5, foot[1], exit_[1], "h", color=CORE_C, side="right")

    # depths on the left
    dim_arrow(ax, -4, P[1], foot[1], f"s_c\n= {s_c:.0f}", side="on", fs=9)
    dim_arrow(ax, -12, P[1], entry[1], f"entry depth\ns_c - h = {s_c - plan['core_length'] / 2:.0f}",
              side="left", fs=9)
    dim_arrow(ax, -12, entry[1], exit_[1],
              f"tissue core\n2h = {plan['core_length']:.0f}",
              color=CORE_C, side="left", fs=9)
    dim_arrow(ax, -20, P[1], exit_[1],
              f"insertion\ndepth\ns_c + h\n= {plan['insertion_depth']:.0f}",
              side="left", fs=9)

    ax.text(-28, -22,
            "s_c = (C - P)·u              where the axis passes closest to C\n"
            "d   = |(C - P) - s_c·u|       targeting error\n"
            "h   = sqrt(R² - d²)          half chord  (only if d ≤ R)\n"
            f"Example drawn: d = {d:.0f}, R = {R:.0f}  →  h = 8 (a 6-8-10 triangle)",
            fontsize=9.5, family="monospace",
            bbox=dict(boxstyle="round", fc="#f4f4f4", ec="#bbbbbb"))

    ax.set_xlim(-30, 24)
    ax.set_ylim(-26, 46)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Biopsy plan geometry (plane containing the needle axis "
                 "and the tumor centre)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "biopsy_geometry.png", dpi=150)
    plt.close(fig)


# Figure 2: scenarios

SCENARIOS = [
    ("Intersect (through centre)", [0.0, 10.0, 0.0], [0, -1, 0]),
    ("Intersect (off centre)", [0.6, 10.0, 0.0], [0, -1, 0]),
    ("Tangent (touches)", [1.0, 10.0, 0.0], [0, -1, 0]),
    ("Miss", [2.0, 10.0, 0.0], [0, -1, 0]),
    ("Tumor behind tip", [0.0, 10.0, 0.0], [0, 1, 0]),
]


def figure_scenarios():
    C = np.zeros(3)
    R = 1.0
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(16, 5.6))

    for ax, (title, P, v) in zip(axes, SCENARIOS):
        P = np.array(P)
        v = np.array(v, float)
        plan = compute_biopsy_plan(C, R, P, v)

        ax.add_patch(Circle((0, 0), R, fc=TUMOR_FC, ec=TUMOR_EC, lw=1.5))
        ax.scatter(0, 0, color=TUMOR_EC, s=12, zorder=5)

        # The tip is 10 mm away, so the view is zoomed on the tumor
        # and the tip is shown as an off-panel marker at the top.
        top = 3.0
        if v[1] < 0:
            ax.plot([P[0], P[0]], [top, -1.8], color=TRAJ_C, ls="--", lw=1.2)
            ax.annotate("", xy=(P[0], top - 0.6), xytext=(P[0], top),
                        arrowprops=dict(arrowstyle="-|>", color="#ff7f0e", lw=2))
            note = f"tip P:\n{P[1]:g} mm above"
        else:
            note = f"tip P:\n{P[1]:g} mm above,\npoints away"
            ax.annotate("", xy=(P[0], top), xytext=(P[0], top - 0.6),
                        arrowprops=dict(arrowstyle="-|>", color="#ff7f0e", lw=2))
        ax.text(-2.1, top - 0.05, note, fontsize=8, va="top", color="#444444")

        # targeting-error segment (C to the axis)
        if plan["targeting_error"] > 1e-9:
            ax.plot([0, P[0]], [0, 0], color=D_C, lw=1.8)

        if plan["core_length"] > 0:
            e, x = plan["entry_point"], plan["exit_point"]
            ax.plot([e[0], x[0]], [e[1], x[1]], color=CORE_C, lw=4,
                    solid_capstyle="butt", zorder=4)
        elif plan["scenario"] == "tangent":
            ax.scatter(*plan["entry_point"][:2], color=CORE_C, s=40, zorder=6)

        depth = plan["insertion_depth"]
        depth_s = "N/A" if depth is None else f"{depth:.2f} mm"
        ax.set_title(title, fontsize=11)
        ax.text(0.5, -0.02,
                f"P = ({P[0]:g}, {P[1]:g}, {P[2]:g})\n"
                f"v = ({v[0]:g}, {v[1]:g}, {v[2]:g})\n\n"
                f"scenario: {plan['scenario']}\n"
                f"error d : {plan['targeting_error']:.2f} mm\n"
                f"core    : {plan['core_length']:.2f} mm\n"
                f"depth   : {depth_s}",
                transform=ax.transAxes, ha="center", va="top", fontsize=9.5,
                family="monospace")
        ax.set_xlim(-2.2, 3.2)
        ax.set_ylim(-1.8, 3.2)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#cccccc")

    fig.suptitle("Biopsy-plan test scenarios: unit tumor at C = (0, 0, 0), "
                 "R = 1 mm  (red = harvested core, purple = targeting error)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.27, 1, 0.94))
    fig.savefig(FIG_DIR / "biopsy_scenarios.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    figure_geometry()
    figure_scenarios()
    print("Figures written to", FIG_DIR)
