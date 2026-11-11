"""
Purpose of this file:
    Draw the explanatory figures for the registration README.
    Each drawn CT pose is produced by register_tool() itself, so
    the figures show what the code actually computes.

    figures/registration_concept.png  - method overview (2D)
    figures/registration_tests.png    - the three test cases (3D)

Run with:  python3 make_figures.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  

from registration import register_tool 

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tool_calibration"))
from geometry import rotation_matrix  
from simulation import MARKERS, P_TRUE, V_TRUE  

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

MARKER_C = "#1b9e77"
TIP_C = "#d62728"
AXIS_C = "#ff7f0e"
SHAFT_C = "#4a6fa5"
GHOST_C = "#9a9a9a"

SHAFT_TOP = np.array([0.0, 26.0, 0.0])   # top of the needle shaft (model)
ARM_JOINT = np.array([0.0, 18.0, 0.0])   # where the marker arm meets the shaft


def tool_points(R, t):
    """Model-frame tool features mapped by (R, t)."""
    f = lambda p: R @ p + t  # noqa: E731
    return {
        "markers": MARKERS @ R.T + t,
        "tip": f(P_TRUE),
        "top": f(SHAFT_TOP),
        "joint": f(ARM_JOINT),
        "axis": R @ V_TRUE,
    }


# Figure 1: concept

def draw_tool_2d(ax, pts, label_suffix, ghost=False):
    c_shaft = GHOST_C if ghost else SHAFT_C
    c_mark = GHOST_C if ghost else MARKER_C
    m = pts["markers"]
    ax.plot(*zip(pts["tip"][:2], pts["top"][:2]), color=c_shaft, lw=4,
            solid_capstyle="round", zorder=1)
    ax.plot(*zip(pts["joint"][:2], m[0][:2]), color=c_shaft, lw=2, zorder=1)
    tri = np.vstack([m, m[:1]])
    ax.plot(tri[:, 0], tri[:, 1], color=c_mark, lw=1.2, zorder=2)
    ax.scatter(m[:, 0], m[:, 1], s=90, color=c_mark, edgecolor="k", zorder=3)
    if ghost:
        return
    for i, p in enumerate(m):
        ax.annotate(f"M{i + 1}{label_suffix}", p[:2], xytext=(6, 4),
                    textcoords="offset points", fontsize=10)
    ax.scatter(*pts["tip"][:2], s=70, color=TIP_C, zorder=4)
    ax.annotate(f"P{label_suffix}", pts["tip"][:2], xytext=(-44, -4),
                textcoords="offset points", color=TIP_C, fontsize=11,
                fontweight="bold")
    a = pts["axis"] * 9
    ax.annotate("", xy=pts["tip"][:2] + a[:2], xytext=pts["tip"][:2],
                arrowprops=dict(arrowstyle="-|>", color=AXIS_C, lw=2.5))
    ax.annotate(f"v{label_suffix}", pts["tip"][:2] + a[:2], xytext=(6, -2),
                textcoords="offset points", color=AXIS_C, fontsize=11,
                fontweight="bold")


def draw_frame(ax, origin, name, length=8):
    ax.annotate("", xy=origin + [length, 0], xytext=origin,
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2))
    ax.annotate("", xy=origin + [0, length], xytext=origin,
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2))
    ax.text(*(origin + [length + 1, -1]), "x", fontsize=9)
    ax.text(*(origin + [-1.5, length + 1]), "y", fontsize=9)
    ax.text(*(origin + [-6, -4]), name, fontsize=11, fontweight="bold")


def figure_concept():
    # A planar pose so everything can be drawn in the x-y plane.
    R = rotation_matrix([0, 0, 1], np.radians(35))
    t = np.array([45.0, 12.0, 0.0])
    ct_markers = MARKERS @ R.T + t
    reg = register_tool(MARKERS, P_TRUE, V_TRUE, ct_markers)
    model = tool_points(np.eye(3), np.zeros(3))
    ct = tool_points(reg["R"], reg["t"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2),
                             gridspec_kw=dict(width_ratios=[1, 1.5]))

    ax = axes[0]
    draw_frame(ax, np.array([0.0, 0.0]), "")
    draw_tool_2d(ax, model, "tool")
    ax.set_title("Inputs in the tool model frame\n(from Part 1 calibration)",
                 fontsize=12)
    ax.text(-14, -16,
            "M1tool=(2,18,0)  M2tool=(8,18,0)  M3tool=(2,24,0)\n"
            "Ptool=(0,0,0)      vtool=(0,-1,0)", fontsize=9, family="monospace")
    ax.set_xlim(-16, 22)
    ax.set_ylim(-19, 30)

    ax = axes[1]
    draw_frame(ax, np.array([0.0, 0.0]), "F_ct")
    draw_tool_2d(ax, model, "", ghost=True)
    ax.text(-3, 28, "tool model pose\n(for reference)", color=GHOST_C, fontsize=8)
    draw_tool_2d(ax, ct, "ct")
    # translation arrow: origin -> P_ct
    ax.annotate("", xy=reg["P_ct"][:2], xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="purple", lw=1.5, ls="--"))
    ax.text(*(reg["P_ct"][:2] * 0.5 + [-2, -5]), "t", color="purple",
            fontsize=12, fontweight="bold")
    # "direction is free": the same v drawn at the origin
    vv = reg["v_ct"][:2] * 9
    ax.annotate("", xy=vv + [66, -2], xytext=(66, -2),
                arrowprops=dict(arrowstyle="-|>", color=AXIS_C, lw=2.5, alpha=0.5))
    ax.text(70, -6, "same v_ct drawn\nelsewhere: a direction\nhas no position,\n"
            "so t is not added", fontsize=8, color=AXIS_C)
    ax.set_title("Registration result in the CT frame", fontsize=12)
    ax.text(-14, -19,
            "1) [R, t] = SVD fit of  M_ct = R·M_tool + t  (3 markers)\n"
            "2) P_ct = R·P_tool + t      tip = point:     rotate + translate\n"
            "3) v_ct = R·v_tool          axis = direction: rotate only",
            fontsize=10, family="monospace",
            bbox=dict(boxstyle="round", fc="#f4f4f4", ec="#bbbbbb"))
    ax.set_xlim(-16, 82)
    ax.set_ylim(-22, 45)

    for ax in axes:
        ax.set_aspect("equal")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "registration_concept.png", dpi=150)
    plt.close(fig)


# Figure 2: test cases in 3D

TEST_CASES = [
    ("Test 1: pure translation\nR = I,  t = (50, -20, 100)",
     np.array([[52, -2, 100], [58, -2, 100], [52, 4, 100]], float),
     "Pct = (50, -20, 100)\nvct = (0, -1, 0)"),
    ("Test 2: pure rotation\nRz(+90°),  t = 0",
     np.array([[-18, 2, 0], [-18, 8, 0], [-24, 2, 0]], float),
     "Pct = (0, 0, 0)\nvct = (1, 0, 0)"),
    ("Test 3: rotation + translation\nRx(+90°),  t = (50, -20, 100)",
     np.array([[52, -20, 118], [58, -20, 118], [52, -20, 124]], float),
     "Pct = (50, -20, 100)\nvct = (0, 0, -1)"),
]


def draw_tool_3d(ax, pts, ghost=False):
    c_shaft = GHOST_C if ghost else SHAFT_C
    c_mark = GHOST_C if ghost else MARKER_C
    ls = "--" if ghost else "-"
    m = pts["markers"]
    ax.plot(*zip(pts["tip"], pts["top"]), color=c_shaft, lw=3, ls=ls)
    ax.plot(*zip(pts["joint"], m[0]), color=c_shaft, lw=1.5, ls=ls)
    tri = np.vstack([m, m[:1]])
    ax.plot(tri[:, 0], tri[:, 1], tri[:, 2], color=c_mark, lw=1, ls=ls)
    ax.scatter(*m.T, s=30, color=c_mark, edgecolor="k", depthshade=False)
    if not ghost:
        ax.scatter(*pts["tip"], s=40, color=TIP_C, depthshade=False)
        ax.quiver(*pts["tip"], *(pts["axis"] * 15), color=AXIS_C, lw=2.5,
                  arrow_length_ratio=0.3)
        ax.text(*(pts["tip"] + pts["axis"] * 17), "vct", color=AXIS_C,
                fontsize=9, fontweight="bold")
        ax.text(*pts["tip"], "  Pct", color=TIP_C, fontsize=9,
                fontweight="bold")


def figure_tests():
    fig = plt.figure(figsize=(15, 7.2))
    model = tool_points(np.eye(3), np.zeros(3))
    for k, (title, ct_markers, expected) in enumerate(TEST_CASES):
        reg = register_tool(MARKERS, P_TRUE, V_TRUE, ct_markers)
        ct = tool_points(reg["R"], reg["t"])

        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        draw_tool_3d(ax, model, ghost=True)
        draw_tool_3d(ax, ct)
        fig.text((k + 0.5) / 3, 0.07,
                 f"Predicted on paper:\n{expected}\n"
                 f"Computed:\nPct = {np.round(reg['P_ct'], 6) + 0.0}\n"
                 f"vct = {np.round(reg['v_ct'], 6) + 0.0}",
                 ha="center", fontsize=9, family="monospace",
                 bbox=dict(boxstyle="round", fc="white", ec="#bbbbbb"))
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x_ct (mm)")
        ax.set_ylabel("y_ct (mm)")
        ax.set_zlabel("z_ct (mm)")

        # equal scaling so rotations look like rotations
        allp = np.vstack([model["markers"], ct["markers"], model["tip"],
                          ct["tip"], model["top"], ct["top"]])
        mid = (allp.max(0) + allp.min(0)) / 2
        half = (allp.max(0) - allp.min(0)).max() / 2 + 5
        ax.set_xlim(mid[0] - half, mid[0] + half)
        ax.set_ylim(mid[1] - half, mid[1] + half)
        ax.set_zlim(mid[2] - half, mid[2] + half)
        ax.view_init(elev=22, azim=-60)

    fig.text(0.5, 0.01,
             "Grey dashed = tool in its model pose.  Colour = registered tool "
             "in CT (green markers, red tip Pct, orange axis vct).",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.22, 1, 1))
    fig.savefig(FIG_DIR / "registration_tests.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    figure_concept()
    figure_tests()
    print("Figures written to", FIG_DIR)
