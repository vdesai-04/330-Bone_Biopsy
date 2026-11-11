"""
Purpose of this file:
    Validate the biopsy-plan computation (biopsy_plan.py) and the
    geometry utilities (line_geometry.py).

    Ground-truth scene (simple but general enough, as the
    assignment suggests):
        Tumor   : unit sphere, C = (0, 0, 0), R = 1 mm
        Needle  : tip P = (x0, 10, 0), direction v = (0, -1, 0)
                  (pointing "down" toward the tumor as in Fig. 6)
    Moving the tip sideways by x0 sweeps the needle axis across
    the tumor, so x0 is exactly the targeting error.

        x0 = 0    through the centre     d=0, core 2,   depth 11
        x0 = 0.6  off-centre             d=0.6, core 1.6, depth 10.8
        x0 = 1    touches (tangent)      d=1, core 0,   depth 10
        x0 = 2    misses                 d=2, core 0,   depth N/A

    Off-centre case by hand (a 3-4-5 triangle scaled by 0.2):
        h = sqrt(1 - 0.6^2) = 0.8,   s_c = 10
        entry 10 - 0.8 = 9.2,  exit 10 + 0.8 = 10.8
        core = 1.6,  depth = 10.8

    Extra cases: oblique needle, tip already inside the tumor,
    tumor behind the tip, near-tangent tolerance, invariance
    under any rigid motion of the whole scene, a brute-force
    check, and a registration -> plan end-to-end case.

Run with:  python3 test_biopsy_plan.py   (or pytest)
"""

import sys
from pathlib import Path

import numpy as np

from line_geometry import (
    distance_point_from_line,
    closest_point_on_line,
    intersect_line_sphere,
)
from biopsy_plan import compute_biopsy_plan, format_plan

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "registration"))
sys.path.insert(0, str(_ROOT / "tool_calibration"))

from registration import register_tool  # noqa: E402
from geometry import rotation_matrix  # noqa: E402
from simulation import MARKERS, P_TRUE, V_TRUE  # noqa: E402

TOL = 1e-9
np.set_printoptions(precision=6, suppress=True)

C_UNIT = np.array([0.0, 0.0, 0.0])
R_UNIT = 1.0
V_DOWN = np.array([0.0, -1.0, 0.0])


def run_case(title, C, R, P, v, expected):
    """Compute the plan, print it next to the expectation, assert."""
    plan = compute_biopsy_plan(C, R, P, v)

    exp_depth = expected["insertion_depth"]
    exp_depth_str = "N/A" if exp_depth is None else f"{exp_depth:.4f} mm"

    print(f"\n--- {title} ---")
    print(f"Tumor C = {C}, R = {R} mm;  Needle P = {P}, v = {v}")
    print(format_plan(plan))
    print(f"Expected:         {expected['scenario']}, "
          f"error {expected['targeting_error']:.4f} mm, "
          f"core {expected['core_length']:.4f} mm, "
          f"depth {exp_depth_str}")

    assert plan["scenario"] == expected["scenario"]
    assert np.isclose(plan["targeting_error"], expected["targeting_error"], atol=TOL)
    assert np.isclose(plan["core_length"], expected["core_length"], atol=TOL)
    if exp_depth is None:
        assert plan["insertion_depth"] is None
    else:
        assert np.isclose(plan["insertion_depth"], exp_depth, atol=TOL)
    for key in ("entry_point", "exit_point"):
        if key in expected:
            assert np.allclose(plan[key], expected[key], atol=TOL)
    return plan


# Utility tests

def test_line_utilities():
    """Check the two geometry utilities on hand-worked examples."""
    # Point (3, 4, 0) from the x-axis: distance 4, foot (3, 0, 0).
    d = distance_point_from_line([3, 4, 0], [0, 0, 0], [1, 0, 0])
    foot, s = closest_point_on_line([3, 4, 0], [0, 0, 0], [2, 0, 0])
    assert np.isclose(d, 4.0) and np.allclose(foot, [3, 0, 0]) and np.isclose(s, 3.0)

    # Point on the line: distance 0.
    assert np.isclose(distance_point_from_line([5, 5, 5], [0, 0, 0], [1, 1, 1]), 0.0)

    # Line along x through a sphere of radius 5 centred at (0, 3, 0):
    # d = 3, h = 4, crossings at s = -4 and +4.
    kind, s_vals = intersect_line_sphere([0, 0, 0], [1, 0, 0], [0, 3, 0], 5)
    assert kind == "secant" and np.allclose(s_vals, [-4.0, 4.0])

    # Tolerance: tiny round-off from d = R still counts as tangent,
    # but a real 1 micrometre gap is a miss.
    kind, _ = intersect_line_sphere([1 - 1e-12, 10, 0], V_DOWN, C_UNIT, 1.0)
    assert kind == "tangent"
    kind, _ = intersect_line_sphere([1 + 1e-3, 10, 0], V_DOWN, C_UNIT, 1.0)
    assert kind == "miss"

    print("\n--- Geometry Utilities ---")
    print("distance_point_from_line, closest_point_on_line, "
          "intersect_line_sphere: passed")


# The three scenarios required by the assignment

def test_intersects_through_centre():
    run_case(
        "Intersect: Through Centre", C_UNIT, R_UNIT,
        np.array([0.0, 10.0, 0.0]), V_DOWN,
        dict(scenario="intersects", targeting_error=0.0,
             core_length=2.0, insertion_depth=11.0,
             entry_point=[0, 1, 0], exit_point=[0, -1, 0]))


def test_intersects_off_centre():
    run_case(
        "Intersect: Off Centre (d = 0.6)", C_UNIT, R_UNIT,
        np.array([0.6, 10.0, 0.0]), V_DOWN,
        dict(scenario="intersects", targeting_error=0.6,
             core_length=1.6, insertion_depth=10.8,
             entry_point=[0.6, 0.8, 0], exit_point=[0.6, -0.8, 0]))


def test_tangent():
    run_case(
        "Tangent: Needle Touches Tumor", C_UNIT, R_UNIT,
        np.array([1.0, 10.0, 0.0]), V_DOWN,
        dict(scenario="tangent", targeting_error=1.0,
             core_length=0.0, insertion_depth=10.0,
             entry_point=[1, 0, 0], exit_point=[1, 0, 0]))


def test_miss():
    run_case(
        "Miss: Needle Misses Tumor", C_UNIT, R_UNIT,
        np.array([2.0, 10.0, 0.0]), V_DOWN,
        dict(scenario="miss", targeting_error=2.0,
             core_length=0.0, insertion_depth=None))


# Edge cases

def test_oblique_needle():
    """
    Tip at (5, 5, 0) aimed at the centre, using the UNNORMALIZED
    direction (-1, -1, 0). Distance to the centre is 5*sqrt(2),
    so entry 5*sqrt(2) - 1, exit 5*sqrt(2) + 1, core 2.
    """
    dist = 5.0 * np.sqrt(2.0)
    run_case(
        "Oblique Needle (unnormalized direction)", C_UNIT, R_UNIT,
        np.array([5.0, 5.0, 0.0]), np.array([-1.0, -1.0, 0.0]),
        dict(scenario="intersects", targeting_error=0.0,
             core_length=2.0, insertion_depth=dist + 1.0))


def test_tip_inside_tumor():
    """
    Tip at (0, 0.5, 0), already inside. Crossings at s = -0.5 and
    s = 1.5; only the forward part [0, 1.5] can be cored.
    """
    run_case(
        "Edge: Tip Already Inside Tumor", C_UNIT, R_UNIT,
        np.array([0.0, 0.5, 0.0]), V_DOWN,
        dict(scenario="tip_inside", targeting_error=0.0,
             core_length=1.5, insertion_depth=1.5,
             entry_point=[0, 0.5, 0], exit_point=[0, -1, 0]))


def test_tumor_behind_tip():
    """
    Needle points AWAY from the tumor. The infinite axis goes
    through the centre (targeting error 0), but forward
    insertion never reaches it. The plan must not report a core.
    """
    run_case(
        "Edge: Tumor Behind the Tip", C_UNIT, R_UNIT,
        np.array([0.0, 10.0, 0.0]), np.array([0.0, 1.0, 0.0]),
        dict(scenario="behind", targeting_error=0.0,
             core_length=0.0, insertion_depth=None))


# Generality

def test_rigid_invariance(n_trials=200):
    """
    The plan outputs are distances, so they must not change when
    the WHOLE scene (tumor + needle) is rotated and translated.
    This shows the simple axis-aligned test scenes stand for
    every pose.
    """
    rng = np.random.default_rng(42)
    base_cases = [
        (np.array([0.0, 10.0, 0.0]), V_DOWN),
        (np.array([0.6, 10.0, 0.0]), V_DOWN),
        (np.array([1.0, 10.0, 0.0]), V_DOWN),
        (np.array([2.0, 10.0, 0.0]), V_DOWN),
        (np.array([0.0, 0.5, 0.0]), V_DOWN),
        (np.array([0.0, 10.0, 0.0]), -V_DOWN),
    ]
    for _ in range(n_trials):
        Rm = rotation_matrix(rng.normal(size=3), rng.uniform(-np.pi, np.pi))
        tm = rng.uniform(-200, 200, size=3)
        for P, v in base_cases:
            ref = compute_biopsy_plan(C_UNIT, R_UNIT, P, v)
            moved = compute_biopsy_plan(Rm @ C_UNIT + tm, R_UNIT,
                                        Rm @ P + tm, Rm @ v)
            assert moved["scenario"] == ref["scenario"]
            for key in ("targeting_error", "core_length"):
                assert np.isclose(moved[key], ref[key], atol=1e-7)
            if ref["insertion_depth"] is not None:
                assert np.isclose(moved["insertion_depth"],
                                  ref["insertion_depth"], atol=1e-7)

    print(f"\n--- Rigid Invariance ({n_trials} random scene poses) ---")
    print("All scenarios give identical outputs: passed")


def test_brute_force_core_length(n_trials=200):
    """
    Independent check of core length without any formulas:
    sample the needle path every 1 um from the tip forward and
    count the samples that lie inside the sphere.
    """
    rng = np.random.default_rng(1)
    step = 1e-3
    worst = 0.0
    for _ in range(n_trials):
        C = rng.uniform(-20, 20, size=3)
        R = rng.uniform(2, 15)
        P = rng.uniform(-60, 60, size=3)
        aim = C + rng.uniform(-1.3, 1.3, size=3) * R  # sometimes misses
        v = aim - P
        if np.linalg.norm(v) < 1e-6:
            continue
        plan = compute_biopsy_plan(C, R, P, v)

        u = v / np.linalg.norm(v)
        s = np.arange(0.0, 250.0, step)
        inside = np.linalg.norm(P + s[:, None] * u - C, axis=1) <= R
        brute = inside.sum() * step
        worst = max(worst, abs(brute - plan["core_length"]))

    print(f"\n--- Brute-Force Core Length ({n_trials} random scenes) ---")
    print(f"Worst difference: {worst:.4f} mm (sampling step {step} mm)")
    assert worst < 3 * step


def test_bad_inputs():
    for args in [
        (C_UNIT, 0.0, [0, 10, 0], V_DOWN),       # zero radius
        (C_UNIT, -1.0, [0, 10, 0], V_DOWN),      # negative radius
        (C_UNIT, 1.0, [0, 10, 0], [0, 0, 0]),    # zero direction
    ]:
        try:
            compute_biopsy_plan(*args)
            raise AssertionError(f"Bad input accepted: {args}")
        except ValueError:
            pass
    print("\n--- Bad Inputs ---")
    print("Zero/negative radius and zero direction rejected: passed")


# Registration -> plan

def test_end_to_end_with_registration():
    """
    Use registration Test 3: markers seen at
        M1ct = (52, -20, 118), M2ct = (58, -20, 118), M3ct = (52, -20, 124)
    which gives Pct = (50, -20, 100) and vct = (0, 0, -1).

    Place a clinically sized tumor (R = 10 mm) 80 mm ahead of the
    tip and 6 mm off the axis in x:  C = (56, -20, 20).
    On paper (a 6-8-10 triangle):
        targeting error = 6
        h = sqrt(10^2 - 6^2) = 8
        entry 80 - 8 = 72,  exit 80 + 8 = 88
        core = 16 mm,  insertion depth = 88 mm
    """
    ct_markers = np.array([
        [52.0, -20.0, 118.0],
        [58.0, -20.0, 118.0],
        [52.0, -20.0, 124.0],
    ])
    reg = register_tool(MARKERS, P_TRUE, V_TRUE, ct_markers)

    run_case(
        "End-to-End: Registration -> Biopsy Plan",
        np.array([56.0, -20.0, 20.0]), 10.0, reg["P_ct"], reg["v_ct"],
        dict(scenario="intersects", targeting_error=6.0,
             core_length=16.0, insertion_depth=88.0,
             entry_point=[50, -20, 28], exit_point=[50, -20, 12]))


if __name__ == "__main__":
    test_line_utilities()
    test_intersects_through_centre()
    test_intersects_off_centre()
    test_tangent()
    test_miss()
    test_oblique_needle()
    test_tip_inside_tumor()
    test_tumor_behind_tip()
    test_rigid_invariance()
    test_brute_force_core_length()
    test_bad_inputs()
    test_end_to_end_with_registration()
    print("\n--- All Biopsy-Plan Tests Passed ---")
