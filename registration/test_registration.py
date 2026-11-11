"""
Purpose of this file:
    Validate tool-to-CT registration (registration.py).

    The three tests the assignment asks for:
        1. pure translation
        2. pure rotation
        3. rotation + translation
    For each, the ground-truth tool is placed in a known CT
    pose, the expected Pct and vct are worked out BY HAND
    (written out in the comments and printed as "Predicted"),
    and the registration output is checked against them.

    Extra checks:
        4. off-origin tip, so the rotation acts on the tip too
        5. a direction must not be translated
        6. cross-check against Part 1's marker-frame method
        7. 500 random poses
        8. bad inputs (collinear markers, corrupted marker)
        9. Part 1 calibration output -> registration
       10. M2/M3 label swap goes undetected (documented limitation)

Run with:  python3 test_registration.py   (or pytest)
"""

import sys
from pathlib import Path

import numpy as np

from registration import (
    register_tool,
    transform_point,
    homogeneous_matrix,
)

_CALIBRATION_DIR = Path(__file__).resolve().parent.parent / "tool_calibration"
sys.path.insert(0, str(_CALIBRATION_DIR))

from geometry import marker_frame, rotation_matrix  # noqa: E402
from simulation import (  # noqa: E402
    MARKERS,
    P_TRUE,
    V_TRUE,
    random_pose,
    simulate_pivot,
    simulate_axis_rotation,
    simulate_forward_motion,
)
from calibration import calibrate_tool  # noqa: E402


# Ground-truth tool model (Fig. 2 of the assignment), in mm:
#   M1 = (2, 18, 0), M2 = (8, 18, 0), M3 = (2, 24, 0)
#   P  = (0, 0, 0),  v  = (0, -1, 0)
TOL = 1e-9  # "numerical precision" tolerance (mm, or unit vector)


def report(title, R_true, t_true, ct_markers, predicted_P, predicted_v,
           result, p_tool=P_TRUE):
    """Print the paper prediction next to the computed result."""
    np.set_printoptions(precision=6, suppress=True)
    print(f"\n--- {title} ---")
    print("True R:\n", R_true)
    print("True t:", t_true)
    print("CT markers (M1ct, M2ct, M3ct):\n", ct_markers)
    print("Tool tip in model frame:", p_tool)
    # Rounding + 0.0 hides harmless "-0." round-off in the printout.
    shown_P = np.round(result["P_ct"], 9) + 0.0
    shown_v = np.round(result["v_ct"], 9) + 0.0
    print(f"Predicted Pct: {predicted_P}    Computed Pct: {shown_P}")
    print(f"Predicted vct: {predicted_v}    Computed vct: {shown_v}")
    print(f"Recovered R error: {np.max(np.abs(result['R'] - R_true)):.2e}")
    print(f"Recovered t error: {np.max(np.abs(result['t'] - t_true)):.2e} mm")
    print(f"Marker FRE (RMS):  {result['fre_rms']:.2e} mm")


def check(result, predicted_P, predicted_v, R_true, t_true):
    assert np.allclose(result["P_ct"], predicted_P, atol=TOL)
    assert np.allclose(result["v_ct"], predicted_v, atol=TOL)
    assert np.allclose(result["R"], R_true, atol=TOL)
    assert np.allclose(result["t"], t_true, atol=TOL)
    assert np.isclose(np.linalg.det(result["R"]), 1.0, atol=TOL)
    assert result["fre_rms"] < TOL
    assert result["consistent"]


# Test 1: pure translation

def test_pure_translation():
    """
    Pose: R = I, t = (50, -20, 100) mm.

    On paper (M_ct = M_tool + t):
        M1ct = (2, 18, 0)  + t = (52, -2, 100)
        M2ct = (8, 18, 0)  + t = (58, -2, 100)
        M3ct = (2, 24, 0)  + t = (52,  4, 100)
        Pct  = (0, 0, 0)   + t = (50, -20, 100)
        vct  = (0, -1, 0)        <- a direction ignores translation
    """
    R_true = np.eye(3)
    t_true = np.array([50.0, -20.0, 100.0])

    ct_markers = np.array([
        [52.0, -2.0, 100.0],
        [58.0, -2.0, 100.0],
        [52.0, 4.0, 100.0],
    ])
    predicted_P = np.array([50.0, -20.0, 100.0])
    predicted_v = np.array([0.0, -1.0, 0.0])

    result = register_tool(MARKERS, P_TRUE, V_TRUE, ct_markers)
    report("Test 1: Pure Translation", R_true, t_true, ct_markers,
           predicted_P, predicted_v, result)
    check(result, predicted_P, predicted_v, R_true, t_true)


# Test 2: pure rotation

def test_pure_rotation():
    """
    Pose: rotate +90 deg about z_ct, t = 0.

        Rz(90) = | 0 -1  0 |      (x, y, z) -> (-y, x, z)
                 | 1  0  0 |
                 | 0  0  1 |

    On paper:
        M1ct = (-18, 2, 0)
        M2ct = (-18, 8, 0)
        M3ct = (-24, 2, 0)
        Pct  = (0, 0, 0)    tip sits on the rotation centre
        vct  = (1, 0, 0)    needle now points along +x
    """
    R_true = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    t_true = np.zeros(3)

    ct_markers = np.array([
        [-18.0, 2.0, 0.0],
        [-18.0, 8.0, 0.0],
        [-24.0, 2.0, 0.0],
    ])
    predicted_P = np.array([0.0, 0.0, 0.0])
    predicted_v = np.array([1.0, 0.0, 0.0])

    result = register_tool(MARKERS, P_TRUE, V_TRUE, ct_markers)
    report("Test 2: Pure Rotation (+90 deg about z)", R_true, t_true,
           ct_markers, predicted_P, predicted_v, result)
    check(result, predicted_P, predicted_v, R_true, t_true)


# Test 3: rotation + translation

def test_rotation_and_translation():
    """
    Pose: rotate +90 deg about x_ct, then t = (50, -20, 100).
    A different axis from Test 2, so x, y and z all get used.

        Rx(90) = | 1  0  0 |      (x, y, z) -> (x, -z, y)
                 | 0  0 -1 |
                 | 0  1  0 |

    On paper (rotate first, then add t):
        M1: (2, 18, 0) -> (2, 0, 18) -> (52, -20, 118)
        M2: (8, 18, 0) -> (8, 0, 18) -> (58, -20, 118)
        M3: (2, 24, 0) -> (2, 0, 24) -> (52, -20, 124)
        Pct: (0, 0, 0) -> (0, 0, 0)  -> (50, -20, 100)
        vct: (0, -1, 0) -> (0, 0, -1)    rotation only
    """
    R_true = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ])
    t_true = np.array([50.0, -20.0, 100.0])

    ct_markers = np.array([
        [52.0, -20.0, 118.0],
        [58.0, -20.0, 118.0],
        [52.0, -20.0, 124.0],
    ])
    predicted_P = np.array([50.0, -20.0, 100.0])
    predicted_v = np.array([0.0, 0.0, -1.0])

    result = register_tool(MARKERS, P_TRUE, V_TRUE, ct_markers)
    report("Test 3: Rotation (+90 deg about x) + Translation", R_true,
           t_true, ct_markers, predicted_P, predicted_v, result)
    check(result, predicted_P, predicted_v, R_true, t_true)


# Test 4: tip away from the model origin

def test_offset_tip():
    """
    The ground-truth tip is at the model origin, so in Tests 1-3
    Pct = R @ 0 + t = t. The rotation never touches the tip,
    so a bug such as "Pct = Ptool + t" would still pass.

    Here we reuse Test 3's pose with the off-origin tip from
    Part 1, Ptool = (3, -2, 5):
        Rx(90) @ (3, -2, 5) = (3, -5, -2)
        Pct = (3, -5, -2) + (50, -20, 100) = (53, -25, 98)
    """
    R_true = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ])
    t_true = np.array([50.0, -20.0, 100.0])
    p_tool = np.array([3.0, -2.0, 5.0])

    ct_markers = np.array([
        [52.0, -20.0, 118.0],
        [58.0, -20.0, 118.0],
        [52.0, -20.0, 124.0],
    ])
    predicted_P = np.array([53.0, -25.0, 98.0])
    predicted_v = np.array([0.0, 0.0, -1.0])

    result = register_tool(MARKERS, p_tool, V_TRUE, ct_markers)
    report("Test 4: Off-Origin Tip (3, -2, 5)", R_true, t_true,
           ct_markers, predicted_P, predicted_v, result, p_tool=p_tool)
    check(result, predicted_P, predicted_v, R_true, t_true)


# Test 5: why the axis must not be translated

def test_direction_is_not_translated():
    """
    In Test 1 (pure translation), applying the full 4x4
    transform to v as if it were a point ([v, 1]) gives
    (0, -1, 0) + t = (50, -21, 100): a wrong answer.
    Using [v, 0] (a direction) gives the correct (0, -1, 0).
    """
    T = homogeneous_matrix(np.eye(3), np.array([50.0, -20.0, 100.0]))

    as_point = (T @ np.append(V_TRUE, 1.0))[:3]
    as_direction = (T @ np.append(V_TRUE, 0.0))[:3]

    print("\n--- Test 5: Direction vs Point Under Translation ---")
    print("v treated as point     [v,1] ->", as_point, " (WRONG)")
    print("v treated as direction [v,0] ->", as_direction, " (correct)")

    assert np.allclose(as_point, [50.0, -21.0, 100.0])
    assert np.allclose(as_direction, V_TRUE)


# Test 6: independent cross-check

def test_matches_marker_frame_method():
    """
    Part 1's marker_frame() builds an orthonormal frame on the
    markers themselves (origin M1, x toward M2, z normal to the
    triangle). With error-free markers, mapping the model frame
    onto the CT frame gives a second, independent R and t:
        R = F_ct @ F_mod^T,   t = o_ct - R @ o_mod
    It must agree with the SVD solution.
    """
    rng = np.random.default_rng(6)
    R_true, t_true = random_pose(rng)
    ct_markers = MARKERS @ R_true.T + t_true

    F_mod, o_mod = marker_frame(MARKERS)
    F_ct, o_ct = marker_frame(ct_markers)
    R_frame = F_ct @ F_mod.T
    t_frame = o_ct - R_frame @ o_mod

    result = register_tool(MARKERS, P_TRUE, V_TRUE, ct_markers)

    print("\n--- Test 6: SVD vs Marker-Frame Registration ---")
    print(f"Max |R_svd - R_frame|: {np.max(np.abs(result['R'] - R_frame)):.2e}")
    print(f"Max |t_svd - t_frame|: {np.max(np.abs(result['t'] - t_frame)):.2e} mm")

    assert np.allclose(result["R"], R_frame, atol=TOL)
    assert np.allclose(result["t"], t_frame, atol=TOL)


# Test 7: many random poses

def test_random_poses(n_trials=500):
    """Random rotations (any axis, +/-180 deg) and translations."""
    rng = np.random.default_rng(7)
    p_tool = np.array([3.0, -2.0, 5.0])
    worst_P, worst_v = 0.0, 0.0

    for _ in range(n_trials):
        R_true, t_true = random_pose(rng)
        ct_markers = MARKERS @ R_true.T + t_true

        result = register_tool(MARKERS, p_tool, V_TRUE, ct_markers)

        worst_P = max(worst_P, np.linalg.norm(
            result["P_ct"] - (R_true @ p_tool + t_true)))
        worst_v = max(worst_v, np.linalg.norm(
            result["v_ct"] - R_true @ V_TRUE))

    print(f"\n--- Test 7: {n_trials} Random Poses ---")
    print(f"Worst tip error:  {worst_P:.2e} mm")
    print(f"Worst axis error: {worst_v:.2e}")

    assert worst_P < TOL and worst_v < TOL


# Test 8: invalid / inconsistent inputs

def test_bad_inputs():
    """
    (a) Collinear model markers cannot fix the rotation about
        their own line, so registration must refuse them.
    (b) A CT marker moved 5 mm cannot come from a rigid motion;
        the result must be flagged as inconsistent.
    (c) A zero axis vector is rejected.
    """
    collinear = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    try:
        register_tool(collinear, P_TRUE, V_TRUE, collinear)
        raise AssertionError("Collinear markers were accepted.")
    except ValueError:
        pass

    corrupted = MARKERS.copy()
    corrupted[2] += np.array([5.0, 0.0, 0.0])
    result = register_tool(MARKERS, P_TRUE, V_TRUE, corrupted)
    assert not result["consistent"]

    try:
        register_tool(MARKERS, P_TRUE, np.zeros(3), MARKERS)
        raise AssertionError("Zero axis was accepted.")
    except ValueError:
        pass

    print("\n--- Test 8: Bad Inputs ---")
    print("Collinear markers rejected: yes")
    print(f"Corrupted marker flagged: distance mismatch = "
          f"{result['distance_mismatch']:.2f} mm, FRE = "
          f"{result['fre_rms']:.2f} mm")
    print("Zero axis rejected: yes")


# Test 9: Part 1 -> Part 2 pipeline

def test_with_part1_calibration():
    """
    Run Part 1's calibrate_tool() on perfect simulated data with
    the off-origin tip, then pass ITS outputs (not the ground
    truth) into register_tool(). This shows the two parts
    connect in the expected frames.
    """
    rng = np.random.default_rng(9)
    tip_true = np.array([3.0, -2.0, 5.0])

    pivot = simulate_pivot(n_poses=30, tip=tip_true, rng=rng)
    base_R, base_t = random_pose(rng)
    axis_obs = simulate_axis_rotation(
        n_poses=36, tip=tip_true,
        base_rotation=base_R, base_translation=base_t)
    before, after = simulate_forward_motion(
        distance=5.0, base_rotation=base_R, base_translation=base_t)

    calib = calibrate_tool(MARKERS, pivot, axis_obs, before, after)

    # Place the tool in CT with a known pose.
    R_ct = rotation_matrix([1.0, 2.0, 3.0], np.radians(40.0))
    t_ct = np.array([120.0, -60.0, 250.0])
    ct_markers = MARKERS @ R_ct.T + t_ct

    result = register_tool(
        MARKERS, calib["tip_model"], calib["axis_model"], ct_markers)

    expected_P = R_ct @ tip_true + t_ct
    expected_v = R_ct @ V_TRUE

    print("\n--- Test 9: Part 1 Calibration -> Registration ---")
    print("Calibrated tip (model):", calib["tip_model"])
    print("Calibrated axis (model):", calib["axis_model"])
    print("Pct:", result["P_ct"], " expected:", expected_P)
    print("vct:", result["v_ct"], " expected:", expected_v)

    assert np.allclose(result["P_ct"], expected_P, atol=1e-8)
    assert np.allclose(result["v_ct"], expected_v, atol=1e-8)


# Test 10: marker-labelling ambiguity (a known limitation)

def test_swapped_m2_m3_is_undetectable():
    """
    |M1M2| = |M1M3| = 6 mm, so M2 and M3 cannot be told apart by
    distances. Swapping them mirrors the flat marker triangle,
    and for three coplanar points a mirror image can always be
    reproduced by a proper rotation (flip the triangle over).
    Registration therefore returns FRE = 0 and passes the
    consistency check, yet the pose is wrong.

    Using Test 1's markers with M2/M3 swapped: the tool is flipped
    180 deg about the M1->(M2+M3)/2 bisector. The tip lands at
    (34, -4, 100) instead of (50, -20, 100) and the axis points
    along -x instead of -y.

    This test documents the limitation: correct marker labels
    must come from somewhere else (e.g. an asymmetric marker
    layout or a fourth marker).
    """
    ct = MARKERS + np.array([50.0, -20.0, 100.0])
    swapped = ct[[0, 2, 1]]
    result = register_tool(MARKERS, P_TRUE, V_TRUE, swapped)

    tip_shift = np.linalg.norm(result["P_ct"] - np.array([50.0, -20.0, 100.0]))

    print("\n--- Test 10: M2/M3 Swapped (labelling ambiguity) ---")
    print(f"FRE: {result['fre_rms']:.2e} mm, consistent: {result['consistent']}")
    print(f"Pct: {np.round(result['P_ct'], 6) + 0.0}  (true (50, -20, 100); "
          f"off by {tip_shift:.1f} mm)")
    print(f"vct: {np.round(result['v_ct'], 6) + 0.0}  (true (0, -1, 0))")

    assert result["fre_rms"] < TOL and result["consistent"]
    assert np.allclose(result["P_ct"], [34.0, -4.0, 100.0], atol=TOL)
    assert np.allclose(result["v_ct"], [-1.0, 0.0, 0.0], atol=TOL)


if __name__ == "__main__":
    test_pure_translation()
    test_pure_rotation()
    test_rotation_and_translation()
    test_offset_tip()
    test_direction_is_not_translated()
    test_matches_marker_frame_method()
    test_random_poses()
    test_bad_inputs()
    test_with_part1_calibration()
    test_swapped_m2_m3_is_undetectable()
    print("\n--- All Registration Tests Passed ---")
