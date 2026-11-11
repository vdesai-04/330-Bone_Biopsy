"""
Purpose of this file:
    Register the calibrated biopsy tool to the CT image frame.

    Inputs (assignment notation):
        M1tool, M2tool, M3tool : markers in the tool model frame
        Ptool                  : tool tip in the tool model frame
        vtool                  : tool axis direction in the model frame
        M1ct, M2ct, M3ct       : the same markers located in CT

    Outputs:
        Pct : tool tip in the CT frame
        vct : tool axis direction in the CT frame

    Method (three steps):
        1. Find the rigid transform (R, t) with
               M_ct = R @ M_tool + t     for every marker.
           We reuse the SVD (Arun/Kabsch) least-squares solver
           written for Part 1 (tool_calibration/geometry.py).
        2. Transform the tip as a POINT:
               Pct = R @ Ptool + t
        3. Transform the axis as a DIRECTION:
               vct = R @ vtool          (no translation!)
"""

import sys
from pathlib import Path

import numpy as np

# Part 1 already implemented and validated rigid registration.
# Rather than duplicate it, we import it from the
# tool_calibration folder that sits next to this one.
_CALIBRATION_DIR = Path(__file__).resolve().parent.parent / "tool_calibration"
if str(_CALIBRATION_DIR) not in sys.path:
    sys.path.insert(0, str(_CALIBRATION_DIR))

from geometry import rigid_registration  # noqa: E402  (path set above)


# Tolerance used to flag CT markers that do not match the
# rigid marker geometry of the tool (e.g. a marker was
# mislabelled or badly localized). 1.0 mm is the size of the
# 95% marker localization error used throughout the assignment.
MARKER_CONSISTENCY_TOL_MM = 1.0


# Transforming points versus directions

def transform_point(R, t, point):
    """
    Map a POINT from the tool model frame into the CT frame.

    A point has a location, so it is both rotated and shifted:
        p_ct = R @ p_tool + t
    """
    return R @ np.asarray(point, dtype=float) + t


def transform_direction(R, direction):
    """
    Map a DIRECTION vector from the tool model frame into CT.

    A direction is the difference of two points, a - b.
    Under a rigid transform:
        (R a + t) - (R b + t) = R (a - b)
    so the translation cancels and ONLY the rotation applies.

    Equivalent homogeneous view: a point is [x, y, z, 1]
    and a direction is [x, y, z, 0]; the 0 switches off the
    translation column of the 4x4 matrix.
    """
    v_ct = R @ np.asarray(direction, dtype=float)

    # R preserves length, so this only removes round-off.
    return v_ct / np.linalg.norm(v_ct)


def homogeneous_matrix(R, t):
    """
    Pack (R, t) into a 4x4 homogeneous transform.

        | R  t |
        | 0  1 |

    Provided for reporting and for the point-vs-direction
    demonstration in the tests.
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# Input checks

def marker_distance_mismatch(model_markers, ct_markers):
    """
    Largest difference between corresponding inter-marker
    distances in the model and CT frames (mm).

    A rigid transform preserves all distances, so for a correct
    marker correspondence |Mi - Mj| must be the same in both
    frames. A large mismatch means the CT markers cannot come
    from a rigid motion of the tool (wrong marker order, a
    missing marker, or a gross localization error).
    """
    def pairwise(points):
        return np.array([
            np.linalg.norm(points[i] - points[j])
            for i, j in ((0, 1), (0, 2), (1, 2))
        ])

    return float(np.max(np.abs(pairwise(model_markers) - pairwise(ct_markers))))


# Main registration routine

def register_tool(model_markers, p_tool, v_tool, ct_markers):
    """
    Register the tracked biopsy tool to the CT frame.

    Parameters
    ----------
    model_markers : (3, 3) array
        Rows are M1tool, M2tool, M3tool (mm).
    p_tool : (3,) array
        Tool tip in the tool model frame (mm), from Part 1.
    v_tool : (3,) array
        Tool axis direction in the model frame, from Part 1.
        It does not need to be unit length.
    ct_markers : (3, 3) array
        Rows are M1ct, M2ct, M3ct (mm), in the SAME order as
        model_markers. The order defines the correspondence.

    Returns
    -------
    dict with
        "P_ct"    : tool tip in CT (mm)
        "v_ct"    : unit tool axis direction in CT
        "R", "t"  : rigid transform, M_ct = R @ M_tool + t
        "T"       : the same transform as a 4x4 matrix
        "fre_rms" : RMS marker residual after registration (mm),
                    the fiducial registration error (FRE)
        "distance_mismatch" : see marker_distance_mismatch (mm)
        "consistent" : False if the CT markers do not fit the
                       tool geometry within MARKER_CONSISTENCY_TOL_MM
    """
    model_markers = np.asarray(model_markers, dtype=float)
    ct_markers = np.asarray(ct_markers, dtype=float)
    p_tool = np.asarray(p_tool, dtype=float)
    v_tool = np.asarray(v_tool, dtype=float)

    if model_markers.shape != (3, 3) or ct_markers.shape != (3, 3):
        raise ValueError("Expected three 3D markers in each frame.")

    if p_tool.shape != (3,) or v_tool.shape != (3,):
        raise ValueError("Tool tip and axis must be 3D vectors.")

    if np.linalg.norm(v_tool) < 1e-10:
        raise ValueError("Tool axis direction must be nonzero.")

    # Step 1: rigid transform from model markers to CT markers.
    # rigid_registration (Part 1) centres both marker sets,
    # solves for R with SVD, forces det(R) = +1 so a mirror
    # image is never returned, and then sets
    #     t = centroid_ct - R @ centroid_model.
    # It raises ValueError if the model markers are collinear,
    # because a line does not fix the rotation about itself.
    R, t, fre_rms = rigid_registration(
        model_markers, ct_markers, return_rms=True
    )

    # Step 2: the tip is a point -> rotate AND translate.
    P_ct = transform_point(R, t, p_tool)

    # Step 3: the axis is a direction -> rotate ONLY.
    v_ct = transform_direction(R, v_tool / np.linalg.norm(v_tool))

    mismatch = marker_distance_mismatch(model_markers, ct_markers)

    return {
        "P_ct": P_ct,
        "v_ct": v_ct,
        "R": R,
        "t": t,
        "T": homogeneous_matrix(R, t),
        "fre_rms": fre_rms,
        "distance_mismatch": mismatch,
        "consistent": mismatch <= MARKER_CONSISTENCY_TOL_MM,
    }
