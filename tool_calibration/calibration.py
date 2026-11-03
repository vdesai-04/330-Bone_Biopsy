
"""
CISC/CMPE 330 - CT-Guided Bone Biopsy Planning
Part 1: Tool Calibration

File: calibration.py

Purpose:
    Estimate the biopsy tool's tip position and axis direction
    from optical-tracker marker measurements.

    Tool-tip calibration uses pivot motion and least squares.
    Tool-axis calibration uses plane fitting during rotation
    about the needle's longitudinal axis.

    The resulting estimates are expressed in the marker-defined
    tool model frame for comparison with the ground-truth model.
"""

import numpy as np

from geometry import marker_frame, fit_plane_normal


# ---------------------------------------------------------
# 1. TOOL-TIP CALIBRATION
# ---------------------------------------------------------

def calibrate_tip(rotations, translations):
    """
    Estimate the tool tip using pivot calibration.

    Motion constraint:
        The tool rotates through varied orientations while
        its tip remains stationary in the tracker frame.

    For each pose:
        R_i @ p_marker + t_i = p_pivot

    Stacking these equations gives a linear least-squares
    problem with six unknowns: the marker-frame tip position
    and the tracker-frame pivot position.

    Returns:
        tip: Tip position in the marker frame (mm).
        pivot: Fixed pivot position in the tracker frame (mm).
    """
    rotations = np.asarray(rotations, dtype=float)
    translations = np.asarray(translations, dtype=float)

    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError("Rotations must have shape (n, 3, 3).")

    if translations.shape != (len(rotations), 3):
        raise ValueError("Translations must have shape (n, 3).")

    if len(rotations) < 3:
        raise ValueError("At least three pivot poses are required.")

    A = []
    b = []

    # Construct the stacked system A @ x = b.
    for R, t in zip(rotations, translations):
        A.append(np.hstack((R, -np.eye(3))))
        b.append(-t)

    A = np.vstack(A)
    b = np.concatenate(b)

    # Check that the six unknowns can be uniquely estimated.
    if np.linalg.matrix_rank(A) < 6:
        raise ValueError(
            "Pivot poses do not sufficiently constrain calibration."
        )

    # Solve using the Moore-Penrose least-squares formulation.
    solution, _, _, _ = np.linalg.lstsq(
        A, b, rcond=None
    )

    tip = solution[:3]
    pivot = solution[3:]

    return tip, pivot


# ---------------------------------------------------------
# 2. TOOL-TIP COORDINATE TRANSFORMATION
# ---------------------------------------------------------

def tip_to_model_frame(tip_marker, marker_origin):
    """
    Express the calibrated tip in the tool model frame.

    For the assignment's ground-truth geometry, the marker
    frame is centred at Marker 1 and its axes are aligned
    with the tool model frame.

    Therefore, the conversion requires only translation.

    Returns:
        Tip position in the tool model frame (mm).
    """
    tip_marker = np.asarray(tip_marker, dtype=float)
    marker_origin = np.asarray(marker_origin, dtype=float)

    return tip_marker + marker_origin


# ---------------------------------------------------------
# 3. TOOL-AXIS CALIBRATION
# ---------------------------------------------------------

def calibrate_axis(observations):
    """
    Estimate the needle-axis orientation using plane fitting.

    Motion constraint:
        The tool rotates around its fixed longitudinal axis.

    Each marker traces a circular trajectory in a plane
    perpendicular to that axis.

    Method:
        Fit a plane to each marker's observations using SVD,
        align the plane normals, and average them.

    Returns:
        Unit axis vector in tracker coordinates.
        The positive/negative direction remains ambiguous.
    """
    observations = np.asarray(observations, dtype=float)

    if observations.ndim != 3 or observations.shape[1:] != (3, 3):
        raise ValueError(
            "Observations must have shape (n_poses, 3, 3)."
        )

    if len(observations) < 3:
        raise ValueError("At least three poses are required.")

    normals = []

    for marker_index in range(3):
        # Extract one marker's trajectory across all poses.
        points = observations[:, marker_index, :]

        # The fitted plane normal estimates the rotation axis.
        normal = fit_plane_normal(points)

        # Resolve relative sign ambiguity between plane normals.
        if normals and np.dot(normal, normals[0]) < 0:
            normal = -normal

        normals.append(normal)

    # Average the consistently oriented plane normals.
    axis = np.mean(normals, axis=0)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-10:
        raise ValueError("Unable to estimate a valid tool axis.")

    return axis / axis_norm


# ---------------------------------------------------------
# 4. TOOL-AXIS DIRECTION
# ---------------------------------------------------------

def orient_axis(axis, forward_displacement):
    """
    Resolve the axis-sign ambiguity using known forward motion.

    Motion constraint:
        The tool translates along its insertion direction
        without changing orientation.

    The observed forward marker displacement identifies
    which of the two possible axis directions points toward
    the needle tip.

    Returns:
        Unit axis vector directed toward the needle tip.
    """
    axis = np.asarray(axis, dtype=float)
    forward_displacement = np.asarray(
        forward_displacement, dtype=float
    )

    axis_norm = np.linalg.norm(axis)
    displacement_norm = np.linalg.norm(forward_displacement)

    if axis_norm < 1e-10:
        raise ValueError("Estimated axis must be nonzero.")

    if displacement_norm < 1e-10:
        raise ValueError("Forward displacement must be nonzero.")

    axis = axis / axis_norm

    # The forward motion must have a component along the axis.
    alignment = np.dot(
        axis, forward_displacement / displacement_norm
    )

    if abs(alignment) < 1e-8:
        raise ValueError(
            "Forward displacement is perpendicular to the tool axis."
        )

    # Choose the sign consistent with the known forward motion.
    if alignment < 0:
        axis = -axis

    return axis


# ---------------------------------------------------------
# 5. TOOL-AXIS COORDINATE TRANSFORMATION
# ---------------------------------------------------------

def axis_to_model_frame(axis_tracker, reference_markers):
    """
    Express a tracker-frame axis in the tool model frame.

    The reference markers must correspond to the same
    tool orientation as the estimated tracker-frame axis.

    For the assignment geometry, the marker-frame axes
    are aligned with the model-frame axes.

    Only rotation is applied because the axis is a
    direction vector, not a position.

    Returns:
        Unit axis direction in the tool model frame.
    """
    R, _ = marker_frame(reference_markers)

    axis_tracker = np.asarray(axis_tracker, dtype=float)

    # Transform the direction from tracker to marker/model frame.
    axis_model = R.T @ axis_tracker

    axis_norm = np.linalg.norm(axis_model)

    if axis_norm < 1e-10:
        raise ValueError("Transformed axis must be nonzero.")

    return axis_model / axis_norm
