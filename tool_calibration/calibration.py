
"""
Purpose of this file:
    Calibrate the biopsy tool's tip position and axis direction
    using optical-tracker marker observations.

    Tip calibration uses pivot motion and least squares.
    Axis calibration uses plane fitting during axial rotation.

    We used rigid registration to express both estimates in
    the tool model frame.
"""

import numpy as np

from geometry import rigid_registration, fit_common_plane_normal



# Tool-Pose Estimation

def poses_from_observations(model_markers, observations):
    """
    Recover tool poses by registering the known model markers
    to their corresponding tracker observations.
    """
    model_markers = np.asarray(model_markers, dtype=float)
    observations = np.asarray(observations, dtype=float)

    if model_markers.shape != (3, 3):
        raise ValueError("Model markers must have shape (3, 3).")

    if observations.ndim != 3 or observations.shape[1:] != (3, 3):
        raise ValueError(
            "Observations must have shape (n_poses, 3, 3)."
        )

    if len(observations) == 0:
        raise ValueError("At least one observation is required.")

    rotations = []
    translations = []
    rms_errors = []

    for markers in observations:
        R, t, rms = rigid_registration(
            model_markers,
            markers,
            return_rms=True
        )

        rotations.append(R)
        translations.append(t)
        rms_errors.append(rms)

    return (
        np.asarray(rotations),
        np.asarray(translations),
        np.asarray(rms_errors)
    )



# Tool Tip Calibration

def calibrate_tip(rotations, translations):
    """
    Estimate the tool-tip position using pivot calibration.

    Motion constraint:
        The needle tip remains stationary while the tool
        rotates through sufficiently varied orientations.

    The poses map model coordinates into tracker coordinates,
    so the estimated tip p is directly in the model frame.
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

    # Stack the fixed-tip equations from all poses.
    for R, t in zip(rotations, translations):
        A.append(np.hstack((R, -np.eye(3))))
        b.append(-t)

    A = np.vstack(A)
    b = np.concatenate(b)

    # A full-rank system is needed to estimate all six unknowns.
    if np.linalg.matrix_rank(A) < 6:
        raise ValueError(
            "Pivot poses do not uniquely constrain calibration."
        )

    # The condition number indicates sensitivity to noise.
    condition_number = np.linalg.cond(A)

    # Solve for the tip and pivot using least squares.
    solution, _, _, _ = np.linalg.lstsq(
        A, b, rcond=None
    )

    tip_model = solution[:3]
    pivot_tracker = solution[3:]

    # Evaluate how well the estimated tip remains stationary.
    pivot_positions = np.array([
        R @ tip_model + t
        for R, t in zip(rotations, translations)
    ])

    residuals = pivot_positions - pivot_tracker

    pivot_rms = np.sqrt(
        np.mean(np.sum(residuals**2, axis=1))
    )

    return tip_model, pivot_tracker, pivot_rms, condition_number


# Tool Axis Calibration

def calibrate_axis(observations):
    """
    Estimate the tool axis from circular marker trajectories.

    Motion constraint:
        The tool rotates around its fixed longitudinal axis.
        Each marker follows a circular path in a plane
        perpendicular to the axis.

    A common plane normal is estimated using the trajectories
    of all three markers, centred independently.
    """
    observations = np.asarray(observations, dtype=float)

    if observations.ndim != 3 or observations.shape[1:] != (3, 3):
        raise ValueError(
            "Observations must have shape (n_poses, 3, 3)."
        )

    if len(observations) < 3:
        raise ValueError("At least three rotation poses are required.")

    # Group the observations by marker trajectory.
    trajectories = observations.transpose(1, 0, 2)

    # Fit one common normal to the centred trajectories.
    axis_tracker, plane_rms = fit_common_plane_normal(
        trajectories,
        return_rms=True
    )

    return axis_tracker, plane_rms

# Axis Transofrmation into model-frame

def axis_to_model_frame(axis_tracker, rotations):
    """
    Convert the tracker-frame axis into the tool model frame.

    Each rotation maps model coordinates into tracker
    coordinates, so its transpose maps directions back.
    """
    axis_tracker = np.asarray(axis_tracker, dtype=float)
    rotations = np.asarray(rotations, dtype=float)

    if axis_tracker.shape != (3,):
        raise ValueError("Axis must be a 3D vector.")

    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError("Rotations must have shape (n, 3, 3).")

    if len(rotations) == 0:
        raise ValueError("At least one rotation is required.")

    axis_norm = np.linalg.norm(axis_tracker)

    if axis_norm < 1e-10:
        raise ValueError("Estimated axis must be nonzero.")

    axis_tracker = axis_tracker / axis_norm

    # Transform the axis into model coordinates using each pose.
    model_axes = np.array([
        R.T @ axis_tracker
        for R in rotations
    ])

    # Average and normalize the model-frame estimates.
    axis_model = np.mean(model_axes, axis=0)
    model_norm = np.linalg.norm(axis_model)

    if model_norm < 1e-10:
        raise ValueError("Unable to recover a model-frame axis.")

    return axis_model / model_norm


# Axis Direction Calibration

def orient_axis(axis_model, forward_displacement_model):
    """
    Resolve the axis sign using a known forward translation.

    Motion constraint:
        The tool moves along the insertion direction without
        changing its orientation.

    Both vectors must be expressed in the model frame.
    """
    axis_model = np.asarray(axis_model, dtype=float)
    displacement = np.asarray(
        forward_displacement_model,
        dtype=float
    )

    if axis_model.shape != (3,) or displacement.shape != (3,):
        raise ValueError("Expected two 3D vectors.")

    axis_norm = np.linalg.norm(axis_model)
    displacement_norm = np.linalg.norm(displacement)

    if axis_norm < 1e-10:
        raise ValueError("Estimated axis must be nonzero.")

    if displacement_norm < 1e-10:
        raise ValueError("Forward displacement must be nonzero.")

    axis_model = axis_model / axis_norm
    displacement = displacement / displacement_norm

    # Check that forward motion is sufficiently aligned
    # with the estimated tool axis.
    alignment = np.dot(axis_model, displacement)

    if abs(alignment) < 1e-3:
        raise ValueError(
            "Forward motion is nearly perpendicular to the tool axis."
        )

    if alignment < 0:
        axis_model = -axis_model

    return axis_model

# Complete Tool Calibration

def calibrate_tool(
    model_markers,
    pivot_observations,
    axis_observations,
    forward_before,
    forward_after
):
    """
    Calibrate the tool tip and directed axis from marker data.
    """
    model_markers = np.asarray(model_markers, dtype=float)
    forward_before = np.asarray(forward_before, dtype=float)
    forward_after = np.asarray(forward_after, dtype=float)

    if model_markers.shape != (3, 3):
        raise ValueError("Model markers must have shape (3, 3).")

    if forward_before.shape != (3, 3):
        raise ValueError("Forward-before markers must have shape (3, 3).")

    if forward_after.shape != (3, 3):
        raise ValueError("Forward-after markers must have shape (3, 3).")

    # Tip calibration

    pivot_R, pivot_t, pivot_registration_rms = (
        poses_from_observations(
            model_markers,
            pivot_observations
        )
    )

    tip_model, pivot_tracker, pivot_rms, pivot_condition = (
        calibrate_tip(pivot_R, pivot_t)
    )

    # Axis Calibration
    axis_R, _, axis_registration_rms = (
        poses_from_observations(
            model_markers,
            axis_observations
        )
    )

    axis_tracker, plane_rms = calibrate_axis(
        axis_observations
    )

    axis_model = axis_to_model_frame(
        axis_tracker,
        axis_R
    )

    # Register the reference pose before forward translation.
    R_before, _ = rigid_registration(
        model_markers,
        forward_before
    )

    # Estimate the tracker-frame forward displacement.
    displacement_tracker = (
        np.mean(forward_after, axis=0)
        - np.mean(forward_before, axis=0)
    )

    # Convert the displacement into model coordinates.
    displacement_model = R_before.T @ displacement_tracker

    directed_axis_model = orient_axis(
        axis_model,
        displacement_model
    )

    return {
        "tip_model": tip_model,
        "axis_model": directed_axis_model,
        "axis_tracker": axis_tracker,
        "pivot_tracker": pivot_tracker,
        "pivot_rms": pivot_rms,
        "pivot_condition": pivot_condition,
        "plane_rms": plane_rms,

        # Average registration residuals.
        "pivot_registration_rms": np.mean(
            pivot_registration_rms
        ),
        "axis_registration_rms": np.mean(
            axis_registration_rms
        ),

        # Per-pose residuals for additional diagnostics.
        "pivot_registration_rms_per_pose": pivot_registration_rms,
        "axis_registration_rms_per_pose": axis_registration_rms
    }
