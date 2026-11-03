
"""
CISC/CMPE 330 - CT-Guided Bone Biopsy Planning
Part 1: Tool Calibration

File: test_calibration.py

Purpose:
    Validate tool-tip and tool-axis calibration using simulated
    optical-tracker observations.

    The tests evaluate:
        1. Perfect-data calibration accuracy
        2. Calibration with Gaussian marker noise
        3. Recovery under a rotated tool orientation
        4. Error statistics over 200 independent noise trials
        5. Motion constraints for tip and axis calibration

    Estimated parameters are compared with the ground-truth
    tool model provided in Assignment Figure 2.
"""

import numpy as np

from geometry import marker_frame, rotation_matrix

from calibration import (
    calibrate_tip,
    calibrate_axis,
    orient_axis,
    tip_to_model_frame,
    axis_to_model_frame
)

from simulation import (
    simulate_pivot,
    simulate_axis_rotation,
    simulate_forward_motion,
    add_marker_noise,
    M1,
    P_TRUE,
    V_TRUE
)


# ---------------------------------------------------------
# 1. CALIBRATION TEST UTILITIES
# ---------------------------------------------------------

def estimate_tip(observations):
    """Recover the tool tip from a collection of pivot poses."""
    rotations = []
    translations = []

    for markers in observations:
        R, t = marker_frame(markers)
        rotations.append(R)
        translations.append(t)

    tip_marker, pivot = calibrate_tip(
        rotations, translations
    )

    return tip_marker, pivot, rotations, translations


def angular_error_degrees(estimated, expected):
    """Calculate the angle between two unit directions."""
    estimated = estimated / np.linalg.norm(estimated)
    expected = expected / np.linalg.norm(expected)

    alignment = np.clip(
        np.dot(estimated, expected), -1.0, 1.0
    )

    return np.degrees(np.arccos(alignment))


def print_error_statistics(errors, units):
    """Print summary statistics for repeated calibration trials."""
    print(f"Mean error: {np.mean(errors):.4f} {units}")
    print(f"Standard deviation: {np.std(errors):.4f} {units}")
    print(
        f"95th-percentile error: "
        f"{np.percentile(errors, 95):.4f} {units}"
    )


# ---------------------------------------------------------
# 2. PERFECT-DATA TOOL-TIP CALIBRATION
# ---------------------------------------------------------

def test_tip_calibration():
    """Verify recovery of the ground-truth tip without noise."""
    observations = simulate_pivot(n_poses=30)

    tip_marker, pivot, rotations, translations = estimate_tip(
        observations
    )

    # Convert the estimated tip to the tool model frame.
    tip_model = tip_to_model_frame(tip_marker, M1)
    tip_error = np.linalg.norm(tip_model - P_TRUE)

    # Verify the tip remains stationary in tracker coordinates.
    pivot_positions = np.array([
        R @ tip_marker + t
        for R, t in zip(rotations, translations)
    ])

    pivot_deviations = np.linalg.norm(
        pivot_positions - pivot, axis=1
    )

    print("\n--- Perfect Tool-Tip Calibration ---")
    print("Ground-truth tip (model frame):", P_TRUE)
    print("Estimated tip (model frame):", tip_model)
    print("Estimated pivot (tracker frame):", pivot)
    print(f"Tool-tip error: {tip_error:.10f} mm")
    print(
        f"Maximum pivot deviation: "
        f"{np.max(pivot_deviations):.10f} mm"
    )

    assert np.allclose(tip_model, P_TRUE, atol=1e-8)
    assert np.max(pivot_deviations) < 1e-8


# ---------------------------------------------------------
# 3. NOISY TOOL-TIP CALIBRATION
# ---------------------------------------------------------

def test_tip_calibration_noisy():
    """Evaluate the tip estimate with Gaussian marker noise."""
    observations = simulate_pivot(n_poses=30)
    noisy_observations = add_marker_noise(
        observations, seed=42
    )

    tip_marker, _, _, _ = estimate_tip(noisy_observations)

    tip_model = tip_to_model_frame(tip_marker, M1)
    tip_error = np.linalg.norm(tip_model - P_TRUE)

    print("\n--- Noisy Tool-Tip Calibration ---")
    print("Ground-truth tip (model frame):", P_TRUE)
    print("Estimated tip (model frame):", tip_model)
    print(f"Tool-tip error: {tip_error:.4f} mm")

    assert np.isfinite(tip_error)


# ---------------------------------------------------------
# 4. PERFECT-DATA TOOL-AXIS CALIBRATION
# ---------------------------------------------------------

def test_axis_calibration():
    """Verify recovery of the directed axis without noise."""
    observations = simulate_axis_rotation(n_poses=36)

    # Estimate the undirected axis from marker trajectories.
    axis_tracker = calibrate_axis(observations)

    # Resolve its sign using a known forward movement.
    before, after = simulate_forward_motion()
    displacement = np.mean(after - before, axis=0)

    directed_axis = orient_axis(
        axis_tracker, displacement
    )

    # Convert into the tool model coordinate frame.
    axis_model = axis_to_model_frame(
        directed_axis, observations[0]
    )

    error = angular_error_degrees(axis_model, V_TRUE)

    print("\n--- Perfect Tool-Axis Calibration ---")
    print("Ground-truth axis (model frame):", V_TRUE)
    print("Estimated axis (model frame):", axis_model)
    print(f"Directed-axis error: {error:.10f} degrees")

    assert error < 1e-5


# ---------------------------------------------------------
# 5. NOISY TOOL-AXIS CALIBRATION
# ---------------------------------------------------------

def test_axis_calibration_noisy():
    """Evaluate the directed axis with noisy marker observations."""
    observations = simulate_axis_rotation(n_poses=36)
    noisy_observations = add_marker_noise(
        observations, seed=123
    )

    axis_tracker = calibrate_axis(noisy_observations)

    # Simulate independently noisy forward-motion observations.
    before, after = simulate_forward_motion()
    noisy_before = add_marker_noise(before, seed=124)
    noisy_after = add_marker_noise(after, seed=125)

    displacement = np.mean(
        noisy_after - noisy_before, axis=0
    )

    directed_axis = orient_axis(
        axis_tracker, displacement
    )

    # Include uncertainty from the noisy reference marker frame.
    axis_model = axis_to_model_frame(
        directed_axis, noisy_observations[0]
    )

    error = angular_error_degrees(axis_model, V_TRUE)

    print("\n--- Noisy Tool-Axis Calibration ---")
    print("Ground-truth axis (model frame):", V_TRUE)
    print("Estimated axis (model frame):", axis_model)
    print(f"Tool-axis angular error: {error:.4f} degrees")

    assert np.isfinite(error)


# ---------------------------------------------------------
# 6. ROTATED-POSE AXIS VALIDATION
# ---------------------------------------------------------

def test_axis_rotated_pose():
    """Verify axis recovery after rotating the complete scene."""
    R_test = rotation_matrix(
        np.array([1.0, 0.0, 0.0]),
        np.radians(45.0)
    )

    observations = simulate_axis_rotation(n_poses=36)

    # Apply the same rigid rotation to all marker observations.
    rotated_observations = observations @ R_test.T

    axis_tracker = calibrate_axis(rotated_observations)

    # Check the undirected axis line in the model frame.
    axis_model = axis_to_model_frame(
        axis_tracker, rotated_observations[0]
    )

    line_alignment = abs(np.dot(axis_model, V_TRUE))
    line_error = np.degrees(
        np.arccos(np.clip(line_alignment, 0.0, 1.0))
    )

    # Rotate the independently simulated forward movement.
    before, after = simulate_forward_motion()

    before_rotated = before @ R_test.T
    after_rotated = after @ R_test.T

    displacement = np.mean(
        after_rotated - before_rotated, axis=0
    )

    directed_axis = orient_axis(
        axis_tracker, displacement
    )

    directed_model = axis_to_model_frame(
        directed_axis, rotated_observations[0]
    )

    direction_error = angular_error_degrees(
        directed_model, V_TRUE
    )

    print("\n--- Rotated-Pose Axis Validation ---")
    print("Estimated axis (model frame):", directed_model)
    print(f"Axis-line error: {line_error:.10f} degrees")
    print(f"Directed-axis error: {direction_error:.10f} degrees")

    assert line_error < 1e-5
    assert direction_error < 1e-5


# ---------------------------------------------------------
# 7. REPEATED NOISY TOOL-TIP CALIBRATION
# ---------------------------------------------------------

def evaluate_tip_noise(n_trials=200):
    """Quantify tip-position error over independent noise trials."""
    observations = simulate_pivot(n_poses=30)
    errors = []

    for trial in range(n_trials):
        noisy_observations = add_marker_noise(
            observations, seed=1000 + trial
        )

        tip_marker, _, _, _ = estimate_tip(noisy_observations)
        tip_model = tip_to_model_frame(tip_marker, M1)

        errors.append(np.linalg.norm(tip_model - P_TRUE))

    errors = np.asarray(errors)

    print("\n--- Repeated Noisy Tool-Tip Calibration ---")
    print(f"Independent noise trials: {n_trials}")
    print_error_statistics(errors, "mm")

    return errors


# ---------------------------------------------------------
# 8. REPEATED NOISY TOOL-AXIS CALIBRATION
# ---------------------------------------------------------

def evaluate_axis_noise(n_trials=200):
    """
    Quantify directed-axis errors over independent noise trials.

    Model-frame error includes reconstruction of a noisy
    reference marker frame. Tracker-frame error evaluates
    the axis estimate before that frame conversion.
    """
    observations = simulate_axis_rotation(n_poses=36)
    before, after = simulate_forward_motion()

    model_errors = []
    tracker_errors = []

    for trial in range(n_trials):
        noisy_observations = add_marker_noise(
            observations, seed=2000 + trial
        )
        noisy_before = add_marker_noise(
            before, seed=3000 + trial
        )
        noisy_after = add_marker_noise(
            after, seed=4000 + trial
        )

        # Estimate the tracker-frame axis from noisy trajectories.
        axis_tracker = calibrate_axis(noisy_observations)

        displacement = np.mean(
            noisy_after - noisy_before, axis=0
        )

        directed_axis = orient_axis(
            axis_tracker, displacement
        )

        # Model-frame error with a noisy reference observation.
        axis_model = axis_to_model_frame(
            directed_axis, noisy_observations[0]
        )

        model_errors.append(
            angular_error_degrees(axis_model, V_TRUE)
        )

        # The reference orientation is identity in this simulation.
        # Evaluate the tracker-frame estimate separately.
        tracker_errors.append(
            angular_error_degrees(directed_axis, V_TRUE)
        )

    model_errors = np.asarray(model_errors)
    tracker_errors = np.asarray(tracker_errors)

    print("\n--- Repeated Noisy Tool-Axis Calibration ---")
    print(f"Independent noise trials: {n_trials}")

    print("\nModel-frame angular error:")
    print_error_statistics(model_errors, "degrees")

    print("\nTracker-frame angular error:")
    print_error_statistics(tracker_errors, "degrees")

    return model_errors, tracker_errors


# ---------------------------------------------------------
# 9. TOOL-TIP MOTION CONSTRAINTS
# ---------------------------------------------------------

def test_pivot_angular_coverage():
    """Check that pivot poses span widely separated orientations."""
    observations = simulate_pivot(n_poses=30)

    directions = []

    for markers in observations:
        R, _ = marker_frame(markers)
        directions.append(R @ V_TRUE)

    directions = np.asarray(directions)

    # Pairwise angular separation between needle directions.
    alignment = np.clip(
        directions @ directions.T, -1.0, 1.0
    )

    angles = np.degrees(np.arccos(alignment))
    max_angle = np.max(angles)

    print("\n--- Pivot Angular Coverage ---")
    print(f"Maximum angular separation: {max_angle:.2f} degrees")

    assert max_angle >= 60.0


# ---------------------------------------------------------
# 10. TOOL-AXIS MOTION CONSTRAINTS
# ---------------------------------------------------------

def test_axis_motion_constraints():
    """Verify planar, nondegenerate marker trajectories."""
    observations = simulate_axis_rotation(n_poses=36)

    print("\n--- Axis Motion Constraints ---")

    for marker_index in range(3):
        points = observations[:, marker_index, :]
        centered = points - np.mean(points, axis=0)

        # Two significant singular values indicate planar spread.
        # The third should be approximately zero for exact data.
        _, singular_values, _ = np.linalg.svd(
            centered, full_matrices=False
        )

        print(
            f"Marker {marker_index + 1} singular values: "
            f"{singular_values}"
        )

        assert singular_values[1] > 1e-8
        assert singular_values[2] < 1e-8


# ---------------------------------------------------------
# RUN ALL CALIBRATION TESTS
# ---------------------------------------------------------

if __name__ == "__main__":
    test_tip_calibration()
    test_tip_calibration_noisy()

    test_axis_calibration()
    test_axis_calibration_noisy()
    test_axis_rotated_pose()

    evaluate_tip_noise()
    evaluate_axis_noise()

    test_pivot_angular_coverage()
    test_axis_motion_constraints()
