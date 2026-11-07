
"""
Purpose of this file:
    Validate tool-tip and tool-axis calibration using perfect
    and noisy simulated optical-tracker observations.

    Tests include rigid registration, motion constraints,
    Gaussian noise, calibration accuracy, conditioning,
    and comparisons between original and improved methods.

    Ground-truth values are used only for simulation and
    evaluation, not inside the calibration algorithms.
"""

import numpy as np

from geometry import (
    marker_frame,
    rigid_registration,
    fit_plane_normal,
    angle_between_deg,
    rotation_error_deg
)

from calibration import (
    poses_from_observations,
    calibrate_tip,
    calibrate_axis,
    axis_to_model_frame,
    orient_axis,
    calibrate_tool
)

from simulation import (
    MARKERS,
    P_TRUE,
    V_TRUE,
    DEFAULT_TRANSLATION,
    NOISE_SIGMA,
    random_pose,
    simulate_pivot,
    simulate_axis_rotation,
    simulate_forward_motion,
    add_marker_noise
)


# Experiment Settings

N_TRIALS = 200
N_CONDITION_TRIALS = 100
N_REDRAW_TRIALS = 100

OFFSET_TIP = np.array([3.0, -2.0, 5.0])

# Assumed insertion depth for an illustrative calculation.
# NOT a measured clinical insertion depth.
INSERTION_DEPTH_MM = 100.0

# Empirical regression thresholds based on previous runs.
# The bounds are approximately 1.5-2 times the measured
# mean errors from the fixed and redrawn motion experiments.
TIP_MEAN_ERROR_LIMIT = 3.5
AXIS_MEAN_ERROR_LIMIT = 1.5


# Helper functions

def error_statistics(errors):
    """Calculate summary statistics for calibration errors."""
    errors = np.asarray(errors, dtype=float)

    return {
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "std": float(np.std(errors)),
        "p95": float(np.percentile(errors, 95))
    }


def print_error_statistics(errors, unit):
    stats = error_statistics(errors)

    print(f"Mean error: {stats['mean']:.4f} {unit}")
    print(f"Median error: {stats['median']:.4f} {unit}")
    print(f"Standard deviation: {stats['std']:.4f} {unit}")
    print(f"95th-percentile error: {stats['p95']:.4f} {unit}")


def marker_frame_pose(model_markers, observed_markers):
    """
    Estimate a model-to-tracker pose using the original
    Gram-Schmidt marker-frame construction.

    This method is retained as a comparison baseline.
    """
    R_model, t_model = marker_frame(model_markers)
    R_observed, t_observed = marker_frame(observed_markers)

    R = R_observed @ R_model.T
    t = t_observed - R @ t_model

    return R, t


def estimate_poses_marker_frame(model_markers, observations):
    """Estimate multiple poses using the original frame method."""
    rotations = []
    translations = []

    for markers in observations:
        R, t = marker_frame_pose(model_markers, markers)

        rotations.append(R)
        translations.append(t)

    return np.asarray(rotations), np.asarray(translations)


def original_axis_estimate(observations):
    """
    Estimate the tracker-frame axis using the original
    method of fitting three separate trajectory planes.

    Plane normals are aligned before averaging.
    """
    normals = []

    for marker_index in range(3):
        trajectory = observations[:, marker_index, :]
        normal = fit_plane_normal(trajectory)
        normals.append(normal)

    reference = normals[0]

    for i in range(1, len(normals)):
        if np.dot(normals[i], reference) < 0:
            normals[i] = -normals[i]

    axis = np.mean(normals, axis=0)

    return axis / np.linalg.norm(axis)


def forward_direction_model(model_markers, before, after):
    """
    Convert a measured forward displacement from tracker
    coordinates into tool model coordinates.
    """
    R_before, _ = rigid_registration(model_markers, before)

    displacement_tracker = (
        np.mean(after, axis=0)
        - np.mean(before, axis=0)
    )

    return R_before.T @ displacement_tracker


def motion_coverage(rotations, axis=V_TRUE):
    """
    Calculate the maximum pairwise angular separation
    between tool-axis directions during pivot motion.
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    directions = np.array([
        R @ axis for R in rotations
    ])

    max_angle = 0.0

    for i in range(len(directions)):
        for j in range(i + 1, len(directions)):
            angle = angle_between_deg(
                directions[i], directions[j]
            )

            max_angle = max(max_angle, angle)

    return max_angle


def maximum_nominal_tilt(rotations, axis=V_TRUE):
    """
    Measure the largest tilt away from the nominal
    tool-axis direction.
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    return max(
        angle_between_deg(R @ axis, axis)
        for R in rotations
    )


def make_perfect_experiments(
    rng,
    tip=OFFSET_TIP,
    max_tilt_deg=None
):
    """
    Generate perfect pivot, axial-rotation, and forward
    translation observations.
    """
    pivot = simulate_pivot(
        n_poses=30,
        tip=tip,
        max_tilt_deg=max_tilt_deg,
        rng=rng
    )

    base_R, base_t = random_pose(rng)

    axis = simulate_axis_rotation(
        n_poses=36,
        tip=tip,
        base_rotation=base_R,
        base_translation=base_t
    )

    before, after = simulate_forward_motion(
        distance=5.0,
        base_rotation=base_R,
        base_translation=base_t
    )

    return pivot, axis, before, after


def add_experiment_noise(pivot, axis, before, after, rng):
    """
    Add independent Gaussian marker-localization noise
    using one continuously advancing random generator.
    """
    return (
        add_marker_noise(pivot, rng=rng),
        add_marker_noise(axis, rng=rng),
        add_marker_noise(before, rng=rng),
        add_marker_noise(after, rng=rng)
    )


# Rigid-Registration Tests

def test_rigid_registration():
    """Verify exact rigid-pose recovery with perfect marker data."""
    rng = np.random.default_rng(101)

    R_true, t_true = random_pose(rng)

    observed = MARKERS @ R_true.T + t_true

    R_est, t_est, rms = rigid_registration(
        MARKERS, observed, return_rms=True
    )

    rotation_error = rotation_error_deg(R_est, R_true)
    translation_error = np.linalg.norm(t_est - t_true)

    assert rotation_error < 1e-8
    assert translation_error < 1e-8
    assert rms < 1e-8

    assert np.allclose(
        R_est.T @ R_est, np.eye(3), atol=1e-10
    )

    assert np.isclose(
        np.linalg.det(R_est), 1.0, atol=1e-10
    )

    print("\n--- Rigid-Registration Validation ---")
    print(f"Rotation error: {rotation_error:.10f} degrees")
    print(f"Translation error: {translation_error:.10f} mm")
    print(f"Registration RMS: {rms:.10f} mm")
    print(f"Rotation determinant: {np.linalg.det(R_est):.10f}")


def test_reflection_correction():
    """
    Verify that rigid registration enforces a proper rotation.

    Four non-coplanar points are used because a mirrored
    planar point set can also be matched by a proper rotation.
    """
    model_points = np.array([
        [0.0, 0.0, 0.0],
        [6.0, 0.0, 0.0],
        [0.0, 6.0, 0.0],
        [0.0, 0.0, 6.0]
    ])

    reflection = np.diag([-1.0, 1.0, 1.0])

    mirrored_points = model_points @ reflection.T

    R, _, rms = rigid_registration(
        model_points,
        mirrored_points,
        return_rms=True
    )

    assert np.allclose(
        R.T @ R, np.eye(3), atol=1e-10
    )

    assert np.isclose(
        np.linalg.det(R), 1.0, atol=1e-10
    )

    # A proper rotation cannot exactly reproduce this
    # reflection of non-coplanar points.
    assert rms > 0.1

    print("\n--- Non-Coplanar Reflection Test ---")
    print(f"Rotation determinant: {np.linalg.det(R):.6f}")
    print(f"Registration RMS: {rms:.4f} mm")
    print("Reflection correction: passed")


def compare_registration_methods(n_trials=300):
    """
    Compare SVD rigid registration and Gram-Schmidt frame
    construction using identical noisy observations.
    """
    rng = np.random.default_rng(102)

    svd_errors = []
    frame_errors = []
    rms_values = []

    for _ in range(n_trials):
        R_true, t_true = random_pose(rng)

        perfect = MARKERS @ R_true.T + t_true
        noisy = add_marker_noise(perfect, rng=rng)

        R_svd, _, rms = rigid_registration(
            MARKERS, noisy, return_rms=True
        )

        R_frame, _ = marker_frame_pose(MARKERS, noisy)

        svd_errors.append(
            rotation_error_deg(R_svd, R_true)
        )

        frame_errors.append(
            rotation_error_deg(R_frame, R_true)
        )

        rms_values.append(rms)

    svd_mean = np.mean(svd_errors)
    frame_mean = np.mean(frame_errors)

    print("\n--- Rigid-Registration Noise Comparison ---")
    print(f"Independent trials: {n_trials}")
    print(f"SVD mean rotation error: {svd_mean:.4f} degrees")
    print(f"Gram-Schmidt mean error: {frame_mean:.4f} degrees")
    print(f"Mean registration RMS: {np.mean(rms_values):.4f} mm")

    # Regression bounds include a margin above the
    # previously measured results for this seeded test.
    assert svd_mean < frame_mean
    assert svd_mean < 8.0


# Gaussian noise validation

def test_marker_noise():
    """Verify the 95% 3D marker-localization requirement."""
    rng = np.random.default_rng(103)

    perfect = np.zeros((100000, 3))
    noisy = add_marker_noise(perfect, rng=rng)

    errors = np.linalg.norm(noisy - perfect, axis=1)

    percentile_95 = np.percentile(errors, 95)
    fraction_below = np.mean(errors < 1.0)

    print("\n--- Gaussian Marker-Noise Validation ---")
    print(f"Coordinate sigma: {NOISE_SIGMA:.6f} mm")
    print(f"95th-percentile error: {percentile_95:.4f} mm")
    print(f"Fraction below 1 mm: {fraction_below:.4%}")

    # Allow sampling variation around the theoretical 95%.
    assert abs(fraction_below - 0.95) < 0.005


# Perfect Tool-Tip Calibration

def test_tip_calibration():
    """Verify exact tip recovery using unrestricted pivot motion."""
    rng = np.random.default_rng(104)

    observations = simulate_pivot(
        n_poses=30, rng=rng
    )

    R, t, _ = poses_from_observations(MARKERS, observations)

    tip, pivot, pivot_rms, condition = calibrate_tip(R, t)

    error = np.linalg.norm(tip - P_TRUE)
    coverage = motion_coverage(R)

    print("\n--- Perfect Tool-Tip Calibration ---")
    print("Ground-truth tip:", P_TRUE)
    print("Estimated tip:", tip)
    print(f"Tip error: {error:.10f} mm")
    print(f"Pivot RMS: {pivot_rms:.10f} mm")
    print(f"Condition number: {condition:.4f}")
    print(f"Maximum angular separation: {coverage:.2f} degrees")

    assert error < 1e-8
    assert pivot_rms < 1e-8

    assert np.linalg.norm(
        pivot - DEFAULT_TRANSLATION
    ) < 1e-8

    assert coverage > 60.0


def test_offset_tip_calibration():
    """Verify exact recovery with a nonzero ground-truth tip."""
    rng = np.random.default_rng(105)

    observations = simulate_pivot(
        n_poses=30,
        tip=OFFSET_TIP,
        rng=rng
    )

    R, t, _ = poses_from_observations(MARKERS, observations)

    tip, pivot, pivot_rms, _ = calibrate_tip(R, t)

    error = np.linalg.norm(tip - OFFSET_TIP)

    print("\n--- Nonzero-Tip Calibration ---")
    print("Ground-truth tip:", OFFSET_TIP)
    print("Estimated tip:", tip)
    print(f"Tool-tip error: {error:.10f} mm")

    assert error < 1e-8
    assert pivot_rms < 1e-8

    assert np.linalg.norm(
        pivot - DEFAULT_TRANSLATION
    ) < 1e-8


# Perfect tool axis calibration

def test_axis_calibration():
    """Verify axis recovery under a random starting orientation."""
    rng = np.random.default_rng(106)

    base_R, base_t = random_pose(rng)

    observations = simulate_axis_rotation(
        n_poses=36,
        base_rotation=base_R,
        base_translation=base_t
    )

    before, after = simulate_forward_motion(
        distance=5.0,
        base_rotation=base_R,
        base_translation=base_t
    )

    rotations, _, _ = poses_from_observations(
        MARKERS, observations
    )

    tracker_axis, plane_rms = calibrate_axis(observations)

    model_axis = axis_to_model_frame(
        tracker_axis, rotations
    )

    displacement_model = forward_direction_model(
        MARKERS, before, after
    )

    directed_axis = orient_axis(
        model_axis, displacement_model
    )

    tracker_error = angle_between_deg(
        tracker_axis,
        base_R @ V_TRUE,
        directed=False
    )

    model_error = angle_between_deg(
        directed_axis, V_TRUE
    )

    print("\n--- Perfect Tool-Axis Calibration ---")
    print("Ground-truth axis:", V_TRUE)
    print("Estimated axis:", directed_axis)
    print(f"Tracker-frame axis-line error: {tracker_error:.10f} degrees")
    print(f"Directed model-frame error: {model_error:.10f} degrees")
    print(f"Plane RMS: {plane_rms:.10f} mm")

    assert tracker_error < 1e-6
    assert model_error < 1e-6
    assert plane_rms < 1e-8


def test_axis_motion_constraints():
    """Verify planar circular marker trajectories."""
    rng = np.random.default_rng(107)

    base_R, base_t = random_pose(rng)

    observations = simulate_axis_rotation(
        n_poses=36,
        base_rotation=base_R,
        base_translation=base_t
    )

    print("\n--- Axis Motion Constraints ---")

    for marker_index in range(3):
        points = observations[:, marker_index, :]
        centered = points - np.mean(points, axis=0)

        _, singular_values, _ = np.linalg.svd(
            centered, full_matrices=False
        )

        print(
            f"Marker {marker_index + 1} singular values: "
            f"{singular_values}"
        )

        assert singular_values[1] > 1e-8
        assert singular_values[2] < 1e-8

    print("Axial rotation: 36 poses over one full turn")

# Repeated noisy tool tip calibration

def evaluate_tip_noise(n_trials=N_TRIALS):
    """
    Compare original and improved tip calibration using
    identical noisy marker observations.
    """
    motion_rng = np.random.default_rng(108)
    noise_rng = np.random.default_rng(109)

    perfect = simulate_pivot(
        n_poses=30, rng=motion_rng
    )

    svd_errors = []
    original_errors = []

    pivot_residuals = []
    conditions = []
    registration_residuals = []

    for _ in range(n_trials):
        noisy = add_marker_noise(perfect, rng=noise_rng)

        R, t, pose_rms = poses_from_observations(
            MARKERS, noisy
        )

        tip, _, pivot_rms, condition = calibrate_tip(R, t)

        svd_errors.append(
            np.linalg.norm(tip - P_TRUE)
        )

        pivot_residuals.append(pivot_rms)
        conditions.append(condition)
        registration_residuals.append(np.mean(pose_rms))

        # Original Gram-Schmidt pose estimation.
        old_R, old_t = estimate_poses_marker_frame(
            MARKERS, noisy
        )

        old_tip, _, _, _ = calibrate_tip(old_R, old_t)

        original_errors.append(
            np.linalg.norm(old_tip - P_TRUE)
        )

    print("\n--- Repeated Noisy Tool-Tip Calibration ---")
    print(f"Independent noise trials: {n_trials}")

    print("\nSVD rigid-registration method:")
    print_error_statistics(svd_errors, "mm")

    print("\nOriginal Gram-Schmidt method:")
    print_error_statistics(original_errors, "mm")

    print(
        f"\nMean pivot RMS: "
        f"{np.mean(pivot_residuals):.4f} mm"
    )

    print(
        f"Mean registration RMS: "
        f"{np.mean(registration_residuals):.4f} mm"
    )

    print(
        f"Mean condition number: "
        f"{np.mean(conditions):.4f}"
    )

    # Empirical regression bounds for this seeded experiment.
    assert np.mean(svd_errors) < 0.7
    assert np.mean(svd_errors) < np.mean(original_errors)

    assert np.all(np.isfinite(svd_errors))
    assert np.all(np.isfinite(conditions))

    return np.asarray(svd_errors)


# Repeated noisy tool-axis calibration

def evaluate_axis_noise(n_trials=N_TRIALS):
    """
    Compare original and improved axis-calibration methods
    using identical noisy observations in each trial.
    """
    motion_rng = np.random.default_rng(110)
    noise_rng = np.random.default_rng(111)

    base_R, base_t = random_pose(motion_rng)

    perfect_axis = simulate_axis_rotation(
        n_poses=36,
        base_rotation=base_R,
        base_translation=base_t
    )

    perfect_before, perfect_after = simulate_forward_motion(
        distance=5.0,
        base_rotation=base_R,
        base_translation=base_t
    )

    true_tracker_axis = base_R @ V_TRUE

    tracker_errors_old = []
    tracker_errors_new = []

    model_errors_original = []
    model_errors_single = []
    model_errors_averaged = []

    plane_residuals = []
    registration_residuals = []

    for _ in range(n_trials):
        noisy_axis = add_marker_noise(
            perfect_axis, rng=noise_rng
        )

        noisy_before = add_marker_noise(
            perfect_before, rng=noise_rng
        )

        noisy_after = add_marker_noise(
            perfect_after, rng=noise_rng
        )

        rotations, _, pose_rms = poses_from_observations(
            MARKERS, noisy_axis
        )

        displacement_model = forward_direction_model(
            MARKERS, noisy_before, noisy_after
        )

        old_tracker_axis = original_axis_estimate(
            noisy_axis
        )

        new_tracker_axis, plane_rms = calibrate_axis(
            noisy_axis
        )

        tracker_errors_old.append(
            angle_between_deg(
                old_tracker_axis,
                true_tracker_axis,
                directed=False
            )
        )

        tracker_errors_new.append(
            angle_between_deg(
                new_tracker_axis,
                true_tracker_axis,
                directed=False
            )
        )

        # Original pipeline:
        # separate planes and one Gram-Schmidt pose.
        old_R, _ = marker_frame_pose(
            MARKERS, noisy_axis[0]
        )

        old_model_axis = orient_axis(
            old_R.T @ old_tracker_axis,
            displacement_model
        )

        model_errors_original.append(
            angle_between_deg(old_model_axis, V_TRUE)
        )

        # Common-plane fitting with one SVD pose.
        single_model_axis = orient_axis(
            rotations[0].T @ new_tracker_axis,
            displacement_model
        )

        model_errors_single.append(
            angle_between_deg(single_model_axis, V_TRUE)
        )

        # Common-plane fitting with all 36 SVD poses.
        averaged_model_axis = axis_to_model_frame(
            new_tracker_axis, rotations
        )

        averaged_model_axis = orient_axis(
            averaged_model_axis,
            displacement_model
        )

        model_errors_averaged.append(
            angle_between_deg(averaged_model_axis, V_TRUE)
        )

        plane_residuals.append(plane_rms)
        registration_residuals.append(np.mean(pose_rms))

    print("\n--- Repeated Noisy Tool-Axis Calibration ---")
    print(f"Independent noise trials: {n_trials}")

    print("\nTracker-frame: original plane normals")
    print_error_statistics(tracker_errors_old, "degrees")

    print("\nTracker-frame: common-plane fitting")
    print_error_statistics(tracker_errors_new, "degrees")

    print("\nModel-frame: original pipeline")
    print_error_statistics(model_errors_original, "degrees")

    print("\nModel-frame: common plane + single pose")
    print_error_statistics(model_errors_single, "degrees")

    print("\nModel-frame: common plane + averaged poses")
    print_error_statistics(model_errors_averaged, "degrees")

    print(
        f"\nMean plane RMS: "
        f"{np.mean(plane_residuals):.4f} mm"
    )

    print(
        f"Mean registration RMS: "
        f"{np.mean(registration_residuals):.4f} mm"
    )

    # Empirical regression bounds based on previous runs.
    assert np.mean(model_errors_averaged) < 1.5

    assert (
        np.mean(model_errors_averaged)
        < np.mean(model_errors_original)
    )

    assert (
        np.mean(tracker_errors_new)
        < np.mean(tracker_errors_old)
    )

    return np.asarray(model_errors_averaged)

# Pivot Conditioning Sweep

def test_pivot_conditioning():
    """
    Examine the effect of restricted pivot motion.

    Each tilt setting uses one fixed perfect motion sequence
    and 100 independent Gaussian-noise realizations.

    Mean, median and 95th-percentile tip errors are reported
    because narrow pivot cones can be poorly conditioned.
    """
    print("\n--- Pivot Conditioning Sweep ---")
    print(f"Noise trials per tilt: {N_CONDITION_TRIALS}")

    print(
        f"{'Tilt':>7} "
        f"{'Coverage':>10} "
        f"{'Cond(A)':>10} "
        f"{'Mean':>10} "
        f"{'Median':>10} "
        f"{'95th pct':>10}"
    )

    condition_values = []

    for max_tilt in [10, 20, 30, 45, 90, 180]:
        motion_rng = np.random.default_rng(200 + max_tilt)
        noise_rng = np.random.default_rng(300 + max_tilt)

        perfect = simulate_pivot(
            n_poses=30,
            max_tilt_deg=max_tilt,
            rng=motion_rng
        )

        perfect_R, perfect_t, _ = poses_from_observations(
            MARKERS, perfect
        )

        coverage = motion_coverage(perfect_R)

        _, _, _, condition = calibrate_tip(
            perfect_R, perfect_t
        )

        errors = []

        for _ in range(N_CONDITION_TRIALS):
            noisy = add_marker_noise(
                perfect, rng=noise_rng
            )

            R, t, _ = poses_from_observations(
                MARKERS, noisy
            )

            tip, _, _, _ = calibrate_tip(R, t)

            errors.append(np.linalg.norm(tip - P_TRUE))

        stats = error_statistics(errors)
        condition_values.append(condition)

        print(
            f"{max_tilt:>6}° "
            f"{coverage:>9.2f}° "
            f"{condition:>10.2f} "
            f"{stats['mean']:>9.4f} "
            f"{stats['median']:>10.4f} "
            f"{stats['p95']:>10.4f}"
        )

        assert np.all(np.isfinite(errors))
        assert condition > 0

    # Narrow pivot cones should have worse conditioning
    # than sufficiently wide motion sequences.
    assert condition_values[0] > condition_values[-1]

    print("\nTip errors are reported in millimetres.")
    print("Condition numbers use the perfect simulated poses.")
    print("Each tilt uses fixed motions and 100 noise realizations.")

# Perfect end to end calibration

def test_complete_calibration():
    """
    Validate the full calibration pipeline using a nonzero
    tip and non-identity tool orientations.

    Forward motion is performed at a different orientation
    to check coordinate-frame consistency.
    """
    rng = np.random.default_rng(112)

    pivot, axis, _, _ = make_perfect_experiments(
        rng, tip=OFFSET_TIP
    )

    forward_R, forward_t = random_pose(rng)

    before, after = simulate_forward_motion(
        distance=5.0,
        base_rotation=forward_R,
        base_translation=forward_t
    )

    result = calibrate_tool(
        MARKERS,
        pivot,
        axis,
        before,
        after
    )

    tip_error = np.linalg.norm(
        result["tip_model"] - OFFSET_TIP
    )

    axis_error = angle_between_deg(
        result["axis_model"], V_TRUE
    )

    
    # insertion points from marker centroid toward the tip.
    marker_centroid = np.mean(MARKERS, axis=0)
    tip_direction = result["tip_model"] - marker_centroid

    sign_alignment = np.dot(
        result["axis_model"], tip_direction
    )

    print("\n--- Perfect End-to-End Calibration ---")
    print("Ground-truth tip:", OFFSET_TIP)
    print("Estimated tip:", result["tip_model"])
    print("Estimated directed axis:", result["axis_model"])
    print(f"Tool-tip error: {tip_error:.10f} mm")
    print(f"Tool-axis error: {axis_error:.10f} degrees")
    print(f"Pivot condition number: {result['pivot_condition']:.4f}")
    print(f"Pivot RMS: {result['pivot_rms']:.10f} mm")
    print(f"Plane RMS: {result['plane_rms']:.10f} mm")

    assert tip_error < 1e-8
    assert axis_error < 1e-6
    assert sign_alignment > 0

    assert len(
        result["pivot_registration_rms_per_pose"]
    ) == 30

    assert len(
        result["axis_registration_rms_per_pose"]
    ) == 36


# Noisy end to end calibration

def evaluate_noisy_end_to_end(
    n_trials=N_TRIALS,
    redraw_motions=False
):
    """
    Evaluate the complete calibration pipeline under noise.

    Simulation settings:
        - Nonzero tool-tip position.
        - Pivot motion limited to a 30-degree cone.
        - 36 axial-rotation poses.
        - Random non-identity starting orientation.
        - Forward translation of 5 mm.
    """
    motion_rng = np.random.default_rng(
        501 if redraw_motions else 401
    )

    noise_rng = np.random.default_rng(
        502 if redraw_motions else 402
    )

    if not redraw_motions:
        perfect_data = make_perfect_experiments(
            motion_rng,
            tip=OFFSET_TIP,
            max_tilt_deg=30.0
        )

    # Error measurements.
    tip_errors = []
    axial_tip_errors = []
    perpendicular_tip_errors = []
    axis_errors = []

    # Calibration diagnostics.
    pivot_residuals = []
    plane_residuals = []
    pivot_conditions = []

    pivot_registration_rms = []
    axis_registration_rms = []

    # Motion-coverage measurements.
    coverages = []
    achieved_tilts = []

    # Ground-truth axis is used only for evaluating errors.
    true_axis = V_TRUE / np.linalg.norm(V_TRUE)

    for _ in range(n_trials):
        if redraw_motions:
            perfect_data = make_perfect_experiments(
                motion_rng,
                tip=OFFSET_TIP,
                max_tilt_deg=30.0
            )

        pivot, axis, before, after = perfect_data

        # Measure actual coverage using perfect pivot poses.
        perfect_R, _, _ = poses_from_observations(
            MARKERS, pivot
        )

        coverages.append(
            motion_coverage(perfect_R)
        )

        achieved_tilts.append(
            maximum_nominal_tilt(perfect_R)
        )

        # Add independent noise to all tracked observations.
        noisy_data = add_experiment_noise(
            pivot,
            axis,
            before,
            after,
            noise_rng
        )

        # Run the complete calibration pipeline.
        result = calibrate_tool(
            MARKERS,
            *noisy_data
        )

       # Tool Tip precision error

        tip_error_vector = (
            result["tip_model"] - OFFSET_TIP
        )

        tip_error = np.linalg.norm(tip_error_vector)

        # Component along the true needle axis.
        axial_component = np.dot(
            tip_error_vector, true_axis
        )

        # Component perpendicular to the true needle axis.
        perpendicular_component = (
            tip_error_vector - axial_component * true_axis
        )

        axial_tip_error = abs(axial_component)

        perpendicular_tip_error = np.linalg.norm(
            perpendicular_component
        )

        tip_errors.append(tip_error)
        axial_tip_errors.append(axial_tip_error)
        perpendicular_tip_errors.append(
            perpendicular_tip_error
        )

        # Directed tool-axis angular error

        axis_error = angle_between_deg(
            result["axis_model"], V_TRUE
        )

        axis_errors.append(axis_error)

        # Calibration diagnostics

        pivot_residuals.append(
            result["pivot_rms"]
        )

        plane_residuals.append(
            result["plane_rms"]
        )

        pivot_conditions.append(
            result["pivot_condition"]
        )

        pivot_registration_rms.append(
            result["pivot_registration_rms"]
        )

        axis_registration_rms.append(
            result["axis_registration_rms"]
        )

    # Print experiment results

    if redraw_motions:
        print(
            "\n--- Noisy End-to-End Calibration: "
            "Redrawn Motions ---"
        )
    else:
        print(
            "\n--- Noisy End-to-End Calibration: "
            "Fixed Motions ---"
        )

    print(f"Independent noise trials: {n_trials}")
    print("Ground-truth tip:", OFFSET_TIP)
    print("Requested maximum pivot tilt: 30 degrees")

    if redraw_motions:
        print(
            f"Mean achieved maximum tilt: "
            f"{np.mean(achieved_tilts):.2f} degrees"
        )

        print(
            f"Mean maximum pairwise angular separation: "
            f"{np.mean(coverages):.2f} degrees"
        )
    else:
        print(
            f"Achieved maximum tilt: "
            f"{achieved_tilts[0]:.2f} degrees"
        )

        print(
            f"Maximum pairwise angular separation: "
            f"{coverages[0]:.2f} degrees"
        )

    print("\nTotal tool-tip position error:")
    print_error_statistics(tip_errors, "mm")

    print("\nAbsolute axial tip-position error:")
    print_error_statistics(axial_tip_errors, "mm")

    print("\nPerpendicular tip-position error:")
    print_error_statistics(perpendicular_tip_errors, "mm")

    print("\nDirected model-frame axis error:")
    print_error_statistics(axis_errors, "degrees")

    print(
        f"\nMean pivot condition number: "
        f"{np.mean(pivot_conditions):.4f}"
    )

    print(
        f"Mean pivot RMS: "
        f"{np.mean(pivot_residuals):.4f} mm"
    )

    print(
        f"Mean plane RMS: "
        f"{np.mean(plane_residuals):.4f} mm"
    )

    print(
        f"Mean pivot registration RMS: "
        f"{np.mean(pivot_registration_rms):.4f} mm"
    )

    print(
        f"Mean axis registration RMS: "
        f"{np.mean(axis_registration_rms):.4f} mm"
    )

    # 
    # Illustrative insertion-depth calculation

    # Lateral deviation due to angular error alone,
    # assuming straight insertion over a 100 mm depth.
    lateral_deviations = (
        INSERTION_DEPTH_MM
        * np.tan(np.radians(axis_errors))
    )

    print(
        f"\n--- Illustrative Lateral Deviation "
        f"at {INSERTION_DEPTH_MM:.0f} mm ---"
    )

    print_error_statistics(lateral_deviations, "mm")

    print(
        "This calculation assumes straight insertion over "
        "the stated depth and isolates axis-angle error."
    )

    print(
        "Tip-position error is reported separately. "
        "These values are not clinical targeting errors."
    )

    
    # Validation assertions
    

    assert np.all(np.isfinite(tip_errors))
    assert np.all(np.isfinite(axial_tip_errors))
    assert np.all(np.isfinite(perpendicular_tip_errors))
    assert np.all(np.isfinite(axis_errors))
    assert np.all(np.isfinite(pivot_conditions))

    # Verify the geometric decomposition:
    # total_error^2 = axial_error^2 + perpendicular_error^2.
    assert np.allclose(
        np.asarray(tip_errors)**2,
        np.asarray(axial_tip_errors)**2
        + np.asarray(perpendicular_tip_errors)**2,
        atol=1e-8
    )

    # Empirical regression thresholds based on the measured
    # fixed-motion and redrawn-motion simulation results.
    # Bounds allow approximately 1.5-2x the measured means.
    assert np.mean(tip_errors) < TIP_MEAN_ERROR_LIMIT
    assert np.mean(axis_errors) < AXIS_MEAN_ERROR_LIMIT

    return {
        "tip_errors": np.asarray(tip_errors),
        "axial_tip_errors": np.asarray(axial_tip_errors),
        "perpendicular_tip_errors": np.asarray(
            perpendicular_tip_errors
        ),
        "axis_errors": np.asarray(axis_errors),
        "lateral_deviations": np.asarray(lateral_deviations)
    }


# RUN ALL TESTS

if __name__ == "__main__":

    # Rigid registration and noise-model validation.
    test_rigid_registration()
    test_reflection_correction()
    compare_registration_methods()
    test_marker_noise()

    # Perfect tool-tip and tool-axis calibration.
    test_tip_calibration()
    test_offset_tip_calibration()
    test_axis_calibration()
    test_axis_motion_constraints()

    # Original versus improved calibration comparisons.
    evaluate_tip_noise()
    evaluate_axis_noise()

    # Restricted pivot-motion conditioning experiment.
    test_pivot_conditioning()

    # Perfect end-to-end calibration.
    test_complete_calibration()

    # Noisy end-to-end calibration using fixed motions.
    evaluate_noisy_end_to_end(
        n_trials=N_TRIALS,
        redraw_motions=False
    )

    # Additional experiment with redrawn motions.
    evaluate_noisy_end_to_end(
        n_trials=N_REDRAW_TRIALS,
        redraw_motions=True
    )

    print("\n--- All Calibration Tests Passed ---")
