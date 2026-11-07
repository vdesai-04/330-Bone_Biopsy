
"""
Purpose of this file:
    Generate simulated optical-tracker observations from the
    ground-truth biopsy tool model.

    Simulations include pivot motion, axial rotation, forward
    translation, and Gaussian marker-localization noise.
"""

import numpy as np

from geometry import rotation_matrix, rigid_registration

# Ground-Truth Tool Model

# Marker positions in the tool model frame (mm).
M1 = np.array([2.0, 18.0, 0.0])
M2 = np.array([8.0, 18.0, 0.0])
M3 = np.array([2.0, 24.0, 0.0])

MARKERS = np.array([M1, M2, M3])

# Ground-truth tip position and axis direction.
P_TRUE = np.array([0.0, 0.0, 0.0])
V_TRUE = np.array([0.0, -1.0, 0.0])

# Default tracker-frame translation (mm).
DEFAULT_TRANSLATION = np.array([10.0, 20.0, 30.0])

# For isotropic 3D Gaussian noise:
# ||error||^2 / sigma^2 follows a chi-squared distribution
# with three degrees of freedom.
#
# The 95th percentile of chi-squared(3) is approximately
# 7.8147279. Setting the 95th-percentile 3D error to 1 mm
# gives sigma = 1 / sqrt(7.8147279).
NOISE_SIGMA = 1.0 / np.sqrt(7.814727903251179)


# Random tool pose

def random_pose(rng):
    """
    Generate a random rigid tool pose.

    The rotation is generated using a random axis and angle.
    Translation components are sampled uniformly between
    -50 mm and 50 mm.
    """
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)

    angle = rng.uniform(-np.pi, np.pi)

    R = rotation_matrix(axis, angle)

    t = rng.uniform(-50.0, 50.0, size=3)

    return R, t


# pivot-motion simulation

def simulate_pivot(
    n_poses=30,
    tip=None,
    axis=None,
    max_tilt_deg=None,
    rng=None
):
    """
    Simulate rotation about a stationary tool tip.

    Motion constraint:
        The tool changes orientation while the needle tip
        remains fixed in tracker coordinates.
    """
    if n_poses < 3:
        raise ValueError("At least three pivot poses are required.")

    if rng is None:
        rng = np.random.default_rng()

    if tip is None:
        tip = P_TRUE

    if axis is None:
        axis = V_TRUE

    tip = np.asarray(tip, dtype=float)
    axis = np.asarray(axis, dtype=float)

    if tip.shape != (3,) or axis.shape != (3,):
        raise ValueError("Tip and axis must be 3D vectors.")

    if np.linalg.norm(axis) < 1e-10:
        raise ValueError("Nominal tool axis must be nonzero.")

    axis = axis / np.linalg.norm(axis)

    if max_tilt_deg is not None:
        if not 0 < max_tilt_deg <= 180:
            raise ValueError(
                "Maximum tilt must be between 0 and 180 degrees."
            )

        # Construct two perpendicular directions spanning
        # the plane normal to the nominal tool axis.
        reference = np.eye(3)[np.argmin(np.abs(axis))]

        u = np.cross(axis, reference)
        u /= np.linalg.norm(u)

        v = np.cross(axis, u)

    # Stationary pivot position in tracker coordinates.
    pivot = DEFAULT_TRANSLATION.copy()

    observations = []

    for _ in range(n_poses):

        if max_tilt_deg is None:
            # Unrestricted random 3D orientation.
            rotation_axis = rng.normal(size=3)
            rotation_axis /= np.linalg.norm(rotation_axis)

            angle = rng.uniform(-np.pi, np.pi)

            R = rotation_matrix(rotation_axis, angle)

        else:
            # Choose a random tilt direction perpendicular
            # to the nominal tool axis.
            azimuth = rng.uniform(0.0, 2.0 * np.pi)

            tilt_axis = (
                np.cos(azimuth) * u
                + np.sin(azimuth) * v
            )

            # Sample the tilt uniformly in angle.
            tilt_angle = rng.uniform(
                0.0, np.radians(max_tilt_deg)
            )

            R_tilt = rotation_matrix(
                tilt_axis, tilt_angle
            )

            # Allow independent spinning around the tool axis.
            spin_angle = rng.uniform(0.0, 2.0 * np.pi)

            R_spin = rotation_matrix(
                axis, spin_angle
            )

            R = R_tilt @ R_spin

        # Enforce the stationary-tip constraint:
        # R @ tip + t = pivot
        t = pivot - R @ tip

        # Transform model markers into tracker coordinates.
        markers_tracker = MARKERS @ R.T + t

        observations.append(markers_tracker)

    return np.asarray(observations)

# Tool-Axis Rotation Simulation

def simulate_axis_rotation(
    n_poses=36,
    tip=None,
    axis=None,
    base_rotation=None,
    base_translation=None
):
    """
    Simulate rotation about the tool's longitudinal axis.

    Motion constraint:
        The tool rotates around the axis line through its tip,
        while that axis remains fixed in tracker coordinates.

    Each marker traces a circle in a plane perpendicular
    to the tool axis. All poses share the same axis line.
    """
    if n_poses < 3:
        raise ValueError("At least three rotation poses are required.")

    if tip is None:
        tip = P_TRUE

    if axis is None:
        axis = V_TRUE

    if base_rotation is None:
        base_rotation = np.eye(3)

    if base_translation is None:
        base_translation = DEFAULT_TRANSLATION

    tip = np.asarray(tip, dtype=float)
    axis = np.asarray(axis, dtype=float)
    base_rotation = np.asarray(base_rotation, dtype=float)
    base_translation = np.asarray(base_translation, dtype=float)

    if tip.shape != (3,) or axis.shape != (3,):
        raise ValueError("Tip and axis must be 3D vectors.")

    if np.linalg.norm(axis) < 1e-10:
        raise ValueError("Tool axis must be nonzero.")

    if base_rotation.shape != (3, 3):
        raise ValueError("Base rotation must have shape (3, 3).")

    if base_translation.shape != (3,):
        raise ValueError("Base translation must be a 3D vector.")

    axis = axis / np.linalg.norm(axis)

    # Use a full rotation without repeating the first pose.
    angles = np.linspace(
        0.0, 2.0 * np.pi, n_poses, endpoint=False
    )

    observations = []

    for angle in angles:
        # Rotate around the model-frame axis direction.
        R_axis = rotation_matrix(axis, angle)

        # Rotate markers about the axis line through the tip.
        rotated_markers = (
            (MARKERS - tip) @ R_axis.T + tip
        )

        # Apply the shared model-to-tracker base pose.
        markers_tracker = (
            rotated_markers @ base_rotation.T
            + base_translation
        )

        observations.append(markers_tracker)

    return np.asarray(observations)

# Known forward motion simulation

def simulate_forward_motion(
    distance=5.0,
    axis=None,
    base_rotation=None,
    base_translation=None
):
    """
    Simulate forward translation along the tool axis.

    Motion constraint:
        The tool translates in its insertion direction
        without changing orientation.
    """
    if distance <= 0:
        raise ValueError("Forward distance must be positive.")

    if axis is None:
        axis = V_TRUE

    if base_rotation is None:
        base_rotation = np.eye(3)

    if base_translation is None:
        base_translation = DEFAULT_TRANSLATION

    axis = np.asarray(axis, dtype=float)
    base_rotation = np.asarray(base_rotation, dtype=float)
    base_translation = np.asarray(base_translation, dtype=float)

    if axis.shape != (3,):
        raise ValueError("Tool axis must be a 3D vector.")

    if np.linalg.norm(axis) < 1e-10:
        raise ValueError("Tool axis must be nonzero.")

    if base_rotation.shape != (3, 3):
        raise ValueError("Base rotation must have shape (3, 3).")

    if base_translation.shape != (3,):
        raise ValueError("Base translation must be a 3D vector.")

    axis = axis / np.linalg.norm(axis)

    # Initial marker positions in tracker coordinates.
    before = MARKERS @ base_rotation.T + base_translation

    # Convert the insertion direction into tracker coordinates.
    direction_tracker = base_rotation @ axis

    # Translate the tool without changing its orientation.
    after = before + distance * direction_tracker

    return before, after


# Gaussian Marker-Localization Noise

def add_marker_noise(observations, sigma=NOISE_SIGMA, rng=None):
    """
    Add independent zero-mean Gaussian marker-localization noise.
    """
    if sigma < 0:
        raise ValueError("Noise standard deviation cannot be negative.")

    if rng is None:
        rng = np.random.default_rng()

    observations = np.asarray(observations, dtype=float)

    noise = rng.normal(
        loc=0.0,
        scale=sigma,
        size=observations.shape
    )

    return observations + noise

# Simulation Validation

if __name__ == "__main__":

    # Separate generators make the validation reproducible
    motion_rng = np.random.default_rng(42)
    noise_rng = np.random.default_rng(123)

    # Standard pivot experiment.
    pivot_observations = simulate_pivot(
        n_poses=30,
        rng=motion_rng
    )

    # Generate a non-identity base pose and reuse it for
    # both axial rotation and forward translation.
    base_R, base_t = random_pose(motion_rng)

    axis_observations = simulate_axis_rotation(
        n_poses=36,
        base_rotation=base_R,
        base_translation=base_t
    )

    before, after = simulate_forward_motion(
        distance=5.0,
        base_rotation=base_R,
        base_translation=base_t
    )

    print("--- Simulated Marker Observations ---")
    print("Pivot observations:", pivot_observations.shape)
    print("Axis observations:", axis_observations.shape)
    print("Forward-before observations:", before.shape)
    print("Forward-after observations:", after.shape)

    # Verify that axial rotation starts from the chosen base pose.
    expected_first_pose = MARKERS @ base_R.T + base_t

    assert np.allclose(
        axis_observations[0],
        expected_first_pose,
        atol=1e-10
    )

    # Verify that forward translation follows the rotated axis.
    displacement = np.mean(after - before, axis=0)
    expected_displacement = 5.0 * (base_R @ V_TRUE)

    assert np.allclose(
        displacement,
        expected_displacement,
        atol=1e-10
    )

    print("\n--- Random Base-Pose Validation ---")
    print("Non-identity base rotation:", not np.allclose(
        base_R, np.eye(3)
    ))
    print("Axial rotation and forward motion: consistent")

    # Cone-limited pivot with an offset tool tip

    offset_tip = np.array([3.0, -2.0, 5.0])

    cone_observations = simulate_pivot(
        n_poses=30,
        tip=offset_tip,
        axis=V_TRUE,
        max_tilt_deg=30.0,
        rng=motion_rng
    )

    # Recover the simulated poses from the known marker geometry.
    # THIS IS ONLY A SIMULATION CHECK
    rotations = []
    translations = []

    for markers in cone_observations:
        R, t = rigid_registration(MARKERS, markers)
        rotations.append(R)
        translations.append(t)

    rotations = np.asarray(rotations)
    translations = np.asarray(translations)

    # Check that the offset tip stays at the fixed pivot.
    tip_positions = np.array([
        R @ offset_tip + t
        for R, t in zip(rotations, translations)
    ])

    stationary_tip_error = np.max(
        np.linalg.norm(
            tip_positions - DEFAULT_TRANSLATION,
            axis=1
        )
    )

    assert stationary_tip_error < 1e-8

    # Measure the actual angular coverage, rather than
    # assuming it equals the requested maximum tilt.
    tool_directions = np.array([
        R @ V_TRUE
        for R in rotations
    ])

    alignments = np.clip(
        tool_directions @ tool_directions.T,
        -1.0,
        1.0
    )

    pairwise_angles = np.degrees(
        np.arccos(alignments)
    )

    max_separation = np.max(pairwise_angles)

    nominal_alignments = np.clip(
        tool_directions @ V_TRUE,
        -1.0,
        1.0
    )

    achieved_tilts = np.degrees(
        np.arccos(nominal_alignments)
    )

    print("\n--- Cone-Limited Pivot Validation ---")
    print("Requested maximum tilt: 30.0 degrees")
    print(
        f"Largest achieved tilt: "
        f"{np.max(achieved_tilts):.2f} degrees"
    )
    print(
        f"Maximum pairwise angular separation: "
        f"{max_separation:.2f} degrees"
    )
    print(
        f"Maximum stationary-tip error: "
        f"{stationary_tip_error:.10f} mm"
    )

    assert np.max(achieved_tilts) <= 30.0 + 1e-6

    # Gaussian noise validation 

    zero_positions = np.zeros((100000, 3))

    noisy_positions = add_marker_noise(
        zero_positions,
        rng=noise_rng
    )

    errors = np.linalg.norm(noisy_positions, axis=1)

    percentile_95 = np.percentile(errors, 95)
    fraction_below = np.mean(errors < 1.0)

    print("\n--- Gaussian Marker-Error Validation ---")
    print(f"Coordinate noise sigma: {NOISE_SIGMA:.6f} mm")
    print(
        f"95th-percentile marker error: "
        f"{percentile_95:.4f} mm"
    )
    print(
        f"Fraction below 1 mm: "
        f"{fraction_below:.4%}"
    )

    # Allow normal Monte Carlo sampling variation.
    assert abs(fraction_below - 0.95) < 0.005
