
"""
CISC/CMPE 330 - CT-Guided Bone Biopsy Planning
Part 1: Tool Calibration

File: simulation.py

Purpose:
    Define the ground-truth biopsy tool and simulate optical-
    tracker marker observations for tool calibration.

    Three constrained motions are simulated:
        1. Pivoting around a stationary needle tip.
        2. Rotating around the tool's longitudinal axis.
        3. Translating forward to identify the axis direction.

    Independent Gaussian noise can be added to the simulated
    marker positions to evaluate calibration accuracy.

Reference:
    Assignment Figure 2 - Ground-truth tracked tool model.
"""

import numpy as np

from geometry import rotation_matrix


# ---------------------------------------------------------
# 1. GROUND-TRUTH TOOL MODEL
# ---------------------------------------------------------

# Marker positions in the tool model frame, F_mod (mm).
M1 = np.array([2.0, 18.0, 0.0])
M2 = np.array([8.0, 18.0, 0.0])
M3 = np.array([2.0, 24.0, 0.0])

MARKERS = np.array([M1, M2, M3])

# Tool-tip position (mm) and unit axis direction in F_mod.
P_TRUE = np.array([0.0, 0.0, 0.0])
V_TRUE = np.array([0.0, -1.0, 0.0])


# ---------------------------------------------------------
# 2. PIVOT-MOTION SIMULATION
# ---------------------------------------------------------

def simulate_pivot(n_poses=30, seed=42):
    """
    Generate marker observations during pivot calibration.

    Constraint:
        The needle tip remains stationary in the tracker
        frame while the tool rotates through varied poses.

    Returns:
        Marker observations with shape (n_poses, 3, 3),
        expressed in tracker coordinates (mm).
    """
    if n_poses < 3:
        raise ValueError("At least three pivot poses are required.")

    rng = np.random.default_rng(seed)

    # Stationary pivot point in tracker coordinates (mm).
    pivot = np.array([10.0, 20.0, 30.0])

    observations = []

    for _ in range(n_poses):
        # Generate a random 3D rotation.
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)

        angle = rng.uniform(-np.pi, np.pi)
        R = rotation_matrix(axis, angle)

        # Enforce the stationary-tip constraint:
        # R @ P_TRUE + t = pivot
        t = pivot - R @ P_TRUE

        # Transform the three model markers into tracker space.
        markers = (R @ MARKERS.T).T + t
        observations.append(markers)

    return np.array(observations)


# ---------------------------------------------------------
# 3. TOOL-AXIS ROTATION SIMULATION
# ---------------------------------------------------------

def simulate_axis_rotation(n_poses=36):
    """
    Generate marker trajectories during axial rotation.

    Constraint:
        The tool rotates through 360 degrees around its
        fixed longitudinal axis without translation.

    Each marker traces a circle in a plane perpendicular
    to the rotation axis.

    Returns:
        Marker observations with shape (n_poses, 3, 3),
        expressed in tracker coordinates (mm).
    """
    if n_poses < 3:
        raise ValueError("At least three rotation poses are required.")

    # Fixed translation in tracker coordinates (mm).
    t = np.array([10.0, 20.0, 30.0])

    angles = np.linspace(
        0.0, 2.0 * np.pi, n_poses, endpoint=False
    )

    observations = []

    for angle in angles:
        # Ground-truth axis is used only to simulate rotation.
        R = rotation_matrix(V_TRUE, angle)

        # Compute the corresponding tracked marker positions.
        markers = (R @ MARKERS.T).T + t
        observations.append(markers)

    return np.array(observations)


# ---------------------------------------------------------
# 4. KNOWN FORWARD-MOTION SIMULATION
# ---------------------------------------------------------

def simulate_forward_motion(distance=5.0):
    """
    Simulate forward translation along the needle axis.

    Constraint:
        The tool translates toward its tip without changing
        its orientation.

    The marker displacement provides an independent
    direction constraint to resolve the sign ambiguity
    of the plane-fitted axis.

    Returns:
        before: Marker positions before translation (mm).
        after: Marker positions after translation (mm).
    """
    if distance <= 0:
        raise ValueError("Forward distance must be positive.")

    t = np.array([10.0, 20.0, 30.0])

    # Marker observations before translation.
    before = MARKERS + t

    # Move the tool forward without changing orientation.
    # V_TRUE is used to generate, not calibrate, the motion.
    after = before + distance * V_TRUE

    return before, after


# ---------------------------------------------------------
# 5. GAUSSIAN MARKER-LOCALIZATION ERROR
# ---------------------------------------------------------

def add_marker_noise(observations, sigma=0.358, seed=42):
    """
    Add independent Gaussian noise to marker observations.

    Each coordinate receives zero-mean Gaussian error:
        error_x, error_y, error_z ~ N(0, sigma^2)

    With sigma approximately 0.358 mm, the magnitude of
    the 3D position error is below 1.0 mm with approximately
    95% probability.

    Returns:
        Noisy observations with the same shape as the input.
    """
    if sigma < 0:
        raise ValueError("Noise standard deviation cannot be negative.")

    rng = np.random.default_rng(seed)

    # Generate independent noise for every observed coordinate.
    noise = rng.normal(
        loc=0.0,
        scale=sigma,
        size=np.shape(observations)
    )

    return np.asarray(observations, dtype=float) + noise


# ---------------------------------------------------------
# 6. SIMULATION VALIDATION
# ---------------------------------------------------------

if __name__ == "__main__":
    # Generate perfect marker observations.
    pivot_observations = simulate_pivot()
    axis_observations = simulate_axis_rotation()

    print("--- Simulated Marker Observations ---")
    print("Pivot observations:", pivot_observations.shape)
    print("Axis observations:", axis_observations.shape)

    # Evaluate the 3D Gaussian noise model statistically.
    zero_positions = np.zeros((100000, 3))
    noisy_positions = add_marker_noise(
        zero_positions, seed=123
    )

    # Calculate Euclidean marker-position errors.
    errors = np.linalg.norm(
        noisy_positions, axis=1
    )

    print("\n--- Gaussian Marker-Error Validation ---")
    print(
        f"95th-percentile marker error: "
        f"{np.percentile(errors, 95):.4f} mm"
    )
    print(
        f"Fraction below 1 mm: "
        f"{np.mean(errors < 1.0):.4%}"
    )
