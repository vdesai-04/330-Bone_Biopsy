
"""
CISC/CMPE 330 - CT-Guided Bone Biopsy Planning
Part 1: Tool Calibration

Purpose:
    Define the ground-truth biopsy tool geometry and generate
    simulated optical-tracker observations for calibration.

    Simulated tool motions will be used to evaluate the
    calibration algorithms under both perfect and noisy
    marker measurements.
"""


import numpy as np

# Ground-truth marker positions (mm)
M1 = np.array([2.0, 18.0, 0.0])
M2 = np.array([8.0, 18.0, 0.0])
M3 = np.array([2.0, 24.0, 0.0])

MARKERS = np.array([M1, M2, M3])

# Ground-truth tool tip and axis
P_TRUE = np.array([0.0, 0.0, 0.0])
V_TRUE = np.array([0.0, -1.0, 0.0])

def rotation_matrix(axis, angle):
    """Create a 3D rotation matrix using Rodrigues' formula."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    x, y, z = axis
    K = np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])

    I = np.eye(3)

    return I + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def simulate_pivot(n_poses=30, seed=42):
    """
    Simulate pivot calibration with a stationary tool tip.

    Generate diverse rotations while keeping the needle tip
    fixed at one point in the optical-tracker frame.

    Returns:
        observations: Marker positions for each pose,
                      shape (n_poses, 3, 3), in mm.
    """
    rng = np.random.default_rng(seed)
    pivot = np.array([10.0, 20.0, 30.0])

    observations = []

    for _ in range(n_poses):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)

        angle = rng.uniform(-np.pi, np.pi)
        R = rotation_matrix(axis, angle)

        # Keep the true tool tip stationary during rotation.
        t = pivot - R @ P_TRUE

        # Simulate the three optical-marker measurements.
        markers = (R @ MARKERS.T).T + t

        observations.append(markers)

    return np.array(observations)


def simulate_axis_rotation(n_poses=36):
    """
    Simulate rotation around the tool's longitudinal axis.

    The tool rotates through 360 degrees while its axis
    remains fixed in the tracker coordinate frame.

    Returns:
        observations: Marker positions for each pose,
                      shape (n_poses, 3, 3), in mm.
    """
    observations = []

    # Keep the tool at a fixed position in tracker space.
    t = np.array([10.0, 20.0, 30.0])

    for angle in np.linspace(0, 2 * np.pi, n_poses, endpoint=False):

        # Rotate around the known axis to generate observations.
        R = rotation_matrix(V_TRUE, angle)

        markers = (R @ MARKERS.T).T + t
        observations.append(markers)

    return np.array(observations)


if __name__ == "__main__":
    pivot_observations = simulate_pivot()
    axis_observations = simulate_axis_rotation()

    print("Pivot observations:", pivot_observations.shape)
    print("Axis observations:", axis_observations.shape)


