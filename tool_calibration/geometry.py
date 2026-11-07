
"""
Purpose of this file:
    Provide reusable geometric utilities for tool calibration,
    including coordinate-frame construction, rotations, plane
    fitting, rigid registration, and angular-error calculations.
"""

import numpy as np


def marker_frame(markers):
    """
    Construct a right-handed coordinate frame from three markers.

    Marker 1 defines the origin, M1 to M2 defines the x-axis,
    and the three markers define the xy-plane.
    """
    markers = np.asarray(markers, dtype=float)

    if markers.shape != (3, 3):
        raise ValueError("Expected three 3D marker positions.")

    m1, m2, m3 = markers

    x = m2 - m1
    x_norm = np.linalg.norm(x)

    if x_norm < 1e-10:
        raise ValueError("Markers 1 and 2 must be distinct.")

    x = x / x_norm

    z = np.cross(x, m3 - m1)
    z_norm = np.linalg.norm(z)

    if z_norm < 1e-10:
        raise ValueError("The markers must not be collinear.")

    z = z / z_norm
    y = np.cross(z, x)

    R = np.column_stack((x, y, z))
    t = m1.copy()

    return R, t


def rotation_matrix(axis, angle):
    """
    Construct a 3D rotation matrix using Rodrigues' formula.
    """
    axis = np.asarray(axis, dtype=float)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-10:
        raise ValueError("Rotation axis must be nonzero.")

    axis = axis / axis_norm

    x, y, z = axis

    K = np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0]
    ])

    I = np.eye(3)

    return I + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def fit_plane_normal(points):
    """
    Estimate a best-fit plane normal using SVD.
    """
    points = np.asarray(points, dtype=float)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected an array with shape (n, 3).")

    if len(points) < 3:
        raise ValueError("At least three points are required.")

    centered = points - np.mean(points, axis=0)

    _, singular_values, Vt = np.linalg.svd(
        centered, full_matrices=False
    )

    # The points must span at least two dimensions.
    if singular_values[0] == 0 or (
        singular_values[1] / singular_values[0] < 1e-10
    ):
        raise ValueError("Points do not sufficiently define a plane.")

    normal = Vt[-1]

    return normal / np.linalg.norm(normal)


def fit_common_plane_normal(point_sets, return_rms=False):
    """
    Estimate a common plane normal from multiple point trajectories.

    Each marker trajectory is centered separately before fitting.
    This is necessary because markers may rotate in parallel
    planes at different positions along the tool axis.
    """
    point_sets = np.asarray(point_sets, dtype=float)

    if point_sets.ndim != 3 or point_sets.shape[2] != 3:
        raise ValueError(
            "Expected trajectories with shape (n_markers, n_poses, 3)."
        )

    if point_sets.shape[0] < 1 or point_sets.shape[1] < 3:
        raise ValueError("Insufficient marker trajectories or poses.")

    # Center every marker trajectory independently.
    centered = point_sets - np.mean(
        point_sets, axis=1, keepdims=True
    )

    # Combine centered trajectories for a shared normal estimate.
    stacked = centered.reshape(-1, 3)

    _, singular_values, Vt = np.linalg.svd(
        stacked, full_matrices=False
    )

    if singular_values[0] == 0 or (
        singular_values[1] / singular_values[0] < 1e-10
    ):
        raise ValueError("Trajectories do not define a stable plane.")

    # The smallest singular vector minimizes out-of-plane error.
    normal = Vt[-1]
    normal /= np.linalg.norm(normal)

    if return_rms:
        distances = stacked @ normal
        rms = np.sqrt(np.mean(distances**2))
        return normal, rms

    return normal


def rigid_registration(model_points, observed_points, return_rms=False):
    """
    Estimate a rigid transformation from corresponding 3D points.
    """
    model_points = np.asarray(model_points, dtype=float)
    observed_points = np.asarray(observed_points, dtype=float)

    if model_points.shape != observed_points.shape:
        raise ValueError("Point arrays must have matching shapes.")

    if model_points.ndim != 2 or model_points.shape[1] != 3:
        raise ValueError("Expected arrays with shape (n, 3).")

    if len(model_points) < 3:
        raise ValueError("At least three corresponding points are required.")

    # Compute centroids and center both point sets.
    model_centroid = np.mean(model_points, axis=0)
    observed_centroid = np.mean(observed_points, axis=0)

    model_centered = model_points - model_centroid
    observed_centered = observed_points - observed_centroid

    if np.linalg.matrix_rank(model_centered) < 2:
        raise ValueError("Reference points must not be collinear.")

    # Arun/Kabsch least-squares rigid registration.
    H = model_centered.T @ observed_centered
    U, _, Vt = np.linalg.svd(H)

    # Prevent reflections by enforcing det(R) = +1.
    D = np.eye(3)
    D[-1, -1] = np.linalg.det(Vt.T @ U.T)

    R = Vt.T @ D @ U.T

    # Recover translation after estimating rotation.
    t = observed_centroid - R @ model_centroid

    if return_rms:
        fitted_points = model_points @ R.T + t
        residuals = fitted_points - observed_points

        rms = np.sqrt(np.mean(np.sum(residuals**2, axis=1)))
        return R, t, rms

    return R, t


def angle_between_deg(a, b, directed=True):
    """
    Calculate the angle between two 3D vectors in degrees.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm < 1e-10 or b_norm < 1e-10:
        raise ValueError("Cannot calculate angle with a zero vector.")

    a = a / a_norm
    b = b / b_norm

    cross_norm = np.linalg.norm(np.cross(a, b))
    dot = np.dot(a, b)

    if not directed:
        dot = abs(dot)

    return np.degrees(np.arctan2(cross_norm, dot))


def rotation_error_deg(R1, R2):
    """
    Calculate the angular difference between two 3D rotations.
    """
    R1 = np.asarray(R1, dtype=float)
    R2 = np.asarray(R2, dtype=float)

    if R1.shape != (3, 3) or R2.shape != (3, 3):
        raise ValueError("Expected two 3x3 rotation matrices.")

    R = R1.T @ R2

    # Skew-symmetric part of the relative rotation.
    skew = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ])

    sin_theta = 0.5 * np.linalg.norm(skew)
    cos_theta = 0.5 * (np.trace(R) - 1.0)

    return np.degrees(np.arctan2(sin_theta, cos_theta))
