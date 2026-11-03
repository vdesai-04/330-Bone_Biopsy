
"""
CISC/CMPE 330 - CT-Guided Bone Biopsy Planning
Tool Calibration

Purpose:
    Provide reusable geometric utilities for tool calibration.

    Includes construction of a marker-defined coordinate frame,
    three-dimensional rotation matrices, and least-squares
    plane fitting.

    These utilities support the calibration methods found
    in the course lectures.
"""

import numpy as np


def marker_frame(markers):
    """
    Construct a right-handed orthonormal frame from three markers.

    The frame origin is Marker 1. Its x-axis points toward
    Marker 2, and Marker 3 determines the xy-plane.

    Returns:
        R: Rotation matrix from marker to tracker coordinates.
        t: Marker-frame origin in tracker coordinates (mm).
    """
    markers = np.asarray(markers, dtype=float)

    if markers.shape != (3, 3):
        raise ValueError("Expected three 3D marker positions.")

    m1, m2, m3 = markers

    # Construct the first coordinate axis.
    x = m2 - m1
    x_norm = np.linalg.norm(x)

    if x_norm < 1e-10:
        raise ValueError("Markers 1 and 2 must be distinct.")

    x = x / x_norm

    # Construct the normal to the marker plane.
    z = np.cross(x, m3 - m1)
    z_norm = np.linalg.norm(z)

    if z_norm < 1e-10:
        raise ValueError("The three markers must not be collinear.")

    z = z / z_norm

    # Complete the right-handed orthonormal basis.
    y = np.cross(z, x)

    # Rotation columns are the marker-frame basis vectors.
    R = np.column_stack((x, y, z))
    t = m1.copy()

    return R, t


def rotation_matrix(axis, angle):
    """
    Construct a 3D rotation matrix using Rodrigues' formula.

    Args:
        axis: Rotation-axis vector.
        angle: Rotation angle in radians.

    Returns:
        A 3x3 rotation matrix.
    """
    axis = np.asarray(axis, dtype=float)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-10:
        raise ValueError("Rotation axis must be nonzero.")

    axis = axis / axis_norm

    x, y, z = axis

    # Skew-symmetric matrix for the cross product.
    K = np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0]
    ])

    I = np.eye(3)

    # Rodrigues' rotation formula.
    R = I + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    return R


def fit_plane_normal(points):
    """
    Estimate a plane normal using singular-value decomposition.

    Args:
        points: Array of 3D points with shape (n, 3).

    Returns:
        Unit normal vector of the least-squares fitted plane.

    The sign of the normal is ambiguous and must be resolved
    separately when estimating a directed tool axis.
    """
    points = np.asarray(points, dtype=float)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected an array of 3D points.")

    if len(points) < 3:
        raise ValueError("Plane fitting requires at least three points.")

    # Center the observations around their centroid.
    centered = points - np.mean(points, axis=0)

    # SVD identifies directions of spatial variation.
    _, singular_values, Vt = np.linalg.svd(
        centered, full_matrices=False
    )

    # A valid plane requires variation in two dimensions.
    if singular_values[1] < 1e-10:
        raise ValueError("Points do not sufficiently define a plane.")

    # The smallest singular vector is the fitted plane normal.
    normal = Vt[-1]

    return normal / np.linalg.norm(normal)
