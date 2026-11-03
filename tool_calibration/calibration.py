
"""
CISC/CMPE 330 - CT-Guided Bone Biopsy Planning
Part 1: Tool Calibration

File: calibration.py

Purpose:
    Recover the tool-tip position and tool-axis direction
    from optical-tracker marker observations.
"""

import numpy as np


def marker_frame(markers):
    """
    Construct an orthonormal coordinate frame from
    three observed marker positions.
    """
    m1, m2, m3 = markers

    # First axis points from marker 1 toward marker 2.
    x = m2 - m1
    x = x / np.linalg.norm(x)

    # Third axis is perpendicular to the marker plane.
    z = np.cross(x, m3 - m1)
    z = z / np.linalg.norm(z)

    # Second axis completes the right-handed frame.
    y = np.cross(z, x)

    R = np.column_stack((x, y, z))
    t = m1

    return R, t

def calibrate_tip(rotations, translations):
    """
    Estimate the tool tip using pivot calibration.

    The tool tip remains stationary while the tool
    rotates through multiple poses.
    """
    A = []
    b = []

    for R, t in zip(rotations, translations):
        A.append(np.hstack((R, -np.eye(3))))
        b.append(-t)

    A = np.vstack(A)
    b = np.concatenate(b)

    solution, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    tip = solution[:3]
    pivot = solution[3:]

    return tip, pivot
