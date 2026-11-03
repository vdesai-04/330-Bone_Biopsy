
"""
CISC/CMPE 330 - CT-Guided Bone Biopsy Planning
Part 1: Tool Calibration

Purpose:
    Test the calibration algorithms using simulated
    optical-tracker observations and compare the
    estimated parameters against ground truth.
"""

import numpy as np

from calibration import marker_frame, calibrate_tip
from simulation import simulate_pivot, M1, P_TRUE


def test_tip_calibration():
    """Validate tool-tip calibration using perfect marker data."""

    # Generate marker observations from 30 pivot poses.
    observations = simulate_pivot(n_poses=30)

    rotations = []
    translations = []

    # Recover each tool pose from the observed markers.
    for markers in observations:
        R, t = marker_frame(markers)
        rotations.append(R)
        translations.append(t)

    # Estimate the tool tip and stationary pivot point.
    estimated_tip, estimated_pivot = calibrate_tip(
        rotations, translations
    )

    # Verify the pivot constraint across all simulated poses.
    pivot_positions = np.array([
        R @ estimated_tip + t
        for R, t in zip(rotations, translations)
    ])

    pivot_deviations = np.linalg.norm(
        pivot_positions - estimated_pivot, axis=1
    )

    print(
        f"Maximum pivot deviation: "
        f"{np.max(pivot_deviations):.10f} mm"
    )


    # The calibrated tip is expressed in the marker frame,
    # which has its origin at M1, not the tool-model origin.
    true_tip_marker = P_TRUE - M1

    tip_error = np.linalg.norm(estimated_tip - true_tip_marker)

    print("Expected tip (marker frame):", true_tip_marker)
    print("Estimated tip (marker frame):", estimated_tip)
    print("Estimated pivot (tracker frame):", estimated_pivot)
    print(f"Tool-tip position error: {tip_error:.10f} mm")

    assert np.allclose(estimated_tip, true_tip_marker, atol=1e-8)


if __name__ == "__main__":
    test_tip_calibration()
