"""
Purpose of this file:
    Compute the salient parameters of a biopsy plan for the
    physician ("iterative forward planning").

    Inputs:
        C_tum, R_tum : tumor sphere centre (CT, mm) and radius (mm)
        P_ct, v_ct   : needle tip (CT, mm) and needle direction,
                       from registration (Part 2)

    Plan: push the needle straight ahead along its current axis,
    all the way through the tumor.

    Outputs:
        targeting_error : distance from the needle axis to C_tum
        core_length     : length of the needle path inside the tumor
        insertion_depth : how far the tip must advance from P_ct
                          to harvest that whole core (to the far
                          tumor wall)

    Geometry, with u = unit(v_ct) and the needle line
    X(s) = P_ct + s u   (s = mm ahead of the current tip):

        s_c = (C - P) . u              depth of closest approach
        d   = |(C - P) - s_c u|        targeting error
        h   = sqrt(R^2 - d^2)          half chord, if d <= R
        entry depth s1 = s_c - h,  exit depth s2 = s_c + h
        core length = s2 - s1 = 2h,   insertion depth = s2

    The cases the code distinguishes:

      "intersects"  0 <= s1 < s2  normal case, core = 2h
      "tangent"     d == R        needle grazes the surface: core = 0
      "miss"        d >  R        no intersection: core = 0
      "tip_inside"  s1 < 0 <= s2  tip already in the tumor: only
                                  the part ahead can be cored,
                                  core = s2
      "behind"      s2 < 0        tumor lies BEHIND the tip; the
                                  infinite axis meets it but
                                  forward insertion never will
"""

import numpy as np

from line_geometry import (
    normalize,
    closest_point_on_line,
    distance_point_from_line,
    intersect_line_sphere,
)


def compute_biopsy_plan(tumor_center, tumor_radius, needle_tip,
                        needle_direction, rel_tol=1e-9):
    """
    Compute targeting error, tissue-core length and insertion depth.

    All lengths are in mm. `needle_direction` need not be unit
    length; it must point the way the needle will be advanced.

    Returns a dict:
        scenario          : one of the strings described above
        targeting_error   : distance of needle AXIS from tumor centre
        core_length       : harvestable core length (0 if none)
        insertion_depth   : advance needed from the tip to the far
                            end of the core (None if no contact)
        closest_depth     : s_c, depth where the axis passes closest
                            to the centre (negative = behind tip)
        entry_point, exit_point : CT points where the core starts
                            and ends (None if no core / contact)
        message           : short plain-language summary
    """
    C = np.asarray(tumor_center, dtype=float)
    P = np.asarray(needle_tip, dtype=float)
    u = normalize(needle_direction)

    if C.shape != (3,) or P.shape != (3,):
        raise ValueError("Tumor centre and needle tip must be 3D points.")

    # Targeting error: distance from C to the infinite needle
    # axis. This follows the assignment's definition ("distance
    # of the needle axis from the tumor center"). Whether the
    # tumor is in front of or behind the tip is reported
    # separately through `scenario`.
    targeting_error = distance_point_from_line(C, P, u)
    _, s_c = closest_point_on_line(C, P, u)

    kind, s_vals = intersect_line_sphere(P, u, C, tumor_radius,
                                         rel_tol=rel_tol)

    def point_at(s):
        return P + s * u

    plan = {
        "targeting_error": targeting_error,
        "closest_depth": s_c,
        "core_length": 0.0,
        "insertion_depth": None,
        "entry_point": None,
        "exit_point": None,
    }

    if kind == "miss":
        plan["scenario"] = "miss"
        plan["message"] = (
            f"Needle axis misses the tumor by "
            f"{targeting_error - tumor_radius:.2f} mm. Re-aim the tool."
        )
        return plan

    if kind == "tangent":
        (s_t,) = s_vals
        if s_t < 0:
            plan["scenario"] = "behind"
            plan["message"] = "Tumor is behind the needle tip."
            return plan

        # The needle touches the surface at a single point. The
        # core length is 0 (no tissue). We still report the depth
        # of the touching point so the physician knows where the
        # grazing contact would be.
        plan["scenario"] = "tangent"
        plan["insertion_depth"] = s_t
        plan["entry_point"] = plan["exit_point"] = point_at(s_t)
        plan["message"] = (
            "Needle only grazes the tumor surface; no core. Re-aim."
        )
        return plan

    # Secant: two crossings, entry s1 < exit s2.
    s1, s2 = s_vals

    if s2 < 0:
        # Both crossings are behind the tip.
        plan["scenario"] = "behind"
        plan["message"] = (
            "Tumor is behind the needle tip; forward insertion "
            "cannot reach it."
        )
        return plan

    if s1 < 0:
        # Tip is already inside the tumor. Only the part from
        # the tip (s = 0) to the far wall (s2) can be cored.
        plan["scenario"] = "tip_inside"
        s1 = 0.0
    else:
        plan["scenario"] = "intersects"

    # Core-needle biopsy: the coring tip must reach the far wall
    # (s2) to collect the whole chord from s1 to s2.
    plan["core_length"] = s2 - s1
    plan["insertion_depth"] = s2
    plan["entry_point"] = point_at(s1)
    plan["exit_point"] = point_at(s2)
    plan["message"] = (
        f"Insert {s2:.2f} mm to harvest a {s2 - s1:.2f} mm core."
    )
    return plan


def format_plan(plan):
    """Format the plan as a short report for the physician."""
    depth = plan["insertion_depth"]
    depth_str = "N/A" if depth is None else f"{depth:.4f} mm"
    return "\n".join([
        f"Scenario:         {plan['scenario']}",
        f"Targeting error:  {plan['targeting_error']:.4f} mm",
        f"Tissue core:      {plan['core_length']:.4f} mm",
        f"Insertion depth:  {depth_str}",
        f"Note:             {plan['message']}",
    ])
