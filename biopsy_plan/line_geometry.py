"""
Purpose of this file:
    Reusable line / sphere geometry utilities for biopsy planning:
        - normalize
        - closest_point_on_line
        - distance_point_from_line
        - intersect_line_sphere

    Line convention used everywhere:
        X(s) = P + s * u,     u = unit direction,  s in mm
    Because u is unit length, the parameter s is a signed
    distance along the line from P. In the biopsy plan, P is the
    needle tip and u points in the insertion direction, so
    s > 0 is ahead of the tip and s < 0 is behind it.
"""

import numpy as np


def normalize(v):
    """Return v / |v|; reject zero vectors."""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if v.shape != (3,):
        raise ValueError("Expected a 3D vector.")
    if n < 1e-12:
        raise ValueError("Direction vector must be nonzero.")
    return v / n


def closest_point_on_line(point, line_point, direction):
    """
    Foot of the perpendicular from `point` onto the line.

    Write w = point - P. Split w into a part along u and a
    part perpendicular to u:
        s     = w . u            (projection length, signed)
        foot  = P + s * u
        w_perp = w - s * u       (perpendicular offset)

            point
              *
              |  w_perp  (length = distance to line)
              |
      P ------+--------->  u
        <- s ->
              foot

    Returns (foot, s).
    """
    P = np.asarray(line_point, dtype=float)
    u = normalize(direction)
    w = np.asarray(point, dtype=float) - P
    s = float(np.dot(w, u))
    return P + s * u, s


def distance_point_from_line(point, line_point, direction):
    """
    Shortest distance from `point` to the INFINITE line.

    Computed as |w - (w.u) u|, the length of the perpendicular
    offset. (This is the same as |w x u| for a unit u. The
    subtraction form avoids the cancellation of the textbook
    sqrt(|w|^2 - s^2) when the point is far along the line.)
    """
    foot, _ = closest_point_on_line(point, line_point, direction)
    return float(np.linalg.norm(np.asarray(point, dtype=float) - foot))


def intersect_line_sphere(line_point, direction, center, radius,
                          rel_tol=1e-9):
    """
    Intersect the line X(s) = P + s u with the sphere |X - C| = r.

    Derivation (Pythagoras in the plane of the line and C):
        s_c = (C - P) . u        parameter of the point closest to C
        d   = distance from C to the line
        A point at parameter s is on the sphere when
            d^2 + (s - s_c)^2 = r^2
        so  s = s_c -/+ h,   with  h = sqrt(r^2 - d^2).

                        entry      exit
          P ---------------*----+----*---------> u
                           |<-h-|-h->|
                                | d
                                C  (sphere centre)

    Cases:
        d < r : two intersections ("secant")
        d = r : one touching point ("tangent")
        d > r : none ("miss")

    Exact equality d == r almost never happens in floating
    point, so |d - r| <= rel_tol * max(r, 1) counts as tangent.

    Returns
    -------
    (kind, s_values) where kind is "secant", "tangent" or "miss"
    and s_values is a sorted tuple of 2, 1 or 0 line parameters.
    """
    radius = float(radius)
    if radius <= 0:
        raise ValueError("Sphere radius must be positive.")

    u = normalize(direction)
    _, s_c = closest_point_on_line(center, line_point, u)
    d = distance_point_from_line(center, line_point, u)

    tol = rel_tol * max(radius, 1.0)

    if d > radius + tol:
        return "miss", ()

    if abs(d - radius) <= tol:
        return "tangent", (s_c,)

    h = np.sqrt(radius**2 - d**2)
    return "secant", (s_c - h, s_c + h)
