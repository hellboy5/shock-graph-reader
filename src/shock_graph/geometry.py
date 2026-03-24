"""Geometric and mathematical operations for shock graph edge features."""

import math
from typing import List, Tuple

import numpy as np


ZERO_TOLERANCE = 1e-7


def angle_diff(angle1: float, angle2: float) -> float:
    """Computes the shortest angular difference (angle1 - angle2).

    Uses modulo arithmetic to safely force the result into the [-pi, pi] range
    without using slow while loops.

    Args:
        angle1: The first angle in radians.
        angle2: The second angle in radians.

    Returns:
        The shortest angular difference in radians.
    """
    diff = angle1 - angle2
    return (diff + math.pi) % (2 * math.pi) - math.pi


def translate_point(
    pt: Tuple[float, float], angle: float, length: float
) -> Tuple[float, float]:
    """Translates a point by a given length along a specific angle.

    Args:
        pt: A tuple of (x, y) coordinates.
        angle: The angle of translation in radians.
        length: The distance to translate.

    Returns:
        The new (x, y) coordinates.
    """
    x = pt[0] + length * math.cos(angle)
    y = pt[1] + length * math.sin(angle)
    return x, y


def l2_dist(pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
    """Computes the Euclidean distance between two points.

    Args:
        pt1: The first (x, y) point.
        pt2: The second (x, y) point.

    Returns:
        The distance between the points.
    """
    return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])


# ---------------------------------------------------------------------------
# Resampling Logic (Interpolation & Subsampling)
# ---------------------------------------------------------------------------


def interpolate(
    shock_pts: List[Tuple[float, float]],
    times: List[float],
    thetas: List[float],
    phis: List[float],
    interpolate_ds: float = 1.0,
) -> Tuple[
    List[Tuple[float, float]], List[float], List[float], List[float]
]:
    """Interpolates the shock curve using contact manifold approximation.

    Args:
        shock_pts: List of (x, y) coordinates along the shock curve.
        times: List of radius values at each point.
        thetas: List of tangent angles in radians.
        phis: List of speed-derived angles in radians.
        interpolate_ds: The target arc-length distance between samples.

    Returns:
        A tuple containing the interpolated (points, times, thetas, phis).
    """
    if len(shock_pts) < 2 or interpolate_ds <= 0.0:
        return shock_pts, times, thetas, phis

    interp_pts = [shock_pts[0]]
    interp_times = [times[0]]
    interp_thetas = [thetas[0]]
    interp_phis = [phis[0]]

    for i in range(1, len(shock_pts)):
        dphi = angle_diff(phis[i], phis[i - 1])
        dtheta = angle_diff(thetas[i], thetas[i - 1])
        dx = shock_pts[i][0] - shock_pts[i - 1][0]
        dy = shock_pts[i][1] - shock_pts[i - 1][1]
        ds = math.hypot(dx, dy)
        dt = times[i] - times[i - 1]

        # Contact manifold approximation
        approx_ds = ds + (abs(dtheta) + abs(dphi)) * (
            times[i - 1] + times[i]
        ) / 2.0

        if approx_ds > interpolate_ds:
            num = int(approx_ds / interpolate_ds)
            for j in range(1, num):
                ratio = float(j) / float(num)

                p_int = (
                    shock_pts[i - 1][0] + ratio * dx,
                    shock_pts[i - 1][1] + ratio * dy,
                )
                time_int = times[i - 1] + ratio * dt
                phi_int = phis[i - 1] + ratio * dphi
                theta_int = thetas[i - 1] + ratio * dtheta

                interp_pts.append(p_int)
                interp_times.append(time_int)
                interp_thetas.append(theta_int)
                interp_phis.append(phi_int)

        # Add the current original sample
        interp_pts.append(shock_pts[i])
        interp_times.append(times[i])
        interp_thetas.append(thetas[i])
        interp_phis.append(phis[i])

    return interp_pts, interp_times, interp_thetas, interp_phis


def subsample(
    shock_pts: List[Tuple[float, float]],
    times: List[float],
    thetas: List[float],
    phis: List[float],
    bdry_plus: List[Tuple[float, float]],
    bdry_minus: List[Tuple[float, float]],
    subsample_ds: float = 5.0,
):
    """Subsamples the curve by retaining points that deviate significantly.

    Args:
        shock_pts: List of (x, y) coordinates along the shock curve.
        times: List of radius values at each point.
        thetas: List of tangent angles in radians.
        phis: List of speed-derived angles in radians.
        bdry_plus: List of (x, y) coordinates for the left boundary.
        bdry_minus: List of (x, y) coordinates for the right boundary.
        subsample_ds: The minimum distance threshold for subsampling.

    Returns:
        A tuple of the subsampled lists.
    """
    if len(shock_pts) < 3 or not (
        len(shock_pts) == len(times) == len(thetas) == len(phis)
    ):
        return shock_pts, times, thetas, phis, bdry_plus, bdry_minus

    sub_pts = [shock_pts[0]]
    sub_times = [times[0]]
    sub_thetas = [thetas[0]]
    sub_phis = [phis[0]]
    sub_bdry_p = [bdry_plus[0]]
    sub_bdry_m = [bdry_minus[0]]

    start_pt = shock_pts[0]
    end_pt = shock_pts[-1]
    start_pb = bdry_plus[0]
    end_pb = bdry_plus[-1]
    start_mb = bdry_minus[0]
    end_mb = bdry_minus[-1]

    for i in range(1, len(shock_pts) - 1):
        curr_pt = shock_pts[i]
        curr_pb = bdry_plus[i]
        curr_mb = bdry_minus[i]

        dist_shock_ok = (
            l2_dist(start_pt, curr_pt) > subsample_ds
            and l2_dist(curr_pt, end_pt) > subsample_ds
        )
        dist_pb_ok = (
            l2_dist(start_pb, curr_pb) > subsample_ds
            and l2_dist(curr_pb, end_pb) > subsample_ds
        )
        dist_mb_ok = (
            l2_dist(start_mb, curr_mb) > subsample_ds
            and l2_dist(curr_mb, end_mb) > subsample_ds
        )

        if dist_shock_ok or dist_pb_ok or dist_mb_ok:
            sub_pts.append(shock_pts[i])
            sub_times.append(times[i])
            sub_thetas.append(thetas[i])
            sub_phis.append(phis[i])
            sub_bdry_p.append(bdry_plus[i])
            sub_bdry_m.append(bdry_minus[i])

            start_pt = curr_pt
            start_pb = curr_pb
            start_mb = curr_mb

    # Always keep the last sample
    sub_pts.append(shock_pts[-1])
    sub_times.append(times[-1])
    sub_thetas.append(thetas[-1])
    sub_phis.append(phis[-1])
    sub_bdry_p.append(bdry_plus[-1])
    sub_bdry_m.append(bdry_minus[-1])

    return sub_pts, sub_times, sub_thetas, sub_phis, sub_bdry_p, sub_bdry_m


# ---------------------------------------------------------------------------
# Differential Geometry Stats
# ---------------------------------------------------------------------------


def compute_arc_length(
    curve: List[Tuple[float, float]]
) -> Tuple[List[float], float]:
    """Computes the cumulative arc length and total length of a curve.

    Args:
        curve: A list of (x, y) points representing the curve.

    Returns:
        A tuple containing a list of cumulative lengths and the total length.
    """
    arc_length = [0.0]
    length = 0.0
    px, py = curve[0]
    for cx, cy in curve[1:]:
        dl = math.hypot(cx - px, cy - py)
        length += dl
        arc_length.append(length)
        px, py = cx, cy
    return arc_length, length


def compute_derivatives(
    curve: List[Tuple[float, float]]
) -> Tuple[List[float], List[float]]:
    """Computes the first derivatives (dx/ds, dy/ds) along the curve.

    Args:
        curve: A list of (x, y) points representing the curve.

    Returns:
        A tuple containing lists of dx and dy values.
    """
    dx, dy = [0.0], [0.0]
    px, py = curve[0]
    for cx, cy in curve[1:]:
        dl = math.hypot(cx - px, cy - py)
        if dl > ZERO_TOLERANCE:
            dx.append((cx - px) / dl)
            dy.append((cy - py) / dl)
        else:
            dx.append(dx[-1] if len(dx) > 1 else 0.0)
            dy.append(dy[-1] if len(dy) > 1 else 0.0)
        px, py = cx, cy
        
    # NEW: Backfill the first derivative so the curve doesn't start from a dead stop
    if len(curve) > 1:
        dx[0] = dx[1]
        dy[0] = dy[1]
        
    return dx, dy


def compute_curvatures(
    dx: List[float],
    dy: List[float],
    arc_length: List[float],
    curve_length: int,
) -> Tuple[List[float], float]:
    """Computes the curvature at each point and the total curvature.

    Args:
        dx: List of first derivatives of x.
        dy: List of first derivatives of y.
        arc_length: List of cumulative arc lengths.
        curve_length: The total number of points in the curve.

    Returns:
        A tuple of the curvature array and the total absolute curvature.
    """
    curvature = [0.0]
    total_curvature = 0.0
    for i in range(1, curve_length):
        pdx, pdy = dx[i - 1], dy[i - 1]
        cdx, cdy = dx[i], dy[i]
        dl = arc_length[i] - arc_length[i - 1]

        d2x, d2y = 0.0, 0.0
        if dl > ZERO_TOLERANCE:
            d2x = (cdx - pdx) / dl
            d2y = (cdy - pdy) / dl

        kappa = 0.0
        if abs(cdx) >= ZERO_TOLERANCE or abs(cdy) >= ZERO_TOLERANCE:
            denominator = math.pow((math.pow(cdx, 2) + math.pow(cdy, 2)), 1.5)
            kappa = (d2y * cdx - d2x * cdy) / denominator

        curvature.append(kappa)
        # NEW: Multiply by arc-length segment (dl) to compute the true integral
        total_curvature += kappa * dl
        
    return curvature, total_curvature


def compute_angles(curve: List[Tuple[float, float]]) -> float:
    """Computes the total absolute angle change along the curve."""
    angle = [0.0]
    total_angle_change = 0.0
    px, py = curve[0]
    
    for cx, cy in curve[1:]:
        # NEW: Protect against duplicate points (dl == 0)
        if math.hypot(cx - px, cy - py) > ZERO_TOLERANCE:
            theta = math.atan2(cy - py, cx - px)
            angle.append(theta)
        else:
            # If points are duplicate, the angle hasn't changed
            angle.append(angle[-1] if len(angle) > 1 else 0.0)
        px, py = cx, cy

    if len(angle) > 2:
        angle[0] = angle[1]
        for i in range(1, len(angle)):
            total_angle_change += abs(angle_diff(angle[i], angle[i - 1]))
            
    return total_angle_change


def compute_curve_stats(
    curve: List[Tuple[float, float]]
) -> Tuple[float, float, float]:
    """Computes the length, total curvature, and total angle change.

    Args:
        curve: A list of (x, y) points representing the curve.

    Returns:
        A tuple of (length, total_curvature, total_angle_change).
    """
    if len(curve) < 2:
        return 0.0, 0.0, 0.0
    arc_length, length = compute_arc_length(curve)
    dx, dy = compute_derivatives(curve)
    _, total_curvature = compute_curvatures(dx, dy, arc_length, len(curve))
    total_angle_change = compute_angles(curve)
    return length, total_curvature, total_angle_change


def poly_area(x: np.ndarray, y: np.ndarray) -> float:
    """Computes the area of a polygon using the Shoelace formula.

    Args:
        x: A numpy array of x coordinates.
        y: A numpy array of y coordinates.

    Returns:
        The calculated area.
    """
    if len(x) < 3:
        return 0.0
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    correction = x_centered[-1] * y_centered[0] - y_centered[-1] * x_centered[0]
    main_area = np.dot(x_centered[:-1], y_centered[1:]) - np.dot(
        y_centered[:-1], x_centered[1:]
    )
    return 0.5 * np.abs(main_area + correction)
