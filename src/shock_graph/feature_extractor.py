"""Feature extraction logic for shock graph elements."""

import numpy as np

from . import geometry
from .structures import Edge, EdgeShapeFeatures


class ShockFeatureExtractor:
    """Extracts and computes geometric features for shock graphs."""

    @staticmethod
    def extract_edge_features(edge: Edge) -> EdgeShapeFeatures:
        """Computes geometric properties for a single edge.

        Applies interpolation, boundary reconstruction, and subsampling
        before computing the differential geometry statistics.

        Args:
            edge: The Edge object containing raw SamplePoints.

        Returns:
            An EdgeShapeFeatures object containing the calculated properties.
        """
        if len(edge.samples) < 2:
            return EdgeShapeFeatures()

        # 1. Unpack raw sample points
        raw_pts = [(s.x, s.y) for s in edge.samples]
        raw_times = [s.t for s in edge.samples]
        raw_thetas = [s.theta for s in edge.samples]
        raw_phis = [s.phi for s in edge.samples]

        # 2. Interpolate
        i_pts, i_times, i_thetas, i_phis = geometry.interpolate(
            raw_pts, raw_times, raw_thetas, raw_phis
        )

        # 3. Reconstruct Plus and Minus Boundaries
        raw_p_bdry = []
        raw_m_bdry = []
        for i in range(len(i_pts)):
            pt = i_pts[i]
            time = i_times[i]
            theta = i_thetas[i]
            phi = i_phis[i]

            left_angle = theta + phi
            right_angle = theta - phi

            raw_p_bdry.append(geometry.translate_point(pt, left_angle, time))
            raw_m_bdry.append(geometry.translate_point(pt, right_angle, time))

        # 4. Subsample
        s_pts, _, _, _, s_p_bdry, s_m_bdry = geometry.subsample(
            i_pts, i_times, i_thetas, i_phis, raw_p_bdry, raw_m_bdry
        )

        # 5. Compute Curve Statistics
        s_len, s_curv, s_ang = geometry.compute_curve_stats(s_pts)
        p_len, p_curv, p_ang = geometry.compute_curve_stats(s_p_bdry)
        m_len, m_curv, m_ang = geometry.compute_curve_stats(s_m_bdry)

        # Ensure consistent boundary orientation to prevent arbitrary swapping
        if len(s_p_bdry) > 0 and len(s_m_bdry) > 0:
            if s_p_bdry[0][0] > s_m_bdry[0][0]:
                p_curv, m_curv = m_curv, p_curv
                p_len, m_len = m_len, p_len
                p_ang, m_ang = m_ang, p_ang

        # 6. Compute Polygon Area
        poly_points = (
            [s_pts[0]] + s_p_bdry + [s_pts[-1]] + s_m_bdry[::-1] + [s_pts[0]]
        )
        x_poly, y_poly = zip(*poly_points)
        area = geometry.poly_area(np.array(x_poly), np.array(y_poly))

        return EdgeShapeFeatures(
            s_length=s_len,
            s_curve=s_curv,
            s_angle=s_ang,
            p_length=p_len,
            p_curve=p_curv,
            p_angle=p_ang,
            m_length=m_len,
            m_curve=m_curv,
            m_angle=m_ang,
            poly_area=area,
        )
