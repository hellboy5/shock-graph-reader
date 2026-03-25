"""Unit tests for the shock graph geometry module."""

import math
import unittest

import numpy as np

# Adjust the import based on your actual path structure
from src.shock_graph import geometry


class TestBasicMath(unittest.TestCase):
    """Tests for basic mathematical helper functions."""

    def test_angle_diff(self):
        """Tests angular difference handling across the -pi/pi boundary."""
        self.assertAlmostEqual(geometry.angle_diff(math.pi, -math.pi), 0.0)
        self.assertAlmostEqual(
            geometry.angle_diff(math.pi - 0.1, -math.pi + 0.1), -0.2
        )
        self.assertAlmostEqual(
            geometry.angle_diff(0.0, math.pi / 2), -math.pi / 2
        )
        self.assertAlmostEqual(
            geometry.angle_diff(math.pi / 2, 0.0), math.pi / 2
        )

    def test_translate_point(self):
        """Tests translating a point along a specific angle."""
        pt = (0.0, 0.0)
        # Translate along X axis
        new_pt = geometry.translate_point(pt, 0.0, 5.0)
        self.assertAlmostEqual(new_pt[0], 5.0)
        self.assertAlmostEqual(new_pt[1], 0.0)

        # Translate along Y axis
        new_pt = geometry.translate_point(pt, math.pi / 2.0, 5.0)
        self.assertAlmostEqual(new_pt[0], 0.0)
        self.assertAlmostEqual(new_pt[1], 5.0)

    def test_l2_dist(self):
        """Tests Euclidean distance calculation."""
        self.assertAlmostEqual(geometry.l2_dist((0.0, 0.0), (3.0, 4.0)), 5.0)
        self.assertAlmostEqual(geometry.l2_dist((1.0, 1.0), (1.0, 1.0)), 0.0)


class TestAnalyticalGeometry(unittest.TestCase):
    """Tests differential geometry using known analytical curves."""

    def test_straight_line(self):
        """A straight line should have 0 curvature and 0 angle change."""
        curve = [(x / 10.0, 0.0) for x in range(101)]

        length, total_curv, total_angle = geometry.compute_curve_stats(curve)

        self.assertAlmostEqual(length, 10.0, places=5)
        self.assertAlmostEqual(total_curv, 0.0, places=5)
        self.assertAlmostEqual(total_angle, 0.0, places=5)

    def test_circular_arc(self):
        """A circular arc has constant curvature K = 1/R.
        
        For a semi-circle (pi radians) of radius R:
        - Arc length = pi * R
        - Total signed curvature integral = pi
        - Total absolute angle change = pi
        """
        radius = 5.0
        num_points = 2000
        angles = np.linspace(0, math.pi, num_points)
        curve = [(radius * math.cos(a), radius * math.sin(a)) for a in angles]

        length, total_curv, total_angle = geometry.compute_curve_stats(curve)

        self.assertAlmostEqual(length, math.pi * radius, delta=0.01)
        self.assertAlmostEqual(total_curv, math.pi, delta=0.05)
        self.assertAlmostEqual(total_angle, math.pi, delta=0.05)

    def test_parabola(self):
        """Tests a parabola y = x^2 from x = -1 to 1.
        
        Analytical total curvature:
        Integral of K ds = 2 * arctan(2) ≈ 2.2143.
        """
        num_points = 2000
        x_vals = np.linspace(-1, 1, num_points)
        curve = [(x, x**2) for x in x_vals]

        length, total_curv, total_angle = geometry.compute_curve_stats(curve)
        
        expected_val = 2.0 * math.atan(2.0)

        self.assertAlmostEqual(total_curv, expected_val, delta=0.05)
        self.assertAlmostEqual(total_angle, expected_val, delta=0.05)

    def test_half_sine_wave(self):
        """Tests a sine wave y = sin(x) from x = 0 to pi.
        
        The curve bends downwards, so curvature is negative.
        - Total signed curvature integral = -pi/2.
        - Total absolute angle change = pi/2.
        """
        num_points = 2000
        x_vals = np.linspace(0, math.pi, num_points)
        curve = [(x, math.sin(x)) for x in x_vals]

        length, total_curv, total_angle = geometry.compute_curve_stats(curve)

        self.assertAlmostEqual(total_curv, -math.pi / 2.0, delta=0.05)
        self.assertAlmostEqual(total_angle, math.pi / 2.0, delta=0.05)

    def test_full_ellipse(self):
        """Tests a closed convex loop.
        
        By Hopf's theorem, total signed curvature of ANY closed convex curve 
        is exactly 2*pi.
        """
        a, b = 5.0, 3.0
        num_points = 2000
        angles = np.linspace(0, 2 * math.pi, num_points)
        curve = [(a * math.cos(t), b * math.sin(t)) for t in angles]

        length, total_curv, total_angle = geometry.compute_curve_stats(curve)

        self.assertAlmostEqual(total_curv, 2 * math.pi, delta=0.05)
        self.assertAlmostEqual(total_angle, 2 * math.pi, delta=0.05)

    def test_resolution_invariance(self):
        """Total curvature and angle change must be independent of sampling density."""
        # 100 points
        angles_100 = np.linspace(0, math.pi, 100)
        curve_100 = [(5.0 * math.cos(a), 5.0 * math.sin(a)) for a in angles_100]
        _, curv_100, angle_100 = geometry.compute_curve_stats(curve_100)
        
        # 1000 points
        angles_1000 = np.linspace(0, math.pi, 1000)
        curve_1000 = [(5.0 * math.cos(a), 5.0 * math.sin(a)) for a in angles_1000]
        _, curv_1000, angle_1000 = geometry.compute_curve_stats(curve_1000)

        # These should match perfectly despite the 10x difference in points
        self.assertAlmostEqual(curv_100, curv_1000, delta=0.05)
        self.assertAlmostEqual(angle_100, angle_1000, delta=0.05)

    def test_poly_area_square(self):
        """Tests Shoelace area formula on a perfect square."""
        x = np.array([0.0, 2.0, 2.0, 0.0, 0.0])
        y = np.array([0.0, 0.0, 2.0, 2.0, 0.0])
        
        area = geometry.poly_area(x, y)
        self.assertAlmostEqual(area, 4.0)

    def test_poly_area_triangle(self):
        """Tests Shoelace area formula on a triangle."""
        x = np.array([0.0, 10.0, 5.0, 0.0])
        y = np.array([0.0, 0.0, 5.0, 0.0])
        
        area = geometry.poly_area(x, y)
        self.assertAlmostEqual(area, 25.0)

    def test_poly_area_offset_rectangle(self):
        """Tests a rectangle far from the origin to verify mean-centering."""
        x = np.array([1000.0, 1005.0, 1005.0, 1000.0, 1000.0])
        y = np.array([1000.0, 1000.0, 1002.0, 1002.0, 1000.0])
        
        area = geometry.poly_area(x, y)
        self.assertAlmostEqual(area, 10.0)

    def test_poly_area_concave_l_shape(self):
        """Tests a concave 'L' shaped polygon."""
        x = np.array([0.0, 4.0, 4.0, 1.0, 1.0, 0.0, 0.0])
        y = np.array([0.0, 0.0, 1.0, 1.0, 4.0, 4.0, 0.0])
        
        area = geometry.poly_area(x, y)
        self.assertAlmostEqual(area, 7.0)

    def test_poly_area_regular_hexagon(self):
        """Tests a regular hexagon."""
        radius = 10.0
        angles = np.linspace(0, 2 * math.pi, 7)
        
        x = np.array([radius * math.cos(a) for a in angles])
        y = np.array([radius * math.sin(a) for a in angles])
        
        area = geometry.poly_area(x, y)
        expected_area = (3 * math.sqrt(3) / 2.0) * (radius ** 2)
        
        self.assertAlmostEqual(area, expected_area, places=4)

    def test_poly_area_degenerate_line(self):
        """Tests a flat line folded back on itself (degenerate polygon)."""
        x = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
        y = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
        
        area = geometry.poly_area(x, y)
        self.assertAlmostEqual(area, 0.0)


class TestResampling(unittest.TestCase):
    """Tests the interpolation and subsampling algorithms."""

    def test_interpolate(self):
        """Tests that gaps larger than interpolate_ds are filled."""
        pts = [(0.0, 0.0), (10.0, 0.0)]
        times = [1.0, 1.0]
        thetas = [0.0, 0.0]
        phis = [math.pi / 4, math.pi / 4]

        i_pts, i_times, i_thetas, i_phis = geometry.interpolate(
            pts, times, thetas, phis, interpolate_ds=1.0
        )

        self.assertGreater(len(i_pts), 2)
        self.assertEqual(i_pts[0], (0.0, 0.0))
        self.assertEqual(i_pts[-1], (10.0, 0.0))
        self.assertEqual(len(i_pts), len(i_times))
        self.assertEqual(len(i_pts), len(i_thetas))

    def test_subsample(self):
        """Tests that redundant, closely packed points are removed."""
        pts = [(float(x), 0.0) for x in range(11)]
        times = [1.0] * 11
        thetas = [0.0] * 11
        phis = [math.pi / 4] * 11
        
        p_bdry = [(float(x), 1.0) for x in range(11)]
        m_bdry = [(float(x), -1.0) for x in range(11)]

        s_pts, _, _, _, s_pb, _ = geometry.subsample(
            pts, times, thetas, phis, p_bdry, m_bdry, subsample_ds=5.0
        )

        self.assertLess(len(s_pts), 11)
        self.assertEqual(s_pts[0], (0.0, 0.0))
        self.assertEqual(s_pts[-1], (10.0, 0.0))
        self.assertEqual(len(s_pts), len(s_pb))

class TestEdgeConditions(unittest.TestCase):
    """Tests for messy, degenerate, or invalid input data."""

    def test_empty_and_single_point_curve(self):
        """Curves with < 2 points should safely return zeros without crashing."""
        l1, c1, a1 = geometry.compute_curve_stats([])
        self.assertEqual((l1, c1, a1), (0.0, 0.0, 0.0))

        l2, c2, a2 = geometry.compute_curve_stats([(5.0, 5.0)])
        self.assertEqual((l2, c2, a2), (0.0, 0.0, 0.0))

    def test_duplicate_points(self):
        """Curves with overlapping consecutive points should not divide by zero or spike."""
        # A straight line, but the point (5.0, 0.0) is duplicated 3 times
        curve = [
            (0.0, 0.0), 
            (5.0, 0.0), 
            (5.0, 0.0), 
            (5.0, 0.0), 
            (10.0, 0.0)
        ]
        
        length, total_curv, total_angle = geometry.compute_curve_stats(curve)

        self.assertAlmostEqual(length, 10.0)
        self.assertAlmostEqual(total_curv, 0.0)
        self.assertAlmostEqual(total_angle, 0.0)

    def test_subsample_mismatched_lists(self):
        """Subsample should safely bail out if input lists are out of sync."""
        pts = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
        times = [1.0, 1.0] # Mismatched length!
        thetas = [0.0, 0.0, 0.0]
        phis = [0.0, 0.0, 0.0]
        bp = [(0,1), (1,2), (2,3)]
        bm = [(0,-1), (1,-2), (2,-3)]

        s_pts, s_times, _, _, _, _ = geometry.subsample(
            pts, times, thetas, phis, bp, bm
        )
        
        # Should return the original lists un-altered
        self.assertEqual(len(s_pts), 3)
        self.assertEqual(len(s_times), 2)

        
if __name__ == "__main__":
    unittest.main()
