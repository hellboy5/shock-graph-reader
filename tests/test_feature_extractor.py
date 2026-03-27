"""Unit tests for the geometric feature extraction."""

import os
import sys
import unittest

# Ensure the src directory is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from shock_graph.parser import ShockParser


class TestFeatureExtractorRectangle(unittest.TestCase):
    """Test suite verifying geometric invariants on a perfect rectangle (rec3)."""

    def setUp(self):
        self.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
        self.rec_esf = os.path.join(self.data_dir, 'rec3.esf')

    def test_rectangle_invariants(self):
        """Verifies zero-curvature, area reconstruction, and perimeter reconstruction."""
        
        if not os.path.exists(self.rec_esf):
            self.skipTest("rec3.esf not found in data/ directory. Skipping test.")

        # ---------------------------------------------------------
        # 1. HARDCODED GROUND TRUTH (From rec3.png analysis)
        # ---------------------------------------------------------
        true_area = 8385.0
        true_perimeter = 384.0

        # ---------------------------------------------------------
        # 2. FEATURE EXTRACTION
        # ---------------------------------------------------------
        # Parsing automatically calls ShockFeatureExtractor.process_graph()
        graph = ShockParser(self.rec_esf).parse()
        
        total_poly_area = 0.0
        total_boundary_length = 0.0

        for edge in graph.edges:
            feats = edge.features
            self.assertIsNotNone(feats, "Features should be populated.")
            
            # --- INVARIANT A: Zero Bending ---
            # Using a tight tolerance (0.2 radians) for floating point math
            tol_angle = 0.2 
            
            self.assertLess(feats.s_curve, tol_angle, f"s_curve {feats.s_curve:.3f} exceeds straightness tolerance.")
            self.assertLess(feats.s_angle, tol_angle, f"s_angle {feats.s_angle:.3f} exceeds straightness tolerance.")
            self.assertLess(feats.p_curve, tol_angle, f"p_curve {feats.p_curve:.3f} exceeds straightness tolerance.")
            self.assertLess(feats.p_angle, tol_angle, f"p_angle {feats.p_angle:.3f} exceeds straightness tolerance.")
            self.assertLess(feats.m_curve, tol_angle, f"m_curve {feats.m_curve:.3f} exceeds straightness tolerance.")
            self.assertLess(feats.m_angle, tol_angle, f"m_angle {feats.m_angle:.3f} exceeds straightness tolerance.")
            
            # --- INVARIANT B: Zero Flare ---
            # Straight boundaries cannot flare outwards like a trumpet.
            self.assertLess(feats.total_flare, tol_angle, f"total_flare {feats.total_flare:.3f} exceeds tolerance.")

            # Accumulate global invariants
            total_poly_area += feats.poly_area
            total_boundary_length += (feats.p_length + feats.m_length)

        # ---------------------------------------------------------
        # 3. GLOBAL RECONSTRUCTION INVARIANTS
        # ---------------------------------------------------------
        # We use a 5% relative error tolerance. The continuous medial axis polygons 
        # computed via shoelace formula will slightly differ from a raw discrete pixel count.
        
        area_error = abs(total_poly_area - true_area) / true_area
        self.assertLess(
            area_error, 0.05, 
            f"Area Mismatch! True Pixels: {true_area}, Graph Polygons: {total_poly_area:.2f}"
        )

        perim_error = abs(total_boundary_length - true_perimeter) / true_perimeter
        self.assertLess(
            perim_error, 0.05, 
            f"Perimeter Mismatch! True ArcLength: {true_perimeter:.2f}, Graph Boundaries: {total_boundary_length:.2f}"
        )


if __name__ == "__main__":
    unittest.main()
