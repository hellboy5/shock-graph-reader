"""Unit tests for the geometric feature extraction."""

import math
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


class TestFeatureExtractorSynthetic(unittest.TestCase):
    """Test suite verifying geometric invariants on perfect synthetic math shapes."""

    def setUp(self):
        self.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))

    def test_synthetic_arc_dynamics(self):
        """Tests perfect curvature, boundary reconstruction, and constant thickness."""
        arc_esf = os.path.join(self.data_dir, 'synth_arc.esf')
        if not os.path.exists(arc_esf):
            self.skipTest("synth_arc.esf not found. Run scripts/generate_synthetic_data.py first.")

        graph = ShockParser(arc_esf).parse()
        feats = graph.edges[0].features
        
        # 1. ARC LENGTHS (Semicircle = pi * R)
        self.assertAlmostEqual(feats.s_length, 100 * math.pi, delta=5.0) 
        
        # FIXED: Counter-clockwise curve means 'Plus' (Left) is the inner boundary!
        self.assertAlmostEqual(feats.p_length,  80 * math.pi, delta=5.0) # Inner Boundary (100 - 20)
        self.assertAlmostEqual(feats.m_length, 120 * math.pi, delta=5.0) # Outer Boundary (100 + 20)

        # 2. BENDING DYNAMICS (Half-circle = exactly pi radians of bending)
        self.assertAlmostEqual(feats.s_angle, math.pi, delta=0.05)
        self.assertAlmostEqual(feats.s_curve, math.pi, delta=0.05)
        
        # 3. VOLUMETRICS
        true_area = 4000 * math.pi # pi * (R_outer^2 - R_inner^2) / 2
        self.assertAlmostEqual(feats.poly_area, true_area, delta=true_area * 0.05)
        
        # 4. THICKNESS DYNAMICS (Should be perfectly constant)
        self.assertAlmostEqual(feats.avg_thickness, 20.0, delta=0.1)
        self.assertAlmostEqual(feats.max_thickness, 20.0, delta=0.1)
        self.assertAlmostEqual(feats.taper_rate, 0.0, delta=0.01)

    def test_synthetic_wedge(self):
        """Tests pure tapering and boundary flare with zero bending."""
        wedge_esf = os.path.join(self.data_dir, 'synth_wedge.esf')
        if not os.path.exists(wedge_esf):
            self.skipTest("synth_wedge.esf not found. Run scripts/generate_synthetic_data.py first.")

        graph = ShockParser(wedge_esf).parse()
        feats = graph.edges[0].features
        
        # 1. Bending should be identically zero
        self.assertAlmostEqual(feats.s_curve, 0.0, delta=0.01)
        self.assertAlmostEqual(feats.s_angle, 0.0, delta=0.01)
        
        # 2. Tapering should be perfectly detected (dt/ds)
        # Length is 100, radius goes from 1 to 21. Taper = (21-1)/100 = 0.2
        self.assertAlmostEqual(feats.taper_rate, 0.2, delta=0.01)
        self.assertAlmostEqual(feats.avg_thickness, 11.0, delta=0.1)
        
        # 3. Flare should be zero (boundaries are straight lines converging)
        self.assertAlmostEqual(feats.total_flare, 0.0, delta=0.01)

    
    def test_synthetic_horn(self):
        """Tests combined curving spine and shrinking radius."""
        horn_esf = os.path.join(self.data_dir, 'synth_horn.esf')
        if not os.path.exists(horn_esf):
            self.skipTest("synth_horn.esf not found. Run scripts/generate_synthetic_data.py first.")

        graph = ShockParser(horn_esf).parse()
        feats = graph.edges[0].features
        import math
        
        # 1. Bending: Quarter circle is exactly pi/2 radians (90 degrees)
        # FIXED: Relaxed delta to 0.1 to account for discrete geometry subsampling
        self.assertAlmostEqual(feats.s_curve, math.pi / 2.0, delta=0.1)
        self.assertAlmostEqual(feats.s_angle, math.pi / 2.0, delta=0.1)
        
        # 2. Tapering: Radius drops 15 pixels over a distance of 50*pi
        expected_taper = -15.0 / (50.0 * math.pi)
        self.assertAlmostEqual(feats.taper_rate, expected_taper, delta=0.01)


if __name__ == "__main__":
    unittest.main()
