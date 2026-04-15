"""Unit tests for the graph coarsening module."""

import os
import glob
import sys
import math
import unittest

# Ensure the src directory is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from shock_graph.parser import ShockParser
from shock_graph.coarsener import GraphCoarsener
from shock_graph.structures import Edge, Node, SamplePoint, ShockGraph


class TestGraphCoarsener(unittest.TestCase):
    """Test suite for the degree-2 node compression logic."""

    def setUp(self):
        self.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
        self.esf_files = glob.glob(os.path.join(self.data_dir, '*.esf'))

    def _get_reducible_deg2_nodes(self, graph: ShockGraph) -> list:
        """Helper to find pure degree-2 nodes that SHOULD have been coarsened.
        
        Evaluates purely on undirected topology, ignoring arbitrary Flow labels.
        Ignores degree-2 nodes that were deliberately promoted to Anchors 
        by the Midpoint Protocol to prevent illegal multigraphs or infinite loops.
        """
        reducible_nodes = []
        for node in graph.nodes.values():
            # Treat the graph as undirected for topological assessment
            if node.degree == 2:
                incident = node.incoming_edges + node.outgoing_edges
                
                if len(incident) != 2:
                    continue # Safety failsafe
                
                # Identify the actual neighbor nodes, regardless of edge direction
                n1 = incident[0].source if incident[0].target == node else incident[0].target
                n2 = incident[1].source if incident[1].target == node else incident[1].target
                
                # If collapsing this node creates a self-loop (n1 == n2), it was deliberately saved.
                if n1.id == n2.id:
                    continue
                    
                # If collapsing this node creates parallel edges, it was deliberately saved.
                is_parallel_risk = False
                for e in n1.incoming_edges + n1.outgoing_edges:
                    other_node = e.source if e.target == n1 else e.target
                    if other_node.id == n2.id and e not in incident:
                        is_parallel_risk = True
                        break
                        
                if not is_parallel_risk:
                    reducible_nodes.append(node)
                    
        return reducible_nodes

    def _compute_arc_length(self, samples: list[SamplePoint]) -> float:
        """Helper to compute the physical Euclidean length of a sample array."""
        length = 0.0
        for i in range(1, len(samples)):
            dx = samples[i].x - samples[i-1].x
            dy = samples[i].y - samples[i-1].y
            length += math.hypot(dx, dy)
        return length

    # -----------------------------------------------------------------------
    # 1. SYNTHETIC TESTS (Controlled Environments)
    # -----------------------------------------------------------------------
    
    def test_arc_length_and_splicing(self):
        """Verifies that duplicate junctions are dropped and Arc Length is perfectly additive."""
        graph = ShockGraph()
        
        n_a = Node(1, 'SOURCE')
        n_b = Node(2, 'JUNCT')
        n_c = Node(3, 'TERMINAL')
        graph.nodes = {1: n_a, 2: n_b, 3: n_c}
        
        # 5 distinct locations. Point ID 3 is the shared junction.
        s1 = SamplePoint(1, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0)
        s2 = SamplePoint(2, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0)
        s3 = SamplePoint(3, 2.0, 0.0, 1.0, 2.0, 0.0, 0.0) # Shared overlap at x=2.0
        
        s4 = SamplePoint(4, 3.0, 0.0, 1.0, 2.0, 0.0, 0.0)
        s5 = SamplePoint(5, 4.0, 0.0, 1.0, 2.0, 0.0, 0.0)
        
        e1 = Edge(10, n_a, n_b, [s1, s2, s3])
        e2 = Edge(11, n_b, n_c, [s3, s4, s5])
        graph.edges = [e1, e2]
        
        coarsened = GraphCoarsener.coarsen(graph)
        merged_edge = coarsened.edges[0]
        
        # Assert Splicing Logic (3 + 3 = 6. Minus 1 dropped junction = 5)
        self.assertEqual(len(merged_edge.samples), 5, "Splicing must drop exactly one duplicate junction node.")
        
        # Assert Math Invariant (Arc Length Additivity)
        l1 = self._compute_arc_length(e1.samples)
        l2 = self._compute_arc_length(e2.samples)
        l_merged = self._compute_arc_length(merged_edge.samples)
        
        self.assertAlmostEqual(
            l_merged, l1 + l2, places=5, 
            msg="Macro-edge arc length must perfectly equal the sum of the micro-edges."
        )

    def test_backward_traversal_geometry_flip(self):
        """Verifies theta and phi are mathematically inverted when walking against the edge direction."""
        graph = ShockGraph()
        n_a = Node(1, 'SOURCE')
        n_b = Node(2, 'JUNCT')
        n_c = Node(3, 'TERMINAL')
        graph.nodes = {1: n_a, 2: n_b, 3: n_c}
        
        # Edge 1: A -> B (Forward traversal)
        s1 = SamplePoint(1, 0.0, 0.0, 1.0, 2.0, 0.5, 0.2)
        s2 = SamplePoint(2, 1.0, 0.0, 1.0, 2.0, 0.5, 0.2)
        e1 = Edge(10, n_a, n_b, [s1, s2])
        
        # Edge 2: C -> B (Coarsener must traverse backward from B -> C)
        s3_at_c = SamplePoint(3, 2.0, 0.0, 1.0, 2.0, 0.8, 0.3)
        s4_at_b = SamplePoint(4, 1.0, 0.0, 1.0, 2.0, 0.8, 0.3)
        # Because edge is C->B, samples start at C and end at B
        e2 = Edge(11, n_c, n_b, [s3_at_c, s4_at_b]) 
        graph.edges = [e1, e2]
        
        coarsened = GraphCoarsener.coarsen(graph)
        merged_edge = coarsened.edges[0]
        
        self.assertEqual(merged_edge.source.id, 1)
        self.assertEqual(merged_edge.target.id, 3)
        
        # Merged samples should be: s1, s2, then flipped s3_at_c. (s4_at_b is dropped as duplicate)
        self.assertEqual(len(merged_edge.samples), 3)
        
        flipped_sample = merged_edge.samples[2]
        
        # Expected Math: theta + pi, and pi - phi
        expected_theta = (0.8 + math.pi) % (2 * math.pi)
        expected_phi = math.pi - 0.3
        
        self.assertAlmostEqual(flipped_sample.theta, expected_theta, places=5, msg="Theta failed to rotate 180 degrees.")
        self.assertAlmostEqual(flipped_sample.phi, expected_phi, places=5, msg="Phi failed to invert flow angle.")
        self.assertEqual(flipped_sample.x, 2.0, "Physical coordinates were corrupted during traversal flip.")

    def test_midpoint_protocol_teardrop(self):
        """Verifies that a 3-edge teardrop loop correctly promotes the geometric midpoint to an Anchor."""
        graph = ShockGraph()
        n_a = Node(1, 'JUNCT') # The Base Anchor
        n_b = Node(2, 'JUNCT') # Pass-through
        n_c = Node(3, 'JUNCT') # Pass-through
        n_out = Node(4, 'TERMINAL')
        graph.nodes = {1: n_a, 2: n_b, 3: n_c, 4: n_out}
        
        s = SamplePoint(0,0,0,0,0,0,0)
        
        # Build A -> B -> C -> A (Teardrop)
        e1 = Edge(10, n_a, n_b, [s, s])
        e2 = Edge(11, n_b, n_c, [s, s])
        e3 = Edge(12, n_c, n_a, [s, s])
        e4 = Edge(13, n_a, n_out, [s, s]) # Forces A to act as a permanent Anchor (degree 3)
        graph.edges = [e1, e2, e3, e4]
        
        coarsened = GraphCoarsener.coarsen(graph)
        
        self.assertIn(2, coarsened.nodes, "Midpoint Node B should have been promoted to Anchor!")
        self.assertNotIn(3, coarsened.nodes, "Node C should still be compressed away.")
        self.assertEqual(len(coarsened.edges), 3, "Graph should resolve to exactly 3 edges: A->B, B->A, and A->Out.")

    def test_mock_y_junction_retention(self):
        """Builds a Y-junction (A->B, C->B, B->D). Node B has degree 3 and MUST NOT be removed."""
        graph = ShockGraph()
        
        n_a = Node(1, 'SOURCE')
        n_c = Node(2, 'SOURCE')
        n_b = Node(3, 'JUNCT')
        n_d = Node(4, 'TERMINAL')
        
        graph.nodes = {1: n_a, 2: n_c, 3: n_b, 4: n_d}
        s_mock = [SamplePoint(1, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0), SamplePoint(2, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0)]
        
        e1 = Edge(10, n_a, n_b, s_mock)
        e2 = Edge(11, n_c, n_b, s_mock)
        e3 = Edge(12, n_b, n_d, s_mock)
        graph.edges = [e1, e2, e3]

        coarsened = GraphCoarsener.coarsen(graph)

        self.assertIn(3, coarsened.nodes, "Y-Junction Node B was incorrectly removed! Rule 3 failed.")
        self.assertEqual(len(coarsened.edges), 3, "Edges should not have been merged across a Y-Junction.")

    # -----------------------------------------------------------------------
    # 2. REAL DATA TEST (Iterative verification across the dataset)
    # -----------------------------------------------------------------------
    
    def test_real_data_global_coarsening(self):
        """Iterates over every ESF file and strictly verifies topological invariants if coarsened."""
        self.assertTrue(len(self.esf_files) > 0, "No .esf files found in the data/ folder.")

        files_tested = 0

        for file_path in self.esf_files:
            file_name = os.path.basename(file_path)
            
            with self.subTest(file=file_name):
                graph = ShockParser(file_path).parse()
                deg2_nodes = self._get_reducible_deg2_nodes(graph)
                
                if len(deg2_nodes) == 0:
                    continue
                    
                files_tested += 1
                coarsened_graph = GraphCoarsener.coarsen(graph)

                # ASSERT: Eradication of standard Degree-2 chains
                remaining_deg2 = self._get_reducible_deg2_nodes(coarsened_graph)
                self.assertEqual(
                    len(remaining_deg2), 0, 
                    f"Found reducible degree-2 nodes in {file_name} that the coarsener missed."
                )

                # ASSERT: Compression ratios
                self.assertLess(
                    len(coarsened_graph.nodes), len(graph.nodes),
                    f"Node count did not decrease in {file_name} despite finding reducible nodes."
                )
                self.assertLessEqual(
                    len(coarsened_graph.edges), len(graph.edges),
                    f"Edge count anomalously increased in {file_name}."
                )

                # ASSERT LOCAL: No Dangling Pointers & Degree calculations
                for node_id, new_node in coarsened_graph.nodes.items():
                    # Verify dynamic degree calculation holds
                    self.assertEqual(new_node.degree, new_node.in_degree + new_node.out_degree)
                    
                    for neighbor_id in new_node.get_cw_neighbors():
                        self.assertIn(
                            neighbor_id, 
                            coarsened_graph.nodes,
                            f"DANGLING POINTER in {file_name}: Node {node_id} claims {neighbor_id} "
                            f"is a neighbor, but {neighbor_id} was deleted!"
                        )

        print(f"\n[Test Info] Successfully verified topological invariants across {files_tested} complex ESF files.")


if __name__ == "__main__":
    unittest.main()
