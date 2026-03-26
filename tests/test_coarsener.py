"""Unit tests for the graph coarsening module."""

import os
import glob
import sys
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

    def _get_pure_deg2_nodes(self, graph: ShockGraph) -> list:
        """Helper to find nodes that strictly meet the coarsening criteria."""
        deg2_nodes = []
        for node in graph.nodes.values():
            if (node.degree == 2 and 
                node.in_degree == 1 and 
                node.out_degree == 1 and 
                node.type not in ['SOURCE', 'SINK', 'TERMINAL', 'A3']):
                deg2_nodes.append(node)
        return deg2_nodes

    # -----------------------------------------------------------------------
    # 1. SYNTHETIC TESTS (Controlled Environments)
    # -----------------------------------------------------------------------
    def test_mock_chain_coarsening(self):
        """Builds a manual A->B->C->D chain to prove exact sample concatenation and neighbor routing."""
        graph = ShockGraph()
        
        n_a = Node(1, 'SOURCE')
        n_b = Node(2, 'JUNCT')
        n_c = Node(3, 'JUNCT')
        n_d = Node(4, 'TERMINAL')
        
        n_a.add_neighbor(2)
        n_b.add_neighbor(3)
        n_b.add_neighbor(1)
        n_c.add_neighbor(4)
        n_c.add_neighbor(2)
        n_d.add_neighbor(3)
        
        graph.nodes = {1: n_a, 2: n_b, 3: n_c, 4: n_d}
        
        s1 = SamplePoint(1, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0)
        s2 = SamplePoint(2, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0)
        s3 = SamplePoint(3, 2.0, 0.0, 1.0, 2.0, 0.0, 0.0)
        s4 = SamplePoint(4, 3.0, 0.0, 1.0, 2.0, 0.0, 0.0)
        
        # Instantiating edges automatically populates incoming/outgoing tracking
        e1 = Edge(10, n_a, n_b, [s1, s2])
        e2 = Edge(11, n_b, n_c, [s2, s3])
        e3 = Edge(12, n_c, n_d, [s3, s4])
        graph.edges = [e1, e2, e3]

        coarsened = GraphCoarsener.coarsen(graph)

        self.assertIn(1, coarsened.nodes, "SOURCE node A was incorrectly removed.")
        self.assertIn(4, coarsened.nodes, "TERMINAL node D was incorrectly removed.")
        self.assertNotIn(2, coarsened.nodes, "JUNCT node B was not removed.")
        self.assertNotIn(3, coarsened.nodes, "JUNCT node C was not removed.")
        
        self.assertEqual(len(coarsened.edges), 1, "Should be exactly 1 merged edge remaining.")
        merged_edge = coarsened.edges[0]
        
        self.assertEqual(merged_edge.source.id, 1, "Merged edge must start at Node A.")
        self.assertEqual(merged_edge.target.id, 4, "Merged edge must end at Node D.")
        self.assertEqual(len(merged_edge.samples), 4, "Merged samples should exactly match s1,s2,s3,s4.")
        
        new_n_a = coarsened.nodes[1]
        self.assertIn(4, new_n_a.get_cw_neighbors(), "Node A's neighbors did not fast-forward to Node D.")

    def test_mock_y_junction_retention(self):
        """Builds a Y-junction (A->B, C->B, B->D). Node B has degree 3 and MUST NOT be removed."""
        graph = ShockGraph()
        
        n_a = Node(1, 'SOURCE')
        n_c = Node(2, 'SOURCE')
        n_b = Node(3, 'JUNCT')
        n_d = Node(4, 'TERMINAL')
        
        graph.nodes = {1: n_a, 2: n_c, 3: n_b, 4: n_d}
        
        s_mock = [SamplePoint(1, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0), SamplePoint(2, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0)]
        
        # A and C both flow into B. B flows into D.
        e1 = Edge(10, n_a, n_b, s_mock)
        e2 = Edge(11, n_c, n_b, s_mock)
        e3 = Edge(12, n_b, n_d, s_mock)
        graph.edges = [e1, e2, e3]

        # B is a JUNCT, but its in_degree is 2. (Total degree = 3)
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
                deg2_nodes = self._get_pure_deg2_nodes(graph)
                deg2_count = len(deg2_nodes)
                
                # Only test files that actually have compressible geometry
                if deg2_count == 0:
                    continue
                    
                files_tested += 1
                coarsened_graph = GraphCoarsener.coarsen(graph)

                # ASSERT GLOBAL: Total Node Count Matches Math
                expected_node_count = len(graph.nodes) - deg2_count
                self.assertEqual(
                    len(coarsened_graph.nodes), 
                    expected_node_count, 
                    f"Global node count mismatch in {file_name}! "
                    f"Expected {expected_node_count}, got {len(coarsened_graph.nodes)}."
                )

                # ASSERT GLOBAL: Eradication
                remaining_deg2 = self._get_pure_deg2_nodes(coarsened_graph)
                self.assertEqual(
                    len(remaining_deg2), 
                    0, 
                    f"Found pure degree-2 nodes in {file_name} that the coarsener missed."
                )

                # ASSERT LOCAL: No Dangling Pointers
                for node_id, new_node in coarsened_graph.nodes.items():
                    for neighbor_id in new_node.get_cw_neighbors():
                        self.assertIn(
                            neighbor_id, 
                            coarsened_graph.nodes,
                            f"DANGLING POINTER in {file_name}: Node {node_id} claims {neighbor_id} "
                            f"is a neighbor, but {neighbor_id} was deleted!"
                        )

        print(f"\n[Test Info] Successfully verified topological coarsening invariants across {files_tested} complex ESF files.")


if __name__ == "__main__":
    unittest.main()
