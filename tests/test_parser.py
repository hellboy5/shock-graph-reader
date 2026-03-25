import unittest
import sys
import os
import glob

# Ensure the src directory is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from shock_graph.parser import ShockParser

# -------------------------------------------------------------------
# Configuration for specific hardcoded file checks
# -------------------------------------------------------------------
TEST_CONFIGS = [
    {
        "filename": "bettle.esf", 
        "expected_node_count": 220,
        "expected_edge_count": 219,
        "test_node_id": 129480,
        "expected_node_type": "SINK", 
        "expected_node_neighbors": [128440, 120776, 121083],
        "test_edge_id": 126338,
        "expected_edge_source": 88090,
        "expected_edge_target": 126466 
    }
]

class TestShockParser(unittest.TestCase):
    def setUp(self):
        # Dynamically locate the data/ directory and grab all .esf files
        self.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
        self.esf_files = glob.glob(os.path.join(self.data_dir, '*.esf'))

    def test_parse_all_esf_files(self):
        """Verifies that all ESF files in the data directory parse correctly without errors."""
        self.assertTrue(len(self.esf_files) > 0, "No .esf files found in the data/ folder.")

        for file_path in self.esf_files:
            file_name = os.path.basename(file_path)
            with self.subTest(file=file_name):
                try:
                    graph = ShockParser(file_path).parse()
                except Exception as e:
                    self.fail(f"Parser raised an exception on {file_name}: {e}")
                
                self.assertIsNotNone(graph, f"Parsed graph returned None for {file_name}")
                self.assertTrue(hasattr(graph, 'nodes'), f"Graph missing 'nodes' attribute in {file_name}")
                self.assertTrue(hasattr(graph, 'edges'), f"Graph missing 'edges' attribute in {file_name}")
                self.assertGreater(len(graph.nodes), 0, f"No nodes were parsed for {file_name}")

    def test_node_degrees(self):
        """Verifies total degree counts are mathematically consistent with in/out degrees."""
        for file_path in self.esf_files:
            file_name = os.path.basename(file_path)
            with self.subTest(file=file_name):
                graph = ShockParser(file_path).parse()
                
                for node_id, node in graph.nodes.items():
                    # Check for type correctness
                    self.assertIsInstance(node.in_degree, int)
                    self.assertIsInstance(node.out_degree, int)
                    self.assertIsInstance(node.degree, int)
                    
                    # Logically, total degree = in + out
                    self.assertEqual(
                        node.degree, 
                        node.in_degree + node.out_degree, 
                        f"Degree mismatch on node {node_id} in {file_name}."
                    )

    def test_type_expansion(self):
        """Verifies the type property is correctly parsed and formatted."""
        for file_path in self.esf_files:
            file_name = os.path.basename(file_path)
            with self.subTest(file=file_name):
                graph = ShockParser(file_path).parse()
                
                for node_id, node in graph.nodes.items():
                    self.assertIsNotNone(node.type, f"Node {node_id} in {file_name} has None for type.")
                    self.assertIsInstance(node.type, str, f"Node {node_id} type is not a string in {file_name}.")
                    self.assertTrue(len(node.type) > 0, f"Node {node_id} in {file_name} has an empty type string.")

    def test_edge_references_valid_nodes(self):
        """Ensures every edge points to source and target nodes that actually exist in the graph."""
        for file_path in self.esf_files:
            file_name = os.path.basename(file_path)
            with self.subTest(file=file_name):
                graph = ShockParser(file_path).parse()
                
                for edge in graph.edges:
                    self.assertIn(
                        edge.source.id, graph.nodes, 
                        f"Edge {edge.id} references missing source node {edge.source.id} in {file_name}."
                    )
                    self.assertIn(
                        edge.target.id, graph.nodes, 
                        f"Edge {edge.id} references missing target node {edge.target.id} in {file_name}."
                    )

    def test_topology_matches_degrees(self):
        """
        Calculates the exact topology strictly from the edges and verifies it against the 
        nodes' parsed degree states. This ensures flow (A->B<-C) is correctly mapped.
        """
        for file_path in self.esf_files:
            file_name = os.path.basename(file_path)
            with self.subTest(file=file_name):
                graph = ShockParser(file_path).parse()
                
                # Manually map out the actual edge connections
                actual_in_degrees = {node_id: 0 for node_id in graph.nodes}
                actual_out_degrees = {node_id: 0 for node_id in graph.nodes}
                
                for edge in graph.edges:
                    actual_out_degrees[edge.source.id] += 1
                    actual_in_degrees[edge.target.id] += 1
                    
                # Verify the actual edge topology perfectly matches the parsed node properties
                for node_id, node in graph.nodes.items():
                    self.assertEqual(
                        node.in_degree, actual_in_degrees[node_id], 
                        f"Topology flow error in {file_name}: Node {node_id} claims in_degree "
                        f"{node.in_degree} but actually has {actual_in_degrees[node_id]} incoming edges."
                    )
                    self.assertEqual(
                        node.out_degree, actual_out_degrees[node_id], 
                        f"Topology flow error in {file_name}: Node {node_id} claims out_degree "
                        f"{node.out_degree} but actually has {actual_out_degrees[node_id]} outgoing edges."
                    )

    # -------------------------------------------------------------------
    # DATA-DRIVEN TEST FOR HARDCODED VALUES
    # -------------------------------------------------------------------
    def test_hardcoded_specifics(self):
        """Verifies specific hardcoded values for known files defined in TEST_CONFIGS."""
        for config in TEST_CONFIGS:
            file_name = config["filename"]
            file_path = os.path.join(self.data_dir, file_name)
            
            # Skip if the configured file isn't physically in the data directory
            if not os.path.exists(file_path):
                print(f"\nWarning: Skipping {file_name} in TEST_CONFIGS because it was not found in data/")
                continue
                
            with self.subTest(file=file_name):
                graph = ShockParser(file_path).parse()
                
                # 1. Assert Totals
                self.assertEqual(
                    len(graph.nodes), 
                    config["expected_node_count"], 
                    f"Node count mismatch in {file_name}"
                )
                self.assertEqual(
                    len(graph.edges), 
                    config["expected_edge_count"], 
                    f"Edge count mismatch in {file_name}"
                )

                # 2. Assert Specific Node Type
                target_node_id = config["test_node_id"]
                self.assertIn(target_node_id, graph.nodes, f"Node {target_node_id} missing in {file_name}")
                self.assertEqual(
                    graph.nodes[target_node_id].type, 
                    config["expected_node_type"],
                    f"Node {target_node_id} type mismatch in {file_name}"
                )

                # 3. Assert Specific Edge Source and Target
                target_edge_id = config["test_edge_id"]
                target_edge = next((e for e in graph.edges if e.id == target_edge_id), None)
                self.assertIsNotNone(target_edge, f"Edge {target_edge_id} missing in {file_name}")
                
                # Check Source
                self.assertEqual(
                    target_edge.source.id,
                    config["expected_edge_source"],
                    f"Edge {target_edge_id} source mismatch in {file_name}"
                )
                
                # Check Target
                self.assertEqual(
                    target_edge.target.id,
                    config["expected_edge_target"],
                    f"Edge {target_edge_id} target mismatch in {file_name}"
                )

                # 4. Assert Specific Node Neighbors (Order independent)
                if "expected_node_neighbors" in config:
                    actual_neighbors = graph.nodes[target_node_id].get_cw_neighbors()
                    self.assertCountEqual(
                        actual_neighbors, 
                        config["expected_node_neighbors"],
                        f"Node {target_node_id} neighbors mismatch in {file_name}. "
                        f"Expected {config['expected_node_neighbors']}, got {actual_neighbors}"
                    )

if __name__ == "__main__":
    unittest.main()
