import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from shock_graph.parser import ShockParser

class TestShockParser(unittest.TestCase):
    def setUp(self):
        self.test_file = "data/test.esf"

    def test_node_degrees(self):
        """Verifies in/out/total degree counts."""
        parser = ShockParser(self.test_file)
        graph = parser.parse()
        
        # Node 2 in our test.esf is a junction/middle point
        node2 = graph.nodes[2]
        self.assertEqual(node2.in_degree, 1)
        self.assertEqual(node2.out_degree, 1)
        self.assertEqual(node2.degree, 2)

    def test_type_expansion(self):
        graph = ShockParser(self.test_file).parse()
        self.assertEqual(graph.nodes[1].type, "A3")

if __name__ == "__main__":
    unittest.main()
