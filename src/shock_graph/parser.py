"""Parser for extracting shock graph structures from ESF files."""

import math
import re
from typing import Dict

from .structures import Edge, Node, SamplePoint, ShockGraph


class ShockParser:
    """Parser matching dbsk2d_xshock_graph_fileio.cxx logic."""

    TYPE_MAP = {
        'A': 'A3',
        'S': 'SOURCE',
        'F': 'SINK',
        'J': 'JUNCT',
        'T': 'TERMINAL',
    }

    def __init__(self, filepath: str) -> None:
        """Initializes the parser with the target filepath.

        Args:
            filepath: The path to the ESF file.
        """
        self._filepath = filepath

    def parse(self) -> ShockGraph:
        """Parses the ESF file into a ShockGraph object.

        Returns:
            A populated ShockGraph instance.
        """
        with open(self._filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        graph = ShockGraph()
        all_samples = self._extract_all_samples(content)

        # 1. Load Nodes
        node_block = re.search(
            r'Begin \[NODE DESCRIPTION\](.*?)End \[NODE DESCRIPTION\]',
            content,
            re.S,
        )
        if node_block:
            pattern = r'(\d+)\s+([AFSTJ])\s+[IO]\s+\[(.*?)\]\s+\[(.*?)\]'
            for match in re.finditer(pattern, node_block.group(1), re.S):
                u_id = int(match.group(1))
                u_type = self.TYPE_MAP.get(match.group(2), "UNKNOWN")
                graph.nodes[u_id] = Node(u_id, u_type)
                
                for v_id in [int(x) for x in match.group(3).split()]:
                    graph.nodes[u_id].add_neighbor(v_id)

        # 2. Build Edges & Assign Node Samples
        edge_block = re.search(
            r'Begin \[EDGE DESCRIPTION\](.*?)End \[EDGE DESCRIPTION\]',
            content,
            re.S,
        )
        if edge_block:
            pattern = r'(\d+)\s+I\s+\[(\d+)\s+(\d+)\]\s+\[(.*?)\]'
            for m in re.finditer(pattern, edge_block.group(1)):
                e_id = int(m.group(1))
                u_id = int(m.group(2))
                v_id = int(m.group(3))
                s_ids = [int(i) for i in m.group(4).split()]

                src_node = graph.nodes[u_id]
                tgt_node = graph.nodes[v_id]
                edge_samples = [all_samples[sid] for sid in s_ids]

                if src_node.sample is None:
                    src_node.sample = edge_samples[0]
                if tgt_node.sample is None:
                    tgt_node.sample = edge_samples[-1]

                # Edge instantiation triggers degree updates in nodes
                graph.edges.append(
                    Edge(e_id, src_node, tgt_node, edge_samples)
                )

        return graph

    def _extract_all_samples(self, content: str) -> Dict[int, SamplePoint]:
        """Extracts all raw sample points from the ESF content.

        Args:
            content: The raw string content of the ESF file.

        Returns:
            A dictionary mapping sample IDs to SamplePoint objects.
        """
        samples = {}
        blocks = re.findall(r'Begin SAMPLE(.*?)End SAMPLE', content, re.S)
        
        for b in blocks:
            sid = int(re.search(r'sample_id\s+(\d+)', b).group(1))
            
            x_y_t_match = re.search(r'\(x, y, t\)\s+\((.*?)\)', b)
            x, y, t = map(float, x_y_t_match.group(1).split(','))
            
            speed = float(re.search(r'speed\s+([e\d\.\+\-]+)', b).group(1))
            
            theta_deg = float(re.search(r'theta\s+([\d\.]+)', b).group(1))
            theta = theta_deg * math.pi / 180.0
            
            if 0 < abs(speed) < 99990:
                phi = math.acos(-1.0 / speed)
            else:
                phi = math.pi / 2.0
                
            samples[sid] = SamplePoint(sid, x, y, t, speed, theta, phi)
            
        return samples
