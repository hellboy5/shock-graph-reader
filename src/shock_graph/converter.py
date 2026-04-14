"""Converts ShockGraph objects into various machine learning graph formats."""

from typing import Any, Dict, List, Tuple

from .structures import Edge, Node, ShockGraph


class GraphConverter:
    """Provides utilities to convert ShockGraphs into external formats."""

    @classmethod
    def _extract_raw_data(
        cls,
        graph: ShockGraph,
        coarsened: bool = False,
        bidirectional: bool = False,
    ) -> Tuple[List[List[float]], List[Tuple[int, int]], List[List[float]]]:
        """Extracts standardized lists of nodes, edges, and features.

        Re-indexes arbitrary node IDs to a continuous [0, N-1] range required
        by most ML frameworks.

        Args:
            graph: The parsed and hydrated ShockGraph.
            coarsened: If True, zeroes out the geometrically meaningless Flow Types.
            bidirectional: If True, generates a swapped backward edge for every connection.

        Returns:
            A tuple containing:
                - node_features: List of feature arrays per node [x, y, t, 5 flow, 3 struct].
                - edge_indices: List of (source_idx, target_idx) tuples.
                - edge_features: List of feature arrays per edge.
        """
        node_to_idx: Dict[int, int] = {}
        for idx, original_id in enumerate(sorted(graph.nodes.keys())):
            node_to_idx[original_id] = idx

        # -------------------------------------------------------------------
        # 1. Extract Node Features (Fixed Length: 11)
        # -------------------------------------------------------------------
        node_features: List[List[float]] = []
        
        for original_id in sorted(graph.nodes.keys()):
            node = graph.nodes[original_id]
            x = node.sample.x if node.sample else 0.0
            y = node.sample.y if node.sample else 0.0
            t = node.sample.t if node.sample else 0.0

            # 5 Flow slots, 3 Structural slots
            flow_feats = [0.0, 0.0, 0.0, 0.0, 0.0]  
            struct_feats = [0.0, 0.0, 0.0]          

            # --- RULE 1: Topological Degrees are ALWAYS mathematically valid ---
            if node.degree == 1: struct_feats[0] = 1.0
            elif node.degree == 2: struct_feats[1] = 1.0
            elif node.degree >= 3: struct_feats[2] = 1.0 

            # --- RULE 2: Flow Types are ONLY valid if the graph is uncoarsened ---
            if not coarsened:
                if node.type == 'SOURCE': flow_feats[0] = 1.0
                elif node.type == 'SINK': flow_feats[1] = 1.0
                elif node.type == 'JUNCT': flow_feats[2] = 1.0
                elif node.type == 'TERMINAL': flow_feats[3] = 1.0
                elif node.type == 'A3': flow_feats[4] = 1.0

            # Concatenate to guarantee length 11
            final_node_feat = [x, y, t] + flow_feats + struct_feats
            node_features.append(final_node_feat)
            
        # -------------------------------------------------------------------
        # 2. Extract Edge Indices and Edge Features
        # -------------------------------------------------------------------
        edge_indices: List[Tuple[int, int]] = []
        edge_features: List[List[float]] = []

        # DETERMINISM FIX: Sort the edges based on their new contiguous source/target IDs
        # This guarantees the PyTorch edge_index tensor rows are mathematically locked.
        sorted_edges = sorted(
            graph.edges, 
            key=lambda e: (node_to_idx[e.source.id], node_to_idx[e.target.id])
        )

        for edge in sorted_edges:
            src_idx = node_to_idx[edge.source.id]
            tgt_idx = node_to_idx[edge.target.id]
            
            # Fallback if feature extraction was skipped
            if not edge.features:
                edge_indices.append((src_idx, tgt_idx))
                edge_features.append([0.0] * 14)
                if bidirectional:
                    edge_indices.append((tgt_idx, src_idx))
                    edge_features.append([0.0] * 14)
                continue

            f = edge.features

            # --- FORWARD EDGE (A -> B) ---
            edge_indices.append((src_idx, tgt_idx))
            edge_features.append([
                f.s_length, f.s_curve, f.s_angle,
                f.p_length, f.p_curve, f.p_angle,  # Left
                f.m_length, f.m_curve, f.m_angle,  # Right
                f.poly_area, f.avg_thickness, f.max_thickness,
                f.taper_rate, f.total_flare        # Forward taper
            ])

            # --- BACKWARD EDGE (B -> A) ---
            if bidirectional:
                edge_indices.append((tgt_idx, src_idx))
                edge_features.append([
                    f.s_length, f.s_curve, f.s_angle,
                    f.m_length, f.m_curve, f.m_angle,  # Swapped Left/Right
                    f.p_length, f.p_curve, f.p_angle,  # Swapped Left/Right
                    f.poly_area, f.avg_thickness, f.max_thickness,
                    -f.taper_rate, f.total_flare       # Flipped Taper Sign
                ])

        return node_features, edge_indices, edge_features

    # -----------------------------------------------------------------------
    # In-Memory Conversion Methods
    # -----------------------------------------------------------------------

    @classmethod
    def to_pytorch_geometric(cls, graph: ShockGraph, coarsened: bool = False, bidirectional: bool = False) -> Any:
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError as e:
            raise ImportError("PyTorch Geometric is not installed.") from e

        n_feats, e_indices, e_feats = cls._extract_raw_data(graph, coarsened=coarsened, bidirectional=bidirectional)

        x = torch.tensor(n_feats, dtype=torch.float)
        edge_index = torch.tensor(e_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(e_feats, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    @classmethod
    def to_networkx(cls, graph: ShockGraph, coarsened: bool = False, bidirectional: bool = False) -> Any:
        try:
            import networkx as nx
        except ImportError as e:
            raise ImportError("NetworkX is not installed.") from e

        n_feats, e_indices, e_feats = cls._extract_raw_data(graph, coarsened=coarsened, bidirectional=bidirectional)

        nx_graph = nx.DiGraph()

        for i, feat in enumerate(n_feats):
            nx_graph.add_node(
                i, 
                x=feat[0], y=feat[1], t=feat[2], 
                src=feat[3], snk=feat[4], jnc=feat[5], trm=feat[6], a3=feat[7],
                deg1=feat[8], deg2=feat[9], deg3p=feat[10]
            )

        for i, (src, tgt) in enumerate(e_indices):
            nx_graph.add_edge(src, tgt, edge_attr=e_feats[i])

        return nx_graph

    # -----------------------------------------------------------------------
    # Disk I/O Wrapper Methods
    # -----------------------------------------------------------------------

    @classmethod
    def save_pytorch_geometric(cls, graph: ShockGraph, filepath: str, coarsened: bool = False, bidirectional: bool = False) -> None:
        try:
            import torch
        except ImportError as e:
            raise ImportError("PyTorch is not installed.") from e

        pyg_data = cls.to_pytorch_geometric(graph, coarsened=coarsened, bidirectional=bidirectional)
        torch.save(pyg_data, filepath)
        print(f"Successfully saved PyG data to {filepath}")

    @classmethod
    def save_networkx(cls, graph: ShockGraph, filepath: str, coarsened: bool = False, bidirectional: bool = False) -> None:
        try:
            import networkx as nx
        except ImportError as e:
            raise ImportError("NetworkX is not installed.") from e

        nx_data = cls.to_networkx(graph, coarsened=coarsened, bidirectional=bidirectional)
        nx.write_graphml(nx_data, filepath)
        print(f"Successfully saved NetworkX data to {filepath}")
