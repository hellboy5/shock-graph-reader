"""Converts ShockGraph objects into various machine learning graph formats."""

from typing import Any, Dict, List, Tuple

from .structures import Edge, Node, ShockGraph


class GraphConverter:
    """Provides utilities to convert ShockGraphs into external formats."""

    # ML frameworks require numerical tensors, so we encode the string types.
    # These will remain as floats here and can be one-hot encoded later
    # in your GNN training loop.
    NODE_TYPE_ENCODING = {
        'SOURCE': 0.0,
        'SINK': 1.0,
        'JUNCT': 2.0,
        'TERMINAL': 3.0,
        'A3': 4.0,
        'UNKNOWN': 5.0,
    }

    @classmethod
    def _extract_raw_data(
        cls,
        graph: ShockGraph,
    ) -> Tuple[List[List[float]], List[Tuple[int, int]], List[List[float]]]:
        """Extracts standardized lists of nodes, edges, and features.

        Re-indexes arbitrary node IDs to a continuous [0, N-1] range required
        by most ML frameworks.

        Args:
            graph: The parsed and hydrated ShockGraph.

        Returns:
            A tuple containing:
                - node_features: List of feature arrays per node [x, y, t, type].
                - edge_indices: List of (source_idx, target_idx) tuples.
                - edge_features: List of feature arrays per edge.
        """
        # Maps original node ID to a contiguous 0-indexed integer for PyTorch
        node_to_idx: Dict[int, int] = {}
        # We explicitly sort the keys to guarantee reproducible tensor layouts
        # across different environments and executions.
        for idx, original_id in enumerate(sorted(graph.nodes.keys())):
            node_to_idx[original_id] = idx

        # [FIXED]: Iterate through the nodes in the exact same sorted order
        # so the feature rows match the node_to_idx mapping.
        for original_id in sorted(graph.nodes.keys()):
            node = graph.nodes[original_id]
            x = node.sample.x if node.sample else 0.0
            y = node.sample.y if node.sample else 0.0
            t = node.sample.t if node.sample else 0.0
            node_type = cls.NODE_TYPE_ENCODING.get(node.type, 5.0)

            node_features.append([x, y, t, node_type])
            
        # 2. Extract Edge Indices and Edge Features
        edge_indices: List[Tuple[int, int]] = []
        edge_features: List[List[float]] = []

        for edge in graph.edges:
            src_idx = node_to_idx[edge.source.id]
            tgt_idx = node_to_idx[edge.target.id]
            edge_indices.append((src_idx, tgt_idx))

            # Flatten the EdgeShapeFeatures dataclass into a raw float list
            if edge.features:
                feats = [
                    edge.features.s_length,
                    edge.features.s_curve,
                    edge.features.s_angle,
                    edge.features.p_length,
                    edge.features.p_curve,
                    edge.features.p_angle,
                    edge.features.m_length,
                    edge.features.m_curve,
                    edge.features.m_angle,
                    edge.features.poly_area,
                ]
            else:
                # Fallback if feature extraction was skipped
                feats = [0.0] * 10

            edge_features.append(feats)

        return node_features, edge_indices, edge_features

    # -----------------------------------------------------------------------
    # In-Memory Conversion Methods
    # -----------------------------------------------------------------------

    @classmethod
    def to_pytorch_geometric(cls, graph: ShockGraph) -> Any:
        """Converts the ShockGraph into a PyTorch Geometric Data object.

        Args:
            graph: The ShockGraph instance.

        Returns:
            A torch_geometric.data.Data object.

        Raises:
            ImportError: If torch or torch_geometric is not installed.
        """
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError as e:
            raise ImportError("PyTorch Geometric is not installed.") from e

        n_feats, e_indices, e_feats = cls._extract_raw_data(graph)

        # PyG expects edge_index to be shape [2, num_edges]
        x = torch.tensor(n_feats, dtype=torch.float)
        edge_index = torch.tensor(e_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(e_feats, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    @classmethod
    def to_dgl(cls, graph: ShockGraph) -> Any:
        """Converts the ShockGraph into a Deep Graph Library (DGL) object.

        Args:
            graph: The ShockGraph instance.

        Returns:
            A dgl.DGLGraph object.

        Raises:
            ImportError: If torch or dgl is not installed.
        """
        try:
            import dgl
            import torch
        except ImportError as e:
            raise ImportError("DGL or PyTorch is not installed.") from e

        n_feats, e_indices, e_feats = cls._extract_raw_data(graph)

        src_nodes = [idx[0] for idx in e_indices]
        tgt_nodes = [idx[1] for idx in e_indices]

        dgl_graph = dgl.graph((src_nodes, tgt_nodes), num_nodes=len(n_feats))
        dgl_graph.ndata['x'] = torch.tensor(n_feats, dtype=torch.float)
        dgl_graph.edata['edge_attr'] = torch.tensor(e_feats, dtype=torch.float)

        return dgl_graph

    @classmethod
    def to_networkx(cls, graph: ShockGraph) -> Any:
        """Converts the ShockGraph into a NetworkX DiGraph.

        Args:
            graph: The ShockGraph instance.

        Returns:
            A networkx.DiGraph object.

        Raises:
            ImportError: If networkx is not installed.
        """
        try:
            import networkx as nx
        except ImportError as e:
            raise ImportError("NetworkX is not installed.") from e

        n_feats, e_indices, e_feats = cls._extract_raw_data(graph)

        nx_graph = nx.DiGraph()

        for i, feat in enumerate(n_feats):
            nx_graph.add_node(i, x=feat[0], y=feat[1], t=feat[2], type=feat[3])

        for i, (src, tgt) in enumerate(e_indices):
            nx_graph.add_edge(src, tgt, edge_attr=e_feats[i])

        return nx_graph

    # -----------------------------------------------------------------------
    # Disk I/O Wrapper Methods
    # -----------------------------------------------------------------------

    @classmethod
    def save_pytorch_geometric(cls, graph: ShockGraph, filepath: str) -> None:
        """Converts and saves the graph as a PyTorch Geometric .pt file.

        Args:
            graph: The ShockGraph instance to convert and save.
            filepath: The destination file path (e.g., 'data/graph.pt').

        Raises:
            ImportError: If PyTorch is not installed.
        """
        try:
            import torch
        except ImportError as e:
            raise ImportError("PyTorch is not installed.") from e

        pyg_data = cls.to_pytorch_geometric(graph)
        torch.save(pyg_data, filepath)
        print(f"Successfully saved PyG data to {filepath}")

    @classmethod
    def save_dgl(cls, graph: ShockGraph, filepath: str) -> None:
        """Converts and saves the graph as a DGL .dgl file.

        Args:
            graph: The ShockGraph instance to convert and save.
            filepath: The destination file path (e.g., 'data/graph.dgl').

        Raises:
            ImportError: If DGL is not installed.
        """
        try:
            import dgl
        except ImportError as e:
            raise ImportError("DGL is not installed.") from e

        dgl_data = cls.to_dgl(graph)
        dgl.save_graphs(filepath, [dgl_data])
        print(f"Successfully saved DGL data to {filepath}")

    @classmethod
    def save_networkx(cls, graph: ShockGraph, filepath: str) -> None:
        """Converts and saves the graph as a NetworkX GraphML file.

        Args:
            graph: The ShockGraph instance to convert and save.
            filepath: The destination file path (e.g., 'data/graph.graphml').

        Raises:
            ImportError: If NetworkX is not installed.
        """
        try:
            import networkx as nx
        except ImportError as e:
            raise ImportError("NetworkX is not installed.") from e

        nx_data = cls.to_networkx(graph)
        nx.write_graphml(nx_data, filepath)
        print(f"Successfully saved NetworkX data to {filepath}")
