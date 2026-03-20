"""Graph coarsening utilities to compress degree-2 chains."""

from typing import Dict, List, Set

from .structures import Edge, Node, ShockGraph


class GraphCoarsener:
    """Compresses redundant degree-2 nodes while preserving topology and geometry."""

    @staticmethod
    def coarsen(graph: ShockGraph) -> ShockGraph:
        """Merges pure pass-through degree-2 JUNCT nodes into single edges."""
        kept_nodes_ids: Set[int] = set()

        # 1. Identify Kept Nodes using native Node properties
        for node_id, node in graph.nodes.items():
            # Rule 1: Keep Degree 1 and Degree 3+
            if node.degree != 2:
                kept_nodes_ids.add(node_id)
            # Rule 2: Keep SOURCE, SINK, TERMINAL, A3 (Only merge pure JUNCT)
            elif node.type in ['SOURCE', 'SINK', 'TERMINAL', 'A3']:
                kept_nodes_ids.add(node_id)
            # Rule 3: Must be a perfect 1-in, 1-out flow to be merged
            elif node.in_degree != 1 or node.out_degree != 1:
                kept_nodes_ids.add(node_id)

        # 2. Create FRESH Node instances to reset their connectivity lists
        new_nodes: Dict[int, Node] = {}
        for nid in kept_nodes_ids:
            old_node = graph.nodes[nid]
            new_node = Node(node_id=old_node.id, node_type=old_node.type)
            new_node.sample = old_node.sample
            new_node._cw_neighbors = list(old_node._cw_neighbors)
            new_nodes[nid] = new_node

        new_edges: List[Edge] = []
        visited_edges: Set[Edge] = set()
        edge_id_counter = 0

        # 3. Trace paths from all kept nodes
        for start_id in kept_nodes_ids:
            start_node = graph.nodes[start_id]
            
            for start_edge in start_node.outgoing_edges:
                if start_edge in visited_edges:
                    continue

                current_edge = start_edge
                # Start the merged sequence with the first edge's sample points
                merged_samples = list(current_edge.samples)
                
                while True:
                    visited_edges.add(current_edge)
                    next_node = current_edge.target

                    # If we hit a kept node, the path is complete
                    if next_node.id in kept_nodes_ids:
                        merged_edge = Edge(
                            edge_id=edge_id_counter,
                            source=new_nodes[start_id],  # Pointing to the NEW node
                            target=new_nodes[next_node.id], # Pointing to the NEW node
                            samples=merged_samples
                        )
                        new_edges.append(merged_edge)
                        edge_id_counter += 1
                        break

                    # Otherwise, swallow the degree-2 node's sample point
                    if next_node.sample:
                        merged_samples.append(next_node.sample)
                    
                    # Move to the next edge in the pass-through chain
                    # (Guaranteed exactly 1 outgoing edge by Rule 3)
                    current_edge = next_node.outgoing_edges[0]
                    merged_samples.extend(current_edge.samples)

                    # Cycle protection (prevents infinite loops on closed circles)
                    if current_edge in visited_edges:
                        break

        # 4. Return the newly compressed graph
        coarsened_graph = ShockGraph()
        coarsened_graph.nodes = new_nodes
        coarsened_graph.edges = new_edges
        coarsened_graph.metadata = dict(graph.metadata)
        
        return coarsened_graph
