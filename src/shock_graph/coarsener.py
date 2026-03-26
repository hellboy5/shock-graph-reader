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
            # Start with a clean slate to prevent dangling pointers
            new_node._cw_neighbors = [] 
            new_nodes[nid] = new_node

        new_edges: List[Edge] = []
        visited_edges: Set[Edge] = set()
        edge_id_counter = 0

        # 3. Trace paths from all kept nodes to build new edges
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

                    # Move to the next edge in the pass-through chain
                    # (Guaranteed exactly 1 outgoing edge by Rule 3)
                    current_edge = next_node.outgoing_edges[0]
                    
                    # Extend the curve, but skip the first sample of the new edge 
                    # because it is identical to the last sample of the previous edge!
                    merged_samples.extend(current_edge.samples[1:])

                    # Cycle protection (prevents infinite loops on closed circles)
                    if current_edge in visited_edges:
                        break

        # 4. Remap Clockwise Neighbors safely (Preserves exact planar ordering)
        for nid, new_node in new_nodes.items():
            old_node = graph.nodes[nid]
            for old_neighbor_id in old_node.get_cw_neighbors():
                current_id = old_neighbor_id
                prev_id = nid
                
                visited_for_cycle = set()
                
                # Fast-forward through degree-2 nodes until we hit a kept node
                while current_id not in kept_nodes_ids:
                    if current_id in visited_for_cycle:
                        break # Failsafe: We hit an infinite loop in corrupted data
                    visited_for_cycle.add(current_id)
                    
                    deg2_node = graph.nodes[current_id]
                    neighbors = deg2_node.get_cw_neighbors()
                    
                    # A pure degree-2 node should have exactly 2 neighbors.
                    if len(neighbors) == 2:
                        next_id = neighbors[0] if neighbors[0] != prev_id else neighbors[1]
                    else:
                        break # Fallback for corrupted planar data
                        
                    prev_id = current_id
                    current_id = next_id
                    
                # We reached the end of the chain; register the valid kept node
                if current_id in kept_nodes_ids:
                    new_node.add_neighbor(current_id)

        # 5. Return the newly compressed graph
        coarsened_graph = ShockGraph()
        coarsened_graph.nodes = new_nodes
        coarsened_graph.edges = new_edges
        coarsened_graph.metadata = dict(graph.metadata)
        
        return coarsened_graph
