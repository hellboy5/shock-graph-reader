"""Graph coarsening utilities to compress degree-2 chains."""

from typing import Dict, List, Set

from .structures import Edge, Node, ShockGraph


class GraphCoarsener:
    """Compresses redundant degree-2 nodes while preserving topology and geometry."""

    @staticmethod
    def coarsen(graph: ShockGraph) -> ShockGraph:
        """Merges all degree-2 nodes regardless of flow direction or type.
        
        Because upstream C++ pruning removes pure 1-in/1-out flow junctions, 
        any remaining degree-2 nodes (e.g., 2-in SINKs or 2-out SOURCEs) are 
        flow-terminating but structurally pass-through. This performs a purely 
        topological extraction of the skeleton's branching structure.
        """
        kept_nodes_ids: Set[int] = set()

        # 1. Identify Anchor Nodes (Strictly topological: degree != 2)
        for node_id, node in graph.nodes.items():
            if node.degree != 2:
                kept_nodes_ids.add(node_id)

        # 2. Create FRESH Node instances
        new_nodes: Dict[int, Node] = {}
        for nid in kept_nodes_ids:
            old_node = graph.nodes[nid]
            # Keep original type, though its geometric meaning may now be ambiguous
            new_node = Node(node_id=old_node.id, node_type=old_node.type)
            new_node.sample = old_node.sample
            new_node._cw_neighbors = [] 
            new_nodes[nid] = new_node

        new_edges: List[Edge] = []
        visited_edges: Set[Edge] = set()
        edge_id_counter = 0

        # 3. Trace Undirected Paths from Anchor Nodes
        for start_id in kept_nodes_ids:
            start_node = graph.nodes[start_id]
            all_incident_edges = start_node.incoming_edges + start_node.outgoing_edges
            
            for start_edge in all_incident_edges:
                if start_edge in visited_edges:
                    continue

                active_edge = start_edge
                current_node = start_node
                visited_edges.add(active_edge)
                
                # Initialize the merged samples array and determine step direction
                if current_node == active_edge.source:
                    merged_samples = list(active_edge.samples)
                    current_node = active_edge.target
                else:
                    merged_samples = list(reversed(active_edge.samples))
                    current_node = active_edge.source
                
                # Walk the chain until we hit another anchor node
                while current_node.id not in kept_nodes_ids:
                    # Degree is guaranteed to be 2 here
                    incident = current_node.incoming_edges + current_node.outgoing_edges
                    next_edge = incident[0] if incident[0] != active_edge else incident[1]
                    
                    active_edge = next_edge
                    visited_edges.add(active_edge)
                    
                    # Extend samples, dropping index 0 to avoid duplicating the joint
                    if current_node == active_edge.source:
                        merged_samples.extend(active_edge.samples[1:])
                        current_node = active_edge.target
                    else:
                        merged_samples.extend(list(reversed(active_edge.samples))[1:])
                        current_node = active_edge.source

                # We hit an anchor node. Create the new arbitrary-direction edge
                merged_edge = Edge(
                    edge_id=edge_id_counter,
                    source=new_nodes[start_id],       # Start of our traversal
                    target=new_nodes[current_node.id], # End of our traversal
                    samples=merged_samples
                )
                new_edges.append(merged_edge)
                edge_id_counter += 1

        # 4. Remap Clockwise Neighbors safely
        # The CW list maps naturally through degree-2 nodes regardless of direction
        for nid, new_node in new_nodes.items():
            old_node = graph.nodes[nid]
            for old_neighbor_id in old_node.get_cw_neighbors():
                current_id = old_neighbor_id
                prev_id = nid
                
                visited_for_cycle = set()
                
                while current_id not in kept_nodes_ids:
                    if current_id in visited_for_cycle:
                        break 
                    visited_for_cycle.add(current_id)
                    
                    deg2_node = graph.nodes[current_id]
                    neighbors = deg2_node.get_cw_neighbors()
                    
                    if len(neighbors) == 2:
                        next_id = neighbors[0] if neighbors[0] != prev_id else neighbors[1]
                    else:
                        break 
                        
                    prev_id = current_id
                    current_id = next_id
                    
                if current_id in kept_nodes_ids:
                    new_node.add_neighbor(current_id)

        coarsened_graph = ShockGraph()
        coarsened_graph.nodes = new_nodes
        coarsened_graph.edges = new_edges
        coarsened_graph.metadata = dict(graph.metadata)
        
        return coarsened_graph
