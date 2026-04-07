"""Graph coarsening utilities to compress degree-2 chains."""

from typing import Dict, List, Set, Tuple

from .structures import Edge, Node, ShockGraph


class GraphCoarsener:
    """Compresses redundant degree-2 nodes while preserving topology and geometry."""

    @staticmethod
    def coarsen(graph: ShockGraph) -> ShockGraph:
        """Merges degree-2 nodes into macroscopic edges, preserving simple graph properties."""
        kept_nodes_ids: Set[int] = set()

        # 1. Identify Base Anchors (Strictly topological: degree != 2)
        for node_id, node in graph.nodes.items():
            if node.degree != 2:
                kept_nodes_ids.add(node_id)

        # 2. Extract all macroscopic paths without mutating the graph
        visited_original_edges: Set[Edge] = set()
        # Stores (start_id, end_id, list_of_edges)
        macro_paths: List[Tuple[int, int, List[Edge]]] = []
        
        # DETERMINISM FIX 1: Sort the anchors to guarantee stable iteration order
        for start_id in sorted(kept_nodes_ids):
            start_node = graph.nodes[start_id]
            
            # DETERMINISM FIX 2: Sort the incident edges by ID to lock the walker's pathing
            all_incident = sorted(
                start_node.incoming_edges + start_node.outgoing_edges, 
                key=lambda e: e.id
            )
            
            for start_edge in all_incident:
                if start_edge in visited_original_edges:
                    continue
                
                path_edges = [start_edge]
                visited_original_edges.add(start_edge)
                
                current_node = start_node
                if current_node == start_edge.source:
                    current_node = start_edge.target
                else:
                    current_node = start_edge.source
                
                # Walk the chain until we hit ANY anchor node
                while current_node.id not in kept_nodes_ids:
                    incident = current_node.incoming_edges + current_node.outgoing_edges
                    if len(incident) != 2:
                        break  # Failsafe for corrupted topology
                    
                    next_edge = incident[0] if incident[0] != path_edges[-1] else incident[1]
                    path_edges.append(next_edge)
                    visited_original_edges.add(next_edge)
                    
                    if current_node == next_edge.source:
                        current_node = next_edge.target
                    else:
                        current_node = next_edge.source
                        
                end_id = current_node.id
                macro_paths.append((start_id, end_id, path_edges))

        # CRITICAL EDGE CASE: Isolated Degree-2 Circles (Donuts)
        all_edges = set(graph.edges)
        unvisited = all_edges - visited_original_edges
        
        # DETERMINISM FIX 3: Sort unvisited donut edges by ID to avoid set-pop randomness
        unvisited_sorted = sorted(list(unvisited), key=lambda e: e.id)
        
        for orphan in unvisited_sorted:
            if orphan in visited_original_edges:
                continue
                
            # Forcibly promote to break the infinite circle
            start_id = orphan.source.id
            kept_nodes_ids.add(start_id)
            
            path_edges = [orphan]
            visited_original_edges.add(orphan)
            
            current_node = orphan.target
            while current_node.id != start_id:
                incident = current_node.incoming_edges + current_node.outgoing_edges
                if len(incident) != 2: 
                    break
                next_edge = incident[0] if incident[0] != path_edges[-1] else incident[1]
                path_edges.append(next_edge)
                visited_original_edges.add(next_edge)
                
                if current_node == next_edge.source:
                    current_node = next_edge.target
                else:
                    current_node = next_edge.source
                    
            macro_paths.append((start_id, start_id, path_edges))

        # DETERMINISM FIX 4: Lock the anomaly resolution order
        # Sort by length (shortest paths first), then by start_id and end_id as tie-breakers.
        macro_paths.sort(key=lambda p: (len(p[2]), p[0], p[1]))

        # 3. Resolve Complex Graph Anomalies (Midpoint Protocol)
        established_pairs = set()
        final_macro_paths = []
        
        for start_id, end_id, path_edges in macro_paths:
            pair = tuple(sorted([start_id, end_id]))
            
            is_self_loop = (start_id == end_id)
            is_parallel = (pair in established_pairs)
            
            # If an anomaly is detected AND there is at least one degree-2 node available
            if (is_self_loop or is_parallel) and len(path_edges) > 1:
                
                # Use the geometric midpoint to ensure stable feature extraction later
                mid_idx = len(path_edges) // 2
                e1 = path_edges[mid_idx - 1]
                e2 = path_edges[mid_idx]
                
                # Find the node that connects the two middle edges
                shared_nodes = set([e1.source.id, e1.target.id]).intersection(
                    set([e2.source.id, e2.target.id])
                )
                
                if shared_nodes:
                    # DETERMINISM FIX 5: Sort the shared nodes before grabbing the first one
                    split_node_id = sorted(list(shared_nodes))[0]
                    
                    # Promote the midpoint to an Anchor
                    kept_nodes_ids.add(split_node_id)
                    
                    # Slice the path symmetrically
                    path1 = path_edges[:mid_idx]
                    path2 = path_edges[mid_idx:]
                    
                    final_macro_paths.append((start_id, split_node_id, path1))
                    final_macro_paths.append((split_node_id, end_id, path2))
                    
                    established_pairs.add(tuple(sorted([start_id, split_node_id])))
                    established_pairs.add(tuple(sorted([split_node_id, end_id])))
                else:
                    # Failsafe in case of severe corruption
                    final_macro_paths.append((start_id, end_id, path_edges))
            else:
                final_macro_paths.append((start_id, end_id, path_edges))
                established_pairs.add(pair)

        # 4. Build the new nodes
        new_nodes: Dict[int, Node] = {}
        # DETERMINISM FIX 6: Sort kept_nodes_ids for deterministic node creation
        for nid in sorted(kept_nodes_ids):
            old_node = graph.nodes[nid]
            new_node = Node(node_id=old_node.id, node_type=old_node.type)
            new_node.sample = old_node.sample
            new_node._cw_neighbors = [] 
            new_nodes[nid] = new_node

        # 5. Build the new edges
        new_edges: List[Edge] = []
        edge_id_counter = 0
        
        for start_id, end_id, path_edges in final_macro_paths:
            merged_samples = []
            current_node_id = start_id
            
            for edge in path_edges:
                if current_node_id == edge.source.id:
                    if not merged_samples:
                        merged_samples.extend(edge.samples)
                    else:
                        merged_samples.extend(edge.samples[1:])
                    current_node_id = edge.target.id
                else:
                    if not merged_samples:
                        merged_samples.extend(list(reversed(edge.samples)))
                    else:
                        merged_samples.extend(list(reversed(edge.samples))[1:])
                    current_node_id = edge.source.id
                    
            merged_edge = Edge(
                edge_id=edge_id_counter,
                source=new_nodes[start_id],
                target=new_nodes[end_id],
                samples=merged_samples
            )
            new_edges.append(merged_edge)
            edge_id_counter += 1

        # 6. Remap Clockwise Neighbors (Preserving Planar Ordering)
        # DETERMINISM FIX 7: Sort new_nodes.items() for deterministic edge assignment
        for nid, new_node in sorted(new_nodes.items()):
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
