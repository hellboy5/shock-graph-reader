import argparse
import sys
import os
from collections import Counter
from src.shock_graph.parser import ShockParser

def print_detailed_summary(graph):
    """Prints a comprehensive breakdown of the Shock Graph structure."""
    print("\n" + "="*50)
    print("           SHOCK GRAPH INTEGRITY REPORT")
    print("="*50)
    
    # 1. General Counts
    print(f"{'Total Nodes:':<25} {len(graph.nodes)}")
    print(f"{'Total Edges:':<25} {len(graph.edges)}")
    print("-" * 50)

    # 2. Node Type Distribution
    # This specifically addresses your request to see the count of each type
    type_counts = Counter(node.type for node in graph.nodes.values())
    print("Node Type Distribution:")
    for node_type, count in sorted(type_counts.items()):
        print(f"  - {node_type:<15}: {count}")
    print("-" * 50)

    # 3. Connectivity Statistics
    if graph.nodes:
        max_deg = max(node.degree for node in graph.nodes.values())
        avg_deg = sum(node.degree for node in graph.nodes.values()) / len(graph.nodes)
        print("Connectivity Metrics:")
        print(f"  - Max Degree: {max_deg}")
        print(f"  - Avg Degree: {avg_deg:.2f}")
    
    print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Shock Graph GNN Preprocessor")
    parser.add_argument("file", help="Path to .esf file")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found.")
        sys.exit(1)

    # Parse using the C++ identical logic
    parser_obj = ShockParser(args.file)
    graph = parser_obj.parse()

    # Print the expanded info
    print_detailed_summary(graph)

    # Optional: Print a few specific node details for verification
    print("Previewing first 3 nodes:")
    for nid in list(graph.nodes.keys())[:3]:
        node = graph.nodes[nid]
        print(f"  [Node {nid}] Type: {node.type:<10} | In: {node.in_degree} | Out: {node.out_degree}")

if __name__ == "__main__":
    main()
