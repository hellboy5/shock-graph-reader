"""Main entry point for processing, visualizing, and exporting shock graphs."""

import argparse
import os
import sys

from src.shock_graph.converter import GraphConverter
from src.shock_graph.parser import ShockParser
from src.shock_graph.structures import ShockGraph
from src.shock_graph.visualizer import ShockVisualizer


def print_graph_report(graph: ShockGraph, filename: str) -> None:
    """Prints a detailed topological and geometric summary of the parsed graph."""
    print("\n" + "=" * 50)
    print(f" SHOCK GRAPH REPORT: {os.path.basename(filename)}")
    print("=" * 50)
    print(f"Total Nodes: {len(graph.nodes)}")
    print(f"Total Edges: {len(graph.edges)}")

    # Breakdown of node types
    type_counts = {}
    for node in graph.nodes.values():
        type_counts[node.type] = type_counts.get(node.type, 0) + 1

    print("\nNode Type Breakdown:")
    for ntype, count in sorted(type_counts.items()):
        print(f"  - {ntype.ljust(10)}: {count}")

    # Edge geometric statistics
    if graph.edges:
        avg_s_len = sum(e.s_length for e in graph.edges) / len(graph.edges)
        avg_poly_area = sum(e.poly_area for e in graph.edges) / len(graph.edges)
        print("\nEdge Geometry Averages:")
        print(f"  - Shock Length : {avg_s_len:.4f}")
        print(f"  - Polygon Area : {avg_poly_area:.4f}")
    
    print("=" * 50 + "\n")


def main() -> None:
    """Parses command line arguments and drives the shock graph pipeline."""
    parser = argparse.ArgumentParser(
        description="Extract features from ESF shock graphs and export to ML formats."
    )
    
    # Positional required argument
    parser.add_argument(
        "input_file", 
        type=str, 
        help="Path to the input .esf file."
    )
    
    # Optional arguments
    parser.add_argument(
        "-f", "--format", 
        type=str, 
        choices=["pyg", "dgl", "nx"], 
        default="pyg",
        help="Target export format: 'pyg' (PyTorch Geometric), 'dgl' (Deep Graph Library), or 'nx' (NetworkX). Defaults to 'pyg'."
    )
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        default=None,
        help="Explicit path to save the output file. If omitted, saves next to the input file."
    )
    parser.add_argument(
        "-v", "--visualize", 
        action="store_true", 
        help="Render and display the shock graph using Matplotlib."
    )

    args = parser.parse_args()

    # 1. Validate Input
    if not os.path.exists(args.input_file):
        print(f"Error: Could not find input file '{args.input_file}'")
        sys.exit(1)

    # 2. Parse and Compute Features
    print(f"Parsing '{args.input_file}'...")
    graph = ShockParser.parse(args.input_file)
    
    # 3. Print Detailed Report (Always)
    print_graph_report(graph, args.input_file)

    # 4. Optional Visualization
    if args.visualize:
        print("Opening visualizer...")
        ShockVisualizer.draw(graph)

    # 5. Determine Output Path
    out_path = args.output
    if not out_path:
        base_name, _ = os.path.splitext(args.input_file)
        extension_map = {"pyg": ".pt", "dgl": ".dgl", "nx": ".graphml"}
        out_path = f"{base_name}{extension_map[args.format]}"

    # 6. Export
    print(f"Exporting to {args.format.upper()} format...")
    try:
        if args.format == "pyg":
            GraphConverter.save_pytorch_geometric(graph, out_path)
        elif args.format == "dgl":
            GraphConverter.save_dgl(graph, out_path)
        elif args.format == "nx":
            GraphConverter.save_networkx(graph, out_path)
    except ImportError as e:
        print(f"\nExport Failed: {e}")
        print("Please ensure the requested ML framework is installed in your environment.")
        sys.exit(1)


if __name__ == "__main__":
    main()
