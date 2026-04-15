"""Main entry point for processing, visualizing, and exporting shock graphs."""

import argparse
import os
import sys

from src.shock_graph.converter import GraphConverter
from src.shock_graph.parser import ShockParser
from src.shock_graph.structures import ShockGraph
from src.shock_graph.visualizer import ShockVisualizer
from src.shock_graph.coarsener import GraphCoarsener
from src.shock_graph.feature_extractor import ShockFeatureExtractor


def print_graph_report(graph: ShockGraph, filename: str, stage: str = "FINAL") -> None:
    """Prints a detailed topological and geometric summary of the graph."""
    print("\n" + "=" * 50)
    print(f" SHOCK GRAPH REPORT: {os.path.basename(filename)} ({stage})")
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
    if graph.edges and graph.edges[0].features:
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
        choices=["pyg", "nx"], 
        default="pyg",
        help="Target export format: 'pyg' or 'nx'. Defaults to 'pyg'."
    )
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        default=None,
        help="Explicit path to save the output file. If omitted, saves next to input."
    )
    parser.add_argument(
        "-v", "--visualize", 
        action="store_true", 
        help="Render and display the shock graph using Matplotlib."
    )
    parser.add_argument(
        "--overlay", 
        action="store_true", 
        help="Visualize the shock graph overlaid on its corresponding underlying image."
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug visualization (includes IDs, boundaries, and outlines)."
    )
    parser.add_argument(
        "-c", "--coarse", 
        action="store_true", 
        help="Coarsen the graph by merging degree-2 nodes."
    )
    parser.add_argument(
        "-b", "--bidirectional", 
        action="store_true", 
        help="Force manual bidirectional edge generation (GNN message passing). Default is False."
    )

    args = parser.parse_args()

    # 1. Validate Input
    if not os.path.exists(args.input_file):
        print(f"Error: Could not find input file '{args.input_file}'")
        sys.exit(1)

    # 2. Parse the Graph
    print(f"Parsing '{args.input_file}'...")
    graph = ShockParser(args.input_file).parse()
    
    # 3. Optional Coarsening
    if args.coarse:
        original_node_count = len(graph.nodes)
        print("\nCoarsening graph topology...")
        graph = GraphCoarsener.coarsen(graph)
        print(f"Graph reduced from {original_node_count} to {len(graph.nodes)} nodes.")
        
        print("Re-computing geometric features for merged edges...")
        ShockFeatureExtractor.process_graph(graph) 

    # 4. Print Detailed Report
    print_graph_report(graph, args.input_file, stage="COARSENED" if args.coarse else "ORIGINAL")

    # 5. Optional Visualization
    if args.visualize or args.overlay:
        print("Opening visualizer...")
        image_path = None

        
        # If overlay is requested, try to find a matching image file
        if args.overlay:
            target_dir = os.path.dirname(args.input_file) or '.'
            target_base = os.path.basename(os.path.splitext(args.input_file)[0])
            valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
            
            try:
                for filename in os.listdir(target_dir):
                    name, ext = os.path.splitext(filename)
                    # Match exact base name, but allow case-insensitive extensions (e.g., .PNG, .Jpg)
                    if name == target_base and ext.lower() in valid_extensions:
                        image_path = os.path.join(target_dir, filename)
                        print(f"Found corresponding image: {image_path}")
                        break
            except OSError as e:
                print(f"Error reading directory for overlay images: {e}")
            
            # Print a graceful warning if missing, but continue rendering
            if not image_path:
                print(f"\nWarning: '--overlay' was requested, but no image matching '{target_base}.[png|jpg]' was found.")
                print("Rendering the shock graph on a blank canvas instead.\n")

        # Determine mode: default is 'minimal', triggered to 'debug' by flag
        viz_mode = 'debug' if args.debug else 'minimal'
        ShockVisualizer.draw(graph, coarsened=args.coarse, mode=viz_mode, image_path=image_path)


    # 6. Determine Output Path
    out_path = args.output
    if not out_path:
        base_name, _ = os.path.splitext(args.input_file)
        extension_map = {"pyg": ".pt", "nx": ".graphml"}
        suffix = "_coarse" if args.coarse else ""
        out_path = f"{base_name}{suffix}{extension_map[args.format]}"

    # 7. Export
    print(f"Exporting to {args.format.upper()} format at {out_path}...")
    print(f"  - Bidirectional: {args.bidirectional}")
    
    try:
        if args.format == "pyg":
            GraphConverter.save_pytorch_geometric(graph, out_path, coarsened=args.coarse, bidirectional=args.bidirectional)
        elif args.format == "nx":
            GraphConverter.save_networkx(graph, out_path, coarsened=args.coarse, bidirectional=args.bidirectional)
    except ImportError as e:
        print(f"\nExport Failed: {e}")
        print("Please ensure the requested ML framework is installed in your environment.")
        sys.exit(1)


if __name__ == "__main__":
    main()
