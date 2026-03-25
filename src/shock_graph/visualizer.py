"""Visualizes shock graphs using Matplotlib."""

import os
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from .structures import ShockGraph


class ShockVisualizer:
    """Handles the rendering of ShockGraph objects."""

    @staticmethod
    def draw(
        graph: ShockGraph,
        figsize: tuple = (10, 10),
        save_path: Optional[str] = None,
        image_path: Optional[str] = None,
    ) -> None:
        """Plots the shock graph with nodes, edges, and directional arrows.

        Args:
            graph: The ShockGraph instance to visualize.
            figsize: The dimensions of the matplotlib figure.
            save_path: Optional file path to save the image instead of showing.
            image_path: Optional file path to an underlying image to overlay the graph onto.
        """
        fig, ax = plt.subplots(figsize=figsize)

        # 0. Optionally render the background image
        if image_path and os.path.exists(image_path):
            img = mpimg.imread(image_path)
            # Display image (Note: Matplotlib imshow defaults to y=0 at the top, 
            # which typically matches computer vision image coordinates)
            ax.imshow(img)

        # 1. Plot edges and directional arrows
        for edge in graph.edges:
            if not edge.samples:
                continue

            x_vals = [s.x for s in edge.samples]
            y_vals = [s.y for s in edge.samples]

            # Plot the continuous curve in green
            ax.plot(x_vals, y_vals, color='green', alpha=0.6, linewidth=2)

            # Add directional arrow near the midpoint of the curve
            if len(edge.samples) >= 2:
                mid_idx = len(edge.samples) // 2
                mid_x, mid_y = x_vals[mid_idx], y_vals[mid_idx]
                
                # Search backwards to find a point physically distant enough to draw a vector.
                # This explicitly ignores clumps of identical "degenerate" points.
                tail_idx = mid_idx - 1
                while tail_idx >= 0:
                    # Calculate Euclidean distance
                    dist = ((mid_x - x_vals[tail_idx])**2 + (mid_y - y_vals[tail_idx])**2) ** 0.5
                    if dist > 2.0:  # Minimum physical distance needed for a clean arrow
                        break
                    tail_idx -= 1
                
                # If we couldn't find a valid tail looking backwards, try looking forwards
                if tail_idx < 0:
                    head_idx = mid_idx + 1
                    while head_idx < len(edge.samples):
                        dist = ((x_vals[head_idx] - mid_x)**2 + (y_vals[head_idx] - mid_y)**2) ** 0.5
                        if dist > 2.0:
                            break
                        head_idx += 1
                    
                    if head_idx < len(edge.samples):
                        tail_x, tail_y = mid_x, mid_y
                        head_x, head_y = x_vals[head_idx], y_vals[head_idx]
                    else:
                        continue  # Edge is far too small to draw an arrow
                else:
                    tail_x, tail_y = x_vals[tail_idx], y_vals[tail_idx]
                    head_x, head_y = mid_x, mid_y

                ax.annotate(
                    '',
                    xy=(head_x, head_y),
                    xytext=(tail_x, tail_y),
                    arrowprops=dict(
                        arrowstyle="-|>",       
                        color='darkgreen',      
                        lw=2.0,
                        mutation_scale=25,      
                    ),
                    zorder=5,                   
                )

        # 2. Plot nodes and labels
        for node in graph.nodes.values():
            if not node.sample:
                continue

            x, y = node.sample.x, node.sample.y

            # Draw the node point
            ax.plot(x, y, marker='o', color='black', markersize=6, zorder=6)

            # Label the node with its Type and ID
            ax.text(
                x,
                y,
                f' {node.type}:{node.id}',
                fontsize=10,
                color='black',
                weight='bold',
                va='bottom',
                ha='left',
                zorder=7
            )

        # 3. Format the plot
        ax.set_aspect('equal', adjustable='box')
        ax.set_title("Shock Graph Visualization", fontsize=14, weight='bold')
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        ax.grid(True, linestyle='--', alpha=0.5)

        # 4. Display or Save
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Graph successfully saved to {save_path}")
            # Only close the figure if we are saving headless!
            plt.close(fig)
        else:
            # Leave the figure open for the interactive widget
            plt.show()
