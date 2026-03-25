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

            # Add directional arrow near the midpoint of the curve in green
            if len(edge.samples) >= 2:
                mid_idx = len(edge.samples) // 2
                
                # We point the arrow from the sample just before the midpoint 
                # to the midpoint itself to indicate forward flow.
                ax.annotate(
                    '',
                    xy=(x_vals[mid_idx], y_vals[mid_idx]),
                    xytext=(x_vals[mid_idx - 1], y_vals[mid_idx - 1]),
                    arrowprops=dict(arrowstyle="->", color='green', lw=2.0),
                )

        # 2. Plot nodes and labels
        for node in graph.nodes.values():
            if not node.sample:
                continue

            x, y = node.sample.x, node.sample.y

            # Draw the node point
            ax.plot(x, y, marker='o', color='black', markersize=6)

            # Label the node with its ID
            ax.text(
                x,
                y,
                f' {node.id}',
                fontsize=10,
                color='black',
                weight='bold',
                va='bottom',
                ha='left',
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
