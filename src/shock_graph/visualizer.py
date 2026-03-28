"""Visualizes shock graphs using Matplotlib with interactive widgets."""

import math
import os
import random
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
from matplotlib.widgets import CheckButtons
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

from .structures import ShockGraph


class ShockVisualizer:
    """Handles the rendering of ShockGraph objects."""

    TYPE_REV_MAP = {
        'A3': 'A',
        'SOURCE': 'S',
        'SINK': 'F',
        'JUNCT': 'J',
        'TERMINAL': 'T',
        'UNKNOWN': 'U'
    }

    @staticmethod
    def draw(
        graph: ShockGraph,
        figsize: tuple = (12, 10),
        save_path: Optional[str] = None,
        image_path: Optional[str] = None,
    ) -> None:
        """Plots the shock graph with nodes, edges, boundaries, and interactive toggles."""
        fig, ax = plt.subplots(figsize=figsize)
        plt.subplots_adjust(left=0.25)

        if image_path and os.path.exists(image_path):
            img = mpimg.imread(image_path)
            ax.imshow(img)

        all_p_lines = []
        all_m_lines = []
        all_disk_collections = []

        # 1. Plot edges, directional arrows, boundaries, and shape reconstruction
        for edge in graph.edges:
            if not edge.samples:
                continue

            x_vals = [s.x for s in edge.samples]
            y_vals = [s.y for s in edge.samples]

            # --- SHOCK SPINE ---
            ax.plot(x_vals, y_vals, color='green', alpha=0.8, linewidth=2, zorder=4)

            # --- BOUNDARIES ---
            p_x, p_y = [], []
            m_x, m_y = [], []
            
            for s in edge.samples:
                p_angle = s.theta + s.phi
                p_x.append(s.x + s.t * math.cos(p_angle))
                p_y.append(s.y + s.t * math.sin(p_angle))
                
                m_angle = s.theta - s.phi
                m_x.append(s.x + s.t * math.cos(m_angle))
                m_y.append(s.y + s.t * math.sin(m_angle))

            # Matched alpha to 0.8 and linewidth to 2 to equal the shock spine
            p_line, = ax.plot(p_x, p_y, color='dodgerblue', alpha=0.8, linewidth=2, linestyle='--', zorder=3)
            m_line, = ax.plot(m_x, m_y, color='crimson', alpha=0.8, linewidth=2, linestyle='--', zorder=3)
            
            all_p_lines.append(p_line)
            all_m_lines.append(m_line)

            # --- MAXIMAL DISKS (Reconstruction) ---
            # Create a collection of circles for THIS specific edge
            disk_patches = [Circle((s.x, s.y), s.t) for s in edge.samples]
            
            # Generate a random RGB color
            edge_color = (random.random(), random.random(), random.random())
            
            disk_collection = PatchCollection(
                disk_patches, 
                facecolor=edge_color, 
                alpha=0.4, 
                edgecolor='none', 
                visible=False,
                zorder=2
            )
            ax.add_collection(disk_collection)
            all_disk_collections.append(disk_collection)

            # --- DIRECTIONAL ARROWS ---
            if len(edge.samples) >= 2:
                mid_idx = len(edge.samples) // 2
                mid_x, mid_y = x_vals[mid_idx], y_vals[mid_idx]
                
                tail_idx = mid_idx - 1
                while tail_idx >= 0:
                    dist = ((mid_x - x_vals[tail_idx])**2 + (mid_y - y_vals[tail_idx])**2) ** 0.5
                    if dist > 2.0:  
                        break
                    tail_idx -= 1
                
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
                        continue  
                else:
                    tail_x, tail_y = x_vals[tail_idx], y_vals[tail_idx]
                    head_x, head_y = mid_x, mid_y

                ax.annotate(
                    '',
                    xy=(head_x, head_y),
                    xytext=(tail_x, tail_y),
                    arrowprops=dict(arrowstyle="->", color='green', lw=2.0, mutation_scale=15),
                    zorder=5,                   
                )

        # 2. Plot nodes and labels
        for node in graph.nodes.values():
            if not node.sample:
                continue

            x, y = node.sample.x, node.sample.y
            ax.plot(x, y, marker='o', color='black', markersize=5, zorder=6)
            short_type = ShockVisualizer.TYPE_REV_MAP.get(node.type, '?')
            ax.text(x, y, f' {short_type}:{node.id}', fontsize=8, color='black', weight='bold', va='bottom', ha='left', zorder=7)

        # 3. Format the main plot
        ax.set_aspect('equal', adjustable='box')
        if not ax.yaxis_inverted():
            ax.invert_yaxis()
            
        ax.set_title("Shock Graph & Shape Reconstruction", fontsize=14, weight='bold')
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        ax.grid(True, linestyle='--', alpha=0.5)

        # Updated legend line widths to match
        legend_lines = [
            Line2D([0], [0], color='green', lw=2),
            Line2D([0], [0], color='dodgerblue', lw=2, linestyle='--'),
            Line2D([0], [0], color='crimson', lw=2, linestyle='--')
        ]
        ax.legend(legend_lines, ['Shock Spine', 'Plus Boundary (+)', 'Minus Boundary (-)'], loc='upper right')

        # ---------------------------------------------------------
        # 4. INTERACTIVE WIDGETS (CheckButtons)
        # ---------------------------------------------------------
        ax_toggle = plt.axes([0.02, 0.4, 0.20, 0.15])
        
        labels = ['Plus (+)', 'Minus (-)', 'Maximal Disks']
        visibility = [True, True, False]
        
        check = CheckButtons(ax_toggle, labels, visibility)

        def toggle_visibility(label):
            if label == 'Plus (+)':
                for line in all_p_lines:
                    line.set_visible(not line.get_visible())
            elif label == 'Minus (-)':
                for line in all_m_lines:
                    line.set_visible(not line.get_visible())
            elif label == 'Maximal Disks':
                # Toggle every individual edge's collection
                for collection in all_disk_collections:
                    collection.set_visible(not collection.get_visible())
            
            fig.canvas.draw_idle()

        check.on_clicked(toggle_visibility)
        fig.toggle_widget = check

        # 5. Display or Save
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Graph successfully saved to {save_path}")
            plt.close(fig)
        else:
            plt.show()
