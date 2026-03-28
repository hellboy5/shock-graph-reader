"""Visualizes shock graphs using Matplotlib with interactive widgets."""

import math
import os
import random
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
from matplotlib.widgets import CheckButtons

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
        """Plots the shock graph with unique colored boundary outlines per edge."""
        fig, ax = plt.subplots(figsize=figsize)
        plt.subplots_adjust(left=0.25)

        if image_path and os.path.exists(image_path):
            img = mpimg.imread(image_path)
            ax.imshow(img)

        all_p_lines = []
        all_m_lines = []
        all_outline_loops = []

        # 1. Plot edges, directional arrows, and boundary outlines
        for edge in graph.edges:
            if not edge.samples:
                continue

            x_vals = [s.x for s in edge.samples]
            y_vals = [s.y for s in edge.samples]

            # --- SHOCK SPINE (Green) ---
            ax.plot(x_vals, y_vals, color='green', alpha=0.8, linewidth=2, zorder=4)

            # --- CALCULATE BOUNDARY POINTS ---
            p_x, p_y = [], []
            m_x, m_y = [], []
            
            for s in edge.samples:
                p_angle = s.theta + s.phi
                p_x.append(s.x + s.t * math.cos(p_angle))
                p_y.append(s.y + s.t * math.sin(p_angle))
                
                m_angle = s.theta - s.phi
                m_x.append(s.x + s.t * math.cos(m_angle))
                m_y.append(s.y + s.t * math.sin(m_angle))

            # --- PLUS/MINUS INDIVIDUAL CURVES ---
            p_line, = ax.plot(p_x, p_y, color='dodgerblue', alpha=0.8, linewidth=2, zorder=3)
            m_line, = ax.plot(m_x, m_y, color='crimson', alpha=0.8, linewidth=2, zorder=3)
            
            all_p_lines.append(p_line)
            all_m_lines.append(m_line)

            # --- CLOSED OUTLINE LOOP (The Perimeter) ---
            # Generate a unique color for this specific edge's path
            path_color = (random.random(), random.random(), random.random())
            
            # Path: StartPoint -> PlusCurve -> EndPoint -> MinusCurve(reversed) -> StartPoint
            loop_x = [x_vals[0]] + p_x + [x_vals[-1]] + m_x[::-1] + [x_vals[0]]
            loop_y = [y_vals[0]] + p_y + [y_vals[-1]] + m_y[::-1] + [y_vals[0]]
            
            # Plot as a solid curve with its unique color
            outline, = ax.plot(
                loop_x, 
                loop_y, 
                color=path_color, 
                alpha=0.9, 
                linewidth=2, 
                visible=False, 
                zorder=2
            )
            all_outline_loops.append(outline)

            # --- DIRECTIONAL ARROW ---
            if len(edge.samples) >= 2:
                mid_idx = len(edge.samples) // 2
                ax.annotate(
                    '',
                    xy=(x_vals[mid_idx], y_vals[mid_idx]),
                    xytext=(x_vals[mid_idx-1], y_vals[mid_idx-1]),
                    arrowprops=dict(arrowstyle="->", color='green', lw=2.0, mutation_scale=15),
                    zorder=5,                   
                )

        # 2. Plot nodes
        for node in graph.nodes.values():
            if node.sample:
                ax.plot(node.sample.x, node.sample.y, marker='o', color='black', markersize=5, zorder=6)

        # 3. Format the plot
        ax.set_aspect('equal', adjustable='box')
        if not ax.yaxis_inverted():
            ax.invert_yaxis()
            
        ax.set_title("Shock Graph: Per-Edge Boundary Outlines", fontsize=14, weight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)

        # 4. Interactive Toggles
        ax_toggle = plt.axes([0.02, 0.4, 0.20, 0.15])
        labels = ['Plus (+)', 'Minus (-)', 'Outlines']
        visibility = [True, True, False]
        check = CheckButtons(ax_toggle, labels, visibility)

        def toggle_visibility(label):
            if label == 'Plus (+)':
                for line in all_p_lines: line.set_visible(not line.get_visible())
            elif label == 'Minus (-)':
                for line in all_m_lines: line.set_visible(not line.get_visible())
            elif label == 'Outlines':
                for line in all_outline_loops: line.set_visible(not line.get_visible())
            fig.canvas.draw_idle()

        check.on_clicked(toggle_visibility)
        fig.toggle_widget = check

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
