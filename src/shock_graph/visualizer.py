"""Visualizes shock graphs with robust arrows, labels, and colored outlines."""

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

    # Map full type names back to single-character keys for cleaner labels
    TYPE_REV_MAP = {
        'A3': ('A', 'A3 Point'),
        'SOURCE': ('S', 'Source (Start)'),
        'SINK': ('F', 'Sink (End)'),
        'JUNCT': ('J', 'Junction (Branch)'),
        'TERMINAL': ('T', 'Terminal (End)'),
        'UNKNOWN': ('U', 'Unknown')
    }

    @staticmethod
    def draw(
        graph: ShockGraph,
        mode: str = 'minimal', # 'minimal' or 'debug'
        figsize: tuple = (12, 10),
        save_path: Optional[str] = None,
        image_path: Optional[str] = None,
    ) -> None:
        """Plots the shock graph with robust arrows, node labels, and boundary toggles."""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Only adjust layout for toggles if in debug mode
        if mode == 'debug':
            plt.subplots_adjust(left=0.25)

        if image_path and os.path.exists(image_path):
            img = mpimg.imread(image_path)
            ax.imshow(img)

        all_p_lines = []
        all_m_lines = []
        all_outline_loops = []

        # 1. Process Edges
        for edge in graph.edges:
            if not edge.samples:
                continue

            x_vals = [s.x for s in edge.samples]
            y_vals = [s.y for s in edge.samples]

            # --- SHOCK SPINE ---
            ax.plot(x_vals, y_vals, color='green', alpha=0.8, linewidth=2, zorder=4)

            # --- CALCULATE BOUNDARIES (Only if debug) ---
            if mode == 'debug':
                p_x, p_y = [], []
                m_x, m_y = [], []
                
                for s in edge.samples:
                    p_angle = s.theta + s.phi
                    p_x.append(s.x + s.t * math.cos(p_angle))
                    p_y.append(s.y + s.t * math.sin(p_angle))
                    
                    m_angle = s.theta - s.phi
                    m_x.append(s.x + s.t * math.cos(m_angle))
                    m_y.append(s.y + s.t * math.sin(m_angle))

                # --- PLUS/MINUS CURVES ---
                p_line, = ax.plot(p_x, p_y, color='dodgerblue', alpha=0.8, linewidth=2, linestyle='-', zorder=3)
                m_line, = ax.plot(m_x, m_y, color='crimson', alpha=0.8, linewidth=2, linestyle='-', zorder=3)
                all_p_lines.append(p_line)
                all_m_lines.append(m_line)

                # --- PER-EDGE OUTLINE ---
                path_color = (random.random(), random.random(), random.random())
                loop_x = [x_vals[0]] + p_x + [x_vals[-1]] + m_x[::-1] + [x_vals[0]]
                loop_y = [y_vals[0]] + p_y + [y_vals[-1]] + m_y[::-1] + [y_vals[0]]
                outline, = ax.plot(loop_x, loop_y, color=path_color, alpha=0.9, linewidth=2, visible=False, zorder=2)
                all_outline_loops.append(outline)

            # --- ROBUST DIRECTIONAL ARROWS (Always Show) ---
            if len(edge.samples) >= 2:
                mid_idx = len(edge.samples) // 2
                mid_x, mid_y = x_vals[mid_idx], y_vals[mid_idx]
                tail_idx = mid_idx - 1
                while tail_idx >= 0:
                    dist = math.sqrt((mid_x - x_vals[tail_idx])**2 + (mid_y - y_vals[tail_idx])**2)
                    if dist > 2.0: break
                    tail_idx -= 1
                
                if tail_idx < 0:
                    head_idx = mid_idx + 1
                    while head_idx < len(edge.samples):
                        dist = math.sqrt((x_vals[head_idx] - mid_x)**2 + (y_vals[head_idx] - mid_y)**2)
                        if dist > 2.0: break
                        head_idx += 1
                    if head_idx < len(edge.samples):
                        t_x, t_y, h_x, h_y = mid_x, mid_y, x_vals[head_idx], y_vals[head_idx]
                    else: continue 
                else:
                    t_x, t_y, h_x, h_y = x_vals[tail_idx], y_vals[tail_idx], mid_x, mid_y

                ax.annotate('', xy=(h_x, h_y), xytext=(t_x, t_y),
                            arrowprops=dict(arrowstyle="->", color='green', lw=2.0, mutation_scale=15),
                            zorder=5)

        # 2. Process Nodes (Markers + ID Labels)
        node_color = 'green' if mode == 'minimal' else 'black'
        used_types = set()

        for node in graph.nodes.values():
            if not node.sample:
                continue

            x, y = node.sample.x, node.sample.y
            ax.plot(x, y, marker='o', color=node_color, markersize=5, zorder=6)

            short_type, full_desc = ShockVisualizer.TYPE_REV_MAP.get(node.type, ('U', 'Unknown'))
            used_types.add((short_type, full_desc))

            # Minimal: Type only | Debug: Type + ID
            label = short_type if mode == 'minimal' else f'{short_type}:{node.id}'
            ax.text(x, y, f' {label}', fontsize=9, color=node_color, 
                    weight='bold', va='bottom', ha='left', zorder=7)

        # 3. Formatting
        ax.set_aspect('equal', adjustable='box')
        if not ax.yaxis_inverted(): ax.invert_yaxis()
            
        ax.set_title(f"Shock Graph Analysis ({mode.capitalize()})", fontsize=14, weight='bold')

        # Legend building
        legend_lines = [Line2D([0], [0], color='green', lw=2, label='Shock Spine')]
        for s_t, f_d in sorted(list(used_types)):
            legend_lines.append(Line2D([0], [0], marker='o', color='w', label=f"{s_t}: {f_d}",
                                       markerfacecolor=node_color, markersize=6))
        
        if mode == 'debug':
            legend_lines.extend([
                Line2D([0], [0], color='dodgerblue', lw=2, label='Plus (+)'),
                Line2D([0], [0], color='crimson', lw=2, label='Minus (-)')
            ])
            
        ax.legend(handles=legend_lines, loc='upper right', fontsize='small')

        # 4. Interactive Toggles (Debug Only)
        if mode == 'debug':
            ax_toggle = plt.axes([0.02, 0.4, 0.20, 0.15])
            check = CheckButtons(ax_toggle, ['Plus (+)', 'Minus (-)', 'Outlines'], [True, True, False])
            def toggle(label):
                if label == 'Plus (+)':
                    for l in all_p_lines: l.set_visible(not l.get_visible())
                elif label == 'Minus (-)':
                    for l in all_m_lines: l.set_visible(not l.get_visible())
                elif label == 'Outlines':
                    for l in all_outline_loops: l.set_visible(not l.get_visible())
                fig.canvas.draw_idle()
            check.on_clicked(toggle)
            fig.toggle_widget = check

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
