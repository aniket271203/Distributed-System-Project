"""
Performance Analysis and Visualization
Measures and plots broadcast/gather performance on mesh topologies
Author: Aniket Gupta
"""

import numpy as np
import math

def calculate_theoretical_metrics(size, mesh_type='2D'):
    """
    Calculate theoretical metrics for a given mesh size and type.
    Returns a dictionary with grid_size, diameter, and comm_steps.
    """
    metrics = {}
    
    if mesh_type == '2D':
        # For 2D, we assume a square grid for theoretical analysis
        # or the best possible rectangle
        best_r, best_c = 1, size
        for r in range(1, int(math.sqrt(size)) + 1):
            if size % r == 0:
                c = size // r
                if abs(r - c) < abs(best_r - best_c):
                    best_r, best_c = r, c
        
        metrics['grid_size'] = f"{best_r}x{best_c}"
        # Diameter = (r-1) + (c-1)
        metrics['diameter'] = (best_r - 1) + (best_c - 1)
        # Broadcast steps (DOR) = (r-1) + (c-1)
        metrics['comm_steps'] = (best_r - 1) + (best_c - 1)
        
    elif mesh_type == '3D':
        # For 3D, best cuboid
        best_dims = (1, 1, size)
        min_diff = float('inf')
        
        for x in range(1, int(size**(1/3)) + 2):
            if size % x == 0:
                rem = size // x
                for y in range(1, int(math.sqrt(rem)) + 1):
                    if rem % y == 0:
                        z = rem // y
                        diff = max(abs(x-y), abs(y-z), abs(x-z))
                        if diff < min_diff:
                            min_diff = diff
                            best_dims = sorted((x, y, z))
        
        x, y, z = best_dims
        metrics['grid_size'] = f"{x}x{y}x{z}"
        metrics['diameter'] = (x - 1) + (y - 1) + (z - 1)
        metrics['comm_steps'] = (x - 1) + (y - 1) + (z - 1)
        
    return metrics

def latency_bandwidth_model(ts, tw, m, size, mesh_type='2D'):
    """
    Calculate theoretical time using Latency-Bandwidth Model
    T = steps * ts + (steps * tw * m) ? 
    Actually for pipelined it's different, but for standard store-and-forward:
    T = steps * (ts + m * tw)
    
    ts: Latency (startup time)
    tw: Time per word (1/bandwidth)
    m: Message size (elements)
    size: Number of processes
    """
    metrics = calculate_theoretical_metrics(size, mesh_type)
    steps = metrics['comm_steps']
    
    # Standard Dimension-Ordered Routing (Store-and-Forward)
    # Each step involves sending the full message
    time = steps * (ts + m * tw)
    
    return time, steps
