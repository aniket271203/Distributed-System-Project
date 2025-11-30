"""
Simulation Study: 2D vs 3D Mesh Topology Comparison
Author: Aniket Gupta / Samarth

This module provides a simulation-based comparison of 2D and 3D mesh topologies
for broadcast and gather operations using DOR (Dimension-Ordered Routing) and Flooding.

Key Metric: Sequential Hops
- For 2D mesh (rows x cols): 
  - DOR: (cols-1) hops along row + (rows-1) hops along column = (rows-1) + (cols-1)
  - Flooding: Manhattan distance from root to farthest node
  
- For 3D mesh (x × y × z):
  - DOR: (x-1) + (y-1) + (z-1) hops
  - Flooding: Manhattan distance from root to farthest node
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict
import math


@dataclass
class SimulationResult:
    """Store results from a simulation run"""
    topology: str  # '2D' or '3D'
    dimensions: Tuple  # (rows, cols) or (x, y, z)
    total_nodes: int
    algorithm: str  # 'DOR' or 'Flooding'
    operation: str  # 'broadcast' or 'gather'
    sequential_hops: int
    total_messages: int
    simulated_time: float  # Based on latency-bandwidth model


class MeshSimulator:
    """Simulate mesh topology operations without MPI"""
    
    def __init__(self, latency=1e-6, bandwidth_time=1e-9, message_size=1000):
        """
        Args:
            latency: Startup time per message (ts)
            bandwidth_time: Time per byte (tw)
            message_size: Size of message in bytes
        """
        self.ts = latency
        self.tw = bandwidth_time
        self.m = message_size
    
    def simulate_2d_broadcast_dor(self, rows: int, cols: int, root: Tuple[int, int] = (0, 0)) -> SimulationResult:
        """
        Simulate 2D DOR broadcast
        
        Algorithm:
        1. Root broadcasts along its row: (cols-1) sequential hops
        2. Each node in root's row broadcasts down its column: (rows-1) sequential hops
        
        These happen in sequence, so total hops = (cols-1) + (rows-1)
        """
        total_nodes = rows * cols
        
        # Sequential hops: first along row, then along columns (in parallel but sequentially within each column)
        row_hops = cols - 1  # Hops to reach all nodes in root's row
        col_hops = rows - 1  # Hops to reach all nodes in each column
        sequential_hops = row_hops + col_hops
        
        # Total messages: root sends to (cols-1) nodes, then each of those sends to (rows-1) nodes
        # But in tree-based broadcast within row/column, it's more efficient
        # For simplicity: cols-1 messages in row phase + cols*(rows-1) messages in column phase
        total_messages = (cols - 1) + cols * (rows - 1)
        
        # Time = sequential_hops * (ts + tw * m)
        simulated_time = sequential_hops * (self.ts + self.tw * self.m)
        
        return SimulationResult(
            topology='2D',
            dimensions=(rows, cols),
            total_nodes=total_nodes,
            algorithm='DOR',
            operation='broadcast',
            sequential_hops=sequential_hops,
            total_messages=total_messages,
            simulated_time=simulated_time
        )
    
    def simulate_2d_gather_dor(self, rows: int, cols: int, root: Tuple[int, int] = (0, 0)) -> SimulationResult:
        """
        Simulate 2D DOR gather (reverse of broadcast)
        
        Algorithm:
        1. Each column gathers to row leaders: (rows-1) sequential hops
        2. Row leaders gather along root's row: (cols-1) sequential hops
        """
        total_nodes = rows * cols
        
        col_hops = rows - 1
        row_hops = cols - 1
        sequential_hops = col_hops + row_hops
        
        total_messages = cols * (rows - 1) + (cols - 1)
        simulated_time = sequential_hops * (self.ts + self.tw * self.m)
        
        return SimulationResult(
            topology='2D',
            dimensions=(rows, cols),
            total_nodes=total_nodes,
            algorithm='DOR',
            operation='gather',
            sequential_hops=sequential_hops,
            total_messages=total_messages,
            simulated_time=simulated_time
        )
    
    def simulate_3d_broadcast_dor(self, x_dim: int, y_dim: int, z_dim: int, 
                                   root: Tuple[int, int, int] = (0, 0, 0)) -> SimulationResult:
        """
        Simulate 3D DOR broadcast
        
        Algorithm:
        1. Broadcast along x-axis: (x_dim-1) sequential hops
        2. Broadcast along y-axis (parallel across x): (y_dim-1) sequential hops
        3. Broadcast along z-axis (parallel across xy plane): (z_dim-1) sequential hops
        """
        total_nodes = x_dim * y_dim * z_dim
        
        x_hops = x_dim - 1
        y_hops = y_dim - 1
        z_hops = z_dim - 1
        sequential_hops = x_hops + y_hops + z_hops
        
        # Messages: x_dim-1 in x-phase, x_dim*(y_dim-1) in y-phase, x_dim*y_dim*(z_dim-1) in z-phase
        total_messages = (x_dim - 1) + x_dim * (y_dim - 1) + x_dim * y_dim * (z_dim - 1)
        
        simulated_time = sequential_hops * (self.ts + self.tw * self.m)
        
        return SimulationResult(
            topology='3D',
            dimensions=(x_dim, y_dim, z_dim),
            total_nodes=total_nodes,
            algorithm='DOR',
            operation='broadcast',
            sequential_hops=sequential_hops,
            total_messages=total_messages,
            simulated_time=simulated_time
        )
    
    def simulate_3d_gather_dor(self, x_dim: int, y_dim: int, z_dim: int,
                                root: Tuple[int, int, int] = (0, 0, 0)) -> SimulationResult:
        """
        Simulate 3D DOR gather (reverse of broadcast)
        """
        total_nodes = x_dim * y_dim * z_dim
        
        z_hops = z_dim - 1
        y_hops = y_dim - 1
        x_hops = x_dim - 1
        sequential_hops = z_hops + y_hops + x_hops
        
        total_messages = x_dim * y_dim * (z_dim - 1) + x_dim * (y_dim - 1) + (x_dim - 1)
        simulated_time = sequential_hops * (self.ts + self.tw * self.m)
        
        return SimulationResult(
            topology='3D',
            dimensions=(x_dim, y_dim, z_dim),
            total_nodes=total_nodes,
            algorithm='DOR',
            operation='gather',
            sequential_hops=sequential_hops,
            total_messages=total_messages,
            simulated_time=simulated_time
        )
    
    def simulate_2d_broadcast_flooding(self, rows: int, cols: int, 
                                        root: Tuple[int, int] = (0, 0)) -> SimulationResult:
        """
        Simulate 2D Flooding broadcast
        
        Sequential hops = Manhattan distance to farthest node
        For root at (0,0), farthest is (rows-1, cols-1)
        Distance = (rows-1) + (cols-1)
        
        Message complexity: In flooding, each node sends to ALL its neighbors
        Total messages = sum of all edges used = 2 * (rows * (cols-1) + cols * (rows-1))
        But since each node floods to all neighbors, it's higher
        """
        total_nodes = rows * cols
        
        # Maximum Manhattan distance from root
        max_dist = (rows - 1 - root[0]) + (cols - 1 - root[1])
        # Also check distance to (0,0) if root is not there
        max_dist = max(max_dist, root[0] + root[1])
        max_dist = max(max_dist, (rows - 1 - root[0]) + root[1])
        max_dist = max(max_dist, root[0] + (cols - 1 - root[1]))
        
        sequential_hops = max_dist
        
        # In flooding, each node sends to ALL its neighbors EXCEPT the one it received from
        # This means each edge is used exactly once (in one direction)
        # Total edges in 2D grid = rows*(cols-1) + cols*(rows-1) (horizontal + vertical)
        num_edges = rows * (cols - 1) + cols * (rows - 1)
        # In flooding, messages = number of edges (each edge carries exactly one message)
        total_messages = num_edges
        
        simulated_time = sequential_hops * (self.ts + self.tw * self.m)
        
        return SimulationResult(
            topology='2D',
            dimensions=(rows, cols),
            total_nodes=total_nodes,
            algorithm='Flooding',
            operation='broadcast',
            sequential_hops=sequential_hops,
            total_messages=total_messages,
            simulated_time=simulated_time
        )
    
    def simulate_2d_gather_flooding(self, rows: int, cols: int,
                                     root: Tuple[int, int] = (0, 0)) -> SimulationResult:
        """Simulate 2D Flooding gather - same hops as broadcast"""
        result = self.simulate_2d_broadcast_flooding(rows, cols, root)
        return SimulationResult(
            topology='2D',
            dimensions=(rows, cols),
            total_nodes=result.total_nodes,
            algorithm='Flooding',
            operation='gather',
            sequential_hops=result.sequential_hops,
            total_messages=result.total_messages,
            simulated_time=result.simulated_time
        )
    
    def simulate_3d_broadcast_flooding(self, x_dim: int, y_dim: int, z_dim: int,
                                        root: Tuple[int, int, int] = (0, 0, 0)) -> SimulationResult:
        """
        Simulate 3D Flooding broadcast
        
        Sequential hops = Maximum Manhattan distance in 3D
        Message complexity: In flooding, each node sends to ALL its neighbors (up to 6)
        """
        total_nodes = x_dim * y_dim * z_dim
        
        # Maximum Manhattan distance from root to any corner
        corners = [
            (0, 0, 0), (x_dim-1, 0, 0), (0, y_dim-1, 0), (0, 0, z_dim-1),
            (x_dim-1, y_dim-1, 0), (x_dim-1, 0, z_dim-1), (0, y_dim-1, z_dim-1),
            (x_dim-1, y_dim-1, z_dim-1)
        ]
        max_dist = max(
            abs(c[0] - root[0]) + abs(c[1] - root[1]) + abs(c[2] - root[2])
            for c in corners
        )
        
        sequential_hops = max_dist
        
        # Total edges in 3D mesh:
        # x-direction edges: (x_dim-1) * y_dim * z_dim
        # y-direction edges: x_dim * (y_dim-1) * z_dim
        # z-direction edges: x_dim * y_dim * (z_dim-1)
        num_edges = ((x_dim-1) * y_dim * z_dim + 
                     x_dim * (y_dim-1) * z_dim + 
                     x_dim * y_dim * (z_dim-1))
        # In flooding, each node sends to all neighbors except sender
        # So messages = number of edges (each edge used exactly once)
        total_messages = num_edges
        
        simulated_time = sequential_hops * (self.ts + self.tw * self.m)
        
        return SimulationResult(
            topology='3D',
            dimensions=(x_dim, y_dim, z_dim),
            total_nodes=total_nodes,
            algorithm='Flooding',
            operation='broadcast',
            sequential_hops=sequential_hops,
            total_messages=total_messages,
            simulated_time=simulated_time
        )
    
    def simulate_3d_gather_flooding(self, x_dim: int, y_dim: int, z_dim: int,
                                     root: Tuple[int, int, int] = (0, 0, 0)) -> SimulationResult:
        """Simulate 3D Flooding gather"""
        result = self.simulate_3d_broadcast_flooding(x_dim, y_dim, z_dim, root)
        return SimulationResult(
            topology='3D',
            dimensions=(x_dim, y_dim, z_dim),
            total_nodes=result.total_nodes,
            algorithm='Flooding',
            operation='gather',
            sequential_hops=result.sequential_hops,
            total_messages=result.total_messages,
            simulated_time=result.simulated_time
        )


def get_comparable_configurations() -> List[Tuple[Tuple, Tuple]]:
    """
    Generate comparable 2D and 3D configurations with similar node counts
    Returns list of ((rows, cols), (x, y, z)) tuples
    """
    configs = [
        # (2D config, 3D config, description)
        ((2, 4), (2, 2, 2), "8 nodes"),      # 8 = 8
        ((3, 3), (2, 2, 2), "9 vs 8 nodes"), # Close comparison
        ((4, 4), (2, 2, 4), "16 nodes"),     # 16 = 16
        ((4, 4), (2, 4, 2), "16 nodes alt"), # 16 = 16
        ((5, 5), (3, 3, 3), "25 vs 27"),     # Close comparison
        ((6, 6), (3, 3, 4), "36 nodes"),     # 36 = 36
        ((8, 8), (4, 4, 4), "64 nodes"),     # 64 = 64
        ((10, 10), (4, 5, 5), "100 nodes"),  # 100 = 100
        ((12, 12), (4, 6, 6), "144 nodes"),  # 144 = 144
        ((16, 16), (4, 8, 8), "256 nodes"),  # 256 = 256
        ((16, 16), (8, 8, 4), "256 nodes alt"), # 256 = 256
    ]
    return configs


def run_simulation_study():
    """Run comprehensive simulation comparing 2D and 3D meshes"""
    simulator = MeshSimulator(latency=1e-5, bandwidth_time=1e-8, message_size=8000)
    
    results = []
    configs = get_comparable_configurations()
    
    print("\n" + "="*100)
    print("SIMULATION STUDY: 2D vs 3D Mesh Topology Comparison")
    print("="*100)
    print(f"{'Config':<25} | {'Nodes':<6} | {'Algo':<10} | {'Op':<10} | {'Hops':<6} | {'Time (μs)':<12}")
    print("-"*100)
    
    for config_2d, config_3d, desc in configs:
        rows, cols = config_2d
        x, y, z = config_3d
        
        # 2D simulations
        for algo in ['DOR', 'Flooding']:
            for op in ['broadcast', 'gather']:
                if algo == 'DOR':
                    if op == 'broadcast':
                        result = simulator.simulate_2d_broadcast_dor(rows, cols)
                    else:
                        result = simulator.simulate_2d_gather_dor(rows, cols)
                else:
                    if op == 'broadcast':
                        result = simulator.simulate_2d_broadcast_flooding(rows, cols)
                    else:
                        result = simulator.simulate_2d_gather_flooding(rows, cols)
                results.append(result)
                print(f"2D {rows}x{cols:<20} | {result.total_nodes:<6} | {algo:<10} | {op:<10} | {result.sequential_hops:<6} | {result.simulated_time*1e6:.4f}")
        
        # 3D simulations
        for algo in ['DOR', 'Flooding']:
            for op in ['broadcast', 'gather']:
                if algo == 'DOR':
                    if op == 'broadcast':
                        result = simulator.simulate_3d_broadcast_dor(x, y, z)
                    else:
                        result = simulator.simulate_3d_gather_dor(x, y, z)
                else:
                    if op == 'broadcast':
                        result = simulator.simulate_3d_broadcast_flooding(x, y, z)
                    else:
                        result = simulator.simulate_3d_gather_flooding(x, y, z)
                results.append(result)
                print(f"3D {x}x{y}x{z:<17} | {result.total_nodes:<6} | {algo:<10} | {op:<10} | {result.sequential_hops:<6} | {result.simulated_time*1e6:.4f}")
        
        print("-"*100)
    
    return results


def plot_comparison_results(save_path: str = "results/simulation_comparison.png"):
    """Generate comprehensive comparison plots"""
    simulator = MeshSimulator(latency=1e-5, bandwidth_time=1e-8, message_size=8000)
    
    # Test configurations with increasing node counts
    node_counts = [8, 16, 27, 64, 125, 216, 343, 512]
    
    # For each node count, find best 2D and 3D configurations
    data_2d_dor_bcast = []
    data_3d_dor_bcast = []
    data_2d_flood_bcast = []
    data_3d_flood_bcast = []
    data_2d_dor_gather = []
    data_3d_dor_gather = []
    data_2d_flood_gather = []
    data_3d_flood_gather = []
    
    actual_nodes_2d = []
    actual_nodes_3d = []
    
    for n in node_counts:
        # 2D: Find closest square grid
        side_2d = int(math.ceil(math.sqrt(n)))
        rows = cols = side_2d
        actual_2d = rows * cols
        actual_nodes_2d.append(actual_2d)
        
        # 3D: Find closest cube
        side_3d = int(round(n ** (1/3)))
        x = y = z = side_3d
        actual_3d = x * y * z
        actual_nodes_3d.append(actual_3d)
        
        # Simulate 2D
        r_2d_dor_b = simulator.simulate_2d_broadcast_dor(rows, cols)
        r_2d_dor_g = simulator.simulate_2d_gather_dor(rows, cols)
        r_2d_flood_b = simulator.simulate_2d_broadcast_flooding(rows, cols)
        r_2d_flood_g = simulator.simulate_2d_gather_flooding(rows, cols)
        
        data_2d_dor_bcast.append(r_2d_dor_b.sequential_hops)
        data_2d_dor_gather.append(r_2d_dor_g.sequential_hops)
        data_2d_flood_bcast.append(r_2d_flood_b.sequential_hops)
        data_2d_flood_gather.append(r_2d_flood_g.sequential_hops)
        
        # Simulate 3D
        r_3d_dor_b = simulator.simulate_3d_broadcast_dor(x, y, z)
        r_3d_dor_g = simulator.simulate_3d_gather_dor(x, y, z)
        r_3d_flood_b = simulator.simulate_3d_broadcast_flooding(x, y, z)
        r_3d_flood_g = simulator.simulate_3d_gather_flooding(x, y, z)
        
        data_3d_dor_bcast.append(r_3d_dor_b.sequential_hops)
        data_3d_dor_gather.append(r_3d_dor_g.sequential_hops)
        data_3d_flood_bcast.append(r_3d_flood_b.sequential_hops)
        data_3d_flood_gather.append(r_3d_flood_g.sequential_hops)
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('2D vs 3D Mesh Topology: Sequential Hops Comparison', fontsize=14, fontweight='bold')
    
    x_labels = [f"{n}" for n in node_counts]
    x_pos = np.arange(len(node_counts))
    width = 0.35
    
    # Plot 1: Broadcast DOR comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x_pos - width/2, data_2d_dor_bcast, width, label='2D Mesh', color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x_pos + width/2, data_3d_dor_bcast, width, label='3D Mesh', color='coral', alpha=0.8)
    ax1.set_xlabel('Number of Nodes')
    ax1.set_ylabel('Sequential Hops')
    ax1.set_title('Broadcast - DOR Algorithm')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Gather DOR comparison
    ax2 = axes[0, 1]
    bars3 = ax2.bar(x_pos - width/2, data_2d_dor_gather, width, label='2D Mesh', color='steelblue', alpha=0.8)
    bars4 = ax2.bar(x_pos + width/2, data_3d_dor_gather, width, label='3D Mesh', color='coral', alpha=0.8)
    ax2.set_xlabel('Number of Nodes')
    ax2.set_ylabel('Sequential Hops')
    ax2.set_title('Gather - DOR Algorithm')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars3:
        height = bar.get_height()
        ax2.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars4:
        height = bar.get_height()
        ax2.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Broadcast Flooding comparison
    ax3 = axes[1, 0]
    bars5 = ax3.bar(x_pos - width/2, data_2d_flood_bcast, width, label='2D Mesh', color='steelblue', alpha=0.8)
    bars6 = ax3.bar(x_pos + width/2, data_3d_flood_bcast, width, label='3D Mesh', color='coral', alpha=0.8)
    ax3.set_xlabel('Number of Nodes')
    ax3.set_ylabel('Sequential Hops')
    ax3.set_title('Broadcast - Flooding Algorithm')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(x_labels)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    for bar in bars5:
        height = bar.get_height()
        ax3.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars6:
        height = bar.get_height()
        ax3.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Gather Flooding comparison
    ax4 = axes[1, 1]
    bars7 = ax4.bar(x_pos - width/2, data_2d_flood_gather, width, label='2D Mesh', color='steelblue', alpha=0.8)
    bars8 = ax4.bar(x_pos + width/2, data_3d_flood_gather, width, label='3D Mesh', color='coral', alpha=0.8)
    ax4.set_xlabel('Number of Nodes')
    ax4.set_ylabel('Sequential Hops')
    ax4.set_title('Gather - Flooding Algorithm')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(x_labels)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    for bar in bars7:
        height = bar.get_height()
        ax4.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars8:
        height = bar.get_height()
        ax4.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nComparison plot saved to: {save_path}")


def plot_time_comparison(save_path: str = "results/time_comparison.png"):
    """Generate time comparison plots"""
    simulator = MeshSimulator(latency=1e-5, bandwidth_time=1e-8, message_size=8000)
    
    # Test configurations
    node_counts = [8, 16, 27, 64, 125, 216, 343, 512]
    
    times_2d_dor = []
    times_3d_dor = []
    times_2d_flood = []
    times_3d_flood = []
    
    for n in node_counts:
        # 2D configuration
        side_2d = int(math.ceil(math.sqrt(n)))
        rows = cols = side_2d
        
        # 3D configuration  
        side_3d = int(round(n ** (1/3)))
        x = y = z = side_3d
        
        # Get times (average of broadcast and gather)
        r_2d_dor = (simulator.simulate_2d_broadcast_dor(rows, cols).simulated_time + 
                   simulator.simulate_2d_gather_dor(rows, cols).simulated_time) / 2
        r_3d_dor = (simulator.simulate_3d_broadcast_dor(x, y, z).simulated_time + 
                   simulator.simulate_3d_gather_dor(x, y, z).simulated_time) / 2
        r_2d_flood = (simulator.simulate_2d_broadcast_flooding(rows, cols).simulated_time + 
                     simulator.simulate_2d_gather_flooding(rows, cols).simulated_time) / 2
        r_3d_flood = (simulator.simulate_3d_broadcast_flooding(x, y, z).simulated_time + 
                     simulator.simulate_3d_gather_flooding(x, y, z).simulated_time) / 2
        
        times_2d_dor.append(r_2d_dor * 1e6)  # Convert to microseconds
        times_3d_dor.append(r_3d_dor * 1e6)
        times_2d_flood.append(r_2d_flood * 1e6)
        times_3d_flood.append(r_3d_flood * 1e6)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('2D vs 3D Mesh: Simulated Time Comparison', fontsize=14, fontweight='bold')
    
    # DOR comparison
    ax1.plot(node_counts, times_2d_dor, 'b-o', linewidth=2, markersize=8, label='2D Mesh DOR')
    ax1.plot(node_counts, times_3d_dor, 'r-s', linewidth=2, markersize=8, label='3D Mesh DOR')
    ax1.set_xlabel('Number of Nodes')
    ax1.set_ylabel('Simulated Time (μs)')
    ax1.set_title('DOR Algorithm - Time vs Nodes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Flooding comparison
    ax2.plot(node_counts, times_2d_flood, 'b-o', linewidth=2, markersize=8, label='2D Mesh Flooding')
    ax2.plot(node_counts, times_3d_flood, 'r-s', linewidth=2, markersize=8, label='3D Mesh Flooding')
    ax2.set_xlabel('Number of Nodes')
    ax2.set_ylabel('Simulated Time (μs)')
    ax2.set_title('Flooding Algorithm - Time vs Nodes')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Time comparison plot saved to: {save_path}")


def plot_scalability_analysis(save_path: str = "results/scalability_analysis.png"):
    """Analyze how hops scale with node count"""
    simulator = MeshSimulator()
    
    # Extended range of node counts
    node_counts_2d = [4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400, 441, 484, 529, 576, 625]
    node_counts_3d = [8, 27, 64, 125, 216, 343, 512]
    
    # Calculate hops for 2D
    hops_2d_dor = []
    for n in node_counts_2d:
        side = int(math.sqrt(n))
        result = simulator.simulate_2d_broadcast_dor(side, side)
        hops_2d_dor.append(result.sequential_hops)
    
    # Calculate hops for 3D
    hops_3d_dor = []
    for n in node_counts_3d:
        side = int(round(n ** (1/3)))
        result = simulator.simulate_3d_broadcast_dor(side, side, side)
        hops_3d_dor.append(result.sequential_hops)
    
    # Theoretical curves
    n_theory = np.linspace(4, 625, 100)
    hops_2d_theory = 2 * (np.sqrt(n_theory) - 1)  # 2D: 2(√n - 1)
    hops_3d_theory = 3 * (n_theory ** (1/3) - 1)   # 3D: 3(∛n - 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot actual data points
    ax.scatter(node_counts_2d, hops_2d_dor, color='blue', s=50, label='2D Mesh (simulated)', zorder=5)
    ax.scatter(node_counts_3d, hops_3d_dor, color='red', s=50, label='3D Mesh (simulated)', zorder=5)
    
    # Plot theoretical curves
    ax.plot(n_theory, hops_2d_theory, 'b--', linewidth=2, alpha=0.7, label='2D Theory: 2(√n - 1)')
    ax.plot(n_theory, hops_3d_theory, 'r--', linewidth=2, alpha=0.7, label='3D Theory: 3(∛n - 1)')
    
    ax.set_xlabel('Number of Nodes', fontsize=12)
    ax.set_ylabel('Sequential Hops (DOR)', fontsize=12)
    ax.set_title('Scalability Analysis: Hops vs Node Count', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.annotate('3D scales better\nfor large N', xy=(400, 12), fontsize=11, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Scalability analysis plot saved to: {save_path}")


def plot_comparable_configs(save_path: str = "results/comparable_configs.png"):
    """Plot comparison for exactly comparable configurations"""
    simulator = MeshSimulator(latency=1e-5, bandwidth_time=1e-8, message_size=8000)
    
    # Comparable configurations: same or very similar node counts
    comparisons = [
        ("2×4 vs 2×2×2", (2, 4), (2, 2, 2)),        # 8 nodes
        ("4×4 vs 2×2×4", (4, 4), (2, 2, 4)),        # 16 nodes
        ("4×4 vs 2×4×2", (4, 4), (2, 4, 2)),        # 16 nodes
        ("6×6 vs 3×3×4", (6, 6), (3, 3, 4)),        # 36 nodes
        ("8×8 vs 4×4×4", (8, 8), (4, 4, 4)),        # 64 nodes
        ("10×10 vs 4×5×5", (10, 10), (4, 5, 5)),    # 100 nodes
        ("16×16 vs 4×8×8", (16, 16), (4, 8, 8)),    # 256 nodes
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comparable 2D vs 3D Configurations', fontsize=14, fontweight='bold')
    
    labels = [c[0] for c in comparisons]
    x_pos = np.arange(len(comparisons))
    width = 0.35
    
    # Collect data
    hops_2d_dor, hops_3d_dor = [], []
    hops_2d_flood, hops_3d_flood = [], []
    time_2d_dor, time_3d_dor = [], []
    time_2d_flood, time_3d_flood = [], []
    
    for _, config_2d, config_3d in comparisons:
        rows, cols = config_2d
        x, y, z = config_3d
        
        # DOR
        r_2d = simulator.simulate_2d_broadcast_dor(rows, cols)
        r_3d = simulator.simulate_3d_broadcast_dor(x, y, z)
        hops_2d_dor.append(r_2d.sequential_hops)
        hops_3d_dor.append(r_3d.sequential_hops)
        time_2d_dor.append(r_2d.simulated_time * 1e6)
        time_3d_dor.append(r_3d.simulated_time * 1e6)
        
        # Flooding
        r_2d_f = simulator.simulate_2d_broadcast_flooding(rows, cols)
        r_3d_f = simulator.simulate_3d_broadcast_flooding(x, y, z)
        hops_2d_flood.append(r_2d_f.sequential_hops)
        hops_3d_flood.append(r_3d_f.sequential_hops)
        time_2d_flood.append(r_2d_f.simulated_time * 1e6)
        time_3d_flood.append(r_3d_f.simulated_time * 1e6)
    
    # Plot 1: DOR Hops
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x_pos - width/2, hops_2d_dor, width, label='2D Mesh', color='steelblue')
    bars2 = ax1.bar(x_pos + width/2, hops_3d_dor, width, label='3D Mesh', color='coral')
    ax1.set_ylabel('Sequential Hops')
    ax1.set_title('DOR - Sequential Hops')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Flooding Hops
    ax2 = axes[0, 1]
    ax2.bar(x_pos - width/2, hops_2d_flood, width, label='2D Mesh', color='steelblue')
    ax2.bar(x_pos + width/2, hops_3d_flood, width, label='3D Mesh', color='coral')
    ax2.set_ylabel('Sequential Hops')
    ax2.set_title('Flooding - Sequential Hops')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: DOR Time
    ax3 = axes[1, 0]
    ax3.bar(x_pos - width/2, time_2d_dor, width, label='2D Mesh', color='steelblue')
    ax3.bar(x_pos + width/2, time_3d_dor, width, label='3D Mesh', color='coral')
    ax3.set_ylabel('Simulated Time (μs)')
    ax3.set_title('DOR - Simulated Time')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(labels, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Flooding Time
    ax4 = axes[1, 1]
    ax4.bar(x_pos - width/2, time_2d_flood, width, label='2D Mesh', color='steelblue')
    ax4.bar(x_pos + width/2, time_3d_flood, width, label='3D Mesh', color='coral')
    ax4.set_ylabel('Simulated Time (μs)')
    ax4.set_title('Flooding - Simulated Time')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(labels, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparable configs plot saved to: {save_path}")


def plot_improvement_percentage(save_path: str = "results/improvement_percentage.png"):
    """Plot the percentage improvement of 3D over 2D"""
    simulator = MeshSimulator()
    
    comparisons = [
        ("8 nodes", (2, 4), (2, 2, 2)),
        ("16 nodes", (4, 4), (2, 2, 4)),
        ("36 nodes", (6, 6), (3, 3, 4)),
        ("64 nodes", (8, 8), (4, 4, 4)),
        ("100 nodes", (10, 10), (4, 5, 5)),
        ("256 nodes", (16, 16), (4, 8, 8)),
        ("512 nodes", (22, 23), (8, 8, 8)),
    ]
    
    labels = [c[0] for c in comparisons]
    improvement_dor = []
    improvement_flood = []
    
    for _, config_2d, config_3d in comparisons:
        rows, cols = config_2d
        x, y, z = config_3d
        
        # DOR improvement
        hops_2d = simulator.simulate_2d_broadcast_dor(rows, cols).sequential_hops
        hops_3d = simulator.simulate_3d_broadcast_dor(x, y, z).sequential_hops
        improvement_dor.append((hops_2d - hops_3d) / hops_2d * 100)
        
        # Flooding improvement
        hops_2d_f = simulator.simulate_2d_broadcast_flooding(rows, cols).sequential_hops
        hops_3d_f = simulator.simulate_3d_broadcast_flooding(x, y, z).sequential_hops
        improvement_flood.append((hops_2d_f - hops_3d_f) / hops_2d_f * 100)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, improvement_dor, width, label='DOR', color='forestgreen', alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, improvement_flood, width, label='Flooding', color='darkorange', alpha=0.8)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Improvement (%)', fontsize=12)
    ax.set_title('3D Mesh Improvement over 2D Mesh (Sequential Hops Reduction)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Improvement percentage plot saved to: {save_path}")


def generate_summary_table():
    """Generate a summary table of all comparisons"""
    simulator = MeshSimulator(latency=1e-5, bandwidth_time=1e-8, message_size=8000)
    
    print("\n" + "="*120)
    print("SUMMARY TABLE: 2D vs 3D Mesh Comparison")
    print("="*120)
    print(f"{'Config':<20} | {'Nodes':<6} | {'2D DOR Hops':<12} | {'3D DOR Hops':<12} | {'Improvement':<12} | {'2D Flood':<10} | {'3D Flood':<10}")
    print("-"*120)
    
    comparisons = [
        ("2×4 vs 2×2×2", (2, 4), (2, 2, 2)),
        ("4×4 vs 2×2×4", (4, 4), (2, 2, 4)),
        ("6×6 vs 3×3×4", (6, 6), (3, 3, 4)),
        ("8×8 vs 4×4×4", (8, 8), (4, 4, 4)),
        ("10×10 vs 4×5×5", (10, 10), (4, 5, 5)),
        ("12×12 vs 4×6×6", (12, 12), (4, 6, 6)),
        ("16×16 vs 4×8×8", (16, 16), (4, 8, 8)),
    ]
    
    for label, config_2d, config_3d in comparisons:
        rows, cols = config_2d
        x, y, z = config_3d
        nodes_2d = rows * cols
        nodes_3d = x * y * z
        
        hops_2d_dor = simulator.simulate_2d_broadcast_dor(rows, cols).sequential_hops
        hops_3d_dor = simulator.simulate_3d_broadcast_dor(x, y, z).sequential_hops
        improvement = (hops_2d_dor - hops_3d_dor) / hops_2d_dor * 100
        
        hops_2d_flood = simulator.simulate_2d_broadcast_flooding(rows, cols).sequential_hops
        hops_3d_flood = simulator.simulate_3d_broadcast_flooding(x, y, z).sequential_hops
        
        print(f"{label:<20} | {nodes_2d}/{nodes_3d:<4} | {hops_2d_dor:<12} | {hops_3d_dor:<12} | {improvement:>10.1f}% | {hops_2d_flood:<10} | {hops_3d_flood:<10}")
    
    print("="*120)
    print("\nKey Insights:")
    print("1. 3D mesh consistently requires fewer sequential hops than 2D mesh")
    print("2. Improvement increases with node count (better scalability)")
    print("3. For DOR: 2D needs 2(√n-1) hops, 3D needs 3(∛n-1) hops")
    print("4. For large n, ∛n grows slower than √n, hence 3D advantage")
    print("="*120 + "\n")


def plot_message_complexity(save_path: str = "results/message_complexity.png"):
    """
    Plot message complexity comparison: Flooding vs DOR for both 2D and 3D meshes
    Shows that flooding has significantly more messages than DOR
    """
    simulator = MeshSimulator()
    
    # Comparable configurations
    comparisons = [
        ("8", (2, 4), (2, 2, 2)),
        ("16", (4, 4), (2, 2, 4)),
        ("36", (6, 6), (3, 3, 4)),
        ("64", (8, 8), (4, 4, 4)),
        ("100", (10, 10), (4, 5, 5)),
        ("144", (12, 12), (4, 6, 6)),
        ("256", (16, 16), (4, 8, 8)),
    ]
    
    labels = [c[0] for c in comparisons]
    
    # Collect message counts
    msgs_2d_dor = []
    msgs_2d_flood = []
    msgs_3d_dor = []
    msgs_3d_flood = []
    
    for _, config_2d, config_3d in comparisons:
        rows, cols = config_2d
        x, y, z = config_3d
        
        # 2D
        r_2d_dor = simulator.simulate_2d_broadcast_dor(rows, cols)
        r_2d_flood = simulator.simulate_2d_broadcast_flooding(rows, cols)
        msgs_2d_dor.append(r_2d_dor.total_messages)
        msgs_2d_flood.append(r_2d_flood.total_messages)
        
        # 3D
        r_3d_dor = simulator.simulate_3d_broadcast_dor(x, y, z)
        r_3d_flood = simulator.simulate_3d_broadcast_flooding(x, y, z)
        msgs_3d_dor.append(r_3d_dor.total_messages)
        msgs_3d_flood.append(r_3d_flood.total_messages)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Message Complexity: Flooding vs DOR', fontsize=14, fontweight='bold')
    
    x_pos = np.arange(len(labels))
    width = 0.35
    
    # Plot 1: 2D Mesh comparison
    bars1 = ax1.bar(x_pos - width/2, msgs_2d_dor, width, label='DOR', color='forestgreen', alpha=0.8)
    bars2 = ax1.bar(x_pos + width/2, msgs_2d_flood, width, label='Flooding', color='crimson', alpha=0.8)
    ax1.set_xlabel('Number of Nodes', fontsize=11)
    ax1.set_ylabel('Total Messages', fontsize=11)
    ax1.set_title('2D Mesh: Message Complexity')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_yscale('log')
    
    # Add ratio annotations
    for i, (dor, flood) in enumerate(zip(msgs_2d_dor, msgs_2d_flood)):
        ratio = flood / dor
        ax1.annotate(f'{ratio:.1f}x', xy=(x_pos[i], max(dor, flood)), 
                    xytext=(0, 5), textcoords="offset points", ha='center', fontsize=8)
    
    # Plot 2: 3D Mesh comparison
    bars3 = ax2.bar(x_pos - width/2, msgs_3d_dor, width, label='DOR', color='forestgreen', alpha=0.8)
    bars4 = ax2.bar(x_pos + width/2, msgs_3d_flood, width, label='Flooding', color='crimson', alpha=0.8)
    ax2.set_xlabel('Number of Nodes', fontsize=11)
    ax2.set_ylabel('Total Messages', fontsize=11)
    ax2.set_title('3D Mesh: Message Complexity')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_yscale('log')
    
    # Add ratio annotations
    for i, (dor, flood) in enumerate(zip(msgs_3d_dor, msgs_3d_flood)):
        ratio = flood / dor
        ax2.annotate(f'{ratio:.1f}x', xy=(x_pos[i], max(dor, flood)), 
                    xytext=(0, 5), textcoords="offset points", ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Message complexity plot saved to: {save_path}")


def plot_steps_2d_vs_3d(save_path: str = "results/steps_2d_vs_3d.png"):
    """
    Plot sequential hops comparison: 2D vs 3D mesh
    Shows that 3D requires fewer steps for the same number of nodes
    """
    simulator = MeshSimulator()
    
    # Comparable configurations with same/similar node counts
    comparisons = [
        ("8 nodes\n(2×4 vs 2×2×2)", (2, 4), (2, 2, 2)),
        ("16 nodes\n(4×4 vs 2×2×4)", (4, 4), (2, 2, 4)),
        ("36 nodes\n(6×6 vs 3×3×4)", (6, 6), (3, 3, 4)),
        ("64 nodes\n(8×8 vs 4×4×4)", (8, 8), (4, 4, 4)),
        ("100 nodes\n(10×10 vs 4×5×5)", (10, 10), (4, 5, 5)),
        ("144 nodes\n(12×12 vs 4×6×6)", (12, 12), (4, 6, 6)),
        ("256 nodes\n(16×16 vs 4×8×8)", (16, 16), (4, 8, 8)),
    ]
    
    labels = [c[0] for c in comparisons]
    
    # Collect step counts for DOR (primary algorithm)
    steps_2d_dor = []
    steps_3d_dor = []
    steps_2d_flood = []
    steps_3d_flood = []
    
    for _, config_2d, config_3d in comparisons:
        rows, cols = config_2d
        x, y, z = config_3d
        
        # DOR steps
        steps_2d_dor.append(simulator.simulate_2d_broadcast_dor(rows, cols).sequential_hops)
        steps_3d_dor.append(simulator.simulate_3d_broadcast_dor(x, y, z).sequential_hops)
        
        # Flooding steps
        steps_2d_flood.append(simulator.simulate_2d_broadcast_flooding(rows, cols).sequential_hops)
        steps_3d_flood.append(simulator.simulate_3d_broadcast_flooding(x, y, z).sequential_hops)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Sequential Hops: 2D vs 3D Mesh Comparison', fontsize=14, fontweight='bold')
    
    x_pos = np.arange(len(labels))
    width = 0.35
    
    # Plot 1: DOR Algorithm
    bars1 = ax1.bar(x_pos - width/2, steps_2d_dor, width, label='2D Mesh', color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x_pos + width/2, steps_3d_dor, width, label='3D Mesh', color='coral', alpha=0.8)
    ax1.set_xlabel('Configuration', fontsize=11)
    ax1.set_ylabel('Sequential Hops', fontsize=11)
    ax1.set_title('DOR Algorithm')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels and improvement percentage
    for i, (s2d, s3d) in enumerate(zip(steps_2d_dor, steps_3d_dor)):
        ax1.annotate(f'{s2d}', xy=(x_pos[i] - width/2, s2d), 
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
        ax1.annotate(f'{s3d}', xy=(x_pos[i] + width/2, s3d), 
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
        # Add improvement arrow
        improvement = (s2d - s3d) / s2d * 100
        ax1.annotate(f'↓{improvement:.0f}%', xy=(x_pos[i], (s2d + s3d) / 2), 
                    ha='center', fontsize=8, color='green', fontweight='bold')
    
    # Plot 2: Flooding Algorithm
    bars3 = ax2.bar(x_pos - width/2, steps_2d_flood, width, label='2D Mesh', color='steelblue', alpha=0.8)
    bars4 = ax2.bar(x_pos + width/2, steps_3d_flood, width, label='3D Mesh', color='coral', alpha=0.8)
    ax2.set_xlabel('Configuration', fontsize=11)
    ax2.set_ylabel('Sequential Hops', fontsize=11)
    ax2.set_title('Flooding Algorithm')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels and improvement percentage
    for i, (s2d, s3d) in enumerate(zip(steps_2d_flood, steps_3d_flood)):
        ax2.annotate(f'{s2d}', xy=(x_pos[i] - width/2, s2d), 
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
        ax2.annotate(f'{s3d}', xy=(x_pos[i] + width/2, s3d), 
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
        improvement = (s2d - s3d) / s2d * 100
        ax2.annotate(f'↓{improvement:.0f}%', xy=(x_pos[i], (s2d + s3d) / 2), 
                    ha='center', fontsize=8, color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Steps 2D vs 3D plot saved to: {save_path}")


def plot_combined_analysis(save_path: str = "results/combined_analysis.png"):
    """
    Combined plot showing both message complexity and steps for all algorithms
    """
    simulator = MeshSimulator()
    
    comparisons = [
        ("8", (2, 4), (2, 2, 2)),
        ("16", (4, 4), (2, 2, 4)),
        ("36", (6, 6), (3, 3, 4)),
        ("64", (8, 8), (4, 4, 4)),
        ("100", (10, 10), (4, 5, 5)),
        ("256", (16, 16), (4, 8, 8)),
    ]
    
    labels = [c[0] for c in comparisons]
    x_pos = np.arange(len(labels))
    
    # Collect all data
    data = {
        '2D_DOR_steps': [], '2D_DOR_msgs': [],
        '2D_Flood_steps': [], '2D_Flood_msgs': [],
        '3D_DOR_steps': [], '3D_DOR_msgs': [],
        '3D_Flood_steps': [], '3D_Flood_msgs': [],
    }
    
    for _, config_2d, config_3d in comparisons:
        rows, cols = config_2d
        x, y, z = config_3d
        
        r = simulator.simulate_2d_broadcast_dor(rows, cols)
        data['2D_DOR_steps'].append(r.sequential_hops)
        data['2D_DOR_msgs'].append(r.total_messages)
        
        r = simulator.simulate_2d_broadcast_flooding(rows, cols)
        data['2D_Flood_steps'].append(r.sequential_hops)
        data['2D_Flood_msgs'].append(r.total_messages)
        
        r = simulator.simulate_3d_broadcast_dor(x, y, z)
        data['3D_DOR_steps'].append(r.sequential_hops)
        data['3D_DOR_msgs'].append(r.total_messages)
        
        r = simulator.simulate_3d_broadcast_flooding(x, y, z)
        data['3D_Flood_steps'].append(r.sequential_hops)
        data['3D_Flood_msgs'].append(r.total_messages)
    
    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Complete Analysis: Steps & Message Complexity', fontsize=14, fontweight='bold')
    
    width = 0.2
    
    # Plot 1: Steps comparison - All algorithms
    ax1 = axes[0, 0]
    ax1.bar(x_pos - 1.5*width, data['2D_DOR_steps'], width, label='2D DOR', color='steelblue')
    ax1.bar(x_pos - 0.5*width, data['2D_Flood_steps'], width, label='2D Flooding', color='lightsteelblue')
    ax1.bar(x_pos + 0.5*width, data['3D_DOR_steps'], width, label='3D DOR', color='coral')
    ax1.bar(x_pos + 1.5*width, data['3D_Flood_steps'], width, label='3D Flooding', color='lightsalmon')
    ax1.set_xlabel('Number of Nodes')
    ax1.set_ylabel('Sequential Hops')
    ax1.set_title('Sequential Hops by Algorithm')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels)
    ax1.legend(fontsize=8)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Message complexity - All algorithms
    ax2 = axes[0, 1]
    ax2.bar(x_pos - 1.5*width, data['2D_DOR_msgs'], width, label='2D DOR', color='steelblue')
    ax2.bar(x_pos - 0.5*width, data['2D_Flood_msgs'], width, label='2D Flooding', color='lightsteelblue')
    ax2.bar(x_pos + 0.5*width, data['3D_DOR_msgs'], width, label='3D DOR', color='coral')
    ax2.bar(x_pos + 1.5*width, data['3D_Flood_msgs'], width, label='3D Flooding', color='lightsalmon')
    ax2.set_xlabel('Number of Nodes')
    ax2.set_ylabel('Total Messages')
    ax2.set_title('Message Complexity by Algorithm')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels)
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: DOR only - 2D vs 3D
    ax3 = axes[1, 0]
    width2 = 0.35
    ax3.bar(x_pos - width2/2, data['2D_DOR_steps'], width2, label='2D Mesh', color='steelblue', alpha=0.8)
    ax3.bar(x_pos + width2/2, data['3D_DOR_steps'], width2, label='3D Mesh', color='coral', alpha=0.8)
    ax3.set_xlabel('Number of Nodes')
    ax3.set_ylabel('Sequential Hops')
    ax3.set_title('DOR: 2D vs 3D Mesh')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(labels)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Add improvement percentages
    for i, (s2d, s3d) in enumerate(zip(data['2D_DOR_steps'], data['3D_DOR_steps'])):
        improvement = (s2d - s3d) / s2d * 100
        ax3.annotate(f'{improvement:.0f}%↓', xy=(x_pos[i], s2d + 0.5), 
                    ha='center', fontsize=9, color='green', fontweight='bold')
    
    # Plot 4: Flooding vs DOR message ratio
    ax4 = axes[1, 1]
    ratio_2d = [f/d for f, d in zip(data['2D_Flood_msgs'], data['2D_DOR_msgs'])]
    ratio_3d = [f/d for f, d in zip(data['3D_Flood_msgs'], data['3D_DOR_msgs'])]
    ax4.bar(x_pos - width2/2, ratio_2d, width2, label='2D Mesh', color='steelblue', alpha=0.8)
    ax4.bar(x_pos + width2/2, ratio_3d, width2, label='3D Mesh', color='coral', alpha=0.8)
    ax4.set_xlabel('Number of Nodes')
    ax4.set_ylabel('Message Ratio (Flooding / DOR)')
    ax4.set_title('Flooding Message Overhead vs DOR')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(labels)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    ax4.axhline(y=1, color='black', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Combined analysis plot saved to: {save_path}")


if __name__ == "__main__":
    import os
    
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    
    # Run simulation study
    results = run_simulation_study()
    
    # Generate all plots
    print("\nGenerating plots...")
    plot_comparison_results()
    plot_time_comparison()
    plot_scalability_analysis()
    plot_comparable_configs()
    plot_improvement_percentage()
    
    # New plots for message complexity and steps comparison
    plot_message_complexity()
    plot_steps_2d_vs_3d()
    plot_combined_analysis()
    
    # Generate summary table
    generate_summary_table()
    
    print("\n" + "="*80)
    print("SIMULATION COMPLETE!")
    print("Generated plots in 'results/' directory:")
    print("  - simulation_comparison.png")
    print("  - time_comparison.png")
    print("  - scalability_analysis.png")
    print("  - comparable_configs.png")
    print("  - improvement_percentage.png")
    print("  - message_complexity.png  [NEW]")
    print("  - steps_2d_vs_3d.png      [NEW]")
    print("  - combined_analysis.png   [NEW]")
    print("="*80)
