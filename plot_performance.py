#!/usr/bin/env python3
"""
Script to plot performance data from the Distributed Systems Project.
Visualizes 2D vs 3D mesh topology performance for broadcast and gather operations.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import math

def load_data(results_dir="results"):
    """Load all performance data JSON files."""
    data = {}
    for filename in os.listdir(results_dir):
        if filename.startswith("performance_data_p") and filename.endswith(".json"):
            filepath = os.path.join(results_dir, filename)
            with open(filepath, 'r') as f:
                json_data = json.load(f)
                process_count = json_data['process_count']
                data[process_count] = json_data['tests']
    return data

def plot_time_vs_datasize(data):
    """Plot execution time vs data size for selected process counts."""
    # Select specific process counts to plot detailed views for
    # Only those that support proper 3D meshes (all dims >= 2)
    selected_counts = [8, 12, 16]
    available_counts = sorted([p for p in data.keys() if p in selected_counts])
    
    if not available_counts:
        print("No matching process counts (8, 12, 16) found to plot detailed views.")
        return

    # Create subplots based on number of available counts
    n_plots = len(available_counts)
    cols = min(n_plots, 3)
    rows = math.ceil(n_plots / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    
    # Ensure axes is always iterable
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, p in enumerate(available_counts):
        ax = axes[idx]
        tests = data[p]
        
        data_sizes = [t['data_size'] for t in tests]
        broadcast_2d = [t['2d_broadcast']['time'] * 1000 for t in tests]  # Convert to ms
        broadcast_3d = [t['3d_broadcast']['time'] * 1000 for t in tests]
        gather_2d = [t['2d_gather']['time'] * 1000 for t in tests]
        gather_3d = [t['3d_gather']['time'] * 1000 for t in tests]
        flooding = [t['2d_broadcast_flooding']['time'] * 1000 for t in tests]
        
        ax.plot(data_sizes, broadcast_2d, 'b-o', label='2D Broadcast', linewidth=2, markersize=6)
        ax.plot(data_sizes, broadcast_3d, 'r-s', label='3D Broadcast', linewidth=2, markersize=6)
        ax.plot(data_sizes, gather_2d, 'g-^', label='2D Gather', linewidth=2, markersize=6)
        ax.plot(data_sizes, gather_3d, 'm-d', label='3D Gather', linewidth=2, markersize=6)
        ax.plot(data_sizes, flooding, 'c-x', label='Flooding', linewidth=2, markersize=6)
        
        ax.set_xlabel('Data Size (elements)', fontsize=11)
        ax.set_ylabel('Time (ms)', fontsize=11)
        ax.set_title(f'Performance with {p} Processes', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
    
    # Hide empty subplots if any
    for i in range(n_plots, rows * cols):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/time_vs_datasize.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: results/time_vs_datasize.png")

def plot_speedup_comparison(data):
    """Plot speedup of 3D over 2D for broadcast and gather."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    process_counts = sorted(data.keys())
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(process_counts)))
    
    # Broadcast speedup
    ax = axes[0]
    for idx, p in enumerate(process_counts):
        tests = data[p]
        data_sizes = [t['data_size'] for t in tests]
        speedup = [t['2d_broadcast']['time'] / t['3d_broadcast']['time'] for t in tests]
        ax.plot(data_sizes, speedup, '-o', color=colors[idx], label=f'p={p}', linewidth=2, markersize=6)
    
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='No speedup')
    ax.set_xlabel('Data Size (elements)', fontsize=11)
    ax.set_ylabel('Speedup (2D Time / 3D Time)', fontsize=11)
    ax.set_title('Broadcast: 3D vs 2D Speedup', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # Gather speedup
    ax = axes[1]
    for idx, p in enumerate(process_counts):
        tests = data[p]
        data_sizes = [t['data_size'] for t in tests]
        speedup = [t['2d_gather']['time'] / t['3d_gather']['time'] for t in tests]
        ax.plot(data_sizes, speedup, '-o', color=colors[idx], label=f'p={p}', linewidth=2, markersize=6)
    
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='No speedup')
    ax.set_xlabel('Data Size (elements)', fontsize=11)
    ax.set_ylabel('Speedup (2D Time / 3D Time)', fontsize=11)
    ax.set_title('Gather: 3D vs 2D Speedup', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('results/speedup_2d_vs_3d.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: results/speedup_2d_vs_3d.png")

def plot_flooding_comparison(data):
    """Compare flooding vs dimension-ordered routing for broadcast."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    process_counts = sorted(data.keys())
    x = np.arange(len(process_counts))
    width = 0.25
    
    # Average times across all data sizes
    avg_2d = []
    avg_3d = []
    avg_flood = []
    
    for p in process_counts:
        tests = data[p]
        avg_2d.append(np.mean([t['2d_broadcast']['time'] for t in tests]) * 1000)
        avg_3d.append(np.mean([t['3d_broadcast']['time'] for t in tests]) * 1000)
        avg_flood.append(np.mean([t['2d_broadcast_flooding']['time'] for t in tests]) * 1000)
    
    bars1 = ax.bar(x - width, avg_2d, width, label='2D DOR Broadcast', color='steelblue')
    bars2 = ax.bar(x, avg_3d, width, label='3D DOR Broadcast', color='coral')
    bars3 = ax.bar(x + width, avg_flood, width, label='Flooding Broadcast', color='seagreen')
    
    ax.set_xlabel('Number of Processes', fontsize=11)
    ax.set_ylabel('Average Time (ms)', fontsize=11)
    ax.set_title('Broadcast: DOR vs Flooding (Averaged over Data Sizes)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(process_counts)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/dor_vs_flooding_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: results/dor_vs_flooding_comparison.png")

def plot_message_steps_comparison(data):
    """Compare message counts and steps for different algorithms."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    process_counts = sorted(data.keys())
    
    # Get steps and messages (same for all data sizes within a process count)
    steps_2d_bc = [data[p][0]['2d_broadcast']['steps'] for p in process_counts]
    steps_3d_bc = [data[p][0]['3d_broadcast']['steps'] for p in process_counts]
    steps_flood = [data[p][0]['2d_broadcast_flooding']['steps'] for p in process_counts]
    
    msgs_2d_bc = [data[p][0]['2d_broadcast']['messages'] for p in process_counts]
    msgs_3d_bc = [data[p][0]['3d_broadcast']['messages'] for p in process_counts]
    msgs_flood = [data[p][0]['2d_broadcast_flooding']['messages'] for p in process_counts]
    
    x = np.arange(len(process_counts))
    width = 0.25
    
    # Steps comparison
    ax = axes[0]
    ax.bar(x - width, steps_2d_bc, width, label='2D DOR', color='steelblue')
    ax.bar(x, steps_3d_bc, width, label='3D DOR', color='coral')
    ax.bar(x + width, steps_flood, width, label='Flooding', color='seagreen')
    ax.set_xlabel('Number of Processes', fontsize=11)
    ax.set_ylabel('Number of Steps', fontsize=11)
    ax.set_title('Broadcast Steps Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(process_counts)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Messages comparison
    ax = axes[1]
    ax.bar(x - width, msgs_2d_bc, width, label='2D DOR', color='steelblue')
    ax.bar(x, msgs_3d_bc, width, label='3D DOR', color='coral')
    ax.bar(x + width, msgs_flood, width, label='Flooding', color='seagreen')
    ax.set_xlabel('Number of Processes', fontsize=11)
    ax.set_ylabel('Number of Messages (per node)', fontsize=11)
    ax.set_title('Broadcast Messages Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(process_counts)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/steps_and_messages.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: results/steps_and_messages.png")

def plot_heatmap_summary(data):
    """Create a heatmap showing average execution times."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    process_counts = sorted(data.keys())
    operations = ['2D Broadcast', '3D Broadcast', '2D Gather', '3D Gather', 'Flooding']
    
    heatmap_data = []
    for p in process_counts:
        tests = data[p]
        row = [
            np.mean([t['2d_broadcast']['time'] for t in tests]) * 1000,
            np.mean([t['3d_broadcast']['time'] for t in tests]) * 1000,
            np.mean([t['2d_gather']['time'] for t in tests]) * 1000,
            np.mean([t['3d_gather']['time'] for t in tests]) * 1000,
            np.mean([t['2d_broadcast_flooding']['time'] for t in tests]) * 1000,
        ]
        heatmap_data.append(row)
    
    heatmap_data = np.array(heatmap_data)
    
    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    
    ax.set_xticks(np.arange(len(operations)))
    ax.set_yticks(np.arange(len(process_counts)))
    ax.set_xticklabels(operations, fontsize=10)
    ax.set_yticklabels([f'p={p}' for p in process_counts], fontsize=10)
    
    # Add text annotations
    for i in range(len(process_counts)):
        for j in range(len(operations)):
            text = ax.text(j, i, f'{heatmap_data[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=9)
    
    ax.set_title('Average Execution Time (ms) Heatmap', fontsize=12, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Time (ms)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/performance_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: results/performance_heatmap.png")

def main():
    print("=" * 60)
    print("Distributed Systems Project - Performance Visualization")
    print("=" * 60)
    
    # Load data
    data = load_data()
    print(f"\nLoaded data for process counts: {sorted(data.keys())}")
    
    # Generate all plots
    print("\n1. Plotting Time vs Data Size...")
    plot_time_vs_datasize(data)
    
    print("\n2. Plotting 2D vs 3D Speedup...")
    plot_speedup_comparison(data)
    
    print("\n3. Plotting DOR vs Flooding Comparison...")
    plot_flooding_comparison(data)
    
    print("\n4. Plotting Steps and Messages Comparison...")
    plot_message_steps_comparison(data)
    
    print("\n5. Plotting Performance Heatmap...")
    plot_heatmap_summary(data)
    
    print("\n" + "=" * 60)
    print("All plots saved to the 'results/' directory!")
    print("=" * 60)

if __name__ == "__main__":
    main()
