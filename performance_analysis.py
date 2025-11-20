"""
Performance Analysis and Visualization
Measures and plots broadcast/gather performance on mesh topologies
Author: Aniket Gupta
"""

from mpi4py import MPI
import numpy as np
import time
import json
from mesh_topology import Mesh2D, Mesh3D
from broadcast import broadcast_2d_mesh, broadcast_3d_mesh
from gather import gather_2d_mesh, gather_3d_mesh


def collect_performance_data(comm, mesh_type='2D', operation='broadcast', data_sizes=[100, 500, 1000, 5000]):
    """Collect performance data for different data sizes"""
    rank = comm.Get_rank()
    results = []
    
    for data_size in data_sizes:
        if mesh_type == '2D':
            mesh = Mesh2D(comm)
        else:
            mesh = Mesh3D(comm)
        
        if operation == 'broadcast':
            if rank == 0:
                data = np.random.rand(data_size)
            else:
                data = None
            
            if mesh_type == '2D':
                _, elapsed_time, steps, msgs = broadcast_2d_mesh(mesh, data, root=0)
            else:
                _, elapsed_time, steps, msgs = broadcast_3d_mesh(mesh, data, root=0)
        
        else:  # gather
            data = np.random.rand(data_size)
            
            if mesh_type == '2D':
                _, elapsed_time, steps, msgs = gather_2d_mesh(mesh, data, root=0)
            else:
                _, elapsed_time, steps, msgs = gather_3d_mesh(mesh, data, root=0)
        
        # Gather times from all processes
        all_times = comm.gather(elapsed_time, root=0)
        
        if rank == 0:
            avg_time = np.mean(all_times)
            max_time = np.max(all_times)
            min_time = np.min(all_times)
            
            results.append({
                'data_size': data_size,
                'avg_time': avg_time,
                'max_time': max_time,
                'min_time': min_time,
                'steps': steps,
                'messages': msgs
            })
    
    return results


def analyze_scalability(comm, operation='broadcast', data_size=1000):
    """Analyze how performance scales with number of processes"""
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Test 2D mesh
    mesh_2d = Mesh2D(comm)
    if operation == 'broadcast':
        if rank == 0:
            data = np.random.rand(data_size)
        else:
            data = None
        _, time_2d, steps_2d, _ = broadcast_2d_mesh(mesh_2d, data, root=0)
    else:
        data = np.random.rand(data_size)
        _, time_2d, steps_2d, _ = gather_2d_mesh(mesh_2d, data, root=0)
    
    # Test 3D mesh if possible
    time_3d = 0
    steps_3d = 0
    if size >= 8:
        mesh_3d = Mesh3D(comm)
        if operation == 'broadcast':
            if rank == 0:
                data = np.random.rand(data_size)
            else:
                data = None
            _, time_3d, steps_3d, _ = broadcast_3d_mesh(mesh_3d, data, root=0)
        else:
            data = np.random.rand(data_size)
            _, time_3d, steps_3d, _ = gather_3d_mesh(mesh_3d, data, root=0)
    
    result = {
        'num_processes': size,
        'time_2d': time_2d,
        'steps_2d': steps_2d,
        'time_3d': time_3d,
        'steps_3d': steps_3d
    }
    
    return result


def calculate_theoretical_metrics(num_processes, mesh_type='2D'):
    """Calculate theoretical performance metrics"""
    if mesh_type == '2D':
        grid_size = int(np.sqrt(num_processes))
        diameter = 2 * (grid_size - 1)
        comm_steps = 2 * (grid_size - 1)
        
        # Bisection width for 2D mesh
        bisection_width = grid_size
        
    else:  # 3D
        grid_size = int(round(num_processes ** (1/3)))
        diameter = 3 * (grid_size - 1)
        comm_steps = 3 * (grid_size - 1)
        
        # Bisection width for 3D mesh
        bisection_width = grid_size * grid_size
    
    return {
        'grid_size': grid_size,
        'diameter': diameter,
        'comm_steps': comm_steps,
        'bisection_width': bisection_width
    }


def latency_bandwidth_model(ts, tw, m, p, mesh_type='2D'):
    """
    Calculate communication time using latency-bandwidth model
    T = ts + tw * m
    
    For Broadcast/Gather:
    2D: T = 2(√p - 1)ts + (p - 1)tw*m
    3D: T = 3(∛p - 1)ts + (p - 1)tw*m
    """
    if mesh_type == '2D':
        grid_size = int(np.sqrt(p))
        steps = 2 * (grid_size - 1)
    else:
        grid_size = int(round(p ** (1/3)))
        steps = 3 * (grid_size - 1)
    
    # Total communication time
    T = steps * ts + (p - 1) * tw * m
    
    return T, steps


def generate_performance_report(comm):
    """Generate comprehensive performance report"""
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("\n" + "="*80)
        print("PERFORMANCE ANALYSIS REPORT")
        print("="*80)
        print(f"\nNumber of processes: {size}")
        
        # 2D Mesh Analysis
        print("\n" + "-"*80)
        print("2D MESH TOPOLOGY ANALYSIS")
        print("-"*80)
        metrics_2d = calculate_theoretical_metrics(size, '2D')
        print(f"Grid configuration: {metrics_2d['grid_size']}x{metrics_2d['grid_size']}")
        print(f"Network diameter: {metrics_2d['diameter']}")
        print(f"Communication steps (Bcast/Gather): {metrics_2d['comm_steps']}")
        print(f"Bisection width: {metrics_2d['bisection_width']}")
        
        # 3D Mesh Analysis
        if size >= 8:
            print("\n" + "-"*80)
            print("3D MESH TOPOLOGY ANALYSIS")
            print("-"*80)
            metrics_3d = calculate_theoretical_metrics(size, '3D')
            print(f"Grid configuration: {metrics_3d['grid_size']}x{metrics_3d['grid_size']}x{metrics_3d['grid_size']}")
            print(f"Network diameter: {metrics_3d['diameter']}")
            print(f"Communication steps (Bcast/Gather): {metrics_3d['comm_steps']}")
            print(f"Bisection width: {metrics_3d['bisection_width']}")
            
            # Comparison
            print("\n" + "-"*80)
            print("2D vs 3D COMPARISON")
            print("-"*80)
            print(f"Diameter reduction: {metrics_2d['diameter']} → {metrics_3d['diameter']} "
                  f"({(1 - metrics_3d['diameter']/metrics_2d['diameter'])*100:.1f}% improvement)")
            print(f"Communication steps reduction: {metrics_2d['comm_steps']} → {metrics_3d['comm_steps']} "
                  f"({(1 - metrics_3d['comm_steps']/metrics_2d['comm_steps'])*100:.1f}% improvement)")
            print(f"Bisection width increase: {metrics_2d['bisection_width']} → {metrics_3d['bisection_width']} "
                  f"({(metrics_3d['bisection_width']/metrics_2d['bisection_width']):.1f}x better)")
        
        # Latency-Bandwidth Model
        print("\n" + "-"*80)
        print("LATENCY-BANDWIDTH MODEL")
        print("-"*80)
        print("Model: T = ts + tw * m")
        print("  where ts = latency (startup time)")
        print("        tw = time per word (bandwidth inverse)")
        print("        m = message size")
        
        # Example calculations with assumed parameters
        ts = 1e-5  # 10 microseconds
        tw = 1e-9  # 1 nanosecond per byte
        m = 1000   # message size
        
        T_2d, steps_2d = latency_bandwidth_model(ts, tw, m, size, '2D')
        print(f"\n2D Mesh Broadcast/Gather time (example):")
        print(f"  T = 2(√p - 1)ts + (p - 1)tw*m")
        print(f"  T = 2({metrics_2d['grid_size']} - 1) × {ts} + ({size} - 1) × {tw} × {m}")
        print(f"  T = {T_2d:.9f} seconds")
        
        if size >= 8:
            T_3d, steps_3d = latency_bandwidth_model(ts, tw, m, size, '3D')
            print(f"\n3D Mesh Broadcast/Gather time (example):")
            print(f"  T = 3(∛p - 1)ts + (p - 1)tw*m")
            print(f"  T = 3({metrics_3d['grid_size']} - 1) × {ts} + ({size} - 1) × {tw} × {m}")
            print(f"  T = {T_3d:.9f} seconds")
            print(f"\n  Speedup (3D over 2D): {T_2d/T_3d:.2f}x")
        
        print("\n" + "="*80)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Generate performance report
    generate_performance_report(comm)
    
    # Test with varying data sizes
    if rank == 0:
        print("\nTesting with varying data sizes...")
    
    data_sizes = [100, 500, 1000, 5000, 10000]
    
    # 2D Broadcast
    results_2d_bcast = collect_performance_data(comm, '2D', 'broadcast', data_sizes)
    
    if rank == 0 and results_2d_bcast:
        print("\n" + "-"*80)
        print("2D MESH BROADCAST - Data Size Scaling")
        print("-"*80)
        print(f"{'Data Size':<12} {'Avg Time (s)':<15} {'Max Time (s)':<15} {'Steps':<10}")
        print("-"*80)
        for result in results_2d_bcast:
            print(f"{result['data_size']:<12} {result['avg_time']:<15.6f} "
                  f"{result['max_time']:<15.6f} {result['steps']:<10}")
    
    # 2D Gather
    results_2d_gather = collect_performance_data(comm, '2D', 'gather', data_sizes)
    
    if rank == 0 and results_2d_gather:
        print("\n" + "-"*80)
        print("2D MESH GATHER - Data Size Scaling")
        print("-"*80)
        print(f"{'Data Size':<12} {'Avg Time (s)':<15} {'Max Time (s)':<15} {'Steps':<10}")
        print("-"*80)
        for result in results_2d_gather:
            print(f"{result['data_size']:<12} {result['avg_time']:<15.6f} "
                  f"{result['max_time']:<15.6f} {result['steps']:<10}")
    
    # 3D tests if we have enough processes
    if comm.Get_size() >= 8:
        results_3d_bcast = collect_performance_data(comm, '3D', 'broadcast', data_sizes)
        results_3d_gather = collect_performance_data(comm, '3D', 'gather', data_sizes)
        
        if rank == 0:
            if results_3d_bcast:
                print("\n" + "-"*80)
                print("3D MESH BROADCAST - Data Size Scaling")
                print("-"*80)
                print(f"{'Data Size':<12} {'Avg Time (s)':<15} {'Max Time (s)':<15} {'Steps':<10}")
                print("-"*80)
                for result in results_3d_bcast:
                    print(f"{result['data_size']:<12} {result['avg_time']:<15.6f} "
                          f"{result['max_time']:<15.6f} {result['steps']:<10}")
            
            if results_3d_gather:
                print("\n" + "-"*80)
                print("3D MESH GATHER - Data Size Scaling")
                print("-"*80)
                print(f"{'Data Size':<12} {'Avg Time (s)':<15} {'Max Time (s)':<15} {'Steps':<10}")
                print("-"*80)
                for result in results_3d_gather:
                    print(f"{result['data_size']:<12} {result['avg_time']:<15.6f} "
                          f"{result['max_time']:<15.6f} {result['steps']:<10}")
    
    if rank == 0:
        print("\n" + "="*80)
        print("Performance analysis completed!")
        print("="*80 + "\n")
