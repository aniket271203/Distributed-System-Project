"""
Ablation Study: Dimension-Ordered Routing (DOR) vs Flooding (BFS)
Author: Aniket Gupta
"""

from mpi4py import MPI
import numpy as np
import time
from mesh_topology import Mesh2D, Mesh3D
from broadcast import broadcast_2d_mesh, broadcast_3d_mesh
from gather import gather_2d_mesh, gather_3d_mesh
from flooding import broadcast_flooding, gather_flooding

def run_experiment(comm, mesh_type, operation, algorithm, data_size=1000, root=0):
    rank = comm.Get_rank()
    
    # Setup Mesh
    if mesh_type == '2D':
        mesh = Mesh2D(comm)
    else:
        mesh = Mesh3D(comm)
        
    # Setup Data
    if operation == 'broadcast':
        data = np.random.rand(data_size) if rank == root else None
    else:
        data = np.random.rand(data_size) # Each has own data
        
    # Run Algorithm
    if operation == 'broadcast':
        if algorithm == 'DOR':
            if mesh_type == '2D':
                _, t, steps, msgs = broadcast_2d_mesh(mesh, data, root)
            else:
                _, t, steps, msgs = broadcast_3d_mesh(mesh, data, root)
        else: # Flooding
            _, t, steps, msgs = broadcast_flooding(mesh, data, root)
            
    else: # Gather
        if algorithm == 'DOR':
            if mesh_type == '2D':
                _, t, steps, msgs = gather_2d_mesh(mesh, data, root)
            else:
                _, t, steps, msgs = gather_3d_mesh(mesh, data, root)
        else: # Flooding
            _, t, steps, msgs = gather_flooding(mesh, data, root)
            
    # Aggregate Results
    avg_time = comm.reduce(t, op=MPI.SUM, root=root)
    if rank == root:
        avg_time /= comm.Get_size()
        
    max_steps = comm.reduce(steps, op=MPI.MAX, root=root)
    total_msgs = comm.reduce(msgs, op=MPI.SUM, root=root)
    
    return avg_time, max_steps, total_msgs

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"ABLATION STUDY: DOR vs FLOODING (Processes: {size})")
        print(f"{'='*80}")
        print(f"{'Mesh':<5} | {'Op':<10} | {'Algo':<10} | {'Time (s)':<10} | {'Steps':<5} | {'Msgs':<5}")
        print("-" * 80)
        
    configs = [
        ('2D', 'broadcast', 'DOR'),
        ('2D', 'broadcast', 'Flooding'),
        ('2D', 'gather', 'DOR'),
        ('2D', 'gather', 'Flooding'),
    ]
    
    if size >= 8:
        configs.extend([
            ('3D', 'broadcast', 'DOR'),
            ('3D', 'broadcast', 'Flooding'),
            ('3D', 'gather', 'DOR'),
            ('3D', 'gather', 'Flooding'),
        ])
        
    for mesh_type, op, algo in configs:
        try:
            # Skip 3D if not enough processes (redundant check but safe)
            if mesh_type == '3D' and size < 8:
                continue
                
            t, s, m = run_experiment(comm, mesh_type, op, algo)
            if rank == 0:
                print(f"{mesh_type:<5} | {op:<10} | {algo:<10} | {t:.6f}   | {s:<5} | {m:<5}")
        except Exception as e:
            if rank == 0:
                print(f"{mesh_type:<5} | {op:<10} | {algo:<10} | FAILED ({str(e)})")
            
    if rank == 0:
        print("-" * 80)
        print("Analysis:")
        print("1. DOR (Dimension-Ordered) typically has fewer messages due to structured paths.")
        print("2. Flooding (BFS) has optimal steps (Manhattan distance) but higher message complexity.")
        print("3. 3D topologies significantly reduce steps compared to 2D.")
        print("=" * 80 + "\n")
