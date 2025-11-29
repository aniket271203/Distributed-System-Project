from mpi4py import MPI
import numpy as np
import time
import sys
from mesh_topology import Mesh2D, Mesh3D
from broadcast import broadcast_2d_mesh, broadcast_3d_mesh
from gather import gather_2d_mesh, gather_3d_mesh
from flooding import broadcast_flooding, gather_flooding

def run_experiment(comm, mesh_type, operation, algorithm, data_size=10000000, root=0):
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Setup Mesh
    if mesh_type == '2D':
        mesh = Mesh2D(comm)
    else:
        mesh = Mesh3D(comm)
        
    # Setup Data
    # Use deterministic data for verification: Fill array with Rank ID
    if operation == 'broadcast':
        if rank == root:
            data = np.full(data_size, float(root))
        else:
            data = np.zeros(data_size) # Placeholder
    else:
        # Gather: everyone has data = their rank
        data = np.full(data_size, float(rank))
        
    # Run Algorithm
    t, steps, msgs = 0, 0, 0
    result_data = None
    
    if operation == 'broadcast':
        if algorithm == 'DOR':
            if mesh_type == '2D':
                result_data, t, steps, msgs = broadcast_2d_mesh(mesh, data, root)
            else:
                result_data, t, steps, msgs = broadcast_3d_mesh(mesh, data, root)
        else: # Flooding
            result_data, t, steps, msgs = broadcast_flooding(mesh, data, root)
            
    else: # Gather
        if algorithm == 'DOR':
            if mesh_type == '2D':
                result_data, t, steps, msgs = gather_2d_mesh(mesh, data, root)
            else:
                result_data, t, steps, msgs = gather_3d_mesh(mesh, data, root)
        else: # Flooding
            result_data, t, steps, msgs = gather_flooding(mesh, data, root)
            
    # Aggregate Results
    avg_time = comm.reduce(t, op=MPI.SUM, root=root)
    if rank == root:
        avg_time /= size
        
    max_steps = comm.reduce(steps, op=MPI.MAX, root=root)
    total_msgs = comm.reduce(msgs, op=MPI.SUM, root=root)
    
    # Verification
    is_correct = True
    if operation == 'broadcast':
        # Gather all received data to root to verify
        try:
            all_data = comm.gather(result_data, root=root)
            if rank == root:
                expected_val = float(root)
                for i, d in enumerate(all_data):
                    # Check if data exists and all values equal root rank
                    if d is None or not np.all(d == expected_val):
                        is_correct = False
                        break
        except Exception:
            is_correct = False
            
    else:
        # Gather Verification
        if rank == root:
            if result_data is None:
                is_correct = False
            else:
                # We expect to receive arrays from all ranks 0 to size-1
                # Flatten or iterate to find unique values
                received_ranks = set()
                
                # Helper to extract rank from a data chunk
                def get_rank_from_chunk(chunk):
                    if isinstance(chunk, np.ndarray) and chunk.size > 0:
                        return int(chunk[0])
                    return None

                if isinstance(result_data, list):
                    for chunk in result_data:
                        r = get_rank_from_chunk(chunk)
                        if r is not None:
                            received_ranks.add(r)
                elif isinstance(result_data, np.ndarray):
                    # If flattened, we might need to reshape or just check unique values
                    # Assuming data_size >= 1
                    uniques = np.unique(result_data)
                    for u in uniques:
                        received_ranks.add(int(u))
                
                # Check if we have all ranks from 0 to size-1
                expected_ranks = set(range(size))
                if received_ranks != expected_ranks:
                    is_correct = False
                     
    return avg_time, max_steps, total_msgs, is_correct
                     
    return avg_time, max_steps, total_msgs, is_correct

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    data_sizes = [1000, 100000, 1000000, 10000000] # 1K, 100K, 1M, 10M floats
    
    if rank == 0:
        print("="*100)
        print(f"ABLATION STUDY: DOR vs FLOODING (Processes: {size})")
        print("="*100)
        print(f"{'Data Size':<10} | {'Mesh':<5} | {'Op':<10} | {'Algo':<10} | {'Time (s)':<10} | {'Steps':<5} | {'Msgs':<5} | {'Verified'}")
        print("-" * 100)
        
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
    
    for d_size in data_sizes:
        for mesh_type, op, algo in configs:
            try:
                # Skip 3D if not enough processes
                if mesh_type == '3D' and size < 8:
                    continue
                    
                t, s, m, verified = run_experiment(comm, mesh_type, op, algo, data_size=d_size)
                if rank == 0:
                    status = "PASS" if verified else "FAIL"
                    print(f"{d_size:<10} | {mesh_type:<5} | {op:<10} | {algo:<10} | {t:.6f}   | {s:<5} | {m:<5} | {status}")
            except Exception as e:
                if rank == 0:
                    print(f"{d_size:<10} | {mesh_type:<5} | {op:<10} | {algo:<10} | FAILED ({str(e)})")
        if rank == 0:
            print("-" * 100)
