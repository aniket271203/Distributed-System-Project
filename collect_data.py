"""
Collect real performance data from MPI runs
Author: Aniket Gupta
"""

from mpi4py import MPI
import numpy as np
import json
import time
from mesh_topology import Mesh2D, Mesh3D
from broadcast import broadcast_2d_mesh, broadcast_3d_mesh, broadcast_2d_mesh_pipelined, broadcast_2d_mesh_binary_tree, broadcast_2d_mesh_flooding
from gather import gather_2d_mesh, gather_3d_mesh

def collect_performance_data():
    """Collect performance data for various configurations"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    results = {
        'process_count': size,
        'tests': []
    }
    
    # Test different data sizes
    data_sizes = [100, 500, 1000, 5000, 10000]
    
    if rank == 0:
        print(f"\nCollecting performance data with {size} processes...")
        print(f"Testing data sizes: {data_sizes}")
    
    for data_size in data_sizes:
        test_result = {
            'data_size': data_size,
            '2d_broadcast': None,
            '2d_gather': None,
            '3d_broadcast': None,
            '3d_gather': None
        }
        
        # 2D Mesh Tests
        mesh_2d = Mesh2D(comm)
        
        # 2D Broadcast
        if rank == 0:
            data = np.random.rand(data_size)
        else:
            data = None
        
        _, time_2d_bcast, steps_2d_bcast, msgs_2d_bcast = broadcast_2d_mesh(mesh_2d, data, root=0)
        
        # 2D Broadcast - Flooding
        if rank == 0:
            data_flood = np.random.rand(data_size) # Use a fresh data for flooding to ensure fair comparison
        else:
            data_flood = None
        _, time_flood, steps_flood, msgs_flood = broadcast_2d_mesh_flooding(mesh_2d, data_flood, root=0)
        
        # 2D Gather
        data_gather = np.random.rand(data_size) # Use fresh data for gather
        _, time_2d_gather, steps_2d_gather, msgs_2d_gather = gather_2d_mesh(mesh_2d, data_gather, root=0)
        
        if rank == 0:
            test_result['2d_broadcast'] = {
                'time': time_2d_bcast,
                'steps': steps_2d_bcast,
                'messages': msgs_2d_bcast
            }
            test_result['2d_broadcast_flooding'] = {
                'time': time_flood,
                'steps': steps_flood,
                'messages': msgs_flood
            }
            test_result['2d_gather'] = {
                'time': time_2d_gather,
                'steps': steps_2d_gather,
                'messages': msgs_2d_gather
            }
        
        # 3D Mesh Tests (if enough processes)
        if size >= 4: # Minimum 4 nodes for a reasonable 3D mesh (e.g. 1x2x2)
            mesh_3d = Mesh3D(comm)
            
            # 3D Broadcast
            if rank == 0:
                data = np.random.rand(data_size)
            else:
                data = None
            
            _, time_3d_bcast, steps_3d_bcast, msgs_3d_bcast = broadcast_3d_mesh(mesh_3d, data, root=0)
            
            # 3D Gather
            data = np.random.rand(data_size)
            _, time_3d_gather, steps_3d_gather, msgs_3d_gather = gather_3d_mesh(mesh_3d, data, root=0)
            
            if rank == 0:
                test_result['3d_broadcast'] = {
                    'time': time_3d_bcast,
                    'steps': steps_3d_bcast,
                    'messages': msgs_3d_bcast
                }
                test_result['3d_gather'] = {
                    'time': time_3d_gather,
                    'steps': steps_3d_gather,
                    'messages': msgs_3d_gather
                }
        
        if rank == 0:
            results['tests'].append(test_result)
            print(f"  ✓ Completed tests for data size: {data_size}")
    
    # Save results
    if rank == 0:
        with open(f'results/performance_data_p{size}.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Performance data saved to: results/performance_data_p{size}.json")
        
        # Print summary
        print("\n" + "="*70)
        print(f"PERFORMANCE SUMMARY ({size} processes)")
        print("="*70)
        print(f"\n{'Data Size':<12} {'2D Bcast':<15} {'2D Gather':<15} {'3D Bcast':<15} {'3D Gather':<15}")
        print("-"*70)
        
        for test in results['tests']:
            row = f"{test['data_size']:<12}"
            
            if test['2d_broadcast']:
                row += f"{test['2d_broadcast']['time']*1000:>13.3f} ms "
            else:
                row += f"{'N/A':<15}"
            
            if test['2d_gather']:
                row += f"{test['2d_gather']['time']*1000:>13.3f} ms "
            else:
                row += f"{'N/A':<15}"
            
            if test['3d_broadcast']:
                row += f"{test['3d_broadcast']['time']*1000:>13.3f} ms "
            else:
                row += f"{'N/A':<15}"
            
            if test['3d_gather']:
                row += f"{test['3d_gather']['time']*1000:>13.3f} ms"
            else:
                row += "N/A"
            
            print(row)
        
        print("="*70 + "\n")

if __name__ == "__main__":
    collect_performance_data()
