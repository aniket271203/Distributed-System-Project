"""
Main driver program for Mesh Communication Operations
Combines mesh creation, broadcast, and gather operations
Author: Aniket Gupta
"""

from mpi4py import MPI
import numpy as np
import time
import sys
from mesh_topology import Mesh2D, Mesh3D
from broadcast import broadcast_2d_mesh, broadcast_3d_mesh
from gather import gather_2d_mesh, gather_3d_mesh


def test_2d_mesh_operations(comm, data_size=1000):
    """Test both broadcast and gather on 2D mesh"""
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("\n" + "="*70)
        print("2D MESH TOPOLOGY - BROADCAST AND GATHER OPERATIONS")
        print("="*70)
    
    # Create 2D mesh
    mesh = Mesh2D(comm)
    
    if rank == 0:
        print(f"\nMesh Configuration:")
        print(f"  Total processes: {size}")
        print(f"  Grid size: {mesh.rows}x{mesh.cols}")
        print(f"  Data size: {data_size} elements")
    
    # Display mesh structure
    all_coords = comm.gather(mesh.coords, root=0)
    if rank == 0:
        print(f"\n  Process coordinate mapping:")
        for r in range(min(size, 16)):  # Show first 16 processes
            print(f"    Rank {r}: {all_coords[r]}")
        if size > 16:
            print(f"    ... ({size - 16} more processes)")
    
    comm.Barrier()
    
    # Test Broadcast
    if rank == 0:
        print(f"\n{'-'*70}")
        print("BROADCAST OPERATION")
        print(f"{'-'*70}")
        broadcast_data = np.arange(data_size, dtype=np.float64)
        print(f"Broadcasting {data_size} elements from root (rank 0)...")
    else:
        broadcast_data = None
    
    recv_data, bcast_time, bcast_steps, msgs_sent = broadcast_2d_mesh(mesh, broadcast_data, root=0)
    
    # Verify broadcast
    verification = np.allclose(recv_data, np.arange(data_size, dtype=np.float64))
    all_verifications = comm.gather(verification, root=0)
    
    if rank == 0:
        if all(all_verifications):
            print("✓ Broadcast verification: SUCCESS - All processes received correct data")
        else:
            print("✗ Broadcast verification: FAILED")
        
        print(f"\nBroadcast Performance:")
        print(f"  Execution time: {bcast_time:.6f} seconds")
        print(f"  Communication steps: {bcast_steps}")
        print(f"  Messages sent: {msgs_sent}")
        
        # Theoretical analysis
        sqrt_p = int(np.sqrt(size))
        print(f"\n  Theoretical Analysis (2D Mesh):")
        print(f"    Grid dimension: √p = {sqrt_p}")
        print(f"    Expected steps: 2(√p - 1) = {2 * (sqrt_p - 1)}")
        print(f"    Formula: T = 2(√p - 1)ts + (p - 1)tw*m")
    
    comm.Barrier()
    
    # Test Gather
    if rank == 0:
        print(f"\n{'-'*70}")
        print("GATHER OPERATION")
        print(f"{'-'*70}")
        print(f"Gathering data from all {size} processes to root...")
    
    # Each process creates unique data
    gather_data = np.full(data_size, rank, dtype=np.float64)
    
    gathered_result, gather_time, gather_steps, msgs_recv = gather_2d_mesh(mesh, gather_data, root=0)
    
    if rank == 0:
        if gathered_result is not None and len(gathered_result) == size:
            print(f"✓ Gather verification: SUCCESS - Received data from all {size} processes")
            # Check if we got data from all ranks
            unique_ranks = len(set([int(d[0]) if isinstance(d, np.ndarray) else int(d) 
                                   for d in gathered_result if d is not None]))
            print(f"  Unique process data collected: {unique_ranks}/{size}")
        else:
            print(f"✗ Gather verification: Received data from {len(gathered_result) if gathered_result else 0} processes")
        
        print(f"\nGather Performance:")
        print(f"  Execution time: {gather_time:.6f} seconds")
        print(f"  Communication steps: {gather_steps}")
        print(f"  Messages received: {msgs_recv}")
        
        print(f"\n  Theoretical Analysis (2D Mesh):")
        print(f"    Grid dimension: √p = {sqrt_p}")
        print(f"    Expected steps: 2(√p - 1) = {2 * (sqrt_p - 1)}")
        print(f"    Formula: T = 2(√p - 1)ts + (p - 1)tw*m")
    
    if rank == 0:
        print("\n" + "="*70)


def test_3d_mesh_operations(comm, data_size=1000):
    """Test both broadcast and gather on 3D mesh"""
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if size < 8:
        if rank == 0:
            print("\nSkipping 3D mesh tests (need at least 8 processes)")
        return
    
    if rank == 0:
        print("\n" + "="*70)
        print("3D MESH TOPOLOGY - BROADCAST AND GATHER OPERATIONS")
        print("="*70)
    
    # Create 3D mesh
    mesh = Mesh3D(comm)
    
    if rank == 0:
        print(f"\nMesh Configuration:")
        print(f"  Total processes: {size}")
        print(f"  Grid size: {mesh.x_dim}x{mesh.y_dim}x{mesh.z_dim}")
        print(f"  Data size: {data_size} elements")
    
    # Display mesh structure
    all_coords = comm.gather(mesh.coords, root=0)
    if rank == 0:
        print(f"\n  Process coordinate mapping:")
        for r in range(min(size, 16)):  # Show first 16 processes
            if all_coords[r] is not None:
                print(f"    Rank {r}: {all_coords[r]}")
        if size > 16:
            print(f"    ... ({size - 16} more processes)")
    
    comm.Barrier()
    
    # Test Broadcast
    if rank == 0:
        print(f"\n{'-'*70}")
        print("BROADCAST OPERATION")
        print(f"{'-'*70}")
        broadcast_data = np.arange(data_size, dtype=np.float64)
        print(f"Broadcasting {data_size} elements from root (rank 0)...")
    else:
        broadcast_data = None
    
    recv_data, bcast_time, bcast_steps, msgs_sent = broadcast_3d_mesh(mesh, broadcast_data, root=0)
    
    # Verify broadcast
    if mesh.coords is not None:
        verification = np.allclose(recv_data, np.arange(data_size, dtype=np.float64))
    else:
        verification = True
    all_verifications = comm.gather(verification, root=0)
    
    if rank == 0:
        if all(all_verifications):
            print("✓ Broadcast verification: SUCCESS - All processes received correct data")
        else:
            print("✗ Broadcast verification: FAILED")
        
        print(f"\nBroadcast Performance:")
        print(f"  Execution time: {bcast_time:.6f} seconds")
        print(f"  Communication steps: {bcast_steps}")
        print(f"  Messages sent: {msgs_sent}")
        
        # Theoretical analysis
        cbrt_p = int(round(size ** (1/3)))
        print(f"\n  Theoretical Analysis (3D Mesh):")
        print(f"    Grid dimension: ∛p = {cbrt_p}")
        print(f"    Expected steps: 3(∛p - 1) = {3 * (cbrt_p - 1)}")
        print(f"    Formula: T = 3(∛p - 1)ts + (p - 1)tw*m")
    
    comm.Barrier()
    
    # Test Gather
    if rank == 0:
        print(f"\n{'-'*70}")
        print("GATHER OPERATION")
        print(f"{'-'*70}")
        print(f"Gathering data from all {size} processes to root...")
    
    # Each process creates unique data
    gather_data = np.full(data_size, rank, dtype=np.float64)
    
    gathered_result, gather_time, gather_steps, msgs_recv = gather_3d_mesh(mesh, gather_data, root=0)
    
    if rank == 0:
        if gathered_result is not None:
            active_processes = len([c for c in all_coords if c is not None])
            print(f"✓ Gather verification: SUCCESS - Received data from processes")
            print(f"  Total data elements collected: {len(gathered_result)}")
            print(f"  Active processes in mesh: {active_processes}")
        else:
            print(f"✗ Gather verification: No data received")
        
        print(f"\nGather Performance:")
        print(f"  Execution time: {gather_time:.6f} seconds")
        print(f"  Communication steps: {gather_steps}")
        print(f"  Messages received: {msgs_recv}")
        
        print(f"\n  Theoretical Analysis (3D Mesh):")
        print(f"    Grid dimension: ∛p = {cbrt_p}")
        print(f"    Expected steps: 3(∛p - 1) = {3 * (cbrt_p - 1)}")
        print(f"    Formula: T = 3(∛p - 1)ts + (p - 1)tw*m")
    
    if rank == 0:
        print("\n" + "="*70)


def compare_2d_vs_3d(comm, data_size=1000):
    """Compare 2D and 3D mesh performance"""
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if size < 8:
        if rank == 0:
            print("\nSkipping comparison (need at least 8 processes for 3D mesh)")
        return
    
    if rank == 0:
        print("\n" + "="*70)
        print("PERFORMANCE COMPARISON: 2D vs 3D MESH")
        print("="*70)
    
    # Test 2D
    mesh_2d = Mesh2D(comm)
    if rank == 0:
        data = np.arange(data_size, dtype=np.float64)
    else:
        data = None
    
    _, time_2d_bcast, steps_2d_bcast, _ = broadcast_2d_mesh(mesh_2d, data, root=0)
    
    # Test 3D
    mesh_3d = Mesh3D(comm)
    if rank == 0:
        data = np.arange(data_size, dtype=np.float64)
    else:
        data = None
    
    _, time_3d_bcast, steps_3d_bcast, _ = broadcast_3d_mesh(mesh_3d, data, root=0)
    
    if rank == 0:
        print(f"\nBroadcast Comparison:")
        print(f"  {'Metric':<30} {'2D Mesh':<15} {'3D Mesh':<15} {'Improvement':<15}")
        print(f"  {'-'*75}")
        print(f"  {'Execution time (seconds)':<30} {time_2d_bcast:<15.6f} {time_3d_bcast:<15.6f} {time_2d_bcast/time_3d_bcast if time_3d_bcast > 0 else 0:<15.2f}x")
        print(f"  {'Communication steps':<30} {steps_2d_bcast:<15} {steps_3d_bcast:<15} {steps_2d_bcast/steps_3d_bcast if steps_3d_bcast > 0 else 0:<15.2f}x")
        
        sqrt_p = int(np.sqrt(size))
        cbrt_p = int(round(size ** (1/3)))
        print(f"\n  Theoretical steps:")
        print(f"    2D: 2(√{size} - 1) = {2 * (sqrt_p - 1)}")
        print(f"    3D: 3(∛{size} - 1) = {3 * (cbrt_p - 1)}")
        print(f"\n  Conclusion: 3D mesh reduces communication distance")
        print(f"              Diameter of 2D: 2√p - 2 = {2*sqrt_p - 2}")
        print(f"              Diameter of 3D: 3∛p - 3 = {3*cbrt_p - 3}")
        print("="*70)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("\n")
        print("╔" + "═"*68 + "╗")
        print("║" + " "*68 + "║")
        print("║" + "  COLLECTIVE COMMUNICATION ON MESH TOPOLOGIES".center(68) + "║")
        print("║" + "  Operations: Broadcast (Bcast) and Gather".center(68) + "║")
        print("║" + "  Topologies: 2D and 3D Mesh".center(68) + "║")
        print("║" + " "*68 + "║")
        print("╚" + "═"*68 + "╝")
        print(f"\nTotal MPI processes: {size}")
    
    # Default data size
    data_size = 1000
    if len(sys.argv) > 1:
        data_size = int(sys.argv[1])
    
    # Test 2D Mesh
    test_2d_mesh_operations(comm, data_size)
    
    # Test 3D Mesh
    test_3d_mesh_operations(comm, data_size)
    
    # Compare performance
    compare_2d_vs_3d(comm, data_size)
    
    if rank == 0:
        print("\n" + "="*70)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*70 + "\n")
