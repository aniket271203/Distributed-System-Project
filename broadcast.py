"""
Broadcast (Bcast) Implementation on 2D and 3D Mesh Topologies
Author: Aniket Gupta
"""

from mpi4py import MPI
import time
import numpy as np
from mesh_topology import Mesh2D, Mesh3D


def broadcast_2d_mesh(mesh, data, root=0):
    """
    Broadcast on 2D Mesh
    Algorithm:
    1. Broadcast M along the root's row
    2. Row nodes become column leaders
    3. Each leader broadcasts M down its column
    
    Runtime: T_bcast^2D = 2(√p - 1)ts + (p - 1)tw*m
    """
    comm = mesh.comm
    rank = mesh.get_rank()
    
    start_time = time.time()
    messages_sent = 0
    
    # Step 1: Broadcast along root's row
    root_coords = mesh._rank_to_coords(root)
    root_row = root_coords[0]
    root_col = root_coords[1]
    
    # All processes must participate in Split
    row_color = mesh.coords[0]
    row_comm = comm.Split(color=row_color, key=mesh.coords[1])
    
    if mesh.coords[0] == root_row:  # Process is in root's row
        data = row_comm.bcast(data, root=root_col)
        if rank == root:
            messages_sent += mesh.cols - 1
    
    row_comm.Free()
    
    # Step 2: Each column broadcasts from its leader (who got data in step 1)
    col_color = mesh.coords[1]
    col_comm = comm.Split(color=col_color, key=mesh.coords[0])
    
    # All processes in a column participate; the one in root_row has the data
    data = col_comm.bcast(data, root=root_row)
    if mesh.coords[0] == root_row and mesh.coords[1] != root_col:
        # Column leaders (except root) broadcast down
        messages_sent += mesh.rows - 1
    
    col_comm.Free()
    
    # Sequential hops = hops along row + hops along column
    # For 2D mesh: (cols-1) + (rows-1) sequential hops
    communication_steps = (mesh.cols - 1) + (mesh.rows - 1)
    
    # Synchronize to ensure all processes have received data
    comm.Barrier()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    return data, elapsed_time, communication_steps, messages_sent


def broadcast_3d_mesh(mesh, data, root=0):
    """
    Broadcast on 3D Mesh
    Algorithm:
    1. Broadcast along root's x-axis
    2. Broadcast along y-axis in root's plane
    3. Broadcast along z-axis from each plane leader
    
    Runtime: T_bcast^3D = 3(∛p - 1)ts + (p - 1)tw*m
    """
    comm = mesh.comm
    rank = mesh.get_rank()
    size = mesh.get_size()
    
    # For processes outside the cube, just broadcast using standard MPI
    if mesh.coords is None:
        data = comm.bcast(data, root=root)
        comm.Barrier()
        return data, 0, 0, 0
    
    start_time = time.time()
    messages_sent = 0
    
    root_coords = mesh._rank_to_coords(root)
    root_x, root_y, root_z = root_coords
    
    # Step 1: Broadcast along x-axis - all processes in mesh participate
    x_color = mesh.coords[1] * mesh.z_dim + mesh.coords[2]
    x_comm = comm.Split(color=x_color, key=mesh.coords[0])
    
    # Broadcast along each x-line (those with matching y,z)
    data = x_comm.bcast(data if (mesh.coords[1] == root_y and mesh.coords[2] == root_z) else None, root=root_x)
    if mesh.coords[0] == root_x and mesh.coords[1] == root_y and mesh.coords[2] == root_z:
        messages_sent += mesh.x_dim - 1
    
    x_comm.Free()
    
    # Step 2: Broadcast along y-axis - all processes in mesh participate
    y_color = mesh.coords[0] * mesh.z_dim + mesh.coords[2]
    y_comm = comm.Split(color=y_color, key=mesh.coords[1])
    
    # Broadcast along each y-line (those with matching x,z)
    data = y_comm.bcast(data if mesh.coords[2] == root_z else None, root=root_y)
    if mesh.coords[1] == root_y and mesh.coords[2] == root_z and mesh.coords[0] < mesh.x_dim:
        messages_sent += mesh.y_dim - 1
    
    y_comm.Free()
    
    # Step 3: Broadcast along z-axis - all processes in mesh participate
    z_color = mesh.coords[0] * mesh.y_dim + mesh.coords[1]
    z_comm = comm.Split(color=z_color, key=mesh.coords[2])
    
    # Broadcast along each z-line (those with matching x,y)
    data = z_comm.bcast(data, root=root_z)
    if mesh.coords[2] == root_z and mesh.coords[0] < mesh.x_dim and mesh.coords[1] < mesh.y_dim:
        messages_sent += mesh.z_dim - 1
    
    z_comm.Free()
    
    # Sequential hops = hops along x + hops along y + hops along z
    # For 3D mesh: (x_dim-1) + (y_dim-1) + (z_dim-1) sequential hops
    communication_steps = (mesh.x_dim - 1) + (mesh.y_dim - 1) + (mesh.z_dim - 1)
    
    # Synchronize to ensure all processes have received data
    comm.Barrier()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    return data, elapsed_time, communication_steps, messages_sent


def measure_broadcast_performance(mesh_type='2D', data_size=1000, root=0):
    """
    Measure broadcast performance with latency-bandwidth model
    T = ts + tw * m
    where ts = latency, tw = time per word, m = message size
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Create mesh topology
    if mesh_type == '2D':
        mesh = Mesh2D(comm)
    else:
        mesh = Mesh3D(comm)
    
    # Create data to broadcast
    if rank == root:
        data = np.random.rand(data_size)
    else:
        data = None
    
    # Perform broadcast
    if mesh_type == '2D':
        data, elapsed_time, comm_steps, msgs_sent = broadcast_2d_mesh(mesh, data, root)
    else:
        data, elapsed_time, comm_steps, msgs_sent = broadcast_3d_mesh(mesh, data, root)
    
    # Gather statistics at root
    all_times = comm.gather(elapsed_time, root=root)
    all_steps = comm.gather(comm_steps, root=root)
    
    if rank == root:
        avg_time = np.mean(all_times)
        max_time = np.max(all_times)
        total_steps = max(all_steps)
        
        print(f"\n{'='*60}")
        print(f"Broadcast Performance on {mesh_type} Mesh")
        print(f"{'='*60}")
        print(f"Number of processes: {comm.Get_size()}")
        print(f"Data size: {data_size} elements")
        print(f"Average time: {avg_time:.6f} seconds")
        print(f"Maximum time: {max_time:.6f} seconds")
        print(f"Communication steps: {total_steps}")
        print(f"Messages sent from root: {msgs_sent}")
        
        # Theoretical analysis
        p = comm.Get_size()
        if mesh_type == '2D':
            grid_size = int(np.sqrt(p))
            theoretical_steps = 2 * (grid_size - 1)
            print(f"Theoretical steps (2D): 2(√p - 1) = {theoretical_steps}")
        else:
            grid_size = int(round(p ** (1/3)))
            theoretical_steps = 3 * (grid_size - 1)
            print(f"Theoretical steps (3D): 3(∛p - 1) = {theoretical_steps}")
        
        print(f"{'='*60}\n")
    
    return data


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("\n" + "="*60)
        print("BROADCAST OPERATIONS ON MESH TOPOLOGIES")
        print("="*60)
    
    # Test 2D Mesh Broadcast
    if rank == 0:
        print("\nTesting 2D Mesh Broadcast...")
    measure_broadcast_performance(mesh_type='2D', data_size=1000, root=0)
    
    # Test 3D Mesh Broadcast if we have enough processes
    if size >= 8:
        if rank == 0:
            print("\nTesting 3D Mesh Broadcast...")
        measure_broadcast_performance(mesh_type='3D', data_size=1000, root=0)
    
    if rank == 0:
        print("\nBroadcast tests completed successfully!")
