"""
Gather Implementation on 2D and 3D Mesh Topologies
Author: Aniket Gupta
"""

from mpi4py import MPI
import time
import numpy as np
from mesh_topology import Mesh2D, Mesh3D


def gather_2d_mesh(mesh, data, root=0):
    """
    Gather on 2D Mesh
    Algorithm:
    1. Each row gathers data toward row leaders (first column)
    2. Row leaders send data up the root column to root
    
    Runtime: T_gather^2D = 2(√p - 1)ts + (p - 1)tw*m
    """
    comm = mesh.comm
    rank = mesh.get_rank()
    
    start_time = time.time()
    communication_steps = 0
    messages_received = 0
    
    root_coords = mesh._rank_to_coords(root)
    root_row = root_coords[0]
    root_col = root_coords[1]
    
    # Step 1: Gather data along each row to row leaders (column 0)
    row_comm = comm.Split(color=mesh.coords[0], key=mesh.coords[1])
    row_data = row_comm.gather(data, root=0)
    communication_steps += mesh.cols - 1
    if mesh.coords[1] == 0:  # Row leader
        messages_received += mesh.cols - 1
    row_comm.Free()
    
    # Step 2: All processes participate, but only column 0 processes have data
    col_comm = comm.Split(color=mesh.coords[1], key=mesh.coords[0])
    if mesh.coords[1] == 0:  # Only row leaders gather
        gathered_data = col_comm.gather(row_data, root=root_row)
        communication_steps += mesh.rows - 1
        if mesh.coords[0] == root_row:  # Root process
            messages_received += mesh.rows - 1
    else:
        gathered_data = None
    col_comm.Free()
    
    # Synchronize
    comm.Barrier()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Flatten the gathered data at root
    if rank == root and gathered_data is not None:
        flat_data = []
        for row_list in gathered_data:
            if row_list is not None:
                if isinstance(row_list, list):
                    flat_data.extend(row_list)
                else:
                    flat_data.append(row_list)
        gathered_data = flat_data
    
    return gathered_data, elapsed_time, communication_steps, messages_received


def gather_3d_mesh(mesh, data, root=0):
    """
    Gather on 3D Mesh
    Algorithm:
    1. Gather along x-axis to x=0 plane leaders
    2. Gather along y-axis to (x=0, y=0) line leaders
    3. Gather along z-axis to root
    
    Runtime: T_gather^3D = 3(∛p - 1)ts + (p - 1)tw*m
    """
    comm = mesh.comm
    rank = mesh.get_rank()
    size = mesh.get_size()
    
    # For processes outside the cube, just use standard MPI gather
    if mesh.coords is None:
        gathered = comm.gather(data, root=root)
        comm.Barrier()
        return gathered if rank == root else None, 0, 0, 0
    
    start_time = time.time()
    communication_steps = 0
    messages_received = 0
    
    root_coords = mesh._rank_to_coords(root)
    root_x, root_y, root_z = root_coords
    
    # Step 1: Gather along x-axis to x=0
    x_comm = comm.Split(color=mesh.coords[1] * mesh.z_dim + mesh.coords[2], key=mesh.coords[0])
    x_data = x_comm.gather(data, root=0)
    communication_steps += mesh.x_dim - 1
    if mesh.coords[0] == 0:
        messages_received += mesh.x_dim - 1
    x_comm.Free()
    
    # Step 2: All processes participate, but only x=0 processes have data
    y_comm = comm.Split(color=mesh.coords[0] * mesh.z_dim + mesh.coords[2], key=mesh.coords[1])
    if mesh.coords[0] == 0:
        y_data = y_comm.gather(x_data, root=0)
        communication_steps += mesh.y_dim - 1
        if mesh.coords[1] == 0:
            messages_received += mesh.y_dim - 1
    else:
        y_data = None
    y_comm.Free()
    
    # Step 3: All processes participate, but only x=0,y=0 processes have data
    z_comm = comm.Split(color=mesh.coords[0] * mesh.y_dim + mesh.coords[1], key=mesh.coords[2])
    if mesh.coords[0] == 0 and mesh.coords[1] == 0:
        gathered_data = z_comm.gather(y_data, root=root_z)
        communication_steps += mesh.z_dim - 1
        if mesh.coords[2] == root_z:
            messages_received += mesh.z_dim - 1
    else:
        gathered_data = None
    z_comm.Free()
    
    # Synchronize
    comm.Barrier()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Flatten the gathered data at root
    if rank == root and gathered_data is not None:
        flat_data = []
        for z_list in gathered_data:
            if z_list is not None:
                if isinstance(z_list, list):
                    for y_list in z_list:
                        if y_list is not None:
                            if isinstance(y_list, list):
                                flat_data.extend(y_list)
                            else:
                                flat_data.append(y_list)
                else:
                    flat_data.append(z_list)
        gathered_data = flat_data
    
    return gathered_data, elapsed_time, communication_steps, messages_received


def measure_gather_performance(mesh_type='2D', data_size=1000, root=0):
    """Measure gather performance and return structured metrics (time, messages, rounds)."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Create mesh topology
    if mesh_type == '2D':
        mesh = Mesh2D(comm)
    else:
        mesh = Mesh3D(comm)

    # Per-process data (scaled by rank to differentiate)
    data = np.random.rand(data_size) * rank

    # Perform gather
    if mesh_type == '2D':
        gathered_data, elapsed_time, comm_steps, msgs_recv = gather_2d_mesh(mesh, data, root)
    else:
        gathered_data, elapsed_time, comm_steps, msgs_recv = gather_3d_mesh(mesh, data, root)

    # Collect stats
    all_times = comm.gather(elapsed_time, root=root)
    all_steps = comm.gather(comm_steps, root=root)
    all_msgs = comm.gather(msgs_recv, root=root)

    if rank == root:
        avg_time = float(np.mean(all_times))
        max_time = float(np.max(all_times))
        total_rounds = int(max(all_steps))
        total_messages = int(np.sum(all_msgs))

        p = comm.Get_size()
        if mesh_type == '2D':
            grid_dim = int(np.sqrt(p))
            theoretical_rounds = 2 * (grid_dim - 1)
        else:
            grid_dim = int(round(p ** (1/3)))
            theoretical_rounds = 3 * (grid_dim - 1)
        minimal_messages = p - 1

        metrics = {
            'collective': 'gather',
            'mesh_type': mesh_type,
            'processes': p,
            'data_size_per_process': data_size,
            'avg_time_sec': avg_time,
            'max_time_sec': max_time,
            'rounds_measured': total_rounds,
            'rounds_theoretical': theoretical_rounds,
            'messages_counted': total_messages,
            'messages_minimal': minimal_messages,
            'total_data_elements': len(gathered_data) if gathered_data is not None else None
        }
    else:
        metrics = None

    return metrics


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("\n" + "="*60)
        print("GATHER OPERATIONS ON MESH TOPOLOGIES")
        print("="*60)
    
    tests = [('2D', 100)]
    if size >= 8:
        tests.append(('3D', 100))
    for mesh_type, dsize in tests:
        if rank == 0:
            print(f"\nTesting {mesh_type} Mesh Gather (data_size={dsize})...")
        measure_gather_performance(mesh_type=mesh_type, data_size=dsize, root=0)
    
    if rank == 0:
        print("\nGather tests completed successfully!")
