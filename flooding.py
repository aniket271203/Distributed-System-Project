"""
Flooding (BFS) Implementation on 2D and 3D Mesh Topologies
Author: Aniket Gupta
"""

from mpi4py import MPI
import time
import numpy as np
from mesh_topology import Mesh2D, Mesh3D

def get_neighbors(mesh, rank):
    """Get all valid neighbors for a rank in the mesh"""
    neighbors = []
    coords = mesh._rank_to_coords(rank)
    
    if isinstance(mesh, Mesh2D):
        r, c = coords
        # Check Up, Down, Left, Right
        candidates = [
            (r-1, c), (r+1, c), (r, c-1), (r, c+1)
        ]
        for nr, nc in candidates:
            if 0 <= nr < mesh.rows and 0 <= nc < mesh.cols:
                n_rank = mesh._coords_to_rank(nr, nc)
                if n_rank is not None:
                    neighbors.append(n_rank)
                
    elif isinstance(mesh, Mesh3D):
        x, y, z = coords
        # Check 6 directions
        candidates = [
            (x-1, y, z), (x+1, y, z),
            (x, y-1, z), (x, y+1, z),
            (x, y, z-1), (x, y, z+1)
        ]
        for nx, ny, nz in candidates:
            if 0 <= nx < mesh.x_dim and 0 <= ny < mesh.y_dim and 0 <= nz < mesh.z_dim:
                n_rank = mesh._coords_to_rank(nx, ny, nz)
                if n_rank is not None:
                    neighbors.append(n_rank)
                
    return neighbors

def broadcast_flooding(mesh, data, root=0):
    """
    Broadcast using Flooding (BFS)
    Algorithm:
    - Root sends to neighbors (Level 1)
    - Level 1 nodes send to their unvisited neighbors (Level 2)
    - ... and so on until all nodes receive
    
    Note: In a real async network, this would be faster. 
    Here we simulate levels to avoid message storms and ensure correctness.
    """
    comm = mesh.comm
    rank = mesh.get_rank()
    size = mesh.get_size()
    
    # Handle processes outside mesh
    if mesh.coords is None:
        data = comm.bcast(data, root=root)
        return data, 0, 0, 0

    start_time = time.time()
    steps = 0
    msgs_sent = 0
    
    # BFS Level calculation (to know when to receive/send)
    # We pre-calculate the distance from root to know our "turn"
    # In a real system, this would be dynamic.
    root_coords = mesh._rank_to_coords(root)
    my_coords = mesh.coords
    
    if isinstance(mesh, Mesh2D):
        dist = abs(root_coords[0] - my_coords[0]) + abs(root_coords[1] - my_coords[1])
        max_dist = mesh.rows + mesh.cols
    else:
        dist = abs(root_coords[0] - my_coords[0]) + abs(root_coords[1] - my_coords[1]) + abs(root_coords[2] - my_coords[2])
        max_dist = mesh.x_dim + mesh.y_dim + mesh.z_dim

    # Level-by-level propagation
    # Level 0: Root has data
    # Level 1: Neighbors of root receive
    # ...
    
    received = False
    if rank == root:
        received = True
    
    # Max possible distance is diameter
    for level in range(max_dist):
        # If I have data and am at current level, send to neighbors at level+1
        if dist == level and received:
            neighbors = get_neighbors(mesh, rank)
            for neighbor in neighbors:
                # Calculate neighbor's distance to check if they are next level
                n_coords = mesh._rank_to_coords(neighbor)
                if isinstance(mesh, Mesh2D):
                    n_dist = abs(root_coords[0] - n_coords[0]) + abs(root_coords[1] - n_coords[1])
                else:
                    n_dist = abs(root_coords[0] - n_coords[0]) + abs(root_coords[1] - n_coords[1]) + abs(root_coords[2] - n_coords[2])
                
                if n_dist == level + 1:
                    comm.send(data, dest=neighbor, tag=level)
                    msgs_sent += 1
        
        # If I am at level+1, receive from a neighbor at level
        elif dist == level + 1:
            # Receive from ALL neighbors at distance 'level'
            expected_msgs = 0
            neighbors = get_neighbors(mesh, rank)
            for neighbor in neighbors:
                n_coords = mesh._rank_to_coords(neighbor)
                if isinstance(mesh, Mesh2D):
                    n_dist = abs(root_coords[0] - n_coords[0]) + abs(root_coords[1] - n_coords[1])
                else:
                    n_dist = abs(root_coords[0] - n_coords[0]) + abs(root_coords[1] - n_coords[1]) + abs(root_coords[2] - n_coords[2])
                
                if n_dist == level:
                    expected_msgs += 1
            
            for _ in range(expected_msgs):
                status = MPI.Status()
                data = comm.recv(source=MPI.ANY_SOURCE, tag=level, status=status)
                received = True
            
        # Synchronization to define a "step" clearly for analysis
        # In real flooding, this barrier isn't needed, but helps measure 'steps' as 'levels'
        comm.Barrier()
        steps += 1
        
        # Optimization: Check if everyone is done (not easy in distributed, but we know max_dist)
        if level > max_dist: 
            break

    end_time = time.time()
    return data, end_time - start_time, steps, msgs_sent

def gather_flooding(mesh, data, root=0):
    """
    Gather using Reverse Flooding (Convergecast)
    Algorithm:
    - Leaves (nodes furthest from root) send to parents
    - Parents aggregate and send up
    """
    comm = mesh.comm
    rank = mesh.get_rank()
    
    if mesh.coords is None:
        gathered = comm.gather(data, root=root)
        return gathered if rank == root else None, 0, 0, 0

    start_time = time.time()
    steps = 0
    msgs_recv = 0
    
    root_coords = mesh._rank_to_coords(root)
    my_coords = mesh.coords
    
    if isinstance(mesh, Mesh2D):
        dist = abs(root_coords[0] - my_coords[0]) + abs(root_coords[1] - my_coords[1])
        max_dist = mesh.rows + mesh.cols
    else:
        dist = abs(root_coords[0] - my_coords[0]) + abs(root_coords[1] - my_coords[1]) + abs(root_coords[2] - my_coords[2])
        max_dist = mesh.x_dim + mesh.y_dim + mesh.z_dim

    # Container for gathered data
    # Start with my own data
    my_gathered_data = [data] 
    
    # Reverse levels: from max_dist down to 1
    # Leaves send first
    for level in range(max_dist, 0, -1):
        # If I am at 'level', send to a parent at 'level-1'
        if dist == level:
            neighbors = get_neighbors(mesh, rank)
            # Find a parent (neighbor with dist = level - 1)
            parent = -1
            for neighbor in neighbors:
                n_coords = mesh._rank_to_coords(neighbor)
                if isinstance(mesh, Mesh2D):
                    n_dist = abs(root_coords[0] - n_coords[0]) + abs(root_coords[1] - n_coords[1])
                else:
                    n_dist = abs(root_coords[0] - n_coords[0]) + abs(root_coords[1] - n_coords[1]) + abs(root_coords[2] - n_coords[2])
                
                if n_dist == level - 1:
                    parent = neighbor
                    break
            
            if parent != -1:
                comm.send(my_gathered_data, dest=parent, tag=level)
        
        # If I am at 'level-1', receive from ALL children at 'level'
        elif dist == level - 1:
            neighbors = get_neighbors(mesh, rank)
            for neighbor in neighbors:
                n_coords = mesh._rank_to_coords(neighbor)
                if isinstance(mesh, Mesh2D):
                    n_dist = abs(root_coords[0] - n_coords[0]) + abs(root_coords[1] - n_coords[1])
                else:
                    n_dist = abs(root_coords[0] - n_coords[0]) + abs(root_coords[1] - n_coords[1]) + abs(root_coords[2] - n_coords[2])
                
                if n_dist == level:
                    # Check if this neighbor (child) actually picked ME as parent
                    # To do this, we simulate the child's parent selection logic
                    child_neighbors = get_neighbors(mesh, neighbor)
                    chosen_parent = -1
                    for cn in child_neighbors:
                        cn_coords = mesh._rank_to_coords(cn)
                        if isinstance(mesh, Mesh2D):
                            cn_dist = abs(root_coords[0] - cn_coords[0]) + abs(root_coords[1] - cn_coords[1])
                        else:
                            cn_dist = abs(root_coords[0] - cn_coords[0]) + abs(root_coords[1] - cn_coords[1]) + abs(root_coords[2] - cn_coords[2])
                        
                        if cn_dist == level - 1:
                            chosen_parent = cn
                            break
                    
                    # Only receive if I am the chosen parent
                    if chosen_parent == rank:
                        child_data = comm.recv(source=neighbor, tag=level)
                        my_gathered_data.extend(child_data)
                        msgs_recv += 1
        
        comm.Barrier()
        steps += 1

    end_time = time.time()
    
    # Root holds all data
    final_data = my_gathered_data if rank == root else None
    
    return final_data, end_time - start_time, steps, msgs_recv
