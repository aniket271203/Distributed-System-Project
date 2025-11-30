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
    
    # Track which nodes have been visited to avoid duplicate receives
    visited = set()
    if rank == root:
        visited.add(rank)
    
    # Max possible distance is diameter
    for level in range(max_dist):
        # If I have data and am at current level, send to ALL neighbors (flooding behavior)
        # This is the key difference from DOR - we flood to all neighbors, not just optimal path
        if dist == level and received:
            neighbors = get_neighbors(mesh, rank)
            for neighbor in neighbors:
                # In true flooding, we send to ALL neighbors regardless of their distance
                # This creates redundant messages but ensures delivery
                comm.send(data, dest=neighbor, tag=level)
                msgs_sent += 1
        
        # If I haven't received yet and am at level+1, receive from a neighbor at level
        elif dist == level + 1 and not received:
            # Receive from ANY neighbor at 'level'
            # We only need one copy but may receive multiple in true flooding
            status = MPI.Status()
            data = comm.recv(source=MPI.ANY_SOURCE, tag=level, status=status)
            received = True
            visited.add(rank)
        
        # Handle redundant messages - nodes that already have data still receive
        # to prevent deadlock (sender is waiting, receiver must accept)
        elif received and dist > 0:
            # Check if any neighbor at level is trying to send to us
            neighbors = get_neighbors(mesh, rank)
            for neighbor in neighbors:
                n_coords = mesh._rank_to_coords(neighbor)
                if isinstance(mesh, Mesh2D):
                    n_dist = abs(root_coords[0] - n_coords[0]) + abs(root_coords[1] - n_coords[1])
                else:
                    n_dist = abs(root_coords[0] - n_coords[0]) + abs(root_coords[1] - n_coords[1]) + abs(root_coords[2] - n_coords[2])
                
                # If neighbor is at current level and would flood to us
                if n_dist == level:
                    # Use non-blocking probe to check for incoming message
                    if comm.Iprobe(source=neighbor, tag=level):
                        _ = comm.recv(source=neighbor, tag=level)  # Discard duplicate
            
        # Synchronization to define a "step" clearly for analysis
        # In real flooding, this barrier isn't needed, but helps measure 'steps' as 'levels'
        comm.Barrier()
        steps += 1
        
        # Optimization: Check if everyone is done (not easy in distributed, but we know max_dist)
        if level > max_dist: 
            break
    
    # Calculate actual sequential hops (max Manhattan distance from root)
    # For flooding, this equals the diameter of the mesh from root
    if isinstance(mesh, Mesh2D):
        # Max distance to any corner from root
        corners = [(0, 0), (mesh.rows-1, 0), (0, mesh.cols-1), (mesh.rows-1, mesh.cols-1)]
        max_hops = max(abs(c[0] - root_coords[0]) + abs(c[1] - root_coords[1]) for c in corners)
    else:
        corners = [
            (0, 0, 0), (mesh.x_dim-1, 0, 0), (0, mesh.y_dim-1, 0), (0, 0, mesh.z_dim-1),
            (mesh.x_dim-1, mesh.y_dim-1, 0), (mesh.x_dim-1, 0, mesh.z_dim-1), 
            (0, mesh.y_dim-1, mesh.z_dim-1), (mesh.x_dim-1, mesh.y_dim-1, mesh.z_dim-1)
        ]
        max_hops = max(abs(c[0] - root_coords[0]) + abs(c[1] - root_coords[1]) + abs(c[2] - root_coords[2]) for c in corners)

    end_time = time.time()
    return data, end_time - start_time, max_hops, msgs_sent

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
    msgs_sent = 0  # Track messages sent (for flooding, each node sends to ALL neighbors)
    
    # Reverse levels: from max_dist down to 1
    # Leaves send first
    for level in range(max_dist, 0, -1):
        # If I am at 'level', send to ALL neighbors (flooding behavior)
        # In true flooding gather, each node sends to all neighbors, not just parent
        if dist == level:
            neighbors = get_neighbors(mesh, rank)
            for neighbor in neighbors:
                n_coords = mesh._rank_to_coords(neighbor)
                if isinstance(mesh, Mesh2D):
                    n_dist = abs(root_coords[0] - n_coords[0]) + abs(root_coords[1] - n_coords[1])
                else:
                    n_dist = abs(root_coords[0] - n_coords[0]) + abs(root_coords[1] - n_coords[1]) + abs(root_coords[2] - n_coords[2])
                
                # Send to parent (neighbor closer to root)
                if n_dist == level - 1:
                    comm.send(my_gathered_data, dest=neighbor, tag=level)
                    msgs_sent += 1
        
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
                    # Expect message from this child
                    child_data = comm.recv(source=neighbor, tag=level)
                    my_gathered_data.extend(child_data)
                    msgs_recv += 1
        
        comm.Barrier()
        steps += 1
    
    # Calculate actual sequential hops (max Manhattan distance from root)
    if isinstance(mesh, Mesh2D):
        corners = [(0, 0), (mesh.rows-1, 0), (0, mesh.cols-1), (mesh.rows-1, mesh.cols-1)]
        max_hops = max(abs(c[0] - root_coords[0]) + abs(c[1] - root_coords[1]) for c in corners)
    else:
        corners = [
            (0, 0, 0), (mesh.x_dim-1, 0, 0), (0, mesh.y_dim-1, 0), (0, 0, mesh.z_dim-1),
            (mesh.x_dim-1, mesh.y_dim-1, 0), (mesh.x_dim-1, 0, mesh.z_dim-1), 
            (0, mesh.y_dim-1, mesh.z_dim-1), (mesh.x_dim-1, mesh.y_dim-1, mesh.z_dim-1)
        ]
        max_hops = max(abs(c[0] - root_coords[0]) + abs(c[1] - root_coords[1]) + abs(c[2] - root_coords[2]) for c in corners)

    end_time = time.time()
    
    # Root holds all data
    final_data = my_gathered_data if rank == root else None
    
    # Return msgs_sent instead of msgs_recv to show flooding message complexity
    return final_data, end_time - start_time, max_hops, msgs_sent
