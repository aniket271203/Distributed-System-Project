"""
Mesh Topology Creation for 2D and 3D Meshes
Author: Aniket Gupta
"""

from mpi4py import MPI
import numpy as np
import math


class MeshTopology:
    """Base class for mesh topology"""
    
    def __init__(self, comm):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.coords = None
        self.neighbors = {}
    
    def get_rank(self):
        return self.rank
    
    def get_size(self):
        return self.size
    
    def get_coords(self):
        return self.coords
    
    def get_neighbors(self):
        return self.neighbors


class Mesh2D(MeshTopology):
    """2D Mesh Topology Implementation"""
    
    def __init__(self, comm):
        super().__init__(comm)
        # Calculate grid dimensions - find best factors for R * C = size
        self.rows, self.cols = self._find_best_2d_dims(self.size)
        
        if self.rank == 0:
            print(f"2D Mesh: {self.rows}x{self.cols} = {self.rows*self.cols} nodes")
        
        # Calculate coordinates for this rank
        self.coords = self._rank_to_coords(self.rank)
        
        # Calculate neighbors
        self._calculate_neighbors()
    
    def _find_best_2d_dims(self, size):
        """Find dimensions (r, c) such that r*c = size and |r-c| is minimized"""
        best_r, best_c = 1, size
        for r in range(1, int(math.sqrt(size)) + 1):
            if size % r == 0:
                c = size // r
                if abs(r - c) < abs(best_r - best_c):
                    best_r, best_c = r, c
        return best_r, best_c

    def _rank_to_coords(self, rank):
        """Convert rank to (row, col) coordinates"""
        row = rank // self.cols
        col = rank % self.cols
        return (row, col)
    
    def _coords_to_rank(self, row, col):
        """Convert (row, col) coordinates to rank"""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            rank = row * self.cols + col
            if rank < self.size:
                return rank
        return None
    
    def _calculate_neighbors(self):
        """Calculate neighbors for current rank"""
        row, col = self.coords
        
        # North neighbor (row-1, col)
        north = self._coords_to_rank(row - 1, col)
        if north is not None:
            self.neighbors['north'] = north
        
        # South neighbor (row+1, col)
        south = self._coords_to_rank(row + 1, col)
        if south is not None:
            self.neighbors['south'] = south
        
        # West neighbor (row, col-1)
        west = self._coords_to_rank(row, col - 1)
        if west is not None:
            self.neighbors['west'] = west
        
        # East neighbor (row, col+1)
        east = self._coords_to_rank(row, col + 1)
        if east is not None:
            self.neighbors['east'] = east
    
    def get_row_neighbors(self):
        """Get all neighbors in the same row"""
        row, col = self.coords
        neighbors = []
        for c in range(self.cols):
            rank = self._coords_to_rank(row, c)
            if rank is not None and rank != self.rank:
                neighbors.append(rank)
        return neighbors
    
    def get_column_neighbors(self):
        """Get all neighbors in the same column"""
        row, col = self.coords
        neighbors = []
        for r in range(self.rows):
            rank = self._coords_to_rank(r, col)
            if rank is not None and rank != self.rank:
                neighbors.append(rank)
        return neighbors
    
    def is_in_root_row(self, root=0):
        """Check if current process is in root's row"""
        root_coords = self._rank_to_coords(root)
        return self.coords[0] == root_coords[0]
    
    def is_in_root_column(self, root=0):
        """Check if current process is in root's column"""
        root_coords = self._rank_to_coords(root)
        return self.coords[1] == root_coords[1]
    
    def get_row_leader(self, root=0):
        """Get the row leader (first process in row)"""
        row, col = self.coords
        return self._coords_to_rank(row, 0)
    
    def is_row_leader(self):
        """Check if current process is a row leader"""
        return self.coords[1] == 0


class Mesh3D(MeshTopology):
    """3D Mesh Topology Implementation"""
    
    def __init__(self, comm):
        super().__init__(comm)
        # Calculate grid dimensions - find best factors for X * Y * Z = size
        self.x_dim, self.y_dim, self.z_dim = self._find_best_3d_dims(self.size)
        
        if self.rank == 0:
            print(f"3D Mesh: {self.x_dim}x{self.y_dim}x{self.z_dim} = {self.x_dim*self.y_dim*self.z_dim} nodes")
        
        # Calculate coordinates for this rank
        self.coords = self._rank_to_coords(self.rank)
        
        # Calculate neighbors
        self._calculate_neighbors()

    def _find_best_3d_dims(self, size):
        """Find dimensions (x, y, z) such that x*y*z = size and dimensions are as close as possible"""
        best_dims = (1, 1, size)
        min_diff = float('inf')
        
        # Iterate to find factors
        for x in range(1, int(size**(1/3)) + 2):
            if size % x == 0:
                rem = size // x
                for y in range(1, int(math.sqrt(rem)) + 1):
                    if rem % y == 0:
                        z = rem // y
                        # Calculate "closeness" (e.g., max difference or variance)
                        diff = max(abs(x-y), abs(y-z), abs(x-z))
                        if diff < min_diff:
                            min_diff = diff
                            best_dims = sorted((x, y, z)) # Sort to keep dimensions consistent
        
        return best_dims
    
    def _rank_to_coords(self, rank):
        """Convert rank to (x, y, z) coordinates"""
        if rank >= self.x_dim * self.y_dim * self.z_dim:
            return None
        
        z = rank // (self.x_dim * self.y_dim)
        remainder = rank % (self.x_dim * self.y_dim)
        y = remainder // self.x_dim
        x = remainder % self.x_dim
        return (x, y, z)
    
    def _coords_to_rank(self, x, y, z):
        """Convert (x, y, z) coordinates to rank"""
        if 0 <= x < self.x_dim and 0 <= y < self.y_dim and 0 <= z < self.z_dim:
            rank = z * (self.x_dim * self.y_dim) + y * self.x_dim + x
            if rank < self.size:
                return rank
        return None
    
    def _calculate_neighbors(self):
        """Calculate neighbors for current rank"""
        if self.coords is None:
            return
        
        x, y, z = self.coords
        
        # West/East neighbors (x-direction)
        west = self._coords_to_rank(x - 1, y, z)
        if west is not None:
            self.neighbors['west'] = west
        
        east = self._coords_to_rank(x + 1, y, z)
        if east is not None:
            self.neighbors['east'] = east
        
        # North/South neighbors (y-direction)
        north = self._coords_to_rank(x, y - 1, z)
        if north is not None:
            self.neighbors['north'] = north
        
        south = self._coords_to_rank(x, y + 1, z)
        if south is not None:
            self.neighbors['south'] = south
        
        # Up/Down neighbors (z-direction)
        down = self._coords_to_rank(x, y, z - 1)
        if down is not None:
            self.neighbors['down'] = down
        
        up = self._coords_to_rank(x, y, z + 1)
        if up is not None:
            self.neighbors['up'] = up
    
    def get_xy_plane_neighbors(self, z_plane=None):
        """Get all neighbors in the same XY plane"""
        if self.coords is None:
            return []
        
        x, y, z = self.coords
        if z_plane is None:
            z_plane = z
        
        neighbors = []
        for yi in range(self.y_dim):
            for xi in range(self.x_dim):
                rank = self._coords_to_rank(xi, yi, z_plane)
                if rank is not None and rank != self.rank:
                    neighbors.append(rank)
        return neighbors
    
    def get_x_line_neighbors(self):
        """Get all neighbors along the x-axis (same y, z)"""
        if self.coords is None:
            return []
        
        x, y, z = self.coords
        neighbors = []
        for xi in range(self.x_dim):
            rank = self._coords_to_rank(xi, y, z)
            if rank is not None and rank != self.rank:
                neighbors.append(rank)
        return neighbors
    
    def get_y_line_neighbors(self):
        """Get all neighbors along the y-axis (same x, z)"""
        if self.coords is None:
            return []
        
        x, y, z = self.coords
        neighbors = []
        for yi in range(self.y_dim):
            rank = self._coords_to_rank(x, yi, z)
            if rank is not None and rank != self.rank:
                neighbors.append(rank)
        return neighbors
    
    def get_z_line_neighbors(self):
        """Get all neighbors along the z-axis (same x, y)"""
        if self.coords is None:
            return []
        
        x, y, z = self.coords
        neighbors = []
        for zi in range(self.z_dim):
            rank = self._coords_to_rank(x, y, zi)
            if rank is not None and rank != self.rank:
                neighbors.append(rank)
        return neighbors
    
    def is_in_root_xy_plane(self, root=0):
        """Check if current process is in root's XY plane"""
        root_coords = self._rank_to_coords(root)
        if root_coords is None or self.coords is None:
            return False
        return self.coords[2] == root_coords[2]
    
    def is_xy_plane_leader(self):
        """Check if current process is a plane leader (x=0, y=0)"""
        if self.coords is None:
            return False
        return self.coords[0] == 0 and self.coords[1] == 0
