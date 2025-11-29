import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

class MeshVisualizer:
    def __init__(self, output_dir="visualizations"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def _draw_2d_mesh_state(self, ax, rows, cols, active_nodes, highlighted_edges, title):
        ax.clear()
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Draw edges
        for r in range(rows):
            for c in range(cols):
                # Horizontal edges
                if c < cols - 1:
                    color = 'red' if ((r, c), (r, c+1)) in highlighted_edges or ((r, c+1), (r, c)) in highlighted_edges else 'gray'
                    width = 2 if color == 'red' else 1
                    ax.plot([c, c+1], [rows-1-r, rows-1-r], color=color, linewidth=width, zorder=1)
                
                # Vertical edges
                if r < rows - 1:
                    color = 'red' if ((r, c), (r+1, c)) in highlighted_edges or ((r+1, c), (r, c)) in highlighted_edges else 'gray'
                    width = 2 if color == 'red' else 1
                    ax.plot([c, c], [rows-1-r, rows-1-(r+1)], color=color, linewidth=width, zorder=1)

        # Draw nodes
        for r in range(rows):
            for c in range(cols):
                color = 'green' if (r, c) in active_nodes else 'white'
                circle = plt.Circle((c, rows-1-r), 0.3, facecolor=color, edgecolor='black', zorder=2)
                ax.add_patch(circle)
                ax.text(c, rows-1-r, f"({r},{c})", ha='center', va='center', fontsize=8, zorder=3)

        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(-0.5, rows - 0.5)

    def animate_broadcast_2d(self, rows, cols, root_coords=(0,0), filename="broadcast_2d.gif"):
        fig, ax = plt.subplots(figsize=(8, 8))
        
        steps = []
        
        # Initial state: only root has data
        active_nodes = {root_coords}
        highlighted_edges = set()
        steps.append((set(active_nodes), set(highlighted_edges), "Initial State"))
        
        # Step 1: Broadcast along root's row
        root_r, root_c = root_coords
        for c in range(cols):
            if c != root_c:
                active_nodes.add((root_r, c))
                # Add edge from previous node to this node (simplified for visualization)
                # In reality it's a tree, but we'll show flow along the row
                if c > root_c:
                    highlighted_edges.add(((root_r, c-1), (root_r, c)))
                else:
                    highlighted_edges.add(((root_r, c+1), (root_r, c)))
        
        steps.append((set(active_nodes), set(highlighted_edges), "Step 1: Broadcast along Row"))
        
        # Step 2: Broadcast down columns
        new_edges = set()
        for c in range(cols):
            # From row leader (root_r, c) to all others in column
            for r in range(rows):
                if r != root_r:
                    active_nodes.add((r, c))
                    if r > root_r:
                        new_edges.add(((r-1, c), (r, c)))
                    else:
                        new_edges.add(((r+1, c), (r, c)))
        
        # We keep previous edges red or clear them? Let's clear previous step's edges to show current activity
        # But active nodes stay active.
        steps.append((set(active_nodes), new_edges, "Step 2: Broadcast along Columns"))

        def update(frame_idx):
            nodes, edges, title = steps[frame_idx]
            self._draw_2d_mesh_state(ax, rows, cols, nodes, edges, title)

        ani = animation.FuncAnimation(fig, update, frames=len(steps), interval=1500)
        ani.save(os.path.join(self.output_dir, filename), writer='pillow')
        plt.close(fig)
        print(f"Saved {filename}")

    def animate_gather_2d(self, rows, cols, root_coords=(0,0), filename="gather_2d.gif"):
        fig, ax = plt.subplots(figsize=(8, 8))
        
        steps = []
        
        # Initial state: everyone has data
        active_nodes = set((r, c) for r in range(rows) for c in range(cols))
        highlighted_edges = set()
        steps.append((set(active_nodes), set(highlighted_edges), "Initial State: All nodes have data"))
        
        # Step 1: Gather along columns to row leaders
        root_r, root_c = root_coords
        current_edges = set()
        
        # Visualize data moving TO the row leaders. 
        # We can represent "active" as "has sent data" or "holding aggregated data".
        # Let's say active nodes are those currently holding the *aggregated* chunk we care about.
        # Initially everyone holds their own.
        # After step 1, only row leaders hold the aggregated column data.
        
        row_leaders = set((root_r, c) for c in range(cols))
        
        for c in range(cols):
            for r in range(rows):
                if r != root_r:
                    # Edge towards row leader
                    if r > root_r:
                        current_edges.add(((r, c), (r-1, c))) # Arrow up
                    else:
                        current_edges.add(((r, c), (r+1, c))) # Arrow down
        
        steps.append((row_leaders, current_edges, "Step 1: Gather to Row Leaders"))
        
        # Step 2: Gather along row to root
        final_node = {root_coords}
        row_edges = set()
        for c in range(cols):
            if c != root_c:
                 if c > root_c:
                     row_edges.add(((root_r, c), (root_r, c-1)))
                 else:
                     row_edges.add(((root_r, c), (root_r, c+1)))
        
        steps.append((final_node, row_edges, "Step 2: Gather to Root"))

        def update(frame_idx):
            nodes, edges, title = steps[frame_idx]
            self._draw_2d_mesh_state(ax, rows, cols, nodes, edges, title)

        ani = animation.FuncAnimation(fig, update, frames=len(steps), interval=1500)
        ani.save(os.path.join(self.output_dir, filename), writer='pillow')
        plt.close(fig)
        print(f"Saved {filename}")

    def _draw_3d_mesh_state(self, ax, dim_x, dim_y, dim_z, active_nodes, highlighted_edges, title):
        ax.clear()
        ax.set_title(title)
        ax.set_axis_off()
        
        # Draw nodes and edges
        # We'll use a simple projection or just scatter plot
        
        xs, ys, zs = [], [], []
        colors = []
        
        for x in range(dim_x):
            for y in range(dim_y):
                for z in range(dim_z):
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)
                    if (x, y, z) in active_nodes:
                        colors.append('green')
                    else:
                        colors.append('white')

        # Draw edges - this is expensive to draw all, maybe just draw highlighted ones + skeleton
        # Skeleton:
        for x in range(dim_x):
            for y in range(dim_y):
                for z in range(dim_z):
                    if x < dim_x - 1:
                        c = 'red' if ((x,y,z), (x+1,y,z)) in highlighted_edges or ((x+1,y,z), (x,y,z)) in highlighted_edges else 'lightgray'
                        w = 2 if c == 'red' else 0.5
                        ax.plot([x, x+1], [y, y], [z, z], color=c, linewidth=w, alpha=0.5)
                    if y < dim_y - 1:
                        c = 'red' if ((x,y,z), (x,y+1,z)) in highlighted_edges or ((x,y+1,z), (x,y,z)) in highlighted_edges else 'lightgray'
                        w = 2 if c == 'red' else 0.5
                        ax.plot([x, x], [y, y+1], [z, z], color=c, linewidth=w, alpha=0.5)
                    if z < dim_z - 1:
                        c = 'red' if ((x,y,z), (x,y,z+1)) in highlighted_edges or ((x,y,z+1), (x,y,z)) in highlighted_edges else 'lightgray'
                        w = 2 if c == 'red' else 0.5
                        ax.plot([x, x], [y, y], [z, z+1], color=c, linewidth=w, alpha=0.5)

        ax.scatter(xs, ys, zs, c=colors, s=50, edgecolors='black', depthshade=True)
        
        # Set limits
        ax.set_xlim(-0.5, dim_x - 0.5)
        ax.set_ylim(-0.5, dim_y - 0.5)
        ax.set_zlim(-0.5, dim_z - 0.5)

    def animate_broadcast_3d(self, dim, root_coords=(0,0,0), filename="broadcast_3d.gif"):
        # dim is tuple (x, y, z)
        dim_x, dim_y, dim_z = dim
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        steps = []
        root_x, root_y, root_z = root_coords
        
        # Initial
        active_nodes = {root_coords}
        highlighted_edges = set()
        steps.append((set(active_nodes), set(highlighted_edges), "Initial State"))
        
        # Step 1: X-axis
        x_edges = set()
        for x in range(dim_x):
            if x != root_x:
                active_nodes.add((x, root_y, root_z))
                if x > root_x:
                    x_edges.add(((x-1, root_y, root_z), (x, root_y, root_z)))
                else:
                    x_edges.add(((x+1, root_y, root_z), (x, root_y, root_z)))
        steps.append((set(active_nodes), x_edges, "Step 1: Broadcast along X-axis"))
        
        # Step 2: Y-axis (from X-leaders)
        y_edges = set()
        for x in range(dim_x): # For each X-leader
            for y in range(dim_y):
                if y != root_y:
                    active_nodes.add((x, y, root_z))
                    if y > root_y:
                        y_edges.add(((x, y-1, root_z), (x, y, root_z)))
                    else:
                        y_edges.add(((x, y+1, root_z), (x, y, root_z)))
        steps.append((set(active_nodes), y_edges, "Step 2: Broadcast along Y-axis"))
        
        # Step 3: Z-axis (from XY-leaders)
        z_edges = set()
        for x in range(dim_x):
            for y in range(dim_y):
                for z in range(dim_z):
                    if z != root_z:
                        active_nodes.add((x, y, z))
                        if z > root_z:
                            z_edges.add(((x, y, z-1), (x, y, z)))
                        else:
                            z_edges.add(((x, y, z+1), (x, y, z)))
        steps.append((set(active_nodes), z_edges, "Step 3: Broadcast along Z-axis"))

        def update(frame_idx):
            nodes, edges, title = steps[frame_idx]
            self._draw_3d_mesh_state(ax, dim_x, dim_y, dim_z, nodes, edges, title)

        ani = animation.FuncAnimation(fig, update, frames=len(steps), interval=2000)
        ani.save(os.path.join(self.output_dir, filename), writer='pillow')
        plt.close(fig)
        print(f"Saved {filename}")

    def animate_broadcast_2d_pipelined(self, rows, cols, root_coords=(0,0), filename="broadcast_2d_pipelined.gif"):
        fig, ax = plt.subplots(figsize=(8, 8))
        
        steps = []
        
        # Initial state
        active_nodes = {root_coords}
        highlighted_edges = set()
        steps.append((set(active_nodes), set(highlighted_edges), "Initial State"))
        
        # Pipelined Row Phase
        # We visualize "chunks" moving. Since we can't easily show chunks inside nodes in this simple viz,
        # we'll show the "wave" of activation.
        # In pipelining, node 0 sends to 1, then 0 sends to 1 AND 1 sends to 2...
        
        root_r, root_c = root_coords
        
        # Simplified visualization of the wave
        for i in range(cols + 2): # Arbitrary steps to show flow
            current_edges = set()
            # Calculate who is active sending to whom
            # Wavefront moves right
            
            # Nodes active are those who have received at least one chunk
            # In step i, node c is active if c <= i
            
            for c in range(cols):
                if c <= i and c >= 0:
                    active_nodes.add((root_r, c))
                
                # Edges active
                # Node c sends to c+1 if c <= i
                if c < cols - 1 and c <= i:
                     # Blink edges to show activity
                     current_edges.add(((root_r, c), (root_r, c+1)))
            
            steps.append((set(active_nodes), current_edges, f"Step {i+1}: Pipelined Row Broadcast"))

        # Pipelined Column Phase
        # Row leaders send down
        for i in range(rows + 2):
            current_edges = set()
            for c in range(cols):
                # For each column, wave moves down
                for r in range(rows):
                    if r != root_r:
                         if r <= i: # Simplified condition
                             active_nodes.add((r, c))
                         
                         if r < rows - 1 and r <= i:
                             if r >= root_r: # Downwards
                                 current_edges.add(((r, c), (r+1, c)))
                             else: # Upwards
                                 current_edges.add(((r, c), (r-1, c))) # Simplified
                                 
            steps.append((set(active_nodes), current_edges, f"Step {i+1}: Pipelined Column Broadcast"))

        def update(frame_idx):
            nodes, edges, title = steps[frame_idx]
            self._draw_2d_mesh_state(ax, rows, cols, nodes, edges, title)

        ani = animation.FuncAnimation(fig, update, frames=len(steps), interval=1000)
        ani.save(os.path.join(self.output_dir, filename), writer='pillow')
        plt.close(fig)
        print(f"Saved {filename}")

    def animate_broadcast_2d_binary_tree(self, rows, cols, root_coords=(0,0), filename="broadcast_2d_binary_tree.gif"):
        fig, ax = plt.subplots(figsize=(8, 8))
        
        steps = []
        size = rows * cols
        
        # Map rank to coords
        def get_coords(rank):
            return (rank // cols, rank % cols)
        
        root_rank = root_coords[0] * cols + root_coords[1]
        
        active_nodes = {root_coords}
        highlighted_edges = set()
        steps.append((set(active_nodes), set(highlighted_edges), "Initial State"))
        
        curr_size = 1
        step_count = 1
        while curr_size < size:
            current_edges = set()
            
            # Calculate edges for this step
            # Nodes < curr_size (relative) send to node + curr_size
            
            for r in range(rows):
                for c in range(cols):
                    rank = r * cols + c
                    rel_rank = (rank - root_rank + size) % size
                    
                    if rel_rank < curr_size:
                        dest_rel = rel_rank + curr_size
                        if dest_rel < size:
                            dest_rank = (dest_rel + root_rank) % size
                            dest_coords = get_coords(dest_rank)
                            src_coords = (r, c)
                            
                            active_nodes.add(dest_coords)
                            current_edges.add((src_coords, dest_coords))
            
            steps.append((set(active_nodes), current_edges, f"Step {step_count}: Binary Tree Doubling"))
            curr_size *= 2
            step_count += 1

        def update(frame_idx):
            nodes, edges, title = steps[frame_idx]
            # Draw custom edges for tree (might be long distance)
            ax.clear()
            ax.set_title(title)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Draw underlying mesh grid (faint)
            for r in range(rows):
                for c in range(cols):
                    if c < cols - 1: ax.plot([c, c+1], [rows-1-r, rows-1-r], color='lightgray', linewidth=0.5, zorder=0)
                    if r < rows - 1: ax.plot([c, c], [rows-1-r, rows-1-(r+1)], color='lightgray', linewidth=0.5, zorder=0)

            # Draw active edges (can be diagonal/long)
            for (r1, c1), (r2, c2) in edges:
                ax.plot([c1, c2], [rows-1-r1, rows-1-r2], color='red', linewidth=2, zorder=1)

            # Draw nodes
            for r in range(rows):
                for c in range(cols):
                    color = 'green' if (r, c) in nodes else 'white'
                    circle = plt.Circle((c, rows-1-r), 0.3, facecolor=color, edgecolor='black', zorder=2)
                    ax.add_patch(circle)
                    ax.text(c, rows-1-r, f"({r},{c})", ha='center', va='center', fontsize=8, zorder=3)

            ax.set_xlim(-0.5, cols - 0.5)
            ax.set_ylim(-0.5, rows - 0.5)

        ani = animation.FuncAnimation(fig, update, frames=len(steps), interval=1500)
        ani.save(os.path.join(self.output_dir, filename), writer='pillow')
        plt.close(fig)
        print(f"Saved {filename}")

if __name__ == "__main__":
    # Test run
    viz = MeshVisualizer()
    viz.animate_broadcast_2d(4, 4)
