from visualization import MeshVisualizer

def main():
    print("Starting visualization generation...")
    viz = MeshVisualizer(output_dir="visualizations")
    
    # 2D Mesh (4x4)
    print("Generating 2D Broadcast animation...")
    viz.animate_broadcast_2d(rows=4, cols=4, root_coords=(0,0), filename="broadcast_2d.gif")
    
    print("Generating 2D Pipelined Broadcast animation...")
    viz.animate_broadcast_2d_pipelined(rows=4, cols=4, root_coords=(0,0), filename="broadcast_2d_pipelined.gif")
    
    print("Generating 2D Binary Tree Broadcast animation...")
    viz.animate_broadcast_2d_binary_tree(rows=4, cols=4, root_coords=(0,0), filename="broadcast_2d_binary_tree.gif")
    
    print("Generating 2D Gather animation...")
    viz.animate_gather_2d(rows=4, cols=4, root_coords=(0,0), filename="gather_2d.gif")
    
    # 3D Mesh (3x3x3)
    print("Generating 3D Broadcast animation...")
    viz.animate_broadcast_3d(dim=(3,3,3), root_coords=(0,0,0), filename="broadcast_3d.gif")
    
    # Note: 3D Gather is symmetric to Broadcast, so we'll skip it to save time/space or implement if needed.
    # But for completeness let's just rely on these for now as they demonstrate the concept well.
    
    print("All visualizations generated in 'visualizations/' directory.")

if __name__ == "__main__":
    main()
