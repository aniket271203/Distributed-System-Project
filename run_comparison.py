import subprocess
import sys
import time

# Define specific dimensions for certain process counts to ensure "interesting" comparisons
# Format: {process_count: {'dims_2d': 'r,c', 'dims_3d': 'x,y,z'}}
# If not specified, defaults (auto-calc) are used.
CUSTOM_DIMS = {
    8: {'dims_2d': '4,2', 'dims_3d': '2,2,2'},
    12: {'dims_2d': '4,3', 'dims_3d': '2,2,3'},
    16: {'dims_2d': '4,4', 'dims_3d': '2,2,4'}, # 16: 4x4 vs 2x2x4
    6: {'dims_2d': '3,2', 'dims_3d': '1,2,3'},
    4: {'dims_2d': '2,2', 'dims_3d': '1,2,2'}
}

def run_command(cmd):
    print(f"Running: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        # Don't exit, just continue to next test
        pass

def main():
    print("Starting Focused Comparison (Proper 3D Meshes: p=8, 12, 16)...")
    print("-" * 60)

    # Only run for process counts that support proper 3D meshes (all dims >= 2)
    # 8 = 2x2x2
    # 12 = 2x2x3
    # 16 = 2x2x4
    target_processes = [8, 12, 16]

    for p in target_processes:
        print(f"\n[Benchmarking with {p} processes]")
        
        cmd = f"mpiexec --oversubscribe -n {p} python3 collect_data.py"
        
        if p in CUSTOM_DIMS:
            dims = CUSTOM_DIMS[p]
            if 'dims_2d' in dims:
                cmd += f" --dims_2d {dims['dims_2d']}"
            if 'dims_3d' in dims:
                cmd += f" --dims_3d {dims['dims_3d']}"
        
        run_command(cmd)
        time.sleep(0.5) # Brief pause

    print("\n" + "-" * 60)
    print("Benchmarking complete. Generating plots...")
    
    # Run the plotting script
    run_command("python3 plot_performance.py")
    
    print("\nDone! Results are in the 'results/' directory.")

if __name__ == "__main__":
    main()
