import subprocess
import sys
import time

# Process counts to test on local machine (max 16 cores)
# 2D: 4 (2x2), 9 (3x3), 16 (4x4)
# 3D: 8 (2x2x2)
PROCESS_COUNTS = [4, 8, 9, 16]

def run_command(cmd):
    print(f"Running: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(1)

def main():
    print("Starting Local Benchmarking...")
    print(f"Testing process counts: {PROCESS_COUNTS}")
    print("-" * 50)

    for p in PROCESS_COUNTS:
        print(f"\n[Benchmarking with {p} processes]")
        cmd = f"mpiexec --oversubscribe -n {p} python3 collect_data.py"
        run_command(cmd)
        time.sleep(1) # Brief pause

    print("\n" + "-" * 50)
    print("Benchmarking complete. Generating plots...")
    run_command("python3 plot_real_data.py")
    
    print("\nDone! Results are in the 'results/' directory.")

if __name__ == "__main__":
    main()
