#!/bin/bash
#SBATCH --job-name=mesh_benchmark
#SBATCH --output=benchmark_%j.out
#SBATCH --error=benchmark_%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=24
#SBATCH --time=00:20:00
#SBATCH --partition=debug

# Load MPI module if needed (adjust based on cluster environment)
# module load openmpi/4.1.1

echo "Starting Benchmark Run at $(date)"
echo "--------------------------------"

# Create results directory if it doesn't exist
mkdir -p results

# List of process counts to test
# 2D Perfect Squares: 4, 9, 16, 25, 36, 49, 64, 81
# 3D Perfect Cubes: 8, 27, 64
# We will run a superset of these. Note: Max cores available is 96.

PROCESS_COUNTS=(4 8 9 16 25 27 36 49 64 81)

for p in "${PROCESS_COUNTS[@]}"; do
    echo "Running benchmark with $p processes..."
    mpiexec -n $p python3 collect_data.py
    echo "--------------------------------"
done

echo "Benchmarking complete. Generating plots..."
python3 plot_real_data.py

echo "All done at $(date)"
