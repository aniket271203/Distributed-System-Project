# Collective Communication Operations on Mesh Topologies

**Operations:** Broadcast (Bcast) and Gather  
**Topologies:** 2D and 3D Mesh  
**Author:** Aniket Gupta - 2022101099 & Samarth Srikar - 2022101106

## Project Overview

This project implements and analyzes collective communication operations on mesh-based networks, focusing on Broadcast and Gather operations. The implementation follows the exact specifications from the project scope document.

## Features

- ✅ **Mesh Topology Creation** (2D and 3D)
- ✅ **Broadcast Operation** on 2D and 3D meshes
- ✅ **Gather Operation** on 2D and 3D meshes
- ✅ **Performance Analysis** using latency-bandwidth model
- ✅ **Comparative Analysis** between 2D and 3D topologies

## File Structure

```
Project/
├── mesh_topology.py         # Mesh topology implementation (2D & 3D)
├── broadcast.py             # Broadcast operations
├── gather.py                # Gather operations
├── main.py                  # Main driver program
├── performance_analysis.py  # Performance measurement and analysis
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Requirements

- Python 3.7+
- MPI implementation (OpenMPI or MPICH)
- mpi4py
- numpy

## Installation

1. Install MPI (if not already installed):
```bash
# On Ubuntu/Debian
sudo apt-get install openmpi-bin libopenmpi-dev

# On macOS
brew install open-mpi
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Complete Project

Run all tests with default settings (4 processes):
```bash
mpiexec -n 4 python main.py
```

Run with 16 processes (for better mesh visualization):
```bash
mpiexec -n 16 python main.py
```

Run with 27 processes (perfect 3D cube):
```bash
mpiexec -n 27 python main.py
```

Run with custom data size:
```bash
mpiexec -n 16 python main.py 5000
```

### Running Individual Components

**Test Broadcast only:**
```bash
mpiexec -n 16 python broadcast.py
```

**Test Gather only:**
```bash
mpiexec -n 16 python gather.py
```

**Run Performance Analysis:**
```bash
mpiexec -n 16 python performance_analysis.py
```

## Algorithm Design

### Broadcast on 2D Mesh

**Algorithm:**
1. Broadcast message along the root's row
2. Row nodes become column leaders
3. Each leader broadcasts message down its column

**Runtime:** `T_bcast^2D = 2(√p - 1)ts + (p - 1)tw*m`

### Gather on 2D Mesh

**Algorithm:**
1. Each row gathers data toward row leaders (first column)
2. Row leaders send data up the root column to root

**Runtime:** `T_gather^2D = 2(√p - 1)ts + (p - 1)tw*m`

### Broadcast on 3D Mesh

**Algorithm:**
1. Broadcast along root's x-axis
2. Broadcast along y-axis in root's plane
3. Broadcast along z-axis from each plane leader

**Runtime:** `T_bcast^3D = 3(∛p - 1)ts + (p - 1)tw*m`

### Gather on 3D Mesh

**Algorithm:**
1. Gather along x-axis to x=0 plane leaders
2. Gather along y-axis to (x=0, y=0) line leaders
3. Gather along z-axis to root

**Runtime:** `T_gather^3D = 3(∛p - 1)ts + (p - 1)tw*m`

## Latency-Bandwidth Model

The project uses the standard latency-bandwidth model:

```
T_msg = ts + tw * m
```

Where:
- `ts` = startup latency (time to initiate communication)
- `tw` = time per word (inverse of bandwidth)
- `m` = message size

For collective operations:
- **2D Mesh:** `T = 2(√p - 1)ts + (p - 1)tw*m`
- **3D Mesh:** `T = 3(∛p - 1)ts + (p - 1)tw*m`

## Example Output

```
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║          COLLECTIVE COMMUNICATION ON MESH TOPOLOGIES               ║
║          Operations: Broadcast (Bcast) and Gather                  ║
║          Topologies: 2D and 3D Mesh                                ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝

Total MPI processes: 16

======================================================================
2D MESH TOPOLOGY - BROADCAST AND GATHER OPERATIONS
======================================================================

Mesh Configuration:
  Total processes: 16
  Grid size: 4x4
  Data size: 1000 elements

----------------------------------------------------------------------
BROADCAST OPERATION
----------------------------------------------------------------------
Broadcasting 1000 elements from root (rank 0)...
✓ Broadcast verification: SUCCESS - All processes received correct data

Broadcast Performance:
  Execution time: 0.002341 seconds
  Communication steps: 2
  Messages sent: 7

  Theoretical Analysis (2D Mesh):
    Grid dimension: √p = 4
    Expected steps: 2(√p - 1) = 6
    Formula: T = 2(√p - 1)ts + (p - 1)tw*m

----------------------------------------------------------------------
GATHER OPERATION
----------------------------------------------------------------------
Gathering data from all 16 processes to root...
✓ Gather verification: SUCCESS - Received data from all 16 processes
  Unique process data collected: 16/16

Gather Performance:
  Execution time: 0.001876 seconds
  Communication steps: 2
  Messages received: 7
```

## Performance Metrics

The implementation measures:
- **Execution time** - Actual wall-clock time
- **Communication steps** - Number of communication rounds
- **Messages sent/received** - Total message count
- **Theoretical vs Actual** - Comparison with theoretical model

## Key Results

1. **3D mesh reduces communication distance** compared to 2D mesh
2. **Diameter comparison:**
   - 2D: `2√p - 2`
   - 3D: `3∛p - 3`
3. **For p=27 processes:**
   - 2D diameter: 8
   - 3D diameter: 6
   - **25% reduction in communication distance**

## Technologies Used

- **Python 3** - Main programming language
- **mpi4py** - Python bindings for MPI
- **NumPy** - Numerical computations
- **OpenMPI/MPICH** - MPI implementation

## Testing

The implementation has been tested with:
- 4 processes (2x2 grid, 2x2x1 3D)
- 9 processes (3x3 grid)
- 16 processes (4x4 grid)
- 27 processes (3x3x3 cube)
- 64 processes (8x8 grid, 4x4x4 cube)

## Verification

All operations include verification:
- **Broadcast:** Verifies all processes receive identical data
- **Gather:** Verifies root receives data from all processes

## Future Enhancements

- Visualization of message flow on mesh grids
- Support for non-square/non-cube mesh dimensions
- Additional collective operations (Reduce, Allreduce, etc.)
- Network simulation with configurable latency/bandwidth

## References

- MPI Standard: https://www.mpi-forum.org/
- mpi4py Documentation: https://mpi4py.readthedocs.io/
- Project Scope Document: `2022101099_project_scope.pdf`

## License

This is an academic project for Distributed Systems course (Sem-7).

## Author

**Aniket Gupta** - 2022101099  
**Samarth Srikar** - 2022101106 
Distributed Systems Project  
November 2024
