# Implementation Documentation

## Collective Communication Operations on Mesh Topologies

### Project Overview
This document provides detailed implementation information for the mesh communication project.

---

## File Descriptions

### 1. mesh_topology.py
**Purpose:** Create and manage 2D and 3D mesh topologies

**Classes:**
- `MeshTopology`: Base class for mesh structures
- `Mesh2D`: 2D grid topology (√p × √p)
- `Mesh3D`: 3D cube topology (∛p × ∛p × ∛p)

**Key Methods:**
```python
# 2D Mesh
- __init__(comm): Initialize 2D mesh with MPI communicator
- _rank_to_coords(rank): Convert rank to (row, col) coordinates
- _coords_to_rank(row, col): Convert coordinates to rank
- get_row_neighbors(): Get all processes in same row
- get_column_neighbors(): Get all processes in same column
- is_in_root_row(root): Check if process is in root's row
- is_row_leader(): Check if process is row leader (col=0)

# 3D Mesh
- __init__(comm): Initialize 3D mesh with MPI communicator
- _rank_to_coords(rank): Convert rank to (x, y, z) coordinates
- _coords_to_rank(x, y, z): Convert coordinates to rank
- get_x_line_neighbors(): Get neighbors along x-axis
- get_y_line_neighbors(): Get neighbors along y-axis
- get_z_line_neighbors(): Get neighbors along z-axis
- is_in_root_xy_plane(root): Check if in root's plane
```

**Example Usage:**
```python
from mpi4py import MPI
from mesh_topology import Mesh2D, Mesh3D

comm = MPI.COMM_WORLD
mesh = Mesh2D(comm)
print(f"My coordinates: {mesh.coords}")
print(f"My neighbors: {mesh.neighbors}")
```

---

### 2. broadcast.py
**Purpose:** Implement broadcast operations on mesh topologies

**Functions:**
```python
broadcast_2d_mesh(mesh, data, root=0)
  - Broadcasts data on 2D mesh
  - Returns: (data, time, steps, messages_sent)
  - Algorithm:
    1. Broadcast along root's row
    2. Broadcast down columns from row leaders

broadcast_3d_mesh(mesh, data, root=0)
  - Broadcasts data on 3D mesh
  - Returns: (data, time, steps, messages_sent)
  - Algorithm:
    1. Broadcast along x-axis
    2. Broadcast along y-axis in root's plane
    3. Broadcast along z-axis from plane leaders

measure_broadcast_performance(mesh_type, data_size, root)
  - Measures and reports broadcast performance
```

**Algorithm Details:**

**2D Broadcast:**
```
Step 1: Root broadcasts to its row
  - Uses MPI row communicator
  - Time: (√p - 1) × ts + m × tw

Step 2: Row leaders broadcast down columns
  - Uses MPI column communicator
  - Time: (√p - 1) × ts + m × tw

Total: 2(√p - 1)ts + (√p - 1) × 2 × m × tw
```

**3D Broadcast:**
```
Step 1: Root broadcasts along x-axis
  - Time: (∛p - 1) × ts

Step 2: X-leaders broadcast along y-axis
  - Time: (∛p - 1) × ts

Step 3: XY-leaders broadcast along z-axis
  - Time: (∛p - 1) × ts

Total: 3(∛p - 1)ts + (p - 1) × m × tw
```

---

### 3. gather.py
**Purpose:** Implement gather operations on mesh topologies

**Functions:**
```python
gather_2d_mesh(mesh, data, root=0)
  - Gathers data on 2D mesh
  - Returns: (gathered_data, time, steps, messages_received)
  - Algorithm:
    1. Gather along rows to row leaders
    2. Row leaders gather to root

gather_3d_mesh(mesh, data, root=0)
  - Gathers data on 3D mesh
  - Returns: (gathered_data, time, steps, messages_received)
  - Algorithm:
    1. Gather along x-axis to x=0 plane
    2. Gather along y-axis to (x=0, y=0) line
    3. Gather along z-axis to root

measure_gather_performance(mesh_type, data_size, root)
  - Measures and reports gather performance
```

**Algorithm Details:**

**2D Gather:**
```
Step 1: Each row gathers to column 0
  - Each process sends to row leader
  - Row leader receives (√p - 1) messages

Step 2: Row leaders gather to root
  - Each row leader sends to root
  - Root receives (√p - 1) messages

Total messages at root: p - 1
Total steps: 2(√p - 1)
```

**3D Gather:**
```
Step 1: Gather along x-axis to x=0
  - Each (y,z) line gathers to x=0

Step 2: Gather along y-axis to y=0
  - Each z-line gathers to y=0

Step 3: Gather along z-axis to root
  - Final gather to root

Total steps: 3(∛p - 1)
```

---

### 4. main.py
**Purpose:** Main driver program integrating all components

**Functions:**
```python
test_2d_mesh_operations(comm, data_size)
  - Tests 2D broadcast and gather
  - Verifies correctness
  - Reports performance

test_3d_mesh_operations(comm, data_size)
  - Tests 3D broadcast and gather
  - Verifies correctness
  - Reports performance

compare_2d_vs_3d(comm, data_size)
  - Compares 2D and 3D performance
  - Shows improvement metrics
```

**Output Format:**
```
╔════════════════════════════════════════════╗
║  COLLECTIVE COMMUNICATION ON MESH          ║
╚════════════════════════════════════════════╝

2D MESH - BROADCAST AND GATHER
  ✓ Broadcast verification: SUCCESS
  ✓ Gather verification: SUCCESS
  Performance metrics...

3D MESH - BROADCAST AND GATHER
  ✓ Broadcast verification: SUCCESS
  ✓ Gather verification: SUCCESS
  Performance metrics...

COMPARISON: 2D vs 3D
  Improvements shown...
```

---

### 5. performance_analysis.py
**Purpose:** Performance measurement and theoretical analysis

**Functions:**
```python
collect_performance_data(comm, mesh_type, operation, data_sizes)
  - Collects performance data for varying data sizes
  - Returns list of results with timing info

analyze_scalability(comm, operation, data_size)
  - Analyzes how performance scales with processes
  - Compares 2D vs 3D

calculate_theoretical_metrics(num_processes, mesh_type)
  - Calculates theoretical performance metrics
  - Returns diameter, steps, bisection width

latency_bandwidth_model(ts, tw, m, p, mesh_type)
  - Implements T = ts + tw × m model
  - Calculates expected communication time

generate_performance_report(comm)
  - Generates comprehensive performance report
  - Shows theoretical vs actual comparison
```

**Metrics Calculated:**
- Network diameter
- Communication steps
- Bisection width
- Expected time (latency-bandwidth model)
- Actual measured time
- Speedup (3D over 2D)

---

### 6. test_project.py
**Purpose:** Comprehensive test suite

**Functions:**
```python
test_mesh_creation()
  - Tests topology creation
  - Verifies coordinates and neighbors

test_broadcast()
  - Tests broadcast correctness
  - Verifies all processes receive correct data

test_gather()
  - Tests gather correctness
  - Verifies root receives all data

test_performance_model()
  - Tests theoretical calculations
  - Verifies latency-bandwidth model

run_all_tests()
  - Runs complete test suite
  - Reports pass/fail status
```

---

## Usage Examples

### Basic Usage
```bash
# Install dependencies
pip install -r requirements.txt

# Run with 16 processes
mpiexec -n 16 python3 main.py

# Run with custom data size
mpiexec -n 16 python3 main.py 5000
```

### Using Make
```bash
# Install and test
make install test

# Quick test
make quick

# Medium test (16 processes)
make medium

# Large test (27 processes, 3D cube)
make large

# Performance analysis
make performance

# Run specific operation
make broadcast
make gather
```

### Using Shell Scripts
```bash
# Simple run
./run.sh 16 1000

# Complete test suite
./run_tests.sh
```

---

## Performance Characteristics

### Complexity Analysis

**2D Mesh (p processes, √p × √p grid):**
- Diameter: O(√p)
- Broadcast steps: O(√p)
- Gather steps: O(√p)
- Bisection width: O(√p)

**3D Mesh (p processes, ∛p × ∛p × ∛p cube):**
- Diameter: O(∛p)
- Broadcast steps: O(∛p)
- Gather steps: O(∛p)
- Bisection width: O((∛p)²)

### Scalability

**For p = 64 processes:**
- 2D: 8×8 grid, diameter = 14, steps = 14
- 3D: 4×4×4 cube, diameter = 9, steps = 9
- **Improvement: 36%**

**For p = 512 processes:**
- 2D: 23×23 grid, diameter = 44, steps = 44
- 3D: 8×8×8 cube, diameter = 21, steps = 21
- **Improvement: 52%**

### Communication Patterns

**2D Broadcast:**
```
Root → Row → Columns
  1  →  √p  →  p processes
```

**3D Broadcast:**
```
Root → X-line → Y-lines → Z-lines
  1  →  ∛p   →  (∛p)²   →  p processes
```

---

## Verification Methods

1. **Broadcast Verification:**
   - All processes check if received data matches original
   - Uses `np.allclose()` for floating-point comparison

2. **Gather Verification:**
   - Root checks if data from all processes received
   - Verifies count matches number of processes

3. **Topology Verification:**
   - Checks coordinate mapping consistency
   - Verifies neighbor relationships

4. **Performance Verification:**
   - Compares actual steps with theoretical
   - Validates latency-bandwidth model

---

## Common Issues and Solutions

### Issue: "MPI not found"
**Solution:** Install MPI
```bash
sudo apt-get install openmpi-bin libopenmpi-dev
```

### Issue: "mpi4py not installed"
**Solution:** Install Python packages
```bash
pip install -r requirements.txt
```

### Issue: Non-square number of processes
**Solution:** The code handles this gracefully by rounding to nearest square/cube

### Issue: Different process counts
**Best configurations:**
- 2D: 4, 9, 16, 25, 36, 49, 64, 81, 100
- 3D: 8, 27, 64, 125, 216, 343, 512

---

## Extension Ideas

1. **Visualization:**
   - Add matplotlib plots of mesh structure
   - Animate message flow

2. **Additional Operations:**
   - Reduce, Allreduce
   - Scatter, Allgather
   - All-to-all communication

3. **Optimization:**
   - Pipeline overlapping
   - Non-blocking communication
   - Custom routing algorithms

4. **Analysis:**
   - Network contention modeling
   - Fault tolerance analysis
   - Energy consumption modeling

---

## References

1. MPI Standard 4.0: https://www.mpi-forum.org/
2. mpi4py Documentation: https://mpi4py.readthedocs.io/
3. Parallel Computer Architecture: Culler, Singh, Gupta
4. Introduction to Parallel Computing: Grama et al.

---

## Contact & Support

**Author:** Aniket Gupta  
**Roll No:** 2022101099  
**Course:** Distributed Systems (Sem-7)  
**Institution:** IIIT Hyderabad  
**Date:** November 2024

For questions or issues, refer to:
- README.md for usage instructions
- PROJECT_SUMMARY.md for high-level overview
- This file for implementation details
