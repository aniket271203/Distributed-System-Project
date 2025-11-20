# Project Summary

## Collective Communication Operations on Mesh Topologies
**Author:** Aniket Gupta (2022101099)  
**Course:** Distributed Systems (Sem-7)  
**Date:** November 2024

---

## Project Implementation

This project implements collective communication operations (Broadcast and Gather) on 2D and 3D mesh topologies using MPI (Message Passing Interface).

### Files Created

1. **mesh_topology.py** - Mesh topology creation
   - `Mesh2D` class: 2D grid topology
   - `Mesh3D` class: 3D cube topology
   - Neighbor calculation and coordinate mapping

2. **broadcast.py** - Broadcast operations
   - 2D mesh broadcast algorithm
   - 3D mesh broadcast algorithm
   - Performance measurement

3. **gather.py** - Gather operations
   - 2D mesh gather algorithm
   - 3D mesh gather algorithm
   - Performance measurement

4. **main.py** - Main driver program
   - Integrates all operations
   - Verification and testing
   - Comparative analysis

5. **performance_analysis.py** - Performance analysis
   - Latency-bandwidth model implementation
   - Theoretical vs actual comparison
   - Scalability analysis

6. **test_project.py** - Test suite
   - Unit tests for all components
   - Verification tests

7. **run_tests.sh** - Quick start script
   - Automated testing
   - Multiple configurations

8. **README.md** - Documentation
9. **requirements.txt** - Dependencies

---

## Algorithms Implemented

### 2D Mesh Broadcast
```
Algorithm Mesh_Broadcast(root, M):
1. Broadcast M along the root's row
2. Row nodes become column leaders
3. Each leader broadcasts M down its column

Runtime: T = 2(√p - 1)ts + (p - 1)tw*m
```

### 2D Mesh Gather
```
Algorithm Mesh_Gather(root):
1. Rows gather data toward row leaders
2. Row leaders send data up the root column

Runtime: T = 2(√p - 1)ts + (p - 1)tw*m
```

### 3D Mesh Broadcast
```
Algorithm:
1. Broadcast along root's x-axis
2. Broadcast along y-axis in root's plane
3. Broadcast along z-axis from plane leaders

Runtime: T = 3(∛p - 1)ts + (p - 1)tw*m
```

### 3D Mesh Gather
```
Algorithm:
1. Gather along x-axis to x=0 plane leaders
2. Gather along y-axis to (x=0, y=0) line leaders
3. Gather along z-axis to root

Runtime: T = 3(∛p - 1)ts + (p - 1)tw*m
```

---

## Key Performance Metrics

### Network Properties

| Metric | 2D Mesh | 3D Mesh |
|--------|---------|---------|
| Diameter | 2√p - 2 | 3∛p - 3 |
| Communication Steps (Bcast/Gather) | 2(√p - 1) | 3(∛p - 1) |
| Bisection Width | √p | (∛p)² |

### Example: 27 Processes

| Metric | 2D (6x6) | 3D (3x3x3) | Improvement |
|--------|----------|------------|-------------|
| Diameter | 10 | 6 | 40% |
| Bcast Steps | 10 | 6 | 40% |
| Bisection Width | 6 | 9 | 50% |

---

## Latency-Bandwidth Model

The project uses the standard model:

```
T_msg = ts + tw * m

where:
  ts = startup latency
  tw = time per word
  m = message size
```

For collective operations:
- **2D Mesh:** `T = 2(√p - 1)ts + (p - 1)tw*m`
- **3D Mesh:** `T = 3(∛p - 1)ts + (p - 1)tw*m`

---

## Running Instructions

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
./run_tests.sh

# Or manually:
mpiexec -n 16 python3 main.py
```

### Test Configurations

1. **Small 2D mesh (4 processes):**
   ```bash
   mpiexec -n 4 python3 main.py
   ```

2. **Medium 2D mesh (16 processes):**
   ```bash
   mpiexec -n 16 python3 main.py
   ```

3. **Perfect 3D cube (27 processes):**
   ```bash
   mpiexec -n 27 python3 main.py
   ```

4. **Performance analysis:**
   ```bash
   mpiexec -n 16 python3 performance_analysis.py
   ```

5. **Run test suite:**
   ```bash
   mpiexec -n 16 python3 test_project.py
   ```

---

## Verification

All operations include automatic verification:

✓ **Broadcast:** Verifies all processes receive identical data  
✓ **Gather:** Verifies root receives data from all processes  
✓ **Topology:** Verifies mesh structure and neighbors  
✓ **Performance:** Compares actual vs theoretical metrics

---

## Key Results

1. **3D mesh reduces communication distance** by ~25-40% compared to 2D
2. **Diameter is shorter** in 3D mesh (3∛p - 3 vs 2√p - 2)
3. **Bisection width is higher** in 3D mesh ((∛p)² vs √p)
4. **Better scalability** with 3D topology for large systems

---

## Technologies Used

- **Python 3** with mpi4py
- **MPI** (OpenMPI/MPICH)
- **NumPy** for numerical operations
- **Linux** environment

---

## Project Structure Follows Scope Document

The implementation follows the project scope document exactly:

1. ✅ **Mesh Creation** - 2D and 3D topologies
2. ✅ **Broadcast** - Both 2D and 3D implementations
3. ✅ **Gather** - Both 2D and 3D implementations
4. ✅ **Performance Analysis** - Latency-bandwidth model
5. ✅ **Comparison** - 2D vs 3D analysis
6. ✅ **Documentation** - Complete README and comments

---

## Conclusion

The project successfully demonstrates:
- Implementation of collective operations on mesh topologies
- Performance benefits of 3D over 2D mesh
- Practical application of latency-bandwidth model
- Scalability analysis for parallel systems

The 3D mesh topology provides better performance due to shorter communication paths and higher bisection width, making it more suitable for large-scale parallel systems.

---

## Contact

**Aniket Gupta**  
Roll Number: 2022101099  
Distributed Systems Project  
Semester 7, 2024
