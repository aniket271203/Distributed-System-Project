# Collective Communication Operations on Mesh Topologies

**Operations:** Broadcast (Bcast) and Gather  
**Topologies:** 2D and 3D Mesh  
**Algorithms:** Dimension-Ordered Routing (DOR) and Flooding (BFS)  
**Authors:** Aniket Gupta (2022101099) & Samarth Srikar (2022101106)

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [File Structure](#file-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Algorithm Design](#algorithm-design)
7. [Implementation Details](#implementation-details)
8. [Evaluation Methodology](#evaluation-methodology)
9. [Experimental Results](#experimental-results)
10. [Best Configuration](#best-configuration)
11. [Visualizations](#visualizations)
12. [References](#references)

---

## 🎯 Project Overview

This project implements and analyzes collective communication operations on mesh-based networks. We compare:

- **Topologies:** 2D Mesh vs 3D Mesh
- **Algorithms:** Dimension-Ordered Routing (DOR) vs Flooding (BFS)
- **Operations:** Broadcast and Gather

The goal is to determine the optimal configuration for minimizing latency (sequential hops) and bandwidth usage (message count) in distributed systems.

---

## ✨ Features

- ✅ **Mesh Topology Creation** - 2D and 3D mesh with automatic dimension calculation
- ✅ **Broadcast Operation** - Root disseminates data to all nodes
- ✅ **Gather Operation** - All nodes send data to root
- ✅ **DOR Algorithm** - Dimension-ordered routing for optimal message count
- ✅ **Flooding Algorithm** - BFS-based routing for fault tolerance
- ✅ **Performance Analysis** - Latency-bandwidth model simulation
- ✅ **Comparative Visualization** - Comprehensive plots comparing all configurations
- ✅ **Simulation Mode** - Run experiments without MPI hardware

---

## 📁 File Structure

```
Project/
├── mesh_topology.py         # 2D and 3D mesh topology implementation
├── broadcast.py             # Broadcast operations (DOR)
├── gather.py                # Gather operations (DOR)
├── flooding.py              # Flooding algorithm implementation
├── ablation_study.py        # DOR vs Flooding comparison with MPI
├── simulation_study.py      # Simulation-based experiments (no MPI required)
├── main.py                  # Main driver program
├── performance_analysis.py  # Performance measurement utilities
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── ANALYSIS.md              # Detailed experimental analysis
└── results/                 # Generated plots and reports
    ├── simulation_comparison.png
    ├── message_complexity.png
    ├── steps_2d_vs_3d.png
    ├── combined_analysis.png
    └── ...
```

---

## 🔧 Installation

### Prerequisites

- Python 3.7+
- MPI implementation (OpenMPI or MPICH) - optional for simulation mode

### Install MPI (Optional - for real MPI experiments)

```bash
# Ubuntu/Debian
sudo apt-get install openmpi-bin libopenmpi-dev

# macOS
brew install open-mpi
```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Simulation Mode (No MPI Required)

Run comprehensive simulation comparing all configurations:

```bash
python simulation_study.py
```

This generates:
- Detailed console output with results
- Visualization plots in `results/` directory

### MPI Mode

Run with actual MPI processes:

```bash
# 16 processes (4x4 2D mesh)
mpiexec -n 16 python main.py

# 27 processes (3x3x3 3D cube)
mpiexec -n 27 python main.py

# Ablation study: DOR vs Flooding
mpiexec -n 16 python ablation_study.py
```

### Individual Components

```bash
# Test Broadcast only
mpiexec -n 16 python broadcast.py

# Test Gather only
mpiexec -n 16 python gather.py

# Test Flooding
mpiexec -n 16 python flooding.py
```

---

## 📐 Algorithm Design

### Mesh Topology Construction

#### 2D Mesh

```
Coordinate Mapping:
  rank → (row, col)
  row = rank // cols
  col = rank % cols

Neighbors: North, South, East, West (4 neighbors max)

Example 4×4 Mesh:
┌───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │
├───┼───┼───┼───┤
│ 4 │ 5 │ 6 │ 7 │
├───┼───┼───┼───┤
│ 8 │ 9 │10 │11 │
├───┼───┼───┼───┤
│12 │13 │14 │15 │
└───┴───┴───┴───┘
```

#### 3D Mesh

```
Coordinate Mapping:
  rank → (x, y, z)
  z = rank // (x_dim × y_dim)
  y = (rank % (x_dim × y_dim)) // x_dim
  x = rank % x_dim

Neighbors: ±X, ±Y, ±Z directions (6 neighbors max)
```

### DOR (Dimension-Ordered Routing)

Routes messages by traversing one dimension at a time in a fixed order.

#### 2D DOR Broadcast Algorithm

```
Phase 1: Row Broadcast
  └─ Root broadcasts along its row
  └─ Sequential hops: (cols - 1)

Phase 2: Column Broadcast (Parallel)
  └─ Each row node broadcasts down its column
  └─ Sequential hops: (rows - 1)

Total Hops: (rows - 1) + (cols - 1)
Total Messages: (cols - 1) + cols × (rows - 1) = p - 1
```

**Visual Example (4×4 mesh):**
```
Phase 1:    0 → 1 → 2 → 3     (3 hops)
Phase 2:    ↓   ↓   ↓   ↓     (3 hops)
            4   5   6   7
            ↓   ↓   ↓   ↓
            8   9  10  11
            ↓   ↓   ↓   ↓
           12  13  14  15
           
Total: 6 hops
```

#### 3D DOR Broadcast Algorithm

```
Phase 1: X-axis Broadcast     → (x_dim - 1) hops
Phase 2: Y-axis Broadcast     → (y_dim - 1) hops
Phase 3: Z-axis Broadcast     → (z_dim - 1) hops

Total Hops: (x - 1) + (y - 1) + (z - 1)
Total Messages: p - 1
```

### Flooding (BFS) Algorithm

Each node forwards data to ALL its neighbors.

```
Level 0: Root has data
Level 1: Root → all neighbors
Level 2: Level-1 nodes → all their neighbors (except sender)
...
Level k: All nodes at Manhattan distance k receive data

Sequential Hops: Maximum Manhattan distance from root
Total Messages: Number of edges in mesh
                (each node forwards to all neighbors except the one it received from)
```

**Key Difference from DOR:**

| Aspect | DOR | Flooding |
|--------|-----|----------|
| Message Count | p - 1 | Number of edges |
| Redundancy | None | Minimal |
| Fault Tolerance | Low | High |

---

## 📊 Implementation Details

### Mesh2D Class

```python
class Mesh2D(MeshTopology):
    def __init__(self, comm):
        self.grid_size = int(math.sqrt(size))
        self.rows = self.cols = self.grid_size
        self.coords = self._rank_to_coords(rank)
        self._calculate_neighbors()
    
    def _rank_to_coords(self, rank):
        return (rank // self.cols, rank % self.cols)
    
    def _calculate_neighbors(self):
        # Add north, south, east, west neighbors
```

### Mesh3D Class

```python
class Mesh3D(MeshTopology):
    def __init__(self, comm):
        self.grid_size = int(round(size ** (1/3)))
        self.x_dim = self.y_dim = self.z_dim = self.grid_size
        self.coords = self._rank_to_coords(rank)
        self._calculate_neighbors()
    
    def _rank_to_coords(self, rank):
        z = rank // (x_dim * y_dim)
        y = (rank % (x_dim * y_dim)) // x_dim
        x = rank % x_dim
        return (x, y, z)
```

### Broadcast Implementation

```python
def broadcast_2d_mesh(mesh, data, root=0):
    # Phase 1: Row broadcast using MPI Split
    row_comm = comm.Split(color=row, key=col)
    data = row_comm.bcast(data, root=root_col)
    
    # Phase 2: Column broadcast
    col_comm = comm.Split(color=col, key=row)
    data = col_comm.bcast(data, root=root_row)
    
    # Calculate sequential hops
    communication_steps = (cols - 1) + (rows - 1)
    return data, time, steps, messages
```

### Flooding Implementation

```python
def broadcast_flooding(mesh, data, root=0):
    # BFS level-by-level propagation
    for level in range(max_distance):
        if my_distance == level and have_data:
            # Send to ALL neighbors
            for neighbor in get_neighbors(mesh, rank):
                comm.send(data, dest=neighbor)
                msgs_sent += 1
        
        elif my_distance == level + 1:
            # Receive from any neighbor
            data = comm.recv(source=MPI.ANY_SOURCE)
    
    return data, time, max_hops, msgs_sent
```

---

## 📈 Evaluation Methodology

### Metrics

1. **Sequential Hops (Latency)**
   - Number of sequential message-passing steps
   - Lower is better

2. **Message Complexity (Bandwidth)**
   - Total messages transmitted
   - Lower is better

3. **Simulated Time**
   - Based on latency-bandwidth model: `T = hops × (ts + tw × m)`
   - ts = 10 μs (startup latency)
   - tw = 10 ns/byte (bandwidth time)
   - m = 8000 bytes (message size)

### Comparison Approach

- **Fair Comparison:** Match node counts between 2D and 3D
- **Configurations tested:**

| Nodes | 2D Mesh | 3D Mesh |
|-------|---------|---------|
| 8 | 2×4 | 2×2×2 |
| 16 | 4×4 | 2×2×4 |
| 64 | 8×8 | 4×4×4 |
| 256 | 16×16 | 4×8×8 |

---

## 📊 Experimental Results

### Sequential Hops Comparison

| Nodes | 2D DOR | 3D DOR | Improvement |
|-------|--------|--------|-------------|
| 8 | 4 | 3 | **25.0%** |
| 16 | 6 | 5 | **16.7%** |
| 64 | 14 | 9 | **35.7%** |
| 256 | 30 | 17 | **43.3%** |

### Message Complexity Comparison

| Nodes | DOR Messages | Flooding Messages | Ratio |
|-------|--------------|-------------------|-------|
| 16 | 15 | 24 | 1.6× |
| 64 | 63 | 112 | 1.8× |
| 256 | 255 | 480 | 1.9× |

### Key Findings

1. **3D mesh reduces latency by up to 43%** compared to 2D for large networks
2. **DOR uses 1.5-2× fewer messages** than flooding
3. **Improvement grows with network size** due to O(∛p) vs O(√p) scaling

---

## 🏆 Best Configuration

### Winner: **3D Mesh + DOR Algorithm**

| Criterion | Performance | Score |
|-----------|-------------|-------|
| Latency | Minimum (3(∛p - 1) hops) | ⭐⭐⭐⭐⭐ |
| Bandwidth | Optimal (p - 1 messages) | ⭐⭐⭐⭐⭐ |
| Scalability | Best (O(∛p) growth) | ⭐⭐⭐⭐⭐ |
| Predictability | Deterministic | ⭐⭐⭐⭐⭐ |
| Implementation | Moderate complexity | ⭐⭐⭐⭐ |

### Why 3D + DOR is Optimal

1. **Latency Advantage:**
   - 3D mesh has smaller diameter: 3(∛p - 1) vs 2(√p - 1)
   - For 256 nodes: 17 hops vs 30 hops (43% reduction)

2. **Bandwidth Efficiency:**
   - DOR sends exactly p - 1 messages (minimum possible)
   - Flooding wastes bandwidth with redundant messages

3. **Scalability:**
   - ∛p grows slower than √p
   - Advantage increases with network size

4. **Practical Considerations:**
   - Deterministic routing → predictable performance
   - No message duplication → reduced network congestion

### Recommendation by Use Case

| Use Case | Recommendation |
|----------|----------------|
| HPC Clusters | 3D + DOR |
| Fault-Tolerant Systems | 3D + Flooding |
| Simple Deployments | 2D + DOR |
| Small Networks (<8 nodes) | 2D + DOR |

---

## 📉 Visualizations

Generated plots in `results/` directory:

| File | Description |
|------|-------------|
| `simulation_comparison.png` | 4-panel comparison of all configurations |
| `message_complexity.png` | DOR vs Flooding message count |
| `steps_2d_vs_3d.png` | Sequential hops with improvement % |
| `combined_analysis.png` | Comprehensive 4-panel analysis |
| `scalability_analysis.png` | Theoretical scaling curves |
| `time_comparison.png` | Simulated time vs nodes |
| `comparable_configs.png` | Direct 2D vs 3D comparison |

---

## 🔬 Theoretical Background

### Latency-Bandwidth Model

```
T_msg = ts + tw × m

Where:
  ts = startup latency (time to initiate communication)
  tw = time per word (inverse of bandwidth)
  m  = message size
```

### Complexity Analysis

| Topology | Algorithm | Sequential Hops | Messages |
|----------|-----------|-----------------|----------|
| 2D Mesh | DOR | 2(√p - 1) | p - 1 |
| 2D Mesh | Flooding | 2(√p - 1) | rows×(cols-1) + cols×(rows-1) |
| 3D Mesh | DOR | 3(∛p - 1) | p - 1 |
| 3D Mesh | Flooding | 3(∛p - 1) | (x-1)yz + x(y-1)z + xy(z-1) |

---

## 🧪 Testing

Tested configurations:
- 4 processes (2×2 grid)
- 9 processes (3×3 grid)
- 16 processes (4×4 grid)
- 27 processes (3×3×3 cube)
- 64 processes (8×8 grid, 4×4×4 cube)

Verification:
- **Broadcast:** All processes receive identical data ✓
- **Gather:** Root receives data from all processes ✓

---

## 📚 References

1. MPI Standard: https://www.mpi-forum.org/
2. mpi4py Documentation: https://mpi4py.readthedocs.io/
3. Kumar et al., "Introduction to Parallel Computing"
4. Project Scope Document: `2022101099_project_scope.pdf`

---

## 📄 License

Academic project for Distributed Systems course (Semester 7), IIIT Hyderabad.

---

## 👥 Authors

- **Aniket Gupta** - 2022101099
- **Samarth Srikar** - 2022101106

*November 2024*
