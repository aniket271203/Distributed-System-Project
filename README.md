# Collective Communication Operations on Mesh Topologies

**Operations:** Broadcast (Bcast) and Gather  
**Topologies:** 2D and 3D Mesh  
**Author:** Aniket Gupta - 2022101099 & Samarth Srikar - 2022101106  
**Course:** Distributed Systems (Sem-7)  
**Date:** November 2024

---

## 1. Project Overview

This project implements and analyzes collective communication operations on mesh-based networks, focusing on Broadcast and Gather operations. The implementation follows the exact specifications from the project scope document.

### Features
- ✅ **Mesh Topology Creation** (2D and 3D)
- ✅ **Broadcast Operation** on 2D and 3D meshes
- ✅ **Gather Operation** on 2D and 3D meshes
- ✅ **Performance Analysis** using latency-bandwidth model
- ✅ **Comparative Analysis** between 2D and 3D topologies

---

## 2. File Structure

```
Project/
├── mesh_topology.py         # Mesh topology implementation (2D & 3D)
├── broadcast.py             # Broadcast operations
├── gather.py                # Gather operations
├── main.py                  # Main driver program
├── performance_analysis.py  # Performance measurement and analysis
├── collect_data.py          # Script to collect real performance data
├── plot_real_data.py        # Script to plot results from collected data
├── benchmark.slurm          # SLURM script for cluster benchmarking
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

---

## 3. Implementation Details

### 3.1 Mesh Topologies (`mesh_topology.py`)
- **`Mesh2D`**: Represents a 2D grid topology ($ \sqrt{p} \times \sqrt{p} $).
- **`Mesh3D`**: Represents a 3D cube topology ($ \sqrt[3]{p} \times \sqrt[3]{p} \times \sqrt[3]{p} $).
- Handles neighbor calculation and coordinate mapping (rank $\leftrightarrow$ coords).

### 3.2 Broadcast Operation (`broadcast.py`)

**2D Mesh Broadcast Algorithm:**
1.  Broadcast message along the root's row.
2.  Row nodes become column leaders.
3.  Each leader broadcasts message down its column.
*Runtime:* $ T_{bcast}^{2D} = 2(\sqrt{p} - 1)t_s + (p - 1)t_w \cdot m $

**3D Mesh Broadcast Algorithm:**
1.  Broadcast along root's x-axis.
2.  Broadcast along y-axis in root's plane.
3.  Broadcast along z-axis from each plane leader.
*Runtime:* $ T_{bcast}^{3D} = 3(\sqrt[3]{p} - 1)t_s + (p - 1)t_w \cdot m $

### 3.3 Gather Operation (`gather.py`)

**2D Mesh Gather Algorithm:**
1.  Each row gathers data toward row leaders (first column).
2.  Row leaders send data up the root column to root.
*Runtime:* $ T_{gather}^{2D} = 2(\sqrt{p} - 1)t_s + (p - 1)t_w \cdot m $

**3D Mesh Gather Algorithm:**
1.  Gather along x-axis to $ x=0 $ plane leaders.
2.  Gather along y-axis to $ (x=0, y=0) $ line leaders.
3.  Gather along z-axis to root.
*Runtime:* $ T_{gather}^{3D} = 3(\sqrt[3]{p} - 1)t_s + (p - 1)t_w \cdot m $

---

## 4. Usage

### Installation
1.  Install MPI (if not already installed):
    ```bash
    sudo apt-get install openmpi-bin libopenmpi-dev  # Ubuntu
    ```
2.  Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Project
**Run all tests (default 4 processes):**
```bash
mpiexec -n 4 python main.py
```

**Run with specific process counts:**
```bash
mpiexec -n 16 python main.py      # 4x4 grid
mpiexec -n 27 python main.py      # 3x3x3 cube
```

**Run Performance Analysis:**
```bash
mpiexec -n 16 python performance_analysis.py
```

**Run Simulation (structured efficiency metrics):**
```bash
# Broadcast & Gather on 2D and 3D for message sizes 256 and 1024
mpiexec -n 16 python simulate_mesh.py --topology both --collectives broadcast,gather --data-sizes 256,1024 --algorithms standard,pipelined

# 3D only (requires process count that factors into a 3D mesh)
mpiexec -n 8 python simulate_mesh.py --topology 3d --collectives broadcast --algorithms standard --data-sizes 2048
```
Outputs are saved to `results/simulation_p<process_count>.json` containing:
- `avg_time_sec`, `max_time_sec`
- `rounds_measured` (communication rounds) vs `rounds_theoretical`
- `messages_counted` (sum of sends/receives) vs minimal spanning tree bound `messages_minimal`

**Run Benchmarks on Cluster (SLURM):**
```bash
sbatch benchmark.slurm
```

---

## 5. Performance Analysis & Results

### 5.1 Theoretical Analysis
The project uses the standard latency-bandwidth model: $ T_{msg} = t_s + t_w \cdot m $

| Metric | 2D Mesh | 3D Mesh | Improvement (p=27) |
|--------|---------|---------|-------------------|
| **Diameter** | $ 2\sqrt{p} - 2 $ | $ 3\sqrt[3]{p} - 3 $ | **40% reduction** |
| **Comm. Steps** | $ 2(\sqrt{p} - 1) $ | $ 3(\sqrt[3]{p} - 1) $ | **40% fewer steps** |
| **Bisection Width** | $ \sqrt{p} $ | $ (\sqrt[3]{p})^2 $ | **50% higher** |

### 5.2 Generated Visualizations
The `results/` directory will contain the following graphs after running the analysis:

1.  **`mesh_topology_diagrams.png`**: Visual representation of 2D and 3D mesh structures.
2.  **`2d_vs_3d_comparison.png`**: Theoretical comparison of communication steps.
3.  **`scalability_2d.png`**: Analysis of how 2D mesh scales with process count.
4.  **`latency_bandwidth_model.png`**: Impact of latency and bandwidth on execution time.
5.  **`network_metrics.png`**: Comparison of network diameter and bisection width.
6.  **`real_performance_vs_datasize.png`**: Actual measured performance from MPI runs.
7.  **`real_2d_vs_3d_comparison.png`**: Bar charts comparing actual execution times.
8.  **`speedup_analysis.png`**: Speedup of 3D mesh over 2D mesh.

### 5.3 Key Findings
1.  **3D Mesh is Superior for Large $ p $**: The 3D topology significantly reduces the network diameter and communication steps compared to 2D, leading to better scalability.
2.  **Latency Dominates for Small Messages**: For small data sizes, the startup latency ($ t_s $) is the bottleneck.
3.  **Bandwidth Dominates for Large Messages**: As message size ($ m $) increases, the transmission time ($ t_w \cdot m $) becomes the dominant factor.

---

## 6. References
- MPI Standard: https://www.mpi-forum.org/
- mpi4py Documentation: https://mpi4py.readthedocs.io/
