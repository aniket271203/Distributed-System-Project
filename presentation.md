# Distributed Systems Project: Mesh Topology Analysis

---

## 1. Introduction & Problem Statement

**Objective:**
Implement and analyze collective communication operations on 2D and 3D mesh topologies.

**Key Operations:**
- **Broadcast:** One-to-All communication.
- **Gather:** All-to-One communication.

**Why Mesh?**
- Scalable topology used in high-performance computing (HPC).
- 3D meshes offer lower diameter than 2D meshes, potentially reducing latency.

---

## 2. Approach & Architecture

### Topologies Implemented
- **2D Mesh:** $\sqrt{P} \times \sqrt{P}$ Grid.
- **3D Mesh:** $\sqrt[3]{P} \times \sqrt[3]{P} \times \sqrt[3]{P}$ Cube.

### Algorithms Compared

**1. Dimension-Order Routing (DOR)**
- **Logic:** Propagate along axes sequentially (X $\rightarrow$ Y $\rightarrow$ Z).
- **Pros:** Simple, low message count.
- **Cons:** Higher latency (steps).

**2. Flooding (BFS)**
- **Logic:** Propagate to all unvisited neighbors (Wavefront).
- **Pros:** Optimal latency (Manhattan distance).
- **Cons:** High message redundancy.

---

## 3. Implementation

**Tech Stack:**
- **Core:** Python + MPI (`mpi4py`).
- **Visualization:** HTML5 Canvas + JavaScript.

**Key Components:**
- `mesh_topology.py`: Dynamic topology generation.
- `flooding.py`: BFS implementation.
- `visualization/`: Interactive web tool.

---

## 4. Live Demo (Visualization)

*(Switch to Browser)*

**Features Demonstrated:**
1. **2D vs 3D:** Visualizing the structure.
2. **DOR vs Flooding:**
   - **DOR:** Axis-aligned propagation.
   - **Flooding:** Diamond-shaped wavefront.
3. **Interactive 3D:** Rotation and Zoom.

---

## 5. Experimental Results

### Execution Time
- **Observation:** Flooding is theoretically faster in steps but has higher overhead in Python due to message volume.
- **3D vs 2D:** 3D consistently requires fewer steps.

### Message Complexity
- **DOR:** $O(P)$ messages (Spanning tree).
- **Flooding:** $O(P \times Degree)$ messages (High redundancy).

*(Show plots from results/ directory)*

---

## 6. Conclusion

1. **3D > 2D:** For large systems, 3D topology significantly reduces communication diameter.
2. **Algorithm Choice:**
   - Use **DOR** for bandwidth-constrained networks.
   - Use **Flooding** for latency-critical, low-diameter networks.
3. **Success:** Successfully implemented all requirements and provided a robust visualization tool.

---

## Q&A

**Thank You!**
