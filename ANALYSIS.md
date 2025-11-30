# Detailed Analysis: Collective Communication on Mesh Topologies

**Authors:** Aniket Gupta (2022101099) & Samarth Srikar (2022101106)  
**Date:** November 2024

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Experimental Setup](#experimental-setup)
3. [Algorithm Implementations](#algorithm-implementations)
4. [Metrics and Evaluation Methodology](#metrics-and-evaluation-methodology)
5. [Experimental Results](#experimental-results)
6. [Comparative Analysis](#comparative-analysis)
7. [Key Findings](#key-findings)
8. [Conclusions and Recommendations](#conclusions-and-recommendations)

---

## 1. Executive Summary

This document presents a comprehensive analysis of collective communication operations (Broadcast and Gather) implemented on 2D and 3D mesh topologies. We compare two routing algorithms:

- **DOR (Dimension-Ordered Routing):** Structured routing that traverses dimensions sequentially
- **Flooding (BFS-based):** Unstructured routing where nodes forward to all neighbors

### Key Findings at a Glance

| Metric | Winner | Improvement |
|--------|--------|-------------|
| Sequential Hops (256 nodes) | 3D Mesh | 43.3% fewer hops |
| Message Complexity | DOR | 1.5-2.8x fewer messages |
| Scalability | 3D + DOR | Best asymptotic behavior |
| Implementation Complexity | DOR | Simpler, deterministic |

**Best Configuration: 3D Mesh + DOR Algorithm**

---

## 2. Experimental Setup

### 2.1 Test Configurations

We tested comparable 2D and 3D configurations with matching node counts:

| Nodes | 2D Configuration | 3D Configuration |
|-------|------------------|------------------|
| 8 | 2 × 4 | 2 × 2 × 2 |
| 16 | 4 × 4 | 2 × 2 × 4 |
| 36 | 6 × 6 | 3 × 3 × 4 |
| 64 | 8 × 8 | 4 × 4 × 4 |
| 100 | 10 × 10 | 4 × 5 × 5 |
| 144 | 12 × 12 | 4 × 6 × 6 |
| 256 | 16 × 16 | 4 × 8 × 8 |

### 2.2 Operations Tested

1. **Broadcast:** Root node disseminates data to all nodes
2. **Gather:** All nodes send data to root node

---

## 3. Algorithm Implementations

### 3.1 Mesh Topology Construction

#### 2D Mesh Topology

```
Coordinate System: (row, col)
Rank to Coords: row = rank // cols, col = rank % cols
Neighbors: North (row-1), South (row+1), East (col+1), West (col-1)

Example 4×4 Mesh:
    Col 0   Col 1   Col 2   Col 3
    ┌───┐   ┌───┐   ┌───┐   ┌───┐
Row0│ 0 │───│ 1 │───│ 2 │───│ 3 │
    └─┬─┘   └─┬─┘   └─┬─┘   └─┬─┘
      │       │       │       │
    ┌─┴─┐   ┌─┴─┐   ┌─┴─┐   ┌─┴─┐
Row1│ 4 │───│ 5 │───│ 6 │───│ 7 │
    └─┬─┘   └─┬─┘   └─┬─┘   └─┬─┘
      │       │       │       │
    ┌─┴─┐   ┌─┴─┐   ┌─┴─┐   ┌─┴─┐
Row2│ 8 │───│ 9 │───│10 │───│11 │
    └─┬─┘   └─┬─┘   └─┬─┘   └─┬─┘
      │       │       │       │
    ┌─┴─┐   ┌─┴─┐   ┌─┴─┐   ┌─┴─┐
Row3│12 │───│13 │───│14 │───│15 │
    └───┘   └───┘   └───┘   └───┘
```

#### 3D Mesh Topology

```
Coordinate System: (x, y, z)
Rank to Coords: z = rank // (x_dim * y_dim)
                y = (rank % (x_dim * y_dim)) // x_dim
                x = rank % x_dim
Neighbors: ±X, ±Y, ±Z directions (up to 6 neighbors)

Example 2×2×2 Mesh:
       z=1 plane          z=0 plane
      ┌───┐───┐          ┌───┐───┐
      │ 4 │ 5 │          │ 0 │ 1 │
      ├───┼───┤          ├───┼───┤
      │ 6 │ 7 │          │ 2 │ 3 │
      └───┴───┘          └───┴───┘
           ↑________________↑
           Connected via z-axis edges
```

### 3.2 DOR (Dimension-Ordered Routing) Algorithm

DOR routes messages by traversing one dimension at a time in a fixed order.

#### 2D DOR Broadcast

```
Algorithm: broadcast_2d_dor(root, data)
─────────────────────────────────────────
Phase 1: Row Broadcast
  - Root broadcasts along its row
  - All nodes in root's row receive data
  - Sequential hops: (cols - 1)

Phase 2: Column Broadcast (Parallel)
  - Each node in root's row becomes a column leader
  - Each leader broadcasts down its column
  - Sequential hops: (rows - 1)

Total Sequential Hops: (rows - 1) + (cols - 1)
Total Messages: (cols - 1) + cols × (rows - 1)
```

**Visual Example (4×4 mesh, root=0):**

```
Phase 1: Row broadcast         Phase 2: Column broadcast
    0 → 1 → 2 → 3                  0   1   2   3
                                   ↓   ↓   ↓   ↓
                                   4   5   6   7
                                   ↓   ↓   ↓   ↓
                                   8   9  10  11
                                   ↓   ↓   ↓   ↓
                                  12  13  14  15

Hops: 3 (row) + 3 (col) = 6 total
```

#### 3D DOR Broadcast

```
Algorithm: broadcast_3d_dor(root, data)
─────────────────────────────────────────
Phase 1: X-axis Broadcast
  - Root broadcasts along x-axis
  - Sequential hops: (x_dim - 1)

Phase 2: Y-axis Broadcast (Parallel across X)
  - Each x-line node broadcasts along y-axis
  - Sequential hops: (y_dim - 1)

Phase 3: Z-axis Broadcast (Parallel across XY)
  - Each xy-plane node broadcasts along z-axis
  - Sequential hops: (z_dim - 1)

Total Sequential Hops: (x - 1) + (y - 1) + (z - 1)
Total Messages: (x-1) + x×(y-1) + x×y×(z-1)
```

### 3.3 Flooding (BFS) Algorithm

Flooding uses breadth-first search where each node forwards to ALL neighbors.

#### Flooding Broadcast

```
Algorithm: broadcast_flooding(root, data)
─────────────────────────────────────────
Level 0: Root has data

Level 1: Root sends to ALL neighbors
         Each neighbor receives data

Level 2: Level-1 nodes send to ALL their neighbors
         (excluding the sender node to avoid redundant messages)

... continues until all nodes receive data

Sequential Hops: Maximum Manhattan distance from root
                 = max(|x_root - x_i| + |y_root - y_i| + ...)

Total Messages: Each edge carries exactly one message
                = number of edges in mesh
                (since nodes don't send back to the sender)
```

**Key Difference from DOR:**

| Aspect | DOR | Flooding |
|--------|-----|----------|
| Routing | Dimension-by-dimension | All neighbors simultaneously |
| Message Count | O(p) = p-1 | O(edges) |
| Path Determinism | Deterministic | Non-deterministic |
| Redundancy | None | Minimal (no back-messages) |

---

## 4. Metrics and Evaluation Methodology

### 4.1 Primary Metrics

#### Sequential Hops (Latency Metric)

**Definition:** The maximum number of sequential message-passing steps required for all nodes to receive/send data.

**Formulas:**
- **2D DOR:** `(rows - 1) + (cols - 1) = 2(√p - 1)` for square mesh
- **3D DOR:** `(x - 1) + (y - 1) + (z - 1) = 3(∛p - 1)` for cube mesh
- **Flooding:** Maximum Manhattan distance from root to any node

#### Message Complexity (Bandwidth Metric)

**Definition:** Total number of messages transmitted across the network.

**Formulas:**
- **2D DOR:** `(cols - 1) + cols × (rows - 1) = p - 1`
- **3D DOR:** `(x - 1) + x × (y - 1) + x × y × (z - 1) = p - 1`
- **2D Flooding:** `rows × (cols - 1) + cols × (rows - 1)` (number of edges)
- **3D Flooding:** `(x-1)×y×z + x×(y-1)×z + x×y×(z-1)` (number of edges)

#### Simulated Time

**Latency-Bandwidth Model:**
```
T = sequential_hops × (ts + tw × m)

Where:
  ts = startup latency (10 μs)
  tw = time per byte (10 ns)
  m  = message size (8000 bytes)
```

### 4.2 Comparison Methodology

1. **Fair Comparison:** Same or similar node counts for 2D vs 3D
2. **Scalability Analysis:** Test across increasing node counts
3. **Percentage Improvement:** `(metric_2D - metric_3D) / metric_2D × 100%`

---

## 5. Experimental Results

### 5.1 Sequential Hops Comparison

#### DOR Algorithm Results

| Nodes | 2D Config | 2D Hops | 3D Config | 3D Hops | Improvement |
|-------|-----------|---------|-----------|---------|-------------|
| 8 | 2×4 | 4 | 2×2×2 | 3 | 25.0% |
| 16 | 4×4 | 6 | 2×2×4 | 5 | 16.7% |
| 36 | 6×6 | 10 | 3×3×4 | 7 | 30.0% |
| 64 | 8×8 | 14 | 4×4×4 | 9 | 35.7% |
| 100 | 10×10 | 18 | 4×5×5 | 11 | 38.9% |
| 144 | 12×12 | 22 | 4×6×6 | 13 | 40.9% |
| 256 | 16×16 | 30 | 4×8×8 | 17 | 43.3% |

**Observation:** 3D mesh consistently outperforms 2D, with improvement increasing as node count grows.

#### Flooding Algorithm Results

| Nodes | 2D Hops | 3D Hops | Improvement |
|-------|---------|---------|-------------|
| 8 | 4 | 3 | 25.0% |
| 16 | 6 | 5 | 16.7% |
| 36 | 10 | 7 | 30.0% |
| 64 | 14 | 9 | 35.7% |
| 100 | 18 | 11 | 38.9% |
| 256 | 30 | 17 | 43.3% |

**Note:** For DOR and Flooding, sequential hops are the same because both are bounded by the mesh diameter.

### 5.2 Message Complexity Comparison

| Nodes | 2D DOR | 2D Flooding | Ratio | 3D DOR | 3D Flooding | Ratio |
|-------|--------|-------------|-------|--------|-------------|-------|
| 8 | 7 | 10 | 1.4x | 7 | 12 | 1.7x |
| 16 | 15 | 24 | 1.6x | 13 | 24 | 1.8x |
| 36 | 35 | 60 | 1.7x | 31 | 66 | 2.1x |
| 64 | 63 | 112 | 1.8x | 57 | 144 | 2.5x |
| 100 | 99 | 180 | 1.8x | 91 | 210 | 2.3x |
| 256 | 255 | 480 | 1.9x | 241 | 672 | 2.8x |

**Key Finding:** Flooding requires 1.5-2.8x more messages than DOR. Since each node sends to all neighbors except the sender, the number of messages equals the number of edges in the mesh.

### 5.3 Simulated Time Comparison

Based on latency-bandwidth model (ts=10μs, tw=10ns, m=8000 bytes):

| Nodes | 2D DOR Time | 3D DOR Time | Improvement |
|-------|-------------|-------------|-------------|
| 8 | 360 μs | 270 μs | 25.0% |
| 16 | 540 μs | 450 μs | 16.7% |
| 64 | 1260 μs | 810 μs | 35.7% |
| 256 | 2700 μs | 1530 μs | 43.3% |

### 5.4 Scalability Analysis

**Theoretical Complexity:**

| Topology | Sequential Hops | Growth Rate |
|----------|-----------------|-------------|
| 2D Mesh | 2(√p - 1) | O(√p) |
| 3D Mesh | 3(∛p - 1) | O(∛p) |

**Asymptotic Comparison:**

For large p:
- 2D: √p grows faster
- 3D: ∛p grows slower

**Crossover Point:** 3D becomes more efficient when `3(∛p - 1) < 2(√p - 1)`

Solving: p > 8 (approximately)

This means 3D mesh is more efficient for **all practical sizes (p ≥ 8)**.

---

## 6. Comparative Analysis

### 6.1 2D vs 3D Mesh Topology

| Aspect | 2D Mesh | 3D Mesh | Winner |
|--------|---------|---------|--------|
| Diameter | 2(√p - 1) | 3(∛p - 1) | 3D (for p > 8) |
| Avg. Path Length | Lower for small p | Lower for large p | Depends on p |
| Implementation | Simpler | More complex | 2D |
| Physical Realization | Easier | Harder | 2D |
| Scalability | O(√p) | O(∛p) | 3D |
| Node Degree | 4 (max) | 6 (max) | 2D (fewer connections) |

### 6.2 DOR vs Flooding

| Aspect | DOR | Flooding | Winner |
|--------|-----|----------|--------|
| Sequential Hops | Same as diameter | Same as diameter | Tie |
| Message Count | O(p) = p-1 | O(edges) | DOR |
| Fault Tolerance | Low | High | Flooding |
| Implementation | Simple | Simple | Tie |
| Determinism | Deterministic | Non-deterministic | DOR |
| Network Load | Balanced | Bursty | DOR |

### 6.3 Combined Analysis: Best Configurations

| Configuration | Sequential Hops | Messages | Time | Overall Score |
|---------------|-----------------|----------|------|---------------|
| 2D + DOR | ●●○○ | ●●●● | ●●○○ | Good |
| 2D + Flooding | ●●○○ | ●○○○ | ●●○○ | Fair |
| 3D + DOR | ●●●● | ●●●● | ●●●● | **Excellent** |
| 3D + Flooding | ●●●● | ●●○○ | ●●●● | Good |

---

## 7. Key Findings

### 7.1 Primary Findings

1. **3D Mesh Superiority for Large Networks**
   - 3D mesh requires up to 43% fewer sequential hops for 256 nodes
   - The advantage grows with network size (O(∛p) vs O(√p))
   - Crossover point is approximately p = 8 nodes

2. **DOR is More Efficient than Flooding**
   - DOR uses 1.5-2.8x fewer messages than flooding
   - Both have the same sequential hops (bounded by diameter)
   - DOR provides deterministic, predictable routing
   - Flooding message count = number of edges (each edge used once)

3. **Scalability Advantage of 3D**
   - For 256 nodes: 2D needs 30 hops, 3D needs only 17 hops
   - For 1000 nodes: 2D would need ~62 hops, 3D would need ~27 hops
   - This translates to significant time savings in large-scale systems

### 7.2 Secondary Findings

4. **Message Complexity Matters for Bandwidth**
   - Flooding sends messages on all edges, while DOR uses only p-1 messages
   - DOR optimizes bandwidth usage with minimal messages

5. **Trade-offs Exist**
   - 3D mesh requires more complex address mapping
   - 3D mesh has higher node degree (6 vs 4), requiring more links
   - Flooding provides implicit fault tolerance

### 7.3 Unexpected Observations

6. **Flooding and DOR have Same Latency**
   - Despite different routing strategies, both achieve optimal diameter-bounded latency
   - The difference is purely in message complexity

7. **3D Improvement is Non-linear**
   - The percentage improvement of 3D over 2D increases with p
   - This is because the ratio √p/∛p increases with p

---

## 8. Conclusions and Recommendations

### 8.1 Final Conclusion

**The optimal configuration for collective communication on mesh topologies is:**

## 🏆 3D Mesh + DOR Algorithm

### Justification:

| Criterion | 3D + DOR Performance |
|-----------|---------------------|
| **Latency** | Minimum (3(∛p - 1) hops) |
| **Bandwidth** | Optimal (p - 1 messages) |
| **Scalability** | Best (O(∛p) growth) |
| **Predictability** | Deterministic routing |
| **Implementation** | Moderate complexity |

### 8.2 Recommendations by Use Case

| Use Case | Recommended Configuration | Reason |
|----------|--------------------------|--------|
| **High-Performance Computing** | 3D + DOR | Minimum latency, optimal bandwidth |
| **Fault-Tolerant Systems** | 3D + Flooding | Implicit redundancy |
| **Simple Implementation** | 2D + DOR | Easiest to implement |
| **Small Networks (< 8 nodes)** | 2D + DOR | Simpler with minimal penalty |
| **Large Networks (> 64 nodes)** | 3D + DOR | Significant latency reduction |

### 8.3 Quantitative Summary

For a **256-node network** (16×16 vs 4×8×8):

| Metric | 2D + DOR | 3D + DOR | Savings |
|--------|----------|----------|---------|
| Sequential Hops | 30 | 17 | 43.3% |
| Total Messages | 255 | 241 | 5.5% |
| Simulated Time | 2700 μs | 1530 μs | 43.3% |

### 8.4 Future Work

1. **Non-uniform Mesh Dimensions:** Explore optimal dimension ratios
2. **Hybrid Approaches:** Combine DOR efficiency with flooding fault tolerance
3. **Real MPI Benchmarks:** Validate simulation with actual MPI experiments
4. **Torus Topology:** Compare with wrap-around connections
5. **Adaptive Routing:** Dynamic algorithm selection based on network conditions

---

## Appendix: Generated Visualizations

The following plots are available in the `results/` directory:

| File | Description |
|------|-------------|
| `simulation_comparison.png` | 4-panel comparison of all configurations |
| `time_comparison.png` | Simulated time vs node count |
| `scalability_analysis.png` | Theoretical vs actual scalability curves |
| `comparable_configs.png` | Direct 2D vs 3D comparison |
| `message_complexity.png` | DOR vs Flooding message count |
| `steps_2d_vs_3d.png` | Sequential hops with improvement percentages |
| `combined_analysis.png` | Comprehensive 4-panel analysis |

---

*End of Analysis Document*
