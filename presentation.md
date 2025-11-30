# Distributed Systems Project: Mesh Topology Analysis
## Presentation Overview (8 Minutes)

---

### Slide 1: Introduction (1 Minute)
**Title:** Analysis of Broadcast & Gather on Mesh Topologies
**Speaker:** Aniket Gupta

**Key Points:**
- **Objective:** We built a system to simulate and analyze how computers talk to each other in a grid (Mesh) network.
- **Why it matters:** Supercomputers use this "Mesh" structure. Making it faster means faster scientific calculations.
- **Goal:** Compare different ways to send messages (Algorithms) on 2D grids vs 3D cubes.

---

### Slide 2: The Structure (Topology) (1 Minute)
**Visual:** Show a 2D Grid vs a 3D Cube.

**Key Points:**
- **2D Mesh:** Like a chessboard. Nodes connect to Up, Down, Left, Right.
- **3D Mesh:** Like a Rubik's cube. Nodes also connect Forward and Backward.
- **Insight:** 3D is better because nodes are closer to each other on average (Lower Diameter).

---

### Slide 3: Algorithms - The Basics (1 Minute)
**Visual:** Simple arrows showing path.

**1. Dimension-Order Routing (DOR):**
- **How it works:** Go all the way along X-axis, then Y, then Z.
- **Pros:** Very simple, no deadlocks.
- **Cons:** Can be slow (takes many steps).

**2. Flooding (BFS):**
- **How it works:** Shout to all neighbors, they shout to theirs. Like a ripple in water.
- **Pros:** Fastest possible path (Manhattan Distance).
- **Cons:** Sends WAY too many duplicate messages.

---

### Slide 4: Algorithms - Advanced (1 Minute)
**Visual:** Chunks moving vs Tree structure.

**3. Pipelined Broadcast:**
- **How it works:** Break a large message into small chunks. Send chunk 1, then chunk 2 immediately behind it.
- **Pros:** Great for large data. While node B receives chunk 1, node A is already sending chunk 2.

**4. Binary Tree:**
- **How it works:** 1 node tells 2, 2 tell 4, 4 tell 8. Exponential growth.
- **Pros:** Very efficient, fewer messages than flooding.

---

### Slide 5: Implementation (1 Minute)
**Tech Stack:**
- **Language:** Python (Easy to write).
- **Library:** `mpi4py` (Standard tool for distributed computing).
- **Visualization:** We built a custom web tool to SEE the messages moving.

**Code Structure:**
- `mesh_topology.py`: Builds the grid/cube.
- `broadcast.py`: The logic for sending messages.
- `visualization/`: The website to show the demo.

---

### Slide 6: Live Demo (2 Minutes)
*(Switch to Browser Visualization)*

**What to show:**
1. **2D vs 3D:** Rotate the 3D cube to show connections.
2. **DOR Animation:** Show the "L" shape path (X then Y).
3. **Flooding Animation:** Show the "Diamond" shape expanding wave.
4. **Performance:** Notice how Flooding hits everyone faster but lights up more links.

---

### Slide 7: Results & Analysis (0.5 Minute)
**Visual:** Bar charts of Time and Message Count.

**Findings:**
1. **3D vs 2D:** 3D is consistently faster because the "world" is smaller (shorter paths).
2. **Flooding:** Fastest time, but highest cost (network traffic).
3. **DOR:** Good balance of simplicity and performance.

---

### Slide 8: Conclusion (0.5 Minute)
**Takeaway:**
- For small messages or latency-critical tasks: **Use Flooding** (on 3D).
- For large data: **Use Pipelined**.
- For general use: **DOR** is robust and simple.

**Future Work:**
- Test on real hardware (not just simulation).
- Handle node failures (Fault tolerance).

**Thank You! Questions?**
