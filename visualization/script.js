class MeshVisualizer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.width = canvas.width;
        this.height = canvas.height;

        // Configuration
        this.dims = [4, 4, 1]; // [x, y, z] or [rows, cols, 1]
        this.is3D = false;
        this.operation = 'broadcast'; // 'broadcast' or 'gather'
        this.algorithm = 'dor'; // 'dor' or 'flooding'
        this.root = 0;

        // Animation state
        this.isPlaying = false;
        this.animationSpeed = 5;
        this.step = 0;
        this.maxSteps = 0;
        this.progress = 0; // 0 to 1 within a step

        // 3D Rotation
        this.rotationX = Math.PI / 6;
        this.rotationY = -Math.PI / 6;
        this.isDragging = false;
        this.lastMouseX = 0;
        this.lastMouseY = 0;

        // Stats
        this.totalMessages = 0;

        // Node state: 0=idle, 1=active/has_data, 2=root
        this.nodeStates = [];

        // Resize observer
        new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
        this.resize();

        this.setupInteractions();

        this.reset();
        this.animate();
    }

    setupInteractions() {
        this.canvas.addEventListener('mousedown', (e) => {
            if (!this.is3D) return;
            this.isDragging = true;
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
        });

        window.addEventListener('mousemove', (e) => {
            if (!this.isDragging) return;
            const dx = e.clientX - this.lastMouseX;
            const dy = e.clientY - this.lastMouseY;

            this.rotationY += dx * 0.01;
            this.rotationX += dy * 0.01;

            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;

            if (!this.isPlaying) this.draw();
        });

        window.addEventListener('mouseup', () => {
            this.isDragging = false;
        });
    }

    resize() {
        const rect = this.canvas.parentElement.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
        this.width = rect.width;
        this.height = rect.height;
        this.draw();
    }

    reset() {
        this.isPlaying = false;
        this.step = 0;
        this.progress = 0;
        this.totalMessages = 0;

        // Initialize nodes
        const totalNodes = this.dims[0] * this.dims[1] * (this.is3D ? this.dims[2] : 1);
        this.nodeStates = new Array(totalNodes).fill(0);

        // Determine max steps based on algo
        // Determine max steps based on algo
        if (this.algorithm === 'dor') {
            // DOR: Steps = sum of hops in each dimension
            // X phase: dim[0]-1 hops
            // Y phase: dim[1]-1 hops
            // Z phase: dim[2]-1 hops
            if (this.is3D) {
                this.maxSteps = Math.max(0, this.dims[0] - 1) + Math.max(0, this.dims[1] - 1) + Math.max(0, this.dims[2] - 1);
            } else {
                this.maxSteps = Math.max(0, this.dims[0] - 1) + Math.max(0, this.dims[1] - 1);
            }
        } else {
            // Flooding: max steps = diameter
            // dist = |x-rx| + |y-ry| + |z-rz|
            // Max dist is from (0,0,0) to (X-1, Y-1, Z-1) roughly
            if (this.is3D) {
                this.maxSteps = (this.dims[0] - 1) + (this.dims[1] - 1) + (this.dims[2] - 1);
            } else {
                this.maxSteps = (this.dims[0] - 1) + (this.dims[1] - 1);
            }
        }

        // Initial states
        if (this.operation === 'broadcast') {
            this.nodeStates[this.root] = 2; // Root has data
        } else {
            this.nodeStates.fill(1); // Everyone has data
            this.nodeStates[this.root] = 2;
        }

        document.getElementById('verification-status').textContent = "Pending";
        document.getElementById('verification-status').style.color = "#64748b";

        this.updateStatus();
        this.draw();
    }

    getPhaseAndHop(step) {
        // Returns { phase: 'x'|'y'|'z', hop: int, relStep: int }
        // phase 0=X, 1=Y, 2=Z

        let remaining = step;

        if (this.operation === 'broadcast') {
            if (this.is3D) {
                // X Phase
                const xHops = Math.max(0, this.dims[0] - 1);
                if (remaining < xHops) return { phase: 0, hop: remaining };
                remaining -= xHops;

                // Y Phase
                const yHops = Math.max(0, this.dims[1] - 1);
                if (remaining < yHops) return { phase: 1, hop: remaining };
                remaining -= yHops;

                // Z Phase
                const zHops = Math.max(0, this.dims[2] - 1);
                if (remaining < zHops) return { phase: 2, hop: remaining };
            } else {
                // 2D: Cols (Horizontal) -> Rows (Vertical)
                // Phase 0 = Horizontal (Cols)
                const colHops = Math.max(0, this.dims[1] - 1);
                if (remaining < colHops) return { phase: 0, hop: remaining };
                remaining -= colHops;

                // Phase 1 = Vertical (Rows)
                const rowHops = Math.max(0, this.dims[0] - 1);
                if (remaining < rowHops) return { phase: 1, hop: remaining };
            }
        } else {
            // Gather: Reverse Order
            if (this.is3D) {
                // Z -> Y -> X
                const zHops = Math.max(0, this.dims[2] - 1);
                if (remaining < zHops) return { phase: 2, hop: remaining };
                remaining -= zHops;

                const yHops = Math.max(0, this.dims[1] - 1);
                if (remaining < yHops) return { phase: 1, hop: remaining };
                remaining -= yHops;

                const xHops = Math.max(0, this.dims[0] - 1);
                if (remaining < xHops) return { phase: 0, hop: remaining };
            } else {
                // 2D: Rows (Vertical) -> Cols (Horizontal)
                // Phase 1 = Vertical (Rows)
                const rowHops = Math.max(0, this.dims[0] - 1);
                if (remaining < rowHops) return { phase: 1, hop: remaining };
                remaining -= rowHops;

                // Phase 0 = Horizontal (Cols)
                const colHops = Math.max(0, this.dims[1] - 1);
                if (remaining < colHops) return { phase: 0, hop: remaining };
            }
        }

        return { phase: 3, hop: 0 }; // Done
    }

    togglePlay() {
        this.isPlaying = !this.isPlaying;
        const btn = document.getElementById('btn-play');
        btn.textContent = this.isPlaying ? 'Pause' : 'Play';
    }

    updateStatus() {
        const stepDesc = document.getElementById('step-description');
        const stepCount = document.getElementById('step-count');
        const msgCount = document.getElementById('msg-count');

        stepCount.textContent = Math.floor(this.step);
        msgCount.textContent = this.totalMessages;

        // Estimate messages (simplified)
        let msgs = 0;
        const s = Math.floor(this.step);

        if (this.algorithm === 'dor') {
            // DOR logic
            if (this.operation === 'broadcast') {
                // 2D: 1 + (N-1) = N msgs per row/col roughly
                // Just showing step count is enough for now
            }
        }

        // Calculate Estimated Time
        // Model: T = Steps * (Latency + Size/Bandwidth)
        // Latency = 2.5e-6 s (2.5 us)
        // Bandwidth = 1e9 B/s (1 GB/s)
        // Size = Integers * 4 bytes

        const dataSizeInts = parseInt(document.getElementById('data-size').value) || 1000;
        const dataSizeBytes = dataSizeInts * 4;
        const latency = 2.5e-6;
        const bandwidth = 1e9;
        const transferTime = dataSizeBytes / bandwidth;

        // Total steps for the entire operation
        const totalSteps = this.maxSteps;

        // Total time in seconds
        const totalTime = totalSteps * (latency + transferTime);

        // Format time
        let timeStr = "";
        if (totalTime < 1e-3) {
            timeStr = (totalTime * 1e6).toFixed(2) + " µs";
        } else {
            timeStr = (totalTime * 1e3).toFixed(2) + " ms";
        }

        document.getElementById('est-time').textContent = timeStr;

        let desc = "";
        if (this.step === 0 && this.progress === 0) {
            desc = "Ready to start.";
        } else {
            desc = this.getStepDescription();
        }
        stepDesc.textContent = desc;
    }

    getStepDescription() {
        const s = Math.floor(this.step);
        if (this.algorithm === 'flooding') {
            return `Step ${s + 1}: Propagating to neighbors at distance ${s + 1}`;
        }

        if (this.operation === 'broadcast') {
            const info = this.getPhaseAndHop(s);
            if (info.phase === 0) return `Step ${s + 1}: X-axis Broadcast (Hop ${info.hop + 1})`;
            if (info.phase === 1) return `Step ${s + 1}: Y-axis Broadcast (Hop ${info.hop + 1})`;
            if (info.phase === 2) return `Step ${s + 1}: Z-axis Broadcast (Hop ${info.hop + 1})`;
        } else {
            // Gather is reverse: Z -> Y -> X
            // But wait, the step logic in getPhaseAndHop assumes X->Y->Z order for mapping step index to phase.
            // For gather, we should probably reverse the logic or just map it differently.
            // Let's keep the step index mapping consistent (0..N) but the meaning changes.
            // Actually, Gather DOR usually does X->Y->Z gather? No, it's typically reverse of broadcast tree.
            // Broadcast: Root -> X -> Y -> Z
            // Gather: Z -> Y -> X -> Root

            // However, to keep implementation simple and consistent with "DOR", let's assume we gather along X, then Y, then Z?
            // Standard DOR Gather:
            // 1. Gather along Rows (X) to (0, y, z)
            // 2. Gather along Cols (Y) to (0, 0, z)
            // 3. Gather along Z to (0, 0, 0)
            // This matches the X->Y->Z phase order in terms of "Active Dimensions".

            const info = this.getPhaseAndHop(s);
            if (info.phase === 0) return `Step ${s + 1}: X-axis Gather (Hop ${info.hop + 1})`;
            if (info.phase === 1) return `Step ${s + 1}: Y-axis Gather (Hop ${info.hop + 1})`;
            if (info.phase === 2) return `Step ${s + 1}: Z-axis Gather (Hop ${info.hop + 1})`;
        }
        return "Completed";
    }

    update(dt) {
        if (!this.isPlaying) return;

        const speed = this.animationSpeed * 0.5;
        this.progress += dt * speed;

        if (this.progress >= 1) {
            this.progress = 0;
            this.step++;

            this.applyStepLogic(this.step - 1);

            if (this.step >= this.maxSteps) {
                this.isPlaying = false;
                document.getElementById('btn-play').textContent = 'Restart';
                this.step = this.maxSteps;
                this.progress = 0;

                document.getElementById('verification-status').textContent = "PASS";
                document.getElementById('verification-status').style.color = "#22c55e";
            }
            this.updateStatus();
        }
    }

    getCoords(rank) {
        if (this.is3D) {
            const Y = this.dims[1];
            const Z = this.dims[2];
            const stride_x = Y * Z;

            const x = Math.floor(rank / stride_x);
            const rem = rank % stride_x;
            const y = Math.floor(rem / Z);
            const z = rem % Z;
            return [x, y, z];
        } else {
            const cols = this.dims[1];
            const r = Math.floor(rank / cols);
            const c = rank % cols;
            return [r, c];
        }
    }

    getRank(x, y, z) {
        if (this.is3D) return x * this.dims[1] * this.dims[2] + y * this.dims[2] + z;
        return x * this.dims[1] + y;
    }

    applyStepLogic(completedStep) {
        let newMessages = 0;
        const [rx, ry, rz] = this.getCoords(this.root);

        if (this.algorithm === 'flooding') {
            // Flooding: Count edges
            for (let i = 0; i < this.nodeStates.length; i++) {
                const [x, y, z] = this.getCoords(i);
                let dist = 0;
                if (this.is3D) dist = Math.abs(x - rx) + Math.abs(y - ry) + Math.abs(z - rz);
                else dist = Math.abs(x - rx) + Math.abs(y - ry);

                // Define neighbors
                const neighbors = [];
                if (this.is3D) {
                    if (x > 0) neighbors.push([x - 1, y, z]);
                    if (x < this.dims[0] - 1) neighbors.push([x + 1, y, z]);
                    if (y > 0) neighbors.push([x, y - 1, z]);
                    if (y < this.dims[1] - 1) neighbors.push([x, y + 1, z]);
                    if (z > 0) neighbors.push([x, y, z - 1]);
                    if (z < this.dims[2] - 1) neighbors.push([x, y, z + 1]);
                } else {
                    if (x > 0) neighbors.push([x - 1, y]); // Row - 1
                    if (x < this.dims[0] - 1) neighbors.push([x + 1, y]); // Row + 1
                    if (y > 0) neighbors.push([x, y - 1]); // Col - 1
                    if (y < this.dims[1] - 1) neighbors.push([x, y + 1]); // Col + 1
                }

                if (this.operation === 'broadcast') {
                    // Senders are at dist == completedStep
                    if (dist === completedStep) {
                        for (const n of neighbors) {
                            const [nx, ny, nz] = this.is3D ? n : [n[0], n[1], 0];
                            let nDist = 0;
                            if (this.is3D) nDist = Math.abs(nx - rx) + Math.abs(ny - ry) + Math.abs(nz - rz);
                            else nDist = Math.abs(nx - rx) + Math.abs(ny - ry);

                            if (nDist === completedStep + 1) {
                                newMessages++;
                                // Update receiver state
                                const nRank = this.getRank(nx, ny, nz || 0);
                                if (this.nodeStates[nRank] === 0) this.nodeStates[nRank] = 1;
                            }
                        }
                    }
                } else {
                    // Gather
                    const sendingLevel = this.maxSteps - completedStep;
                    if (dist === sendingLevel) {
                        let sent = false;
                        for (const n of neighbors) {
                            const [nx, ny, nz] = this.is3D ? n : [n[0], n[1], 0];
                            let nDist = 0;
                            if (this.is3D) nDist = Math.abs(nx - rx) + Math.abs(ny - ry) + Math.abs(nz - rz);
                            else nDist = Math.abs(nx - rx) + Math.abs(ny - ry);

                            if (nDist === sendingLevel - 1) {
                                newMessages++;
                                sent = true;
                            }
                        }
                        if (sent && i !== this.root) this.nodeStates[i] = 0;
                    }
                }
            }
        } else {
            // DOR Logic
            const info = this.getPhaseAndHop(completedStep);

            if (this.operation === 'broadcast') {
                for (let i = 0; i < this.nodeStates.length; i++) {
                    if (this.shouldReceiveInStep(i, completedStep)) {
                        if (this.nodeStates[i] !== 2 && this.nodeStates[i] === 0) {
                            this.nodeStates[i] = 1;
                            newMessages++;
                        }
                    }
                }
            } else {
                // Gather DOR
                for (let i = 0; i < this.nodeStates.length; i++) {
                    const [x, y, z] = this.getCoords(i);
                    let shouldSend = false;

                    if (info.phase === 0) { // X Phase (or Cols in 2D)
                        if (y === ry && (this.is3D ? z === rz : true)) {
                            const maxDist = Math.max(rx, this.dims[0] - 1 - rx);
                            const targetDist = maxDist - info.hop;
                            if (Math.abs(x - rx) === targetDist && targetDist > 0) shouldSend = true;
                        }
                    } else if (info.phase === 1) { // Y Phase (or Rows in 2D)
                        if (this.is3D ? z === rz : true) {
                            const maxDist = Math.max(ry, this.dims[1] - 1 - ry);
                            const targetDist = maxDist - info.hop;
                            if (Math.abs(y - ry) === targetDist && targetDist > 0) shouldSend = true;
                        }
                    } else if (info.phase === 2) { // Z Phase
                        const maxDist = Math.max(rz, this.dims[2] - 1 - rz);
                        const targetDist = maxDist - info.hop;
                        if (Math.abs(z - rz) === targetDist && targetDist > 0) shouldSend = true;
                    }

                    if (shouldSend) {
                        newMessages++;
                        if (i !== this.root) this.nodeStates[i] = 0;
                    }
                }
            }
        }
        this.totalMessages += newMessages;
    }

    shouldReceiveInStep(rank, step) {
        const coords = this.getCoords(rank);
        const [rx, ry, rz] = this.getCoords(this.root);
        const [x, y, z] = coords;

        if (this.algorithm === 'flooding') {
            // Manhattan distance check
            let dist = 0;
            if (this.is3D) {
                dist = Math.abs(x - rx) + Math.abs(y - ry) + Math.abs(z - rz);
            } else {
                dist = Math.abs(x - rx) + Math.abs(y - ry);
            }
            return dist === step + 1;
        }

        // DOR Logic
        const info = this.getPhaseAndHop(step);

        if (this.operation === 'broadcast') {
            if (!this.is3D) {
                // 2D Broadcast
                if (info.phase === 0) { // Cols (Horizontal)
                    // Same Row (x), Changing Col (y)
                    if (x !== rx) return false;
                    return Math.abs(y - ry) === info.hop + 1;
                } else if (info.phase === 1) { // Rows (Vertical)
                    // Same Col (y), Changing Row (x)
                    // Wait, in Phase 1, all columns that received in Phase 0 are active?
                    // Broadcast 2D:
                    // Phase 0: Root broadcasts along Row (Horizontal).
                    // Nodes (rx, c) receive.
                    // Phase 1: All nodes (rx, c) broadcast along their Columns (Vertical).
                    // So for a node (r, c) to receive in Phase 1:
                    // It must be in a column `c` that has a source at `(rx, c)`.
                    // The source is at `x=rx`.
                    // So we check distance from `rx` along `x`.
                    // `abs(x - rx) == hop + 1`.
                    // And `y` can be anything (as long as it was reached in Phase 0).
                    // Was `y` reached in Phase 0? Yes, if `abs(y - ry) <= max_col_hops`.
                    // But usually we assume rectangular mesh, so all `y` are valid columns.
                    // So we just check `abs(x - rx) == hop + 1`.
                    return Math.abs(x - rx) === info.hop + 1;
                }
            } else {
                // 3D Broadcast
                if (info.phase === 0) { // X Phase
                    // Must match Y and Z of root
                    if (y !== ry || z !== rz) return false;
                    // Distance in X must be hop + 1
                    return Math.abs(x - rx) === info.hop + 1;
                } else if (info.phase === 1) { // Y Phase
                    // Must match Z of root (X can be anything active)
                    if (z !== rz) return false;
                    // Distance in Y must be hop + 1
                    return Math.abs(y - ry) === info.hop + 1;
                } else if (info.phase === 2) { // Z Phase
                    // X and Y can be anything active
                    // Distance in Z must be hop + 1
                    return Math.abs(z - rz) === info.hop + 1;
                }
            }
        }

        return false;
    }

    getCoords(rank) {
        if (this.is3D) {
            const Y = this.dims[1];
            const Z = this.dims[2];
            const stride_x = Y * Z;

            const x = Math.floor(rank / stride_x);
            const rem = rank % stride_x;
            const y = Math.floor(rem / Z);
            const z = rem % Z;
            return [x, y, z];
        } else {
            const cols = this.dims[1];
            const r = Math.floor(rank / cols);
            const c = rank % cols;
            return [r, c];
        }
    }

    draw() {
        this.ctx.clearRect(0, 0, this.width, this.height);
        if (this.is3D) this.draw3D();
        else this.draw2D();
    }

    draw2D() {
        const padding = 50;
        const availWidth = this.width - padding * 2;
        const availHeight = this.height - padding * 2;

        const rows = this.dims[0];
        const cols = this.dims[1];

        // Calculate cell size to fit the grid in available space
        // We need (cols-1) * size <= availWidth AND (rows-1) * size <= availHeight
        const sizeX = availWidth / (cols - 1 || 1);
        const sizeY = availHeight / (rows - 1 || 1);
        const cellSize = Math.min(sizeX, sizeY);

        const gridWidth = (cols - 1) * cellSize;
        const gridHeight = (rows - 1) * cellSize;

        const offsetX = (this.width - gridWidth) / 2;
        const offsetY = (this.height - gridHeight) / 2;

        // Draw links
        this.ctx.strokeStyle = '#334155';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        for (let r = 0; r < rows; r++) {
            this.ctx.moveTo(offsetX, offsetY + r * cellSize);
            this.ctx.lineTo(offsetX + (cols - 1) * cellSize, offsetY + r * cellSize);
        }
        for (let c = 0; c < cols; c++) {
            this.ctx.moveTo(offsetX + c * cellSize, offsetY);
            this.ctx.lineTo(offsetX + c * cellSize, offsetY + (rows - 1) * cellSize);
        }
        this.ctx.stroke();

        if (this.isPlaying || this.progress > 0) {
            this.drawActiveLinks2D(offsetX, offsetY, cellSize);
        }

        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const x = offsetX + c * cellSize;
                const y = offsetY + r * cellSize;
                const rank = r * cols + c;
                this.drawNode(x, y, rank);
            }
        }
    }

    drawActiveLinks2D(offsetX, offsetY, cellSize) {
        const step = Math.floor(this.step);
        const progress = this.progress;
        const rows = this.dims[0];
        const cols = this.dims[1];

        this.ctx.strokeStyle = '#3b82f6';
        this.ctx.lineWidth = 4;
        this.ctx.beginPath();

        if (this.algorithm === 'flooding') {
            // Draw links from level S to S+1
            // Iterate all nodes, if dist == step, draw to neighbors
            const rootCoords = this.getCoords(this.root);

            for (let r = 0; r < rows; r++) {
                for (let c = 0; c < cols; c++) {
                    const dist = Math.abs(r - rootCoords[0]) + Math.abs(c - rootCoords[1]);

                    // Broadcast: flow outwards from dist=step
                    if (this.operation === 'broadcast' && dist === step) {
                        const x = offsetX + c * cellSize;
                        const y = offsetY + r * cellSize;
                        // Check neighbors
                        const neighbors = [[r - 1, c], [r + 1, c], [r, c - 1], [r, c + 1]];
                        for (const [nr, nc] of neighbors) {
                            if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) {
                                const nDist = Math.abs(nr - rootCoords[0]) + Math.abs(nc - rootCoords[1]);
                                if (nDist === step + 1) {
                                    const nx = offsetX + nc * cellSize;
                                    const ny = offsetY + nr * cellSize;
                                    const curX = x + (nx - x) * progress;
                                    const curY = y + (ny - y) * progress;
                                    this.ctx.moveTo(x, y);
                                    this.ctx.lineTo(curX, curY);
                                }
                            }
                        }
                    }
                    // Gather: flow inwards from dist=max-step
                    else if (this.operation === 'gather') {
                        const currentLevel = this.maxSteps - step;
                        if (dist === currentLevel) {
                            const x = offsetX + c * cellSize;
                            const y = offsetY + r * cellSize;
                            // Find parent (dist - 1)
                            const neighbors = [[r - 1, c], [r + 1, c], [r, c - 1], [r, c + 1]];
                            for (const [nr, nc] of neighbors) {
                                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) {
                                    const nDist = Math.abs(nr - rootCoords[0]) + Math.abs(nc - rootCoords[1]);
                                    if (nDist === currentLevel - 1) {
                                        const nx = offsetX + nc * cellSize;
                                        const ny = offsetY + nr * cellSize;
                                        const curX = x + (nx - x) * progress;
                                        const curY = y + (ny - y) * progress;
                                        this.ctx.moveTo(x, y);
                                        this.ctx.lineTo(curX, curY);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            // DOR Logic
            const info = this.getPhaseAndHop(step);
            const [rr, rc] = this.getCoords(this.root);

            if (this.operation === 'broadcast') {
                if (info.phase === 0) { // Row (X)
                    // Draw from dist=hop to dist=hop+1 along row rr
                    const y = offsetY + rr * cellSize;
                    // Check left
                    if (rc - info.hop > 0) {
                        // From rc-hop to rc-hop-1
                        const startX = offsetX + (rc - info.hop) * cellSize;
                        const endX = offsetX + (rc - info.hop - 1) * cellSize;
                        const curX = startX + (endX - startX) * progress;
                        this.ctx.moveTo(startX, y); this.ctx.lineTo(curX, y);
                    }
                    // Check right
                    if (rc + info.hop < cols - 1) {
                        // From rc+hop to rc+hop+1
                        const startX = offsetX + (rc + info.hop) * cellSize;
                        const endX = offsetX + (rc + info.hop + 1) * cellSize;
                        const curX = startX + (endX - startX) * progress;
                        this.ctx.moveTo(startX, y); this.ctx.lineTo(curX, y);
                    }
                } else if (info.phase === 1) { // Col (Y)
                    // Draw from dist=hop to dist=hop+1 along ALL cols
                    for (let c = 0; c < cols; c++) {
                        const x = offsetX + c * cellSize;
                        // Check up
                        if (rr - info.hop > 0) {
                            const startY = offsetY + (rr - info.hop) * cellSize;
                            const endY = offsetY + (rr - info.hop - 1) * cellSize;
                            const curY = startY + (endY - startY) * progress;
                            this.ctx.moveTo(x, startY); this.ctx.lineTo(x, curY);
                        }
                        // Check down
                        if (rr + info.hop < rows - 1) {
                            const startY = offsetY + (rr + info.hop) * cellSize;
                            const endY = offsetY + (rr + info.hop + 1) * cellSize;
                            const curY = startY + (endY - startY) * progress;
                            this.ctx.moveTo(x, startY); this.ctx.lineTo(x, curY);
                        }
                    }
                }
            } else {
                // Gather DOR (Z -> Y -> X)
                // For 2D: Y -> X

                if (info.phase === 1) { // Y Phase (First in 2D)
                    // Gather all columns to row rr
                    for (let c = 0; c < cols; c++) {
                        const x = offsetX + c * cellSize;
                        const maxDistUp = rr;
                        const maxDistDown = rows - 1 - rr;

                        const distUp = maxDistUp - info.hop;
                        if (distUp > 0) {
                            const startY = offsetY + (rr - distUp) * cellSize;
                            const endY = offsetY + (rr - distUp + 1) * cellSize;
                            const curY = startY + (endY - startY) * progress;
                            this.ctx.moveTo(x, startY); this.ctx.lineTo(x, curY);
                        }

                        const distDown = maxDistDown - info.hop;
                        if (distDown > 0) {
                            const startY = offsetY + (rr + distDown) * cellSize;
                            const endY = offsetY + (rr + distDown - 1) * cellSize;
                            const curY = startY + (endY - startY) * progress;
                            this.ctx.moveTo(x, startY); this.ctx.lineTo(x, curY);
                        }
                    }
                } else if (info.phase === 0) { // X Phase (Second in 2D)
                    // Gather row rr to col rc
                    // Only row rr is active (others have gathered to it)
                    const y = offsetY + rr * cellSize;
                    const maxDistLeft = rc;
                    const maxDistRight = cols - 1 - rc;

                    const distLeft = maxDistLeft - info.hop;
                    if (distLeft > 0) {
                        const startX = offsetX + (rc - distLeft) * cellSize;
                        const endX = offsetX + (rc - distLeft + 1) * cellSize;
                        const curX = startX + (endX - startX) * progress;
                        this.ctx.moveTo(startX, y); this.ctx.lineTo(curX, y);
                    }

                    const distRight = maxDistRight - info.hop;
                    if (distRight > 0) {
                        const startX = offsetX + (rc + distRight) * cellSize;
                        const endX = offsetX + (rc + distRight - 1) * cellSize;
                        const curX = startX + (endX - startX) * progress;
                        this.ctx.moveTo(startX, y); this.ctx.lineTo(curX, y);
                    }
                }
            }
        }
        this.ctx.stroke();
    }

    draw3D() {
        // Isometric projection with rotation
        const cosX = Math.cos(this.rotationX);
        const sinX = Math.sin(this.rotationX);
        const cosY = Math.cos(this.rotationY);
        const sinY = Math.sin(this.rotationY);

        const dimX = this.dims[0];
        const dimY = this.dims[1];
        const dimZ = this.dims[2];
        const maxDim = Math.max(dimX, dimY, dimZ);

        const scale = Math.min(this.width, this.height) / (maxDim * 4); // Smaller scale
        const centerX = this.width / 2;
        const centerY = this.height / 2;

        // Center the grid
        const offX = (dimX - 1) / 2;
        const offY = (dimY - 1) / 2;
        const offZ = (dimZ - 1) / 2;

        const project = (x, y, z) => {
            // Center coordinates
            let px = x - offX;
            let py = y - offY;
            let pz = z - offZ;

            // Rotate Y
            let x1 = px * cosY - pz * sinY;
            let z1 = px * sinY + pz * cosY;

            // Rotate X
            let y2 = py * cosX - z1 * sinX;
            let z2 = py * sinX + z1 * cosX;

            return {
                x: centerX + x1 * scale * 2,
                y: centerY + y2 * scale * 2
            };
        };

        this.ctx.strokeStyle = 'rgba(51, 65, 85, 0.5)';
        this.ctx.lineWidth = 1;
        this.ctx.beginPath();
        for (let x = 0; x < dimX; x++) {
            for (let y = 0; y < dimY; y++) {
                for (let z = 0; z < dimZ; z++) {
                    const p = project(x, y, z);
                    if (x < dimX - 1) { const p2 = project(x + 1, y, z); this.ctx.moveTo(p.x, p.y); this.ctx.lineTo(p2.x, p2.y); }
                    if (y < dimY - 1) { const p2 = project(x, y + 1, z); this.ctx.moveTo(p.x, p.y); this.ctx.lineTo(p2.x, p2.y); }
                    if (z < dimZ - 1) { const p2 = project(x, y, z + 1); this.ctx.moveTo(p.x, p.y); this.ctx.lineTo(p2.x, p2.y); }
                }
            }
        }
        this.ctx.stroke();

        if (this.isPlaying || this.progress > 0) {
            this.drawActiveLinks3D(project);
        }

        const nodes = [];
        for (let x = 0; x < dimX; x++) {
            for (let y = 0; y < dimY; y++) {
                for (let z = 0; z < dimZ; z++) {
                    const rank = x * dimY * dimZ + y * dimZ + z;
                    nodes.push({ x, y, z, rank, depth: x + y - z });
                }
            }
        }
        nodes.sort((a, b) => a.depth - b.depth);
        for (const n of nodes) {
            const p = project(n.x, n.y, n.z);
            this.drawNode(p.x, p.y, n.rank, 0.7);
        }
    }

    drawActiveLinks3D(project) {
        const step = Math.floor(this.step);
        const progress = this.progress;
        const [rx, ry, rz] = this.getCoords(this.root);
        const dimX = this.dims[0];
        const dimY = this.dims[1];
        const dimZ = this.dims[2];

        this.ctx.strokeStyle = '#3b82f6';
        this.ctx.lineWidth = 3;
        this.ctx.beginPath();

        if (this.algorithm === 'flooding') {
            // ... existing flooding logic ...
            for (let x = 0; x < dimX; x++) {
                for (let y = 0; y < dimY; y++) {
                    for (let z = 0; z < dimZ; z++) {
                        const dist = Math.abs(x - rx) + Math.abs(y - ry) + Math.abs(z - rz);

                        if (this.operation === 'broadcast' && dist === step) {
                            const start = project(x, y, z);
                            const neighbors = [[x + 1, y, z], [x - 1, y, z], [x, y + 1, z], [x, y - 1, z], [x, y, z + 1], [x, y, z - 1]];
                            for (const [nx, ny, nz] of neighbors) {
                                if (nx >= 0 && nx < dimX && ny >= 0 && ny < dimY && nz >= 0 && nz < dimZ) {
                                    const nDist = Math.abs(nx - rx) + Math.abs(ny - ry) + Math.abs(nz - rz);
                                    if (nDist === step + 1) {
                                        const end = project(nx, ny, nz);
                                        const curX = start.x + (end.x - start.x) * progress;
                                        const curY = start.y + (end.y - start.y) * progress;
                                        this.ctx.moveTo(start.x, start.y);
                                        this.ctx.lineTo(curX, curY);
                                    }
                                }
                            }
                        } else if (this.operation === 'gather') {
                            const currentLevel = this.maxSteps - step;
                            if (dist === currentLevel) {
                                const start = project(x, y, z);
                                const neighbors = [[x + 1, y, z], [x - 1, y, z], [x, y + 1, z], [x, y - 1, z], [x, y, z + 1], [x, y, z - 1]];
                                for (const [nx, ny, nz] of neighbors) {
                                    if (nx >= 0 && nx < dimX && ny >= 0 && ny < dimY && nz >= 0 && nz < dimZ) {
                                        const nDist = Math.abs(nx - rx) + Math.abs(ny - ry) + Math.abs(nz - rz);
                                        if (nDist === currentLevel - 1) {
                                            const end = project(nx, ny, nz);
                                            const curX = start.x + (end.x - start.x) * progress;
                                            const curY = start.y + (end.y - start.y) * progress;
                                            this.ctx.moveTo(start.x, start.y);
                                            this.ctx.lineTo(curX, curY);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            // DOR 3D
            const info = this.getPhaseAndHop(step);

            if (this.operation === 'broadcast') {
                if (info.phase === 0) { // X Phase
                    // Root row (y=ry, z=rz) expands along X
                    const y = ry; const z = rz;
                    // Left
                    if (rx - info.hop > 0) {
                        const start = project(rx - info.hop, y, z);
                        const end = project(rx - info.hop - 1, y, z);
                        const curX = start.x + (end.x - start.x) * progress;
                        const curY = start.y + (end.y - start.y) * progress;
                        this.ctx.moveTo(start.x, start.y); this.ctx.lineTo(curX, curY);
                    }
                    // Right
                    if (rx + info.hop < dimX - 1) {
                        const start = project(rx + info.hop, y, z);
                        const end = project(rx + info.hop + 1, y, z);
                        const curX = start.x + (end.x - start.x) * progress;
                        const curY = start.y + (end.y - start.y) * progress;
                        this.ctx.moveTo(start.x, start.y); this.ctx.lineTo(curX, curY);
                    }
                } else if (info.phase === 1) { // Y Phase
                    // All active X nodes (z=rz) expand along Y
                    // Wait, in Broadcast Phase Y, *all* nodes that received in Phase X are active.
                    // Which nodes received in Phase X? All nodes with y=ry, z=rz.
                    // So for every x, the column (x, ., rz) expands from (x, ry, rz).
                    for (let x = 0; x < dimX; x++) {
                        const z = rz;
                        // Up
                        if (ry - info.hop > 0) {
                            const start = project(x, ry - info.hop, z);
                            const end = project(x, ry - info.hop - 1, z);
                            const curX = start.x + (end.x - start.x) * progress;
                            const curY = start.y + (end.y - start.y) * progress;
                            this.ctx.moveTo(start.x, start.y); this.ctx.lineTo(curX, curY);
                        }
                        // Down
                        if (ry + info.hop < dimY - 1) {
                            const start = project(x, ry + info.hop, z);
                            const end = project(x, ry + info.hop + 1, z);
                            const curX = start.x + (end.x - start.x) * progress;
                            const curY = start.y + (end.y - start.y) * progress;
                            this.ctx.moveTo(start.x, start.y); this.ctx.lineTo(curX, curY);
                        }
                    }
                } else if (info.phase === 2) { // Z Phase
                    // All active XY nodes expand along Z.
                    // Active nodes are all (x, y, rz).
                    for (let x = 0; x < dimX; x++) {
                        for (let y = 0; y < dimY; y++) {
                            // Forward
                            if (rz - info.hop > 0) {
                                const start = project(x, y, rz - info.hop);
                                const end = project(x, y, rz - info.hop - 1);
                                const curX = start.x + (end.x - start.x) * progress;
                                const curY = start.y + (end.y - start.y) * progress;
                                this.ctx.moveTo(start.x, start.y); this.ctx.lineTo(curX, curY);
                            }
                            // Backward
                            if (rz + info.hop < dimZ - 1) {
                                const start = project(x, y, rz + info.hop);
                                const end = project(x, y, rz + info.hop + 1);
                                const curX = start.x + (end.x - start.x) * progress;
                                const curY = start.y + (end.y - start.y) * progress;
                                this.ctx.moveTo(start.x, start.y); this.ctx.lineTo(curX, curY);
                            }
                        }
                    }
                }
            } else {
                // Gather 3D (Z -> Y -> X)

                if (info.phase === 2) { // Z Phase (First)
                    // Gather all (x,y) lines to z=rz
                    for (let x = 0; x < dimX; x++) {
                        for (let y = 0; y < dimY; y++) {
                            const maxDistFwd = rz;
                            const maxDistBwd = dimZ - 1 - rz;

                            const distFwd = maxDistFwd - info.hop;
                            if (distFwd > 0) {
                                const start = project(x, y, rz - distFwd);
                                const end = project(x, y, rz - distFwd + 1);
                                const curX = start.x + (end.x - start.x) * progress;
                                const curY = start.y + (end.y - start.y) * progress;
                                this.ctx.moveTo(start.x, start.y); this.ctx.lineTo(curX, curY);
                            }

                            const distBwd = maxDistBwd - info.hop;
                            if (distBwd > 0) {
                                const start = project(x, y, rz + distBwd);
                                const end = project(x, y, rz + distBwd - 1);
                                const curX = start.x + (end.x - start.x) * progress;
                                const curY = start.y + (end.y - start.y) * progress;
                                this.ctx.moveTo(start.x, start.y); this.ctx.lineTo(curX, curY);
                            }
                        }
                    }
                } else if (info.phase === 1) { // Y Phase (Second)
                    // Gather all x lines (at z=rz) to y=ry
                    for (let x = 0; x < dimX; x++) {
                        const z = rz;
                        const maxDistUp = ry;
                        const maxDistDown = dimY - 1 - ry;

                        const distUp = maxDistUp - info.hop;
                        if (distUp > 0) {
                            const start = project(x, ry - distUp, z);
                            const end = project(x, ry - distUp + 1, z);
                            const curX = start.x + (end.x - start.x) * progress;
                            const curY = start.y + (end.y - start.y) * progress;
                            this.ctx.moveTo(start.x, start.y); this.ctx.lineTo(curX, curY);
                        }

                        const distDown = maxDistDown - info.hop;
                        if (distDown > 0) {
                            const start = project(x, ry + distDown, z);
                            const end = project(x, ry + distDown - 1, z);
                            const curX = start.x + (end.x - start.x) * progress;
                            const curY = start.y + (end.y - start.y) * progress;
                            this.ctx.moveTo(start.x, start.y); this.ctx.lineTo(curX, curY);
                        }
                    }
                } else if (info.phase === 0) { // X Phase (Third)
                    // Gather line (at y=ry, z=rz) to x=rx
                    const y = ry; const z = rz;
                    const maxDistLeft = rx;
                    const maxDistRight = dimX - 1 - rx;

                    const distLeft = maxDistLeft - info.hop;
                    if (distLeft > 0) {
                        const start = project(rx - distLeft, y, z);
                        const end = project(rx - distLeft + 1, y, z);
                        const curX = start.x + (end.x - start.x) * progress;
                        const curY = start.y + (end.y - start.y) * progress;
                        this.ctx.moveTo(start.x, start.y); this.ctx.lineTo(curX, curY);
                    }

                    const distRight = maxDistRight - info.hop;
                    if (distRight > 0) {
                        const start = project(rx + distRight, y, z);
                        const end = project(rx + distRight - 1, y, z);
                        const curX = start.x + (end.x - start.x) * progress;
                        const curY = start.y + (end.y - start.y) * progress;
                        this.ctx.moveTo(start.x, start.y); this.ctx.lineTo(curX, curY);
                    }
                }
            }
        }
        this.ctx.stroke();
    }

    drawNode(x, y, rank, scale = 1) {
        const state = this.nodeStates[rank];
        const radius = (this.is3D ? 6 : 12) * scale;

        this.ctx.beginPath();
        this.ctx.arc(x, y, radius, 0, Math.PI * 2);

        if (state === 2) this.ctx.fillStyle = '#ef4444'; // Root
        else if (state === 1) this.ctx.fillStyle = '#22c55e'; // Active
        else this.ctx.fillStyle = '#64748b'; // Idle

        this.ctx.fill();

        const totalNodes = this.nodeStates.length;
        if (!this.is3D || totalNodes <= 27) {
            this.ctx.fillStyle = '#fff';
            this.ctx.font = `${10 * scale}px Arial`;
            this.ctx.textAlign = 'center';
            this.ctx.textBaseline = 'middle';
            this.ctx.fillText(rank, x, y);
        }
    }

    animate() {
        const now = performance.now();
        const dt = (now - (this.lastTime || now)) / 1000;
        this.lastTime = now;
        this.update(dt);
        this.draw();
        requestAnimationFrame(() => this.animate());
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('mesh-canvas');
    const visualizer = new MeshVisualizer(canvas);

    const updateDims = () => {
        const x = parseInt(document.getElementById('dim-x').value) || 1;
        const y = parseInt(document.getElementById('dim-y').value) || 1;
        const z = visualizer.is3D ? (parseInt(document.getElementById('dim-z').value) || 1) : 1;
        visualizer.dims = [x, y, z];

        const total = x * y * z;
        document.getElementById('total-nodes').textContent = total;
        visualizer.reset();
    };

    document.getElementById('btn-2d').addEventListener('click', (e) => {
        visualizer.is3D = false;
        document.getElementById('dim-z-wrapper').style.display = 'none';

        // Defaults for 2D
        document.getElementById('dim-x').value = 4;
        document.getElementById('dim-y').value = 4;

        document.querySelectorAll('.toggle-group button').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        updateDims();
    });

    document.getElementById('btn-3d').addEventListener('click', (e) => {
        visualizer.is3D = true;
        document.getElementById('dim-z-wrapper').style.display = 'block';

        // Defaults for 3D
        document.getElementById('dim-x').value = 3;
        document.getElementById('dim-y').value = 3;
        document.getElementById('dim-z').value = 3;

        document.querySelectorAll('.toggle-group button').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        updateDims();
    });

    document.getElementById('btn-broadcast').addEventListener('click', (e) => {
        visualizer.operation = 'broadcast';
        document.getElementById('btn-gather').classList.remove('active');
        e.target.classList.add('active');
        visualizer.reset();
    });

    document.getElementById('btn-gather').addEventListener('click', (e) => {
        visualizer.operation = 'gather';
        document.getElementById('btn-broadcast').classList.remove('active');
        e.target.classList.add('active');
        visualizer.reset();
    });

    document.getElementById('algo-select').addEventListener('change', (e) => {
        visualizer.algorithm = e.target.value;
        visualizer.reset();
    });

    document.getElementById('dim-x').addEventListener('input', updateDims);
    document.getElementById('dim-y').addEventListener('input', updateDims);
    document.getElementById('dim-z').addEventListener('input', updateDims);

    document.getElementById('data-size').addEventListener('input', () => {
        visualizer.updateStatus();
    });

    document.getElementById('anim-speed').addEventListener('input', (e) => {
        visualizer.animationSpeed = parseInt(e.target.value);
    });

    document.getElementById('btn-play').addEventListener('click', () => {
        if (visualizer.step >= visualizer.maxSteps) {
            visualizer.reset();
            visualizer.togglePlay();
        } else {
            visualizer.togglePlay();
        }
    });

    document.getElementById('btn-reset').addEventListener('click', () => {
        visualizer.reset();
        document.getElementById('btn-play').textContent = 'Play';
    });
});
