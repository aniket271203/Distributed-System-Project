class MeshVisualizer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.width = canvas.width;
        this.height = canvas.height;

        // Configuration
        this.gridSize = 4;
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
        const totalNodes = this.is3D ? Math.pow(this.gridSize, 3) : Math.pow(this.gridSize, 2);
        this.nodeStates = new Array(totalNodes).fill(0);

        // Determine max steps based on algo
        if (this.algorithm === 'dor') {
            this.maxSteps = this.is3D ? 3 : 2;
        } else {
            // Flooding: max steps = diameter
            if (this.is3D) {
                // From 0,0,0 to N-1,N-1,N-1
                this.maxSteps = (this.gridSize - 1) * 3;
            } else {
                this.maxSteps = (this.gridSize - 1) * 2;
            }
        }

        // Initial states
        if (this.operation === 'broadcast') {
            this.nodeStates[this.root] = 2; // Root has data
        } else {
            this.nodeStates.fill(1); // Everyone has data
            this.nodeStates[this.root] = 2;
        }

        this.updateStatus();
        this.draw();
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
            if (this.is3D) {
                if (s === 0) return "Step 1: Root broadcasts along X-axis";
                if (s === 1) return "Step 2: Nodes broadcast along Y-axis";
                if (s === 2) return "Step 3: Nodes broadcast along Z-axis";
            } else {
                if (s === 0) return "Step 1: Root broadcasts along Row";
                if (s === 1) return "Step 2: Nodes broadcast along Columns";
            }
        } else {
            if (this.is3D) {
                if (s === 0) return "Step 1: Gathering along X-axis to X=0 plane";
                if (s === 1) return "Step 2: Gathering along Y-axis to (0,0) line";
                if (s === 2) return "Step 3: Gathering along Z-axis to Root";
            } else {
                if (s === 0) return "Step 1: Gathering along Rows to Column 0";
                if (s === 1) return "Step 2: Gathering along Column 0 to Root";
            }
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
            }
            this.updateStatus();
        }
    }

    applyStepLogic(completedStep) {
        let newMessages = 0;

        if (this.operation === 'broadcast') {
            // Update states: anyone who received becomes active
            for (let i = 0; i < this.nodeStates.length; i++) {
                // Check if this node receives in this step
                if (this.shouldReceiveInStep(i, completedStep)) {
                    if (this.nodeStates[i] !== 2) {
                        if (this.nodeStates[i] === 0) {
                            this.nodeStates[i] = 1;
                            newMessages++; // Count message received by this node
                        }
                    }
                }
            }
        } else {
            // Gather: count messages sent to parents/leaders
            // Simplified: count nodes that are active at this level
            for (let i = 0; i < this.nodeStates.length; i++) {
                // If node is sending in this step
                // In gather, flow is from dist=S to dist=S-1
                // So nodes at dist=S send messages
                const coords = this.getCoords(i);
                const rootCoords = this.getCoords(this.root);
                let dist = 0;
                if (this.is3D) dist = Math.abs(coords[0] - rootCoords[0]) + Math.abs(coords[1] - rootCoords[1]) + Math.abs(coords[2] - rootCoords[2]);
                else dist = Math.abs(coords[0] - rootCoords[0]) + Math.abs(coords[1] - rootCoords[1]);

                if (this.algorithm === 'flooding') {
                    // Step S: nodes at dist=Max-S send to dist=Max-S-1
                    const sendingLevel = this.maxSteps - completedStep;
                    if (dist === sendingLevel) newMessages++;
                } else {
                    // DOR Gather logic... simplified count
                    // Just increment for visualization
                    newMessages++;
                }
            }
        }
        this.totalMessages += newMessages;
    }

    shouldReceiveInStep(rank, step) {
        const coords = this.getCoords(rank);
        const rootCoords = this.getCoords(this.root);

        if (this.algorithm === 'flooding') {
            // Manhattan distance check
            let dist = 0;
            if (this.is3D) {
                dist = Math.abs(coords[0] - rootCoords[0]) + Math.abs(coords[1] - rootCoords[1]) + Math.abs(coords[2] - rootCoords[2]);
            } else {
                dist = Math.abs(coords[0] - rootCoords[0]) + Math.abs(coords[1] - rootCoords[1]);
            }
            return dist === step + 1;
        }

        // DOR Logic
        if (this.is3D) {
            const [x, y, z] = coords;
            const [rx, ry, rz] = rootCoords;
            if (step === 0) return y === ry && z === rz; // X-axis
            if (step === 1) return z === rz;             // Y-axis
            if (step === 2) return true;                 // Z-axis
        } else {
            const [r, c] = coords;
            const [rr, rc] = rootCoords;
            if (step === 0) return r === rr;
            if (step === 1) return true;
        }
        return false;
    }

    getCoords(rank) {
        if (this.is3D) {
            const p2 = this.gridSize * this.gridSize;
            const x = Math.floor(rank / p2);
            const rem = rank % p2;
            const y = Math.floor(rem / this.gridSize);
            const z = rem % this.gridSize;
            return [x, y, z];
        } else {
            const r = Math.floor(rank / this.gridSize);
            const c = rank % this.gridSize;
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
        const cellSize = Math.min(availWidth, availHeight) / (this.gridSize - 1 || 1);

        const offsetX = (this.width - cellSize * (this.gridSize - 1)) / 2;
        const offsetY = (this.height - cellSize * (this.gridSize - 1)) / 2;

        // Draw links
        this.ctx.strokeStyle = '#334155';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        for (let r = 0; r < this.gridSize; r++) {
            this.ctx.moveTo(offsetX, offsetY + r * cellSize);
            this.ctx.lineTo(offsetX + (this.gridSize - 1) * cellSize, offsetY + r * cellSize);
        }
        for (let c = 0; c < this.gridSize; c++) {
            this.ctx.moveTo(offsetX + c * cellSize, offsetY);
            this.ctx.lineTo(offsetX + c * cellSize, offsetY + (this.gridSize - 1) * cellSize);
        }
        this.ctx.stroke();

        if (this.isPlaying || this.progress > 0) {
            this.drawActiveLinks2D(offsetX, offsetY, cellSize);
        }

        for (let r = 0; r < this.gridSize; r++) {
            for (let c = 0; c < this.gridSize; c++) {
                const x = offsetX + c * cellSize;
                const y = offsetY + r * cellSize;
                const rank = r * this.gridSize + c;
                this.drawNode(x, y, rank);
            }
        }
    }

    drawActiveLinks2D(offsetX, offsetY, cellSize) {
        const step = Math.floor(this.step);
        const progress = this.progress;
        this.ctx.strokeStyle = '#3b82f6';
        this.ctx.lineWidth = 4;
        this.ctx.beginPath();

        if (this.algorithm === 'flooding') {
            // Draw links from level S to S+1
            // Iterate all nodes, if dist == step, draw to neighbors
            const rootCoords = this.getCoords(this.root);

            for (let r = 0; r < this.gridSize; r++) {
                for (let c = 0; c < this.gridSize; c++) {
                    const dist = Math.abs(r - rootCoords[0]) + Math.abs(c - rootCoords[1]);

                    // Broadcast: flow outwards from dist=step
                    if (this.operation === 'broadcast' && dist === step) {
                        const x = offsetX + c * cellSize;
                        const y = offsetY + r * cellSize;
                        // Check neighbors
                        const neighbors = [[r - 1, c], [r + 1, c], [r, c - 1], [r, c + 1]];
                        for (const [nr, nc] of neighbors) {
                            if (nr >= 0 && nr < this.gridSize && nc >= 0 && nc < this.gridSize) {
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
                    // This is tricky to visualize perfectly reversed, simplified:
                    // Flow from dist=step+1 to dist=step (reverse logic)
                    else if (this.operation === 'gather') {
                        // In gather, we visualize flow TO root.
                        // Step 0: Leaves (max dist) -> Parents
                        // Let's use 'step' as distance from max.
                        // Current active level = maxSteps - step
                        const currentLevel = this.maxSteps - step;
                        if (dist === currentLevel) {
                            const x = offsetX + c * cellSize;
                            const y = offsetY + r * cellSize;
                            // Find parent (dist - 1)
                            const neighbors = [[r - 1, c], [r + 1, c], [r, c - 1], [r, c + 1]];
                            for (const [nr, nc] of neighbors) {
                                if (nr >= 0 && nr < this.gridSize && nc >= 0 && nc < this.gridSize) {
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
            // DOR Logic (Existing)
            const rootCoords = this.getCoords(this.root);
            const [rr, rc] = rootCoords;

            if (this.operation === 'broadcast') {
                if (step === 0) { // Row
                    const y = offsetY + rr * cellSize;
                    const startX = offsetX + rc * cellSize;
                    if (rc > 0) {
                        const targetX = offsetX;
                        const currentX = startX - (startX - targetX) * progress;
                        this.ctx.moveTo(startX, y); this.ctx.lineTo(currentX, y);
                    }
                    if (rc < this.gridSize - 1) {
                        const targetX = offsetX + (this.gridSize - 1) * cellSize;
                        const currentX = startX + (targetX - startX) * progress;
                        this.ctx.moveTo(startX, y); this.ctx.lineTo(currentX, y);
                    }
                } else if (step === 1) { // Col
                    for (let c = 0; c < this.gridSize; c++) {
                        const x = offsetX + c * cellSize;
                        const startY = offsetY + rr * cellSize;
                        if (rr > 0) {
                            const targetY = offsetY;
                            const currentY = startY - (startY - targetY) * progress;
                            this.ctx.moveTo(x, startY); this.ctx.lineTo(x, currentY);
                        }
                        if (rr < this.gridSize - 1) {
                            const targetY = offsetY + (this.gridSize - 1) * cellSize;
                            const currentY = startY + (targetY - startY) * progress;
                            this.ctx.moveTo(x, startY); this.ctx.lineTo(x, currentY);
                        }
                    }
                }
            } else {
                // Gather DOR
                if (step === 0) { // Rows to Col 0
                    for (let r = 0; r < this.gridSize; r++) {
                        const y = offsetY + r * cellSize;
                        const targetX = offsetX; // Col 0
                        for (let c = 1; c < this.gridSize; c++) {
                            const startX = offsetX + c * cellSize;
                            const curr = startX - (startX - targetX) * progress;
                            // Simplified: draw line from node to left
                            // Better: draw from node to node-1? No, gather is to leader.
                            // Let's just draw lines moving left
                            const dist = c * cellSize;
                            const curX = startX - dist * progress;
                            this.ctx.moveTo(startX, y); this.ctx.lineTo(curX, y);
                        }
                    }
                } else if (step === 1) { // Col 0 to Root
                    const x = offsetX;
                    const targetY = offsetY + rr * cellSize;
                    for (let r = 0; r < this.gridSize; r++) {
                        if (r === rr) continue;
                        const startY = offsetY + r * cellSize;
                        const currentY = startY + (targetY - startY) * progress;
                        this.ctx.moveTo(x, startY); this.ctx.lineTo(x, currentY);
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

        const scale = Math.min(this.width, this.height) / (this.gridSize * 4); // Smaller scale
        const centerX = this.width / 2;
        const centerY = this.height / 2;

        // Center the grid
        const offset = (this.gridSize - 1) / 2;

        const project = (x, y, z) => {
            // Center coordinates
            let px = x - offset;
            let py = y - offset;
            let pz = z - offset;

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
        for (let x = 0; x < this.gridSize; x++) {
            for (let y = 0; y < this.gridSize; y++) {
                for (let z = 0; z < this.gridSize; z++) {
                    const p = project(x, y, z);
                    if (x < this.gridSize - 1) { const p2 = project(x + 1, y, z); this.ctx.moveTo(p.x, p.y); this.ctx.lineTo(p2.x, p2.y); }
                    if (y < this.gridSize - 1) { const p2 = project(x, y + 1, z); this.ctx.moveTo(p.x, p.y); this.ctx.lineTo(p2.x, p2.y); }
                    if (z < this.gridSize - 1) { const p2 = project(x, y, z + 1); this.ctx.moveTo(p.x, p.y); this.ctx.lineTo(p2.x, p2.y); }
                }
            }
        }
        this.ctx.stroke();

        if (this.isPlaying || this.progress > 0) {
            this.drawActiveLinks3D(project);
        }

        const nodes = [];
        for (let x = 0; x < this.gridSize; x++) {
            for (let y = 0; y < this.gridSize; y++) {
                for (let z = 0; z < this.gridSize; z++) {
                    const rank = x * this.gridSize * this.gridSize + y * this.gridSize + z;
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

        this.ctx.strokeStyle = '#3b82f6';
        this.ctx.lineWidth = 3;
        this.ctx.beginPath();

        if (this.algorithm === 'flooding') {
            // 3D Flooding
            for (let x = 0; x < this.gridSize; x++) {
                for (let y = 0; y < this.gridSize; y++) {
                    for (let z = 0; z < this.gridSize; z++) {
                        const dist = Math.abs(x - rx) + Math.abs(y - ry) + Math.abs(z - rz);

                        if (this.operation === 'broadcast' && dist === step) {
                            const start = project(x, y, z);
                            const neighbors = [[x + 1, y, z], [x - 1, y, z], [x, y + 1, z], [x, y - 1, z], [x, y, z + 1], [x, y, z - 1]];
                            for (const [nx, ny, nz] of neighbors) {
                                if (nx >= 0 && nx < this.gridSize && ny >= 0 && ny < this.gridSize && nz >= 0 && nz < this.gridSize) {
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
                                    if (nx >= 0 && nx < this.gridSize && ny >= 0 && ny < this.gridSize && nz >= 0 && nz < this.gridSize) {
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
            if (this.operation === 'broadcast') {
                if (step === 0) { // X
                    const start = project(rx, ry, rz);
                    if (rx < this.gridSize - 1) {
                        const end = project(this.gridSize - 1, ry, rz);
                        const currX = start.x + (end.x - start.x) * progress;
                        const currY = start.y + (end.y - start.y) * progress;
                        this.ctx.moveTo(start.x, start.y); this.ctx.lineTo(currX, currY);
                    }
                    if (rx > 0) {
                        const end = project(0, ry, rz);
                        const currX = start.x + (end.x - start.x) * progress;
                        const currY = start.y + (end.y - start.y) * progress;
                        this.ctx.moveTo(start.x, start.y); this.ctx.lineTo(currX, currY);
                    }
                } else if (step === 1) { // Y
                    for (let x = 0; x < this.gridSize; x++) {
                        const start = project(x, ry, rz);
                        if (ry < this.gridSize - 1) {
                            const end = project(x, this.gridSize - 1, rz);
                            const currX = start.x + (end.x - start.x) * progress;
                            const currY = start.y + (end.y - start.y) * progress;
                            this.ctx.moveTo(start.x, start.y); this.ctx.lineTo(currX, currY);
                        }
                    }
                } else if (step === 2) { // Z
                    for (let x = 0; x < this.gridSize; x++) {
                        for (let y = 0; y < this.gridSize; y++) {
                            const start = project(x, y, rz);
                            if (rz < this.gridSize - 1) {
                                const end = project(x, y, this.gridSize - 1);
                                const currX = start.x + (end.x - start.x) * progress;
                                const currY = start.y + (end.y - start.y) * progress;
                                this.ctx.moveTo(start.x, start.y); this.ctx.lineTo(currX, currY);
                            }
                        }
                    }
                }
            } else {
                // Gather 3D (FIXED: Reverse of Broadcast)
                // Step 0: X-axis gather to X=0
                // Step 1: Y-axis gather to Y=0
                // Step 2: Z-axis gather to Root

                if (step === 0) { // X-axis gather to X=0
                    for (let y = 0; y < this.gridSize; y++) {
                        for (let z = 0; z < this.gridSize; z++) {
                            // All nodes on this line gather to x=0
                            // Visual: lines from all x>0 to x=0
                            const target = project(0, y, z);
                            for (let x = 1; x < this.gridSize; x++) {
                                const start = project(x, y, z);
                                const curX = start.x + (target.x - start.x) * progress;
                                const curY = start.y + (target.y - start.y) * progress;
                                this.ctx.moveTo(start.x, start.y); this.ctx.lineTo(curX, curY);
                            }
                        }
                    }
                } else if (step === 1) { // Y-axis gather to Y=0 (only at X=0 plane)
                    for (let z = 0; z < this.gridSize; z++) {
                        const target = project(0, 0, z);
                        for (let y = 1; y < this.gridSize; y++) {
                            const start = project(0, y, z);
                            const curX = start.x + (target.x - start.x) * progress;
                            const curY = start.y + (target.y - start.y) * progress;
                            this.ctx.moveTo(start.x, start.y); this.ctx.lineTo(curX, curY);
                        }
                    }
                } else if (step === 2) { // Z-axis gather to Root (only at X=0, Y=0 line)
                    const target = project(0, 0, rz);
                    for (let z = 0; z < this.gridSize; z++) {
                        if (z === rz) continue;
                        const start = project(0, 0, z);
                        const curX = start.x + (target.x - start.x) * progress;
                        const curY = start.y + (target.y - start.y) * progress;
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

        if (!this.is3D || this.gridSize <= 3) {
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

    document.getElementById('btn-2d').addEventListener('click', (e) => {
        visualizer.is3D = false;
        visualizer.gridSize = 4;
        document.getElementById('grid-size').max = 8;
        document.getElementById('grid-size').value = 4;
        document.getElementById('size-label').textContent = '4x4';
        document.querySelectorAll('.toggle-group button').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        visualizer.reset();
    });

    document.getElementById('btn-3d').addEventListener('click', (e) => {
        visualizer.is3D = true;
        visualizer.gridSize = 3;
        document.getElementById('grid-size').max = 5;
        document.getElementById('grid-size').value = 3;
        document.getElementById('size-label').textContent = '3x3x3';
        document.querySelectorAll('.toggle-group button').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        visualizer.reset();
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

    document.getElementById('grid-size').addEventListener('input', (e) => {
        const val = parseInt(e.target.value);
        visualizer.gridSize = val;
        document.getElementById('size-label').textContent = visualizer.is3D ? `${val}x${val}x${val}` : `${val}x${val}`;
        visualizer.reset();
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
