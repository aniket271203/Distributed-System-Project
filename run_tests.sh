#!/bin/bash

# Quick Start Script for Mesh Communication Project
# Author: Aniket Gupta

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║     COLLECTIVE COMMUNICATION ON MESH TOPOLOGIES                ║"
echo "║     Quick Start Script                                         ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if MPI is installed
if ! command -v mpiexec &> /dev/null; then
    echo "❌ Error: MPI is not installed!"
    echo "Please install OpenMPI or MPICH first:"
    echo "  Ubuntu/Debian: sudo apt-get install openmpi-bin libopenmpi-dev"
    echo "  macOS: brew install open-mpi"
    exit 1
fi

echo "✓ MPI is installed"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed!"
    exit 1
fi

echo "✓ Python 3 is installed"

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip install -q -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo " Running Tests"
echo "════════════════════════════════════════════════════════════════"

# Test 1: 2D Mesh with 4 processes (2x2)
echo ""
echo "Test 1: 2D Mesh with 4 processes (2x2 grid)"
echo "------------------------------------------------------------"
mpiexec -n 4 python3 main.py 1000

# Test 2: 2D Mesh with 16 processes (4x4)
echo ""
echo ""
echo "Test 2: 2D Mesh with 16 processes (4x4 grid)"
echo "------------------------------------------------------------"
mpiexec -n 16 python3 main.py 1000

# Test 3: 3D Mesh with 27 processes (3x3x3 cube)
echo ""
echo ""
echo "Test 3: 3D Mesh with 27 processes (3x3x3 cube)"
echo "------------------------------------------------------------"
mpiexec -n 27 python3 main.py 1000

# Test 4: Performance Analysis with 16 processes
echo ""
echo ""
echo "Test 4: Performance Analysis with 16 processes"
echo "------------------------------------------------------------"
mpiexec -n 16 python3 performance_analysis.py

echo ""
echo "════════════════════════════════════════════════════════════════"
echo " All Tests Completed!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "To run individual tests:"
echo "  mpiexec -n <num_processes> python3 main.py [data_size]"
echo ""
echo "Examples:"
echo "  mpiexec -n 4 python3 main.py         # 2D mesh with 4 processes"
echo "  mpiexec -n 16 python3 main.py 5000   # 2D mesh with custom data"
echo "  mpiexec -n 27 python3 main.py        # 3D cube with 27 processes"
echo ""
