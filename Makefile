# Makefile for Mesh Communication Project
# Author: Aniket Gupta

# Default number of processes
NP ?= 16

# Python interpreter
PYTHON = python3

# MPI executor
MPIEXEC = mpiexec

.PHONY: all install test clean help run-2d run-3d run-all performance analyze

# Default target
all: install test

# Install dependencies
install:
	@echo "Installing dependencies..."
	@pip install -r requirements.txt
	@echo "✓ Dependencies installed"

# Run all tests
test:
	@echo "Running test suite with $(NP) processes..."
	@$(MPIEXEC) -n $(NP) $(PYTHON) test_project.py

# Run main program
run: run-all

# Run with 2D mesh (4 processes)
run-2d:
	@echo "Running 2D mesh tests (4 processes)..."
	@$(MPIEXEC) -n 4 $(PYTHON) main.py

# Run with 3D mesh (27 processes)
run-3d:
	@echo "Running 3D mesh tests (27 processes)..."
	@$(MPIEXEC) -n 27 $(PYTHON) main.py

# Run all configurations
run-all:
	@echo "Running complete test suite..."
	@bash run_tests.sh

# Run performance analysis
performance:
	@echo "Running performance analysis with $(NP) processes..."
	@$(MPIEXEC) -n $(NP) $(PYTHON) performance_analysis.py

# Alias for performance
analyze: performance

# Run broadcast only
broadcast:
	@echo "Testing broadcast operations with $(NP) processes..."
	@$(MPIEXEC) -n $(NP) $(PYTHON) broadcast.py

# Run gather only
gather:
	@echo "Testing gather operations with $(NP) processes..."
	@$(MPIEXEC) -n $(NP) $(PYTHON) gather.py

# Quick test with 4 processes
quick:
	@echo "Quick test with 4 processes..."
	@$(MPIEXEC) -n 4 $(PYTHON) main.py 100

# Test with 16 processes (good for 2D)
medium:
	@echo "Medium test with 16 processes..."
	@$(MPIEXEC) -n 16 $(PYTHON) main.py 1000

# Test with 27 processes (perfect 3D cube)
large:
	@echo "Large test with 27 processes (3x3x3 cube)..."
	@$(MPIEXEC) -n 27 $(PYTHON) main.py 5000

# Clean temporary files
clean:
	@echo "Cleaning temporary files..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@echo "✓ Clean complete"

# Display help
help:
	@echo "Mesh Communication Project - Makefile"
	@echo "====================================="
	@echo ""
	@echo "Available targets:"
	@echo "  make install      - Install Python dependencies"
	@echo "  make test         - Run test suite (default: 16 processes)"
	@echo "  make run          - Run complete test suite"
	@echo "  make run-2d       - Run 2D mesh tests (4 processes)"
	@echo "  make run-3d       - Run 3D mesh tests (27 processes)"
	@echo "  make performance  - Run performance analysis"
	@echo "  make broadcast    - Test broadcast only"
	@echo "  make gather       - Test gather only"
	@echo "  make quick        - Quick test (4 processes, small data)"
	@echo "  make medium       - Medium test (16 processes)"
	@echo "  make large        - Large test (27 processes, 3D cube)"
	@echo "  make clean        - Clean temporary files"
	@echo "  make help         - Display this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make test NP=8    - Run tests with 8 processes"
	@echo "  make performance NP=32  - Performance analysis with 32 processes"
	@echo ""
	@echo "Requirements:"
	@echo "  - Python 3.7+"
	@echo "  - MPI (OpenMPI or MPICH)"
	@echo "  - mpi4py, numpy"
	@echo ""
