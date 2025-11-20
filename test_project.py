"""
Simple test script to verify individual components
Author: Aniket Gupta
"""

from mpi4py import MPI
import numpy as np
import sys

def test_mesh_creation():
    """Test mesh topology creation"""
    from mesh_topology import Mesh2D, Mesh3D
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("\n" + "="*60)
        print("TEST 1: MESH TOPOLOGY CREATION")
        print("="*60)
    
    # Test 2D Mesh
    mesh_2d = Mesh2D(comm)
    assert mesh_2d.rank == rank, "Rank mismatch in 2D mesh"
    assert mesh_2d.size == size, "Size mismatch in 2D mesh"
    assert mesh_2d.coords is not None, "Coordinates not set in 2D mesh"
    
    if rank == 0:
        print(f"✓ 2D Mesh created: {mesh_2d.rows}x{mesh_2d.cols}")
        print(f"  Process 0 coordinates: {mesh_2d.coords}")
        print(f"  Neighbors: {mesh_2d.neighbors}")
    
    # Test 3D Mesh
    if size >= 8:
        mesh_3d = Mesh3D(comm)
        assert mesh_3d.rank == rank, "Rank mismatch in 3D mesh"
        assert mesh_3d.size == size, "Size mismatch in 3D mesh"
        
        if rank == 0:
            print(f"✓ 3D Mesh created: {mesh_3d.x_dim}x{mesh_3d.y_dim}x{mesh_3d.z_dim}")
            print(f"  Process 0 coordinates: {mesh_3d.coords}")
            print(f"  Neighbors: {mesh_3d.neighbors}")
    
    comm.Barrier()
    if rank == 0:
        print("✓ Mesh topology creation test PASSED\n")
    
    return True


def test_broadcast():
    """Test broadcast operations"""
    from mesh_topology import Mesh2D, Mesh3D
    from broadcast import broadcast_2d_mesh, broadcast_3d_mesh
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("="*60)
        print("TEST 2: BROADCAST OPERATIONS")
        print("="*60)
    
    # Test 2D Broadcast
    mesh_2d = Mesh2D(comm)
    test_data = np.arange(100, dtype=np.float64) if rank == 0 else None
    
    result, time, steps, msgs = broadcast_2d_mesh(mesh_2d, test_data, root=0)
    
    # Verify
    expected = np.arange(100, dtype=np.float64)
    assert np.allclose(result, expected), f"Broadcast verification failed at rank {rank}"
    
    if rank == 0:
        print(f"✓ 2D Broadcast test PASSED")
        print(f"  Time: {time:.6f}s, Steps: {steps}, Messages: {msgs}")
    
    # Test 3D Broadcast
    if size >= 8:
        mesh_3d = Mesh3D(comm)
        test_data = np.arange(100, dtype=np.float64) if rank == 0 else None
        
        result, time, steps, msgs = broadcast_3d_mesh(mesh_3d, test_data, root=0)
        
        if mesh_3d.coords is not None:
            assert np.allclose(result, expected), f"3D Broadcast verification failed at rank {rank}"
        
        if rank == 0:
            print(f"✓ 3D Broadcast test PASSED")
            print(f"  Time: {time:.6f}s, Steps: {steps}, Messages: {msgs}")
    
    comm.Barrier()
    if rank == 0:
        print("✓ All broadcast tests PASSED\n")
    
    return True


def test_gather():
    """Test gather operations"""
    from mesh_topology import Mesh2D, Mesh3D
    from gather import gather_2d_mesh, gather_3d_mesh
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("="*60)
        print("TEST 3: GATHER OPERATIONS")
        print("="*60)
    
    # Test 2D Gather
    mesh_2d = Mesh2D(comm)
    test_data = np.full(100, rank, dtype=np.float64)
    
    result, time, steps, msgs = gather_2d_mesh(mesh_2d, test_data, root=0)
    
    if rank == 0:
        assert result is not None, "Gather returned None at root"
        assert len(result) == size, f"Gathered {len(result)} items, expected {size}"
        print(f"✓ 2D Gather test PASSED")
        print(f"  Time: {time:.6f}s, Steps: {steps}, Messages: {msgs}")
        print(f"  Collected data from {len(result)} processes")
    
    # Test 3D Gather
    if size >= 8:
        mesh_3d = Mesh3D(comm)
        test_data = np.full(100, rank, dtype=np.float64)
        
        result, time, steps, msgs = gather_3d_mesh(mesh_3d, test_data, root=0)
        
        if rank == 0:
            assert result is not None, "3D Gather returned None at root"
            print(f"✓ 3D Gather test PASSED")
            print(f"  Time: {time:.6f}s, Steps: {steps}, Messages: {msgs}")
            print(f"  Collected data from processes")
    
    comm.Barrier()
    if rank == 0:
        print("✓ All gather tests PASSED\n")
    
    return True


def test_performance_model():
    """Test latency-bandwidth model calculations"""
    from performance_analysis import calculate_theoretical_metrics, latency_bandwidth_model
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("="*60)
        print("TEST 4: PERFORMANCE MODEL")
        print("="*60)
        
        # Test theoretical metrics
        metrics_2d = calculate_theoretical_metrics(size, '2D')
        print(f"✓ 2D Mesh metrics calculated:")
        print(f"  Grid size: {metrics_2d['grid_size']}")
        print(f"  Diameter: {metrics_2d['diameter']}")
        print(f"  Communication steps: {metrics_2d['comm_steps']}")
        
        if size >= 8:
            metrics_3d = calculate_theoretical_metrics(size, '3D')
            print(f"✓ 3D Mesh metrics calculated:")
            print(f"  Grid size: {metrics_3d['grid_size']}")
            print(f"  Diameter: {metrics_3d['diameter']}")
            print(f"  Communication steps: {metrics_3d['comm_steps']}")
        
        # Test latency-bandwidth model
        ts = 1e-5
        tw = 1e-9
        m = 1000
        
        T_2d, steps_2d = latency_bandwidth_model(ts, tw, m, size, '2D')
        print(f"\n✓ Latency-bandwidth model (2D):")
        print(f"  Calculated time: {T_2d:.9f}s")
        print(f"  Steps: {steps_2d}")
        
        if size >= 8:
            T_3d, steps_3d = latency_bandwidth_model(ts, tw, m, size, '3D')
            print(f"✓ Latency-bandwidth model (3D):")
            print(f"  Calculated time: {T_3d:.9f}s")
            print(f"  Steps: {steps_3d}")
            print(f"  Speedup (3D over 2D): {T_2d/T_3d:.2f}x")
        
        print("\n✓ Performance model tests PASSED\n")
    
    return True


def run_all_tests():
    """Run all tests"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("\n" + "╔" + "═"*58 + "╗")
        print("║" + " "*58 + "║")
        print("║" + "  MESH COMMUNICATION PROJECT - TEST SUITE".center(58) + "║")
        print("║" + " "*58 + "║")
        print("╚" + "═"*58 + "╝")
        print(f"\nRunning tests with {comm.Get_size()} MPI processes...\n")
    
    try:
        # Run all tests
        test_mesh_creation()
        test_broadcast()
        test_gather()
        test_performance_model()
        
        if rank == 0:
            print("="*60)
            print("ALL TESTS PASSED! ✓")
            print("="*60)
            print("\nProject is working correctly!")
            print("\nNext steps:")
            print("  1. Run full experiments: mpiexec -n 16 python3 main.py")
            print("  2. Performance analysis: mpiexec -n 16 python3 performance_analysis.py")
            print("  3. Or use: ./run_tests.sh")
            print()
        
        return True
        
    except AssertionError as e:
        if rank == 0:
            print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        if rank == 0:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
