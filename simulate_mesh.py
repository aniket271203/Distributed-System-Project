"""Mesh Collective Simulation Script

Simulates efficiency metrics (time, messages, rounds) for broadcast and gather
collectives on 2D and/or 3D mesh topologies.

Usage examples:
  mpiexec -n 16 python simulate_mesh.py --topology 2d --collectives broadcast,gather \
      --data-sizes 256,1024 --algorithms standard,pipelined

  mpiexec -n 8 python simulate_mesh.py --topology both --collectives broadcast \
      --algorithms standard,flooding

Results are written (root only) to: results/simulation_p<proc>.json
"""

import argparse
import json
import os
from mpi4py import MPI

from broadcast import measure_broadcast_performance
from gather import measure_gather_performance

BCAST_ALGOS_2D = ["standard", "pipelined", "binary_tree", "flooding"]
BCAST_ALGOS_3D = ["standard"]  # currently only standard supported


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate mesh collectives and record efficiency metrics")
    parser.add_argument("--topology", type=str, default="both", choices=["2d", "3d", "both"],
                        help="Which topology to simulate")
    parser.add_argument("--collectives", type=str, default="broadcast,gather",
                        help="Comma-separated list: broadcast,gather")
    parser.add_argument("--data-sizes", type=str, default="1024",
                        help="Comma-separated list of message sizes (elements) for broadcast; gather uses per-process size")
    parser.add_argument("--algorithms", type=str, default="standard",
                        help="Comma-separated list of broadcast algorithms (2D: standard,pipelined,binary_tree,flooding; 3D: standard)")
    parser.add_argument("--output", type=str, default="results", help="Directory for JSON output")
    return parser.parse_args()


def ensure_output_dir(path: str):
    if MPI.COMM_WORLD.Get_rank() == 0:
        os.makedirs(path, exist_ok=True)


def simulate():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    ensure_output_dir(args.output)

    requested_collectives = [c.strip().lower() for c in args.collectives.split(',') if c.strip()]
    data_sizes = [int(x) for x in args.data_sizes.split(',') if x.strip()]
    requested_algos = [a.strip().lower() for a in args.algorithms.split(',') if a.strip()]

    topo_list = []
    if args.topology in ("2d", "both"):
        topo_list.append("2D")
    if args.topology in ("3d", "both"):
        topo_list.append("3D")

    if rank == 0:
        print(f"\n=== Mesh Collective Simulation (processes={size}) ===")
        print(f"Topologies: {topo_list}")
        print(f"Collectives: {requested_collectives}")
        print(f"Data sizes: {data_sizes}")
        print(f"Algorithms requested: {requested_algos}\n")

    metrics_results = []

    for topo in topo_list:
        for data_size in data_sizes:
            # Gather
            if "gather" in requested_collectives:
                if rank == 0:
                    print(f"Simulating gather topo={topo} size={data_size} ...")
                gather_metrics = measure_gather_performance(mesh_type=topo, data_size=data_size, root=0)
                if rank == 0 and gather_metrics:
                    metrics_results.append(gather_metrics)
            # Broadcast
            if "broadcast" in requested_collectives:
                # Determine valid algorithm list for this topology
                if topo == "2D":
                    algos = [a for a in requested_algos if a in BCAST_ALGOS_2D]
                else:
                    algos = [a for a in requested_algos if a in BCAST_ALGOS_3D]
                if not algos:
                    if rank == 0:
                        print(f"No valid broadcast algorithms for topology {topo} from {requested_algos}")
                for algo in algos:
                    if rank == 0:
                        print(f"Simulating broadcast topo={topo} algo={algo} size={data_size} ...")
                    bcast_metrics = measure_broadcast_performance(mesh_type=topo, data_size=data_size, root=0, algorithm=algo)
                    if rank == 0 and bcast_metrics:
                        metrics_results.append(bcast_metrics)

    # Root writes results
    if rank == 0:
        output_path = os.path.join(args.output, f"simulation_p{size}.json")
        payload = {
            "process_count": size,
            "topologies": topo_list,
            "metrics": metrics_results
        }
        with open(output_path, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"\n✓ Simulation metrics written to {output_path}\n")


if __name__ == "__main__":
    simulate()
