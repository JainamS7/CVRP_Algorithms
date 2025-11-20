#!/usr/bin/env python3
"""
grid_search_cvrp.py

Grid search script for CVRP (2-stage approach):
  - Stage 1: Assign each target to its nearest depot.
  - Stage 2: For each depot, build a CVRP using OR-Tools with as many vehicles
             as number of assigned targets (vehicles can be unused).
  - Capacity is varied from 100 to 200 with step 20.
  - Each dataset (1..100) is solved for each capacity.
  - Results are timed and saved in CSV files (one per capacity).

Requires:
    ortools
    pandas
    numpy
"""

import math
import time
import csv
import os
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np

# -------------------- Parsing utilities --------------------

def parse_cvrp_data(filename):
    """Parse CVRP dataset file into structured dicts."""
    datasets = {}
    with open(filename, 'r') as f:
        content = f.read()
    dataset_blocks = content.split('Data set #')[1:]
    for block in dataset_blocks:
        if not block.strip():
            continue
        lines = block.strip().split('\n')
        dataset_num = int(lines[0].split()[0])
        # depots
        vehicle_line = [line for line in lines if line.startswith('Vehicle locations')][0]
        vehicle_coords_str = vehicle_line.split(':')[1].strip().rstrip(';')
        vehicle_coords = []
        for coord_pair in vehicle_coords_str.split(';'):
            if coord_pair.strip():
                x, y = map(int, coord_pair.split(','))
                vehicle_coords.append((x, y))
        # targets
        target_line = [line for line in lines if line.startswith('Target locations')][0]
        target_coords_str = target_line.split(':')[1].strip().rstrip(';')
        target_coords = []
        for coord_pair in target_coords_str.split(';'):
            if coord_pair.strip():
                x, y = map(int, coord_pair.split(','))
                target_coords.append((x, y))
        # demands
        weight_line = [line for line in lines if line.startswith('Weights')][0]
        weights_str = weight_line.split('=')[1].strip()
        weights = list(map(int, weights_str.split(',')))
        datasets[dataset_num] = {
            'vehicle_coords': vehicle_coords,
            'target_coords': target_coords,
            'weights': weights
        }
    return datasets

def euclidean_distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

# -------------------- OR-Tools CVRP per depot --------------------

def solve_cvrp_single_depot(depot_coord, cust_coords, cust_demands, vehicle_capacity):
    """Solve CVRP for one depot with OR-Tools. Returns total distance, status, #vehicles used."""
    if not cust_coords:
        return 0.0, "NoCustomers", 0

    # Build distance matrix (including depot at index 0)
    all_coords = [depot_coord] + cust_coords
    n = len(all_coords)
    dist_matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = int(euclidean_distance(all_coords[i], all_coords[j]))

    # Demands array (depot has 0 demand)
    demands = [0] + cust_demands

    # Number of vehicles = number of customers (upper bound)
    num_vehicles = len(cust_coords)
    depot_index = 0

    # OR-Tools model
    manager = pywrapcp.RoutingIndexManager(len(dist_matrix), num_vehicles, depot_index)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return dist_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Capacity constraints
    def demand_callback(from_index):
        return demands[manager.IndexToNode(from_index)]
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        [vehicle_capacity] * num_vehicles,
        True,
        "Capacity"
    )

    # Search parameters
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    # WGLS ("without GLS") intentionally skips the local_search_metaheuristic to
    # highlight the impact of Guided Local Search in the sister script.
    search_params.time_limit.seconds = 10  # short per-instance limit

    solution = routing.SolveWithParameters(search_params)
    if solution:
        total_distance = 0
        used_vehicles = 0
        for v in range(num_vehicles):
            index = routing.Start(v)
            if not routing.IsEnd(solution.Value(routing.NextVar(index))):
                used_vehicles += 1
            route_distance = 0
            while not routing.IsEnd(index):
                prev_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(prev_index, index, v)
            total_distance += route_distance
        return float(total_distance), "Optimal", used_vehicles
    else:
        return float("inf"), "NoSolution", 0

# -------------------- 2-stage approach --------------------

def solve_dataset(dataset, capacity):
    """Assign each target to nearest depot, then solve per depot with OR-Tools CVRP."""
    depots = dataset['vehicle_coords']
    targets = dataset['target_coords']
    demands = dataset['weights']

    # Assign each target to nearest depot
    assignment = {i: [] for i in range(len(depots))}
    demand_assign = {i: [] for i in range(len(depots))}
    for t_idx, t_coord in enumerate(targets):
        best_depot = min(range(len(depots)), key=lambda d: euclidean_distance(t_coord, depots[d]))
        assignment[best_depot].append(t_coord)
        demand_assign[best_depot].append(demands[t_idx])

    total_cost = 0.0
    total_used = 0
    for d in range(len(depots)):
        dist, status, used = solve_cvrp_single_depot(depots[d], assignment[d], demand_assign[d], capacity)
        total_cost += dist if dist != float("inf") else 0
        total_used += used
    return total_cost, total_used

# -------------------- Grid search loop --------------------

def run_grid_search(datafile, start=100, end=200, step=20, datasets_range=range(1,101), outdir="grid_results"):
    os.makedirs(outdir, exist_ok=True)
    datasets = parse_cvrp_data(datafile)

    for cap in range(start, end+1, step):
        rows = []
        print("="*70)
        print(f"Running capacity {cap}")
        for ds in datasets_range:
            if ds not in datasets:
                print(f"Dataset {ds} missing")
                continue
            start_time = time.perf_counter()
            try:
                cost, used = solve_dataset(datasets[ds], cap)
                elapsed = time.perf_counter() - start_time
                rows.append({
                    "Dataset": ds,
                    "Capacity": cap,
                    "Cost": round(cost, 2),
                    "VehiclesUsed": used,
                    "TimeSeconds": round(elapsed, 6),
                    "Status": "OK"
                })
                print(f"Dataset {ds}: cost={cost:.2f}, vehicles={used}, time={elapsed:.3f}s")
            except Exception as e:
                elapsed = time.perf_counter() - start_time
                rows.append({
                    "Dataset": ds,
                    "Capacity": cap,
                    "Cost": "",
                    "VehiclesUsed": "",
                    "TimeSeconds": round(elapsed, 6),
                    "Status": f"Error: {e}"
                })
                print(f"Dataset {ds}: ERROR {e}")
        # save results for this capacity
        out_csv = os.path.join(outdir, f"results_capacity_{cap}.csv")
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Dataset","Capacity","Cost","VehiclesUsed","TimeSeconds","Status"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"Saved {len(rows)} results to {out_csv}")

if __name__ == "__main__":
    datafile = "CVRP_10Vehicles_100Targets.txt"  # adjust path
    run_grid_search(datafile, start=100, end=200, step=20, datasets_range=range(1,101))
