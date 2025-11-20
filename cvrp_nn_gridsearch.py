#!/usr/bin/env python3

"""
cvrp_nearest_neighbor_gridsearch.py

Grid search script for CVRP using nearest neighbor heuristic:

Stage 1: Assign each target to its nearest depot.
Stage 2: For each depot, build routes using nearest neighbor heuristic:
    1) Start from depot
    2) Choose nearest unvisited target from current location
    3) Continue until adding next target exceeds capacity
    4) Return to depot and start new tour if unvisited targets remain

- Capacity is varied from 100 to 200 with step 20.
- Each dataset (1..100) is solved for each capacity.
- Results are timed and saved in CSV files (one per capacity).
"""

import math
import time
import csv
import os

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

        # Parse depots
        vehicle_line = [line for line in lines if line.startswith('Vehicle locations')][0]
        vehicle_coords_str = vehicle_line.split(':')[1].strip().rstrip(';')
        vehicle_coords = []
        for coord_pair in vehicle_coords_str.split(';'):
            if coord_pair.strip():
                x, y = map(int, coord_pair.split(','))
                vehicle_coords.append((x, y))

        # Parse targets
        target_line = [line for line in lines if line.startswith('Target locations')][0]
        target_coords_str = target_line.split(':')[1].strip().rstrip(';')
        target_coords = []
        for coord_pair in target_coords_str.split(';'):
            if coord_pair.strip():
                x, y = map(int, coord_pair.split(','))
                target_coords.append((x, y))

        # Parse demands
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
    """Calculate Euclidean distance between two points."""
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

# -------------------- Nearest Neighbor Heuristic --------------------

def solve_cvrp_nearest_neighbor(depot_coord, cust_coords, cust_demands, vehicle_capacity):
    """
    Solve CVRP for one depot using nearest neighbor heuristic.

    Algorithm:
    1) Start from depot
    2) From current location, select nearest unvisited customer
    3) Add to route if capacity permits
    4) Return to depot when no more customers can be added
    5) Start new route if unvisited customers remain

    Returns: (total_distance, num_vehicles_used, routes)
    """
    if not cust_coords:
        return 0.0, 0, []

    n = len(cust_coords)
    unvisited = set(range(n))
    routes = []
    total_distance = 0.0

    # Keep spawning new tours until every assigned customer is routed.
    while unvisited:
        # Start new route from depot
        route = []
        route_load = 0
        route_distance = 0.0
        current_location = depot_coord

        # Find nearest customer to depot to start route
        if unvisited:
            nearest_to_depot = min(unvisited, 
                                   key=lambda i: euclidean_distance(depot_coord, cust_coords[i]))

            # Check if we can serve this customer
            if cust_demands[nearest_to_depot] <= vehicle_capacity:
                route.append(nearest_to_depot)
                route_load += cust_demands[nearest_to_depot]
                route_distance += euclidean_distance(current_location, cust_coords[nearest_to_depot])
                current_location = cust_coords[nearest_to_depot]
                unvisited.remove(nearest_to_depot)
            else:
                # Customer demand exceeds capacity - skip this customer
                # This should not happen with proper data, but handle it
                print(f"Warning: Customer {nearest_to_depot} demand ({cust_demands[nearest_to_depot]}) exceeds capacity ({vehicle_capacity})")
                unvisited.remove(nearest_to_depot)
                continue

        # Continue building route using nearest neighbor
        while unvisited:
            # Find nearest unvisited customer from current location
            feasible_customers = [i for i in unvisited 
                                 if route_load + cust_demands[i] <= vehicle_capacity]

            if not feasible_customers:
                # No more customers can be added to this route
                break

            # Select nearest feasible customer
            nearest = min(feasible_customers, 
                         key=lambda i: euclidean_distance(current_location, cust_coords[i]))

            # Add customer to route
            route.append(nearest)
            route_load += cust_demands[nearest]
            route_distance += euclidean_distance(current_location, cust_coords[nearest])
            current_location = cust_coords[nearest]
            unvisited.remove(nearest)

        # Return to depot
        if route:
            route_distance += euclidean_distance(current_location, depot_coord)
            routes.append({
                'customers': route,
                'distance': route_distance,
                'load': route_load
            })
            total_distance += route_distance

    return total_distance, len(routes), routes

# -------------------- 2-stage approach --------------------

def solve_dataset(dataset, capacity):
    """
    Stage 1: Assign each target to nearest depot.
    Stage 2: Solve CVRP for each depot using nearest neighbor heuristic.

    Returns: (total_cost, total_vehicles_used, detailed_results)
    """
    depots = dataset['vehicle_coords']
    targets = dataset['target_coords']
    demands = dataset['weights']

    # Stage 1: Assign each target to nearest depot
    assignment = {i: [] for i in range(len(depots))}
    demand_assign = {i: [] for i in range(len(depots))}
    target_indices = {i: [] for i in range(len(depots))}  # Track original indices

    for t_idx, t_coord in enumerate(targets):
        best_depot = min(range(len(depots)), 
                        key=lambda d: euclidean_distance(t_coord, depots[d]))
        assignment[best_depot].append(t_coord)
        demand_assign[best_depot].append(demands[t_idx])
        target_indices[best_depot].append(t_idx)

    # Stage 2: Solve CVRP for each depot using nearest neighbor heuristic
    total_cost = 0.0
    total_vehicles = 0
    # Track depot-level rollups so callers can inspect route makeup if needed.
    depot_results = {}

    # Evaluate each depot independently since assignments are already partitioned.
    for d in range(len(depots)):
        if not assignment[d]:
            depot_results[d] = {
                'distance': 0.0,
                'vehicles': 0,
                'routes': [],
                'customers_assigned': 0
            }
            continue

        dist, num_vehicles, routes = solve_cvrp_nearest_neighbor(
            depots[d], 
            assignment[d], 
            demand_assign[d], 
            capacity
        )

        total_cost += dist
        total_vehicles += num_vehicles

        depot_results[d] = {
            'distance': dist,
            'vehicles': num_vehicles,
            'routes': routes,
            'customers_assigned': len(assignment[d]),
            'depot_coord': depots[d]
        }

    return total_cost, total_vehicles, depot_results

# -------------------- Grid search loop --------------------

def run_grid_search(datafile, start=100, end=200, step=20, 
                   datasets_range=range(1, 101), 
                   outdir="nearest_neighbor_results"):
    """
    Run grid search over capacity values.

    Args:
        datafile: Path to CVRP dataset file
        start: Starting capacity value
        end: Ending capacity value (inclusive)
        step: Step size for capacity
        datasets_range: Range of dataset numbers to process
        outdir: Output directory for results
    """
    os.makedirs(outdir, exist_ok=True)

    print("="*80)
    print("CVRP NEAREST NEIGHBOR HEURISTIC - GRID SEARCH")
    print("="*80)
    print(f"Capacity range: {start} to {end} (step {step})")
    print(f"Datasets: {len(list(datasets_range))}")
    print(f"Output directory: {outdir}")
    print("="*80)

    # Parse all datasets once
    print("\nParsing datasets...")
    datasets = parse_cvrp_data(datafile)
    print(f"Loaded {len(datasets)} datasets")

    # Summary statistics across all capacities
    all_results = []

    # Sweep through capacities so we can study sensitivity without touching code.
    for cap in range(start, end + 1, step):
        rows = []

        print(f"\n{'='*70}")
        print(f"Processing Capacity: {cap}")
        print(f"{'='*70}")

        capacity_start_time = time.perf_counter()

        for ds in datasets_range:
            if ds not in datasets:
                print(f"Warning: Dataset {ds} not found, skipping...")
                continue

            dataset_start_time = time.perf_counter()

            try:
                cost, vehicles_used, depot_results = solve_dataset(datasets[ds], cap)
                elapsed = time.perf_counter() - dataset_start_time

                rows.append({
                    "Dataset": ds,
                    "Capacity": cap,
                    "Total_Cost": round(cost, 2),
                    "Vehicles_Used": vehicles_used,
                    "Num_Depots": len(datasets[ds]['vehicle_coords']),
                    "Num_Customers": len(datasets[ds]['target_coords']),
                    "Time_Seconds": round(elapsed, 6),
                    "Status": "OK"
                })

                all_results.append({
                    "Dataset": ds,
                    "Capacity": cap,
                    "Cost": round(cost, 2),
                    "Vehicles": vehicles_used,
                    "Time": round(elapsed, 6)
                })

                if ds % 10 == 0:
                    print(f"Dataset {ds:3d}: cost={cost:8.2f}, vehicles={vehicles_used:3d}, time={elapsed:.3f}s")

            except Exception as e:
                elapsed = time.perf_counter() - dataset_start_time

                rows.append({
                    "Dataset": ds,
                    "Capacity": cap,
                    "Total_Cost": "",
                    "Vehicles_Used": "",
                    "Num_Depots": "",
                    "Num_Customers": "",
                    "Time_Seconds": round(elapsed, 6),
                    "Status": f"Error: {str(e)}"
                })

                print(f"Dataset {ds:3d}: ERROR - {str(e)}")

        capacity_elapsed = time.perf_counter() - capacity_start_time

        # Save results for this capacity
        out_csv = os.path.join(outdir, f"results_capacity_{cap}.csv")
        with open(out_csv, "w", newline="") as f:
            fieldnames = ["Dataset", "Capacity", "Total_Cost", "Vehicles_Used", 
                         "Num_Depots", "Num_Customers", "Time_Seconds", "Status"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        # Calculate statistics for this capacity
        successful = [r for r in rows if r["Status"] == "OK"]
        if successful:
            avg_cost = sum(float(r["Total_Cost"]) for r in successful) / len(successful)
            avg_time = sum(float(r["Time_Seconds"]) for r in successful) / len(successful)
            avg_vehicles = sum(int(r["Vehicles_Used"]) for r in successful) / len(successful)

            print(f"\nCapacity {cap} Summary:")
            print(f"  Successful: {len(successful)}/{len(rows)}")
            print(f"  Avg Cost: {avg_cost:.2f}")
            print(f"  Avg Vehicles: {avg_vehicles:.2f}")
            print(f"  Avg Time: {avg_time:.4f}s")
            print(f"  Total Time: {capacity_elapsed:.2f}s")

        print(f"Saved {len(rows)} results to {out_csv}")

    # Save comprehensive summary
    summary_csv = os.path.join(outdir, "grid_search_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        fieldnames = ["Dataset", "Capacity", "Cost", "Vehicles", "Time"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)

    print(f"\n{'='*80}")
    print("GRID SEARCH COMPLETED")
    print(f"{'='*80}")
    print(f"Total results saved in: {outdir}")
    print(f"Summary file: {summary_csv}")
    print(f"Individual capacity files: results_capacity_*.csv")

if __name__ == "__main__":
    datafile = "CVRP_10Vehicles_100Targets.txt"

    run_grid_search(
        datafile=datafile,
        start=100,
        end=200,
        step=20,
        datasets_range=range(1, 101),
        outdir="nearest_neighbor_results"
    )

    print("\nScript completed successfully!")
