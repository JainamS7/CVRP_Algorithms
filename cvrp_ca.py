
import numpy as np
import pandas as pd
from itertools import combinations, permutations
import math
import subprocess
import sys
import time
import csv

# -----------------------------------------------------------------------------
# Combinatorial Auction baseline for CVRP:
#   1) Parse all datasets from a benchmark text file.
#   2) Enumerate all customer subsets (up to max size) per depot and price them
#      by running a brute-force TSP, respecting vehicle capacity.
#   3) Keep the cheapest depot bid per subset and solve the resulting
#      set-partitioning problem with PuLP (CBC) to assign subsets to depots.
#   4) Dump human-readable assignments and timing summaries for every dataset.
# -----------------------------------------------------------------------------

# Install required packages
try:
    import pulp
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pulp"])
    import pulp

def parse_cvrp_data(filename):
    """Parse the CVRP dataset file and extract coordinates and weights."""
    datasets = {}

    with open(filename, 'r') as f:
        content = f.read()

    # Split by datasets
    dataset_blocks = content.split('Data set #')[1:]  # Skip first empty part

    for block in dataset_blocks:
        if not block.strip():
            continue

        lines = block.strip().split('\n')

        # Extract dataset number
        dataset_num = int(lines[0].split()[0]) if lines[0].split() else 1

        # Parse vehicle locations (depots)
        vehicle_line = [line for line in lines if line.startswith('Vehicle locations')][0]
        vehicle_coords_str = vehicle_line.split(':')[1].strip().rstrip(';')
        vehicle_coords = []
        for coord_pair in vehicle_coords_str.split(';'):
            if coord_pair.strip():
                x, y = map(int, coord_pair.split(','))
                vehicle_coords.append((x, y))

        # Parse target locations (customers)
        target_line = [line for line in lines if line.startswith('Target locations')][0]
        target_coords_str = target_line.split(':')[1].strip().rstrip(';')
        target_coords = []
        for coord_pair in target_coords_str.split(';'):
            if coord_pair.strip():
                x, y = map(int, coord_pair.split(','))
                target_coords.append((x, y))

        # Parse weights (demands)
        weight_line = [line for line in lines if line.startswith('Weights')][0]
        weights_str = weight_line.split('=')[1].strip()
        weights = list(map(int, weights_str.split(',')))

        datasets[dataset_num] = {
            'vehicle_coords': vehicle_coords,
            'target_coords': target_coords,
            'weights': weights
        }

    return datasets

def euclidean_distance(coord1, coord2):
    """Calculate Euclidean distance between two coordinates."""
    return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

def generate_subsets(n_targets, max_size=3):
    """Generate all possible subsets up to max_size."""
    subsets = []
    for size in range(1, min(max_size + 1, n_targets + 1)):
        for subset in combinations(range(n_targets), size):
            subsets.append(subset)
    return subsets

def solve_tsp_brute_force(depot_coord, target_coords, weights, capacity=100):
    """
    Solve TSP for a small subset using brute force.
    Returns the minimum distance and whether the subset is feasible.
    """
    if not target_coords:
        return 0, True

    # Check capacity constraint
    total_weight = sum(weights)
    if total_weight > capacity:
        return float('inf'), False

    # For single target
    if len(target_coords) == 1:
        target_coord = target_coords[0]
        distance = 2 * euclidean_distance(depot_coord, target_coord)
        return distance, True

    # For multiple targets, try all permutations
    min_distance = float('inf')

    for perm in permutations(range(len(target_coords))):
        # Calculate route: depot -> targets in permutation order -> depot
        distance = euclidean_distance(depot_coord, target_coords[perm[0]])

        for i in range(len(perm) - 1):
            distance += euclidean_distance(target_coords[perm[i]], target_coords[perm[i+1]])

        distance += euclidean_distance(target_coords[perm[-1]], depot_coord)

        min_distance = min(min_distance, distance)

    return min_distance, True

def generate_bids_matrix(dataset, max_subset_size=3):
    """
    Generate bids matrix for all depot-subset pairs.
    Returns: bid_matrix, subsets, feasible_subsets
    """
    vehicle_coords = dataset['vehicle_coords']
    target_coords = dataset['target_coords']
    weights = dataset['weights']
    n_vehicles = len(vehicle_coords)
    n_targets = len(target_coords)

    print(f"Generating subsets for {n_targets} targets...")

    # Generate all subsets up to specified size
    subsets = generate_subsets(n_targets, max_size=max_subset_size)
    print(f"Generated {len(subsets)} subsets")

    # Initialize bid matrix: rows = depots, columns = subsets
    bid_matrix = np.full((n_vehicles, len(subsets)), float('inf'))
    feasible_subsets = []

    print("Calculating bids for each depot-subset pair...")

    for depot_idx, depot_coord in enumerate(vehicle_coords):
        if depot_idx % 2 == 0:  # Progress indicator
            print(f"Processing depot {depot_idx}...")

        for subset_idx, subset in enumerate(subsets):
            # Get coordinates and weights for this subset
            subset_coords = [target_coords[i] for i in subset]
            subset_weights = [weights[i] for i in subset]

            # Solve TSP for this depot-subset pair
            distance, is_feasible = solve_tsp_brute_force(
                depot_coord, subset_coords, subset_weights, capacity=100
            )

            if is_feasible:
                bid_matrix[depot_idx, subset_idx] = distance
                if subset_idx not in feasible_subsets:
                    feasible_subsets.append(subset_idx)

    return bid_matrix, subsets, feasible_subsets

def find_best_bids(bid_matrix, subsets, feasible_subsets):
    """
    Find the best (lowest) bid for each subset across all depots.
    Returns: best_bids dictionary {subset_idx: (depot_idx, bid_value)}
    """
    best_bids = {}

    for subset_idx in feasible_subsets:
        # Find the depot with minimum bid for this subset
        depot_bids = bid_matrix[:, subset_idx]
        best_depot = np.argmin(depot_bids)
        best_bid_value = depot_bids[best_depot]

        if best_bid_value != float('inf'):
            best_bids[subset_idx] = {
                'depot': best_depot,
                'bid': best_bid_value,
                'subset': subsets[subset_idx]
            }

    return best_bids
from ortools.linear_solver import pywraplp

# def solve_set_partitioning(best_bids, n_targets):
#     """
#     Solve Set Partitioning using Google OR-Tools CP-SAT solver
#     """
#     print("Solving Set Partitioning Problem using OR-Tools CP-SAT...")
    
#     # Create the solver
#     solver = pywraplp.Solver.CreateSolver('SCIP')  # or 'CBC', 'GLOP'
    
#     if not solver:
#         return None, None, "Solver not available"
    
#     # Create binary variables: x[i] = 1 if subset i is selected
#     x = {}
#     for subset_idx in best_bids.keys():
#         x[subset_idx] = solver.IntVar(0, 1, f'x_{subset_idx}')
    
#     # Objective function: minimize total cost
#     objective = solver.Objective()
#     for subset_idx in best_bids.keys():
#         objective.SetCoefficient(x[subset_idx], best_bids[subset_idx]['bid'])
#     objective.SetMinimization()
    
#     # Constraints: each target must be covered exactly once
#     uncoverable_targets = []
#     for target in range(n_targets):
#         covering_subsets = []
#         for subset_idx, bid_info in best_bids.items():
#             if target in bid_info['subset']:
#                 covering_subsets.append(subset_idx)
        
#         if covering_subsets:
#             constraint = solver.Constraint(1, 1, f'target_{target}')
#             for subset_idx in covering_subsets:
#                 constraint.SetCoefficient(x[subset_idx], 1)
#         else:
#             uncoverable_targets.append(target)
    
#     if uncoverable_targets:
#         print(f"Warning: {len(uncoverable_targets)} targets cannot be covered")
#         return None, None, f"{len(uncoverable_targets)} targets cannot be covered"
    
#     # Set time limit (20 seconds)
#     print("Running OR-Tools solver with 20s time limit...")
#     solver.SetTimeLimit(20000)  # milliseconds
    
#     # Solve
#     status = solver.Solve()
    
#     # Extract solution
#     if status == pywraplp.Solver.OPTIMAL:
#         selected_subsets = []
#         total_cost = 0
        
#         for subset_idx in best_bids.keys():
#             if x[subset_idx].solution_value() == 1:
#                 selected_subsets.append({
#                     'subset_idx': subset_idx,
#                     'subset': best_bids[subset_idx]['subset'],
#                     'depot': best_bids[subset_idx]['depot'],
#                     'bid': best_bids[subset_idx]['bid']
#                 })
#                 total_cost += best_bids[subset_idx]['bid']
        
#         return selected_subsets, total_cost, "Optimal"
    
#     elif status == pywraplp.Solver.FEASIBLE:
#         selected_subsets = []
#         total_cost = 0
        
#         for subset_idx in best_bids.keys():
#             if x[subset_idx].solution_value() == 1:
#                 selected_subsets.append({
#                     'subset_idx': subset_idx,
#                     'subset': best_bids[subset_idx]['subset'],
#                     'depot': best_bids[subset_idx]['depot'],
#                     'bid': best_bids[subset_idx]['bid']
#                 })
#                 total_cost += best_bids[subset_idx]['bid']
        
#         return selected_subsets, total_cost, "Suboptimal (time limit)"
    
#     else:
#         return None, None, "No solution found"

def solve_set_partitioning(best_bids, n_targets):
    """
    Solve the Set Partitioning Problem using Integer Linear Programming.
    """
    print("Solving Set Partitioning Problem...")

    # Create the problem
    prob = pulp.LpProblem("CVRP_Set_Partitioning", pulp.LpMinimize)

    # Decision variables: x[i] = 1 if subset i is selected, 0 otherwise
    x = {}
    for subset_idx in best_bids.keys():
        x[subset_idx] = pulp.LpVariable(f"x_{subset_idx}", cat='Binary')

    # Objective function: minimize total cost
    prob += pulp.lpSum([best_bids[subset_idx]['bid'] * x[subset_idx] 
                       for subset_idx in best_bids.keys()])

    # Constraints: each target must be covered exactly once
    uncoverable_targets = []
    for target in range(n_targets):
        # Find all subsets that contain this target
        covering_subsets = []
        for subset_idx, bid_info in best_bids.items():
            if target in bid_info['subset']:
                covering_subsets.append(subset_idx)

        if covering_subsets:
            prob += pulp.lpSum([x[subset_idx] for subset_idx in covering_subsets]) == 1
        else:
            uncoverable_targets.append(target)

    if uncoverable_targets:
        print(f"Warning: {len(uncoverable_targets)} targets cannot be covered: {uncoverable_targets[:10]}...")
        return None, None, f"{len(uncoverable_targets)} targets cannot be covered"

    # Solve the problem
    print("Running optimization solver...")
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # Extract solution
    if prob.status == pulp.LpStatusOptimal:
        selected_subsets = []
        total_cost = 0

        for subset_idx in best_bids.keys():
            if x[subset_idx].varValue == 1:
                selected_subsets.append({
                    'subset_idx': subset_idx,
                    'subset': best_bids[subset_idx]['subset'],
                    'depot': best_bids[subset_idx]['depot'],
                    'bid': best_bids[subset_idx]['bid']
                })
                total_cost += best_bids[subset_idx]['bid']

        return selected_subsets, total_cost, "Optimal"
    else:
        return None, None, f"Optimization failed with status: {pulp.LpStatus[prob.status]}"
# def solve_set_partitioning(best_bids, n_targets, mip_gap=0.05):
#     """
#     Solve ILP with MIP GAP TOLERANCE (5% = within 5% of optimal)
    
#     Args:
#         mip_gap: Optimality gap tolerance (0.05 = 5%, 0.10 = 10%, etc)
#     """
#     print("Solving Set Partitioning Problem...")
    
#     prob = pulp.LpProblem("CVRP_Set_Partitioning", pulp.LpMinimize)
    
#     x = {}
#     for subset_idx in best_bids.keys():
#         x[subset_idx] = pulp.LpVariable(f"x_{subset_idx}", cat='Binary')
    
#     prob += pulp.lpSum([best_bids[subset_idx]['bid'] * x[subset_idx] 
#                        for subset_idx in best_bids.keys()])
    
#     uncoverable_targets = []
#     for target in range(n_targets):
#         covering_subsets = []
#         for subset_idx, bid_info in best_bids.items():
#             if target in bid_info['subset']:
#                 covering_subsets.append(subset_idx)
        
#         if covering_subsets:
#             prob += pulp.lpSum([x[subset_idx] for subset_idx in covering_subsets]) == 1
#         else:
#             uncoverable_targets.append(target)
    
#     if uncoverable_targets:
#         return None, None, f"{len(uncoverable_targets)} targets cannot be covered"
    
#     print(f"Running solver with {mip_gap*100:.1f}% MIP gap tolerance...")
    
#     # ✅ SET MIP GAP TOLERANCE
#     # CBC doesn't support mipGap directly in PULP_CBC_CMD, so use time+options
#     solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=20,threads=4)
#     prob.solve(solver)
    
#     if prob.status in [pulp.LpStatusOptimal, pulp.LpStatusNotSolved]:
#         selected_subsets = []
#         total_cost = 0
        
#         for subset_idx in best_bids.keys():
#             if x[subset_idx].varValue == 1:
#                 selected_subsets.append({
#                     'subset_idx': subset_idx,
#                     'subset': best_bids[subset_idx]['subset'],
#                     'depot': best_bids[subset_idx]['depot'],
#                     'bid': best_bids[subset_idx]['bid']
#                 })
#                 total_cost += best_bids[subset_idx]['bid']
        
#         status_msg = "Optimal" if prob.status == pulp.LpStatusOptimal else "Suboptimal (gap threshold)"
#         return selected_subsets, total_cost, status_msg
#     else:
#         return None, None, f"Optimization failed"


def format_and_print_results(results, datasets):
    """
    Format results similar to the results.txt example.
    Modified: Number of vehicles at depot = number of subsets assigned to that depot
    """
    out_lines = []

    for dataset_num, result in results.items():
        out_lines.append("================================================================================")
        out_lines.append(f"Dataset: Data set #{dataset_num}")
        n_depots = len(datasets[dataset_num]['vehicle_coords'])
        n_targets = len(datasets[dataset_num]['target_coords'])
        out_lines.append(f"Depots: {n_depots}, Customers: {n_targets}")
        out_lines.append("")

        # Count customers and subsets per depot
        depot_customer_count = {i: 0 for i in range(n_depots)}
        depot_subset_count = {i: 0 for i in range(n_depots)}
        depot_routes = {i: [] for i in range(n_depots)}

        if result['solution']:
            for sol in result['solution']:
                depot_num = sol['depot']
                subset_targets = sol['subset']
                depot_customer_count[depot_num] += len(subset_targets)
                depot_subset_count[depot_num] += 1
                depot_routes[depot_num].append(sol)

        out_lines.append("Assignment summary (customers per depot):")
        for depot, count in depot_customer_count.items():
            out_lines.append(f"  Depot {depot}: {count} customers")

        total_distance = sum(sol['bid'] for sol in result['solution']) if result['solution'] else float('inf')
        out_lines.append("")
        out_lines.append("------------------------------------------------------------")

        # For each depot, list vehicle assignments
        for depot in range(n_depots):
            assigned_routes = depot_routes[depot]
            depot_coord = datasets[dataset_num]['vehicle_coords'][depot]
            depot_load = 0
            depot_dist = 0

            # Number of vehicles = number of subsets assigned to this depot
            num_vehicles_at_depot = depot_subset_count[depot]

            out_lines.append(f"Depot {depot} at ({depot_coord[0]}, {depot_coord[1]})  -> {depot_customer_count[depot]} assigned customers")

            # List each vehicle (subset) assigned to this depot
            for vehicle_idx, route in enumerate(assigned_routes):
                route_load = sum(datasets[dataset_num]['weights'][i] for i in route['subset'])
                depot_load += route_load
                depot_dist += route['bid']
                customer_str = ', '.join(str(i) for i in route['subset'])
                out_lines.append(f"    Vehicle {vehicle_idx}: customers(global idx)=[{customer_str}], load={route_load}, distance={route['bid']:.2f}")

            # If no routes assigned to this depot, show that no vehicles are used
            if num_vehicles_at_depot == 0:
                out_lines.append("    No vehicles assigned to this depot")

            out_lines.append(f"  Depot total distance: {depot_dist:.2f}, total load: {depot_load}")
            out_lines.append("------------------------------------------------------------")

        out_lines.append(f"Grand total distance across depots: {total_distance:.2f}")
        out_lines.append("")

    out_text = '\n'.join(out_lines)

    with open('combinatorial_auction_results.txt', 'w') as f:
        f.write(out_text)

    print(f"\nResults formatted and saved to 'combinatorial_auction_results.txt'")

def solve_cvrp_combinatorial_auction(filename, dataset_nums=None, max_subset_size=3):
    """
    Main function to solve CVRP using combinatorial auction approach.
    """
    print("="*70)
    print("COMBINATORIAL AUCTION CVRP SOLVER")
    print("="*70)

    # Parse datasets
    print("Parsing CVRP datasets...")
    datasets = parse_cvrp_data(filename)
    print(f"Loaded {len(datasets)} datasets")

    # Process specified datasets or all datasets
    if dataset_nums is None:
        dataset_nums = list(datasets.keys())

    results = {}
    timing_results = []
    for dataset_num in dataset_nums:
        if dataset_num not in datasets:
            print(f"Dataset {dataset_num} not found!")
            continue
        dataset_start_time = time.time()
        print(f"\n{'='*50}")
        print(f"PROCESSING DATASET {dataset_num}")
        print(f"{'='*50}")

        dataset = datasets[dataset_num]
        print(f"Depots: {len(dataset['vehicle_coords'])}")
        print(f"Targets: {len(dataset['target_coords'])}")
        print(f"Max subset size: {max_subset_size}")

        try:
            # Step 1: Generate bids matrix
            print(f"\nStep 1: Generating bids matrix...")
            bid_matrix, subsets, feasible_subsets = generate_bids_matrix(dataset, max_subset_size)

            print(f"Bid matrix shape: {bid_matrix.shape}")
            print(f"Feasible subsets: {len(feasible_subsets)} out of {len(subsets)}")

            # Step 2: Find best bids
            print(f"\nStep 2: Finding best bids...")
            best_bids = find_best_bids(bid_matrix, subsets, feasible_subsets)
            print(f"Best bids found for {len(best_bids)} subsets")

            # Step 3: Solve Set Partitioning Problem
            print(f"\nStep 3: Solving Set Partitioning Problem...")
            solution, total_cost, status = solve_set_partitioning(best_bids, len(dataset['target_coords']))
            # Each dataset is independent, so persist intermediate stats immediately.

            # Store results
            results[dataset_num] = {
                'status': status,
                'total_cost': total_cost,
                'solution': solution if solution else [],
                'n_subsets_used': len(solution) if solution else 0,
                'feasible_subsets': len(feasible_subsets),
                'total_subsets': len(subsets)
            }
            dataset_end_time = time.time()
            computation_time = dataset_end_time - dataset_start_time
        
            print(f"Computation time: {computation_time:.4f} seconds")
        
        # Store timing data
            timing_results.append({
            'Dataset': f"Data set #{dataset_num}",
            'Computation_Time_Seconds': round(computation_time, 4),
            'Grand_Total_Distance': round(total_cost, 2) if total_cost else 'FAILED',
            'Status': status,
            'Feasible_Subsets': len(feasible_subsets),
            'Total_Subsets': len(subsets)
            })
            print(f"Dataset {dataset_num}: {status} - Cost: {total_cost if total_cost else 'N/A'}")

        except Exception as e:
            print(f"Error processing dataset {dataset_num}: {str(e)}")
            results[dataset_num] = {
                'status': f'Error: {str(e)}',
                'total_cost': None,
                'solution': [],
                'n_subsets_used': 0,
                'feasible_subsets': 0,
                'total_subsets': 0
            }

    # Format and save results at the end
    format_and_print_results(results, datasets)

    return results

# Main execution
if __name__ == "__main__":
    # Run the combinatorial auction solver for all datasets
    print("Starting Combinatorial Auction CVRP Solver...")
    print("Processing all 100 datasets...")

    results = solve_cvrp_combinatorial_auction(
        'CVRP_10Vehicles_100Targets.txt', 
        dataset_nums=list(range(1, 101)),  # Process all 100 datasets
        max_subset_size=3
    )

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")

    successful_datasets = 0
    total_cost_sum = 0

    for dataset_num, result in results.items():
        status = result['status']
        cost = result.get('total_cost', None)

        if status == "Optimal" and cost is not None:
            successful_datasets += 1
            total_cost_sum += cost
            print(f"Dataset {dataset_num}: {status} - Cost: {cost:.2f}")
        else:
            print(f"Dataset {dataset_num}: {status}")

    if successful_datasets > 0:
        avg_cost = total_cost_sum / successful_datasets
        print(f"\nSuccessfully solved {successful_datasets} out of {len(results)} datasets")
        print(f"Average cost: {avg_cost:.2f}")

    print(f"\nResults saved to 'combinatorial_auction_results.txt'")
    print("Solver completed!")
