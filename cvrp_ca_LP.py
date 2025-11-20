
import numpy as np
import pandas as pd
from itertools import combinations, permutations
import math
import subprocess
import sys
import time
import csv

# -----------------------------------------------------------------------------
# LP-relaxation variant of the combinatorial auction solver.
# Differences vs. `cvrp_ca.py`:
#   * All subsets remain in the bid matrix (infeasible ones receive BIG_BID).
#   * The set-partitioning problem is relaxed to an LP, then rounded using
#     a deterministic 1/K rule followed by local conflict resolution.
#   * Useful when an exact MILP takes too long or cannot find a solution.
# -----------------------------------------------------------------------------

# Install required packages
try:
    import pulp
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pulp"])
    import pulp

BIG_BID=1e9
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

def find_best_bids(bid_matrix, subsets):
    """
    For each subset (column) pick the depot with minimum bid (even if BIG_BID).
    Returns best_bids dict mapping subset_idx -> {depot, bid, subset}
    """
    n_vehicles, n_subsets = bid_matrix.shape
    best_bids = {}
    for subset_idx in range(n_subsets):
        # choose depot with minimum bid (ties -> first)
        depot_idx = int(np.argmin(bid_matrix[:, subset_idx]))
        best_bid_value = float(bid_matrix[depot_idx, subset_idx])
        best_bids[subset_idx] = {
            'depot': depot_idx,
            'bid': best_bid_value,
            'subset': subsets[subset_idx]
        }
    return best_bids
def generate_bids_matrix(dataset, capacity=100, max_subset_size=3):
    """
    Generate bids matrix for all depots and for the full combinatorial universe
    of subsets up to size max_subset_size. Do NOT drop infeasible subsets --
    instead set their bids to BIG_BID.
    Returns: bid_matrix (n_depots x n_subsets), subsets (list of tuples)
    """
    vehicle_coords = dataset['vehicle_coords']
    target_coords = dataset['target_coords']
    weights = dataset['weights']
    n_vehicles = len(vehicle_coords)
    n_targets = len(target_coords)

    subsets = generate_subsets(n_targets, max_size=max_subset_size)
    m = len(subsets)
    bid_matrix = np.full((n_vehicles, m), BIG_BID, dtype=float)  # initialize with BIG_BID

    for depot_idx, depot_coord in enumerate(vehicle_coords):
        for subset_idx, subset in enumerate(subsets):
            subset_coords = [target_coords[i] for i in subset]
            subset_weights = [weights[i] for i in subset]
            dist, feasible = solve_tsp_brute_force(
                depot_coord, subset_coords, subset_weights, capacity
            )
            if feasible:
                bid_matrix[depot_idx, subset_idx] = dist
            else:
                bid_matrix[depot_idx, subset_idx] = BIG_BID  # keep it present, but very costly

    return bid_matrix, subsets

def solve_set_partitioning(best_bids, n_targets):
    """
    Solve the Set Partitioning Problem using LP Relaxation + Rounding.
    
    Steps:
    1. Relax ILP to LP (binary → continuous)
    2. Solve LP to get fractional solution
    3. Round using 1/K threshold strategy
    4. Resolve conflicts to get feasible integer solution
    """
    print("Solving Set Partitioning Problem using LP Relaxation...")
    
    # ============================================================
    # STEP 1: Create LP problem (relaxed version)
    # ============================================================
    prob = pulp.LpProblem("CVRP_Set_Partitioning_LP", pulp.LpMinimize)
    
    # Decision variables: CONTINUOUS (not Binary)
    x = {}
    for subset_idx in best_bids.keys():
        x[subset_idx] = pulp.LpVariable(
            f"x_{subset_idx}", 
            lowBound=0, 
            upBound=1, 
            cat='Continuous'  # ← KEY DIFFERENCE: Continuous instead of Binary
        )
    
    # Objective function: minimize total cost
    prob += pulp.lpSum([best_bids[subset_idx]['bid'] * x[subset_idx] 
                       for subset_idx in best_bids.keys()])
    
    # Constraints: each target must be covered exactly once
    # Also track which subsets cover each target for rounding
    constraint_vars = {}  # Maps target → list of covering subsets
    uncoverable_targets = []
    
    for target in range(n_targets):
        # Find all subsets that contain this target
        covering_subsets = []
        for subset_idx, bid_info in best_bids.items():
            if target in bid_info['subset']:
                covering_subsets.append(subset_idx)
        
        if covering_subsets:
            prob += pulp.lpSum([x[subset_idx] for subset_idx in covering_subsets]) == 1
            constraint_vars[target] = covering_subsets  # Store for rounding
        else:
            uncoverable_targets.append(target)
    
    if uncoverable_targets:
        print(f"Warning: {len(uncoverable_targets)} targets cannot be covered: {uncoverable_targets[:10]}...")
        return None, None, f"{len(uncoverable_targets)} targets cannot be covered"
    
    # ============================================================
    # STEP 2: Solve the relaxed LP problem
    # ============================================================
    print("Running LP solver...")
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    if prob.status != pulp.LpStatusOptimal:
        return None, None, f"LP solve failed with status: {pulp.LpStatus[prob.status]}"
    
    # Get LP solution (fractional values)
    lp_solution = {subset_idx: x[subset_idx].varValue 
                   for subset_idx in best_bids.keys() 
                   if x[subset_idx].varValue is not None}
    
    lp_objective = pulp.value(prob.objective)
    print(f"LP objective value: {lp_objective:.2f}")
    
    # ============================================================
    # STEP 3: Rounding strategy using 1/K threshold
    # ============================================================
    print("Rounding LP solution to integer solution...")
    
    # Calculate threshold for each target: 1/K where K = number of covering subsets
    target_thresholds = {}
    for target, covering_subsets in constraint_vars.items():
        K = len(covering_subsets)
        print(target," ",K)
        target_thresholds[target] = 1.0 / K
    
    # Round each variable based on its threshold
    rounded_solution = {}
    for subset_idx in best_bids.keys():
        if subset_idx not in lp_solution:
            rounded_solution[subset_idx] = 0
            continue
        
        # Find the maximum threshold among all targets in this subset
        max_threshold = 0.0
        for target in best_bids[subset_idx]['subset']:
            if target in target_thresholds:
                max_threshold = max(max_threshold, target_thresholds[target])
        #print(max_threshold, "**")        
        
        # Apply rounding rule
        if lp_solution[subset_idx] >= max_threshold:
            rounded_solution[subset_idx] = 1
        else:
            rounded_solution[subset_idx] = 0
    
    # ============================================================
    # STEP 4: Check feasibility and resolve conflicts
    # ============================================================
    print("Resolving conflicts to ensure feasibility...")
    
    # Track which subsets are selected for each target
    target_coverage = {t: [] for t in range(n_targets)}
    for subset_idx, value in rounded_solution.items():
        if value == 1:
            for target in best_bids[subset_idx]['subset']:
                target_coverage[target].append(subset_idx)
    
    # Resolve conflicts
    final_selected = set()
    infeasible_targets = []
    
    for target in range(n_targets):
        num_covering = len(target_coverage[target])
        
        if num_covering == 0:
            # Target not covered - need to add a subset
            if target in constraint_vars and constraint_vars[target]:
                # Pick the subset with highest LP value
                best_subset = max(constraint_vars[target], 
                                key=lambda s: lp_solution.get(s, 0))
                final_selected.add(best_subset)
                print(f"  Added subset {best_subset} to cover uncovered target {target}")
            else:
                infeasible_targets.append(target)
        
        elif num_covering == 1:
            # Perfect - exactly one subset covers this target
            final_selected.add(target_coverage[target][0])
        
        else:
            # Multiple subsets cover this target - keep only the best one
            # Strategy: Keep the one with lowest cost (bid)
            best_subset = min(target_coverage[target], 
                            key=lambda s: best_bids[s]['bid'])
            final_selected.add(best_subset)
            print(f"  Resolved conflict for target {target}: kept subset {best_subset} (lowest cost)")
    
    if infeasible_targets:
        print(f"Error: {len(infeasible_targets)} targets remain uncovered after rounding")
        return None, None, f"Infeasible after rounding: {len(infeasible_targets)} targets uncovered"
    
    # ============================================================
    # STEP 5: Build final solution in required format
    # ============================================================
    selected_subsets = []
    total_cost = 0
    
    for subset_idx in sorted(final_selected):
        selected_subsets.append({
            'subset_idx': subset_idx,
            'subset': best_bids[subset_idx]['subset'],
            'depot': best_bids[subset_idx]['depot'],
            'bid': best_bids[subset_idx]['bid']
        })
        total_cost += best_bids[subset_idx]['bid']
    
    print(f"Integer solution objective: {total_cost:.2f}")
    print(f"Gap from LP lower bound: {((total_cost - lp_objective) / lp_objective * 100):.2f}%")
    print(f"Number of subsets selected: {len(selected_subsets)}")
    
    return selected_subsets, total_cost, "Optimal (via LP relaxation)"

# def solve_set_partitioning(best_bids, n_targets):
#     """
#     Solve the Set Partitioning Problem using LP Relaxation + Improved Rounding.
    
#     Steps:
#     1. Relax ILP to LP (binary → continuous)
#     2. Solve LP to get fractional solution
#     3. Round using CONSERVATIVE threshold (0.5 instead of 1/K)
#     4. Resolve conflicts using LP VALUES (not cost)
#     5. Return feasible integer solution
#     """
#     print("Solving Set Partitioning Problem using improved LP Relaxation...")
    
#     # ============================================================
#     # STEP 1: Create LP problem (relaxed version)
#     # ============================================================
#     prob = pulp.LpProblem("CVRP_Set_Partitioning_LP", pulp.LpMinimize)
    
#     # Decision variables: CONTINUOUS (not Binary)
#     x = {}
#     for subset_idx in best_bids.keys():
#         x[subset_idx] = pulp.LpVariable(
#             f"x_{subset_idx}", 
#             lowBound=0, 
#             upBound=1, 
#             cat='Continuous'  # ← KEY DIFFERENCE: Continuous instead of Binary
#         )
    
#     # Objective function: minimize total cost
#     prob += pulp.lpSum([best_bids[subset_idx]['bid'] * x[subset_idx] 
#                        for subset_idx in best_bids.keys()])
    
#     # Constraints: each target must be covered exactly once
#     # Also track which subsets cover each target for rounding
#     constraint_vars = {}  # Maps target → list of covering subsets
#     uncoverable_targets = []
    
#     for target in range(n_targets):
#         # Find all subsets that contain this target
#         covering_subsets = []
#         for subset_idx, bid_info in best_bids.items():
#             if target in bid_info['subset']:
#                 covering_subsets.append(subset_idx)
        
#         if covering_subsets:
#             prob += pulp.lpSum([x[subset_idx] for subset_idx in covering_subsets]) == 1
#             constraint_vars[target] = covering_subsets  # Store for rounding
#         else:
#             uncoverable_targets.append(target)
    
#     if uncoverable_targets:
#         print(f"Warning: {len(uncoverable_targets)} targets cannot be covered: {uncoverable_targets[:10]}...")
#         return None, None, f"{len(uncoverable_targets)} targets cannot be covered"
    
#     # ============================================================
#     # STEP 2: Solve the relaxed LP problem
#     # ============================================================
#     print("Running LP solver...")
#     prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
#     if prob.status != pulp.LpStatusOptimal:
#         return None, None, f"LP solve failed with status: {pulp.LpStatus[prob.status]}"
    
#     # Get LP solution (fractional values)
#     lp_solution = {subset_idx: x[subset_idx].varValue 
#                    for subset_idx in best_bids.keys() 
#                    if x[subset_idx].varValue is not None}
    
#     lp_objective = pulp.value(prob.objective)
#     print(f"LP objective value: {lp_objective:.2f}")
    
#     # ============================================================
#     # STEP 3: IMPROVED Rounding strategy - CONSERVATIVE threshold
#     # ============================================================
#     print("Rounding LP solution to integer solution...")
    
#     # FIX 1: Use CONSERVATIVE threshold (0.5) instead of 1/K
#     # This minimizes conflicts in the rounding phase
#     ROUNDING_THRESHOLD = 0.5
    
#     # Round each variable based on conservative threshold
#     rounded_solution = {}
#     for subset_idx in best_bids.keys():
#         if subset_idx not in lp_solution:
#             rounded_solution[subset_idx] = 0
#             continue
        
#         # Apply conservative rounding rule
#         if lp_solution[subset_idx] >= ROUNDING_THRESHOLD:
#             rounded_solution[subset_idx] = 1
#         else:
#             rounded_solution[subset_idx] = 0
    
#     # ============================================================
#     # STEP 4: Check feasibility and resolve conflicts
#     # ============================================================
#     print("Resolving conflicts to ensure feasibility...")
    
#     # Track which subsets are selected for each target
#     target_coverage = {t: [] for t in range(n_targets)}
#     for subset_idx, value in rounded_solution.items():
#         if value == 1:
#             for target in best_bids[subset_idx]['subset']:
#                 target_coverage[target].append(subset_idx)
    
#     # Resolve conflicts
#     final_selected = set()
#     infeasible_targets = []
#     num_conflicts = 0
    
#     for target in range(n_targets):
#         num_covering = len(target_coverage[target])
        
#         if num_covering == 0:
#             # Target not covered - need to add a subset
#             if target in constraint_vars and constraint_vars[target]:
#                 # FIX 2: Pick the subset with HIGHEST LP VALUE (not lowest cost!)
#                 # Trust the LP solver's solution
#                 best_subset = max(constraint_vars[target], 
#                                 key=lambda s: lp_solution.get(s, 0))
#                 final_selected.add(best_subset)
#                 print(f"  Added subset {best_subset} to cover uncovered target {target}")
#             else:
#                 infeasible_targets.append(target)
        
#         elif num_covering == 1:
#             # Perfect - exactly one subset covers this target
#             final_selected.add(target_coverage[target][0])
        
#         else:
#             # Multiple subsets cover this target - keep only the best one
#             # FIX 2: Keep the one with HIGHEST LP VALUE (not lowest cost!)
#             # This respects the LP solver's fractional solution
#             best_subset = max(target_coverage[target], 
#                             key=lambda s: lp_solution.get(s, 0))
#             final_selected.add(best_subset)
#             print(f"  Resolved conflict for target {target}: kept subset {best_subset} (highest LP value={lp_solution.get(best_subset, 0):.4f})")
#             num_conflicts += 1
    
#     if infeasible_targets:
#         print(f"Error: {len(infeasible_targets)} targets remain uncovered after rounding")
#         return None, None, f"Infeasible after rounding: {len(infeasible_targets)} targets uncovered"
    
#     # ============================================================
#     # STEP 5: Build final solution in required format
#     # ============================================================
#     selected_subsets = []
#     total_cost = 0
    
#     for subset_idx in sorted(final_selected):
#         selected_subsets.append({
#             'subset_idx': subset_idx,
#             'subset': best_bids[subset_idx]['subset'],
#             'depot': best_bids[subset_idx]['depot'],
#             'bid': best_bids[subset_idx]['bid']
#         })
#         total_cost += best_bids[subset_idx]['bid']
    
#     gap = ((total_cost - lp_objective) / lp_objective * 100) if lp_objective > 0 else 0
    
#     print(f"Integer solution objective: {total_cost:.2f}")
#     print(f"Gap from LP lower bound: {gap:.2f}%")
#     print(f"Number of subsets selected: {len(selected_subsets)}")
#     print(f"Conflicts resolved: {num_conflicts}")
    
#     return selected_subsets, total_cost, "Optimal (via improved LP relaxation)"


# def format_and_print_results(results, datasets):
#     """
#     Format results similar to the results.txt example.
#     Modified: Number of vehicles at depot = number of subsets assigned to that depot
#     """
#     out_lines = []

#     for dataset_num, result in results.items():
#         out_lines.append("================================================================================")
#         out_lines.append(f"Dataset: Data set #{dataset_num}")
#         n_depots = len(datasets[dataset_num]['vehicle_coords'])
#         n_targets = len(datasets[dataset_num]['target_coords'])
#         out_lines.append(f"Depots: {n_depots}, Customers: {n_targets}")
#         out_lines.append("")

#         # Count customers and subsets per depot
#         depot_customer_count = {i: 0 for i in range(n_depots)}
#         depot_subset_count = {i: 0 for i in range(n_depots)}
#         depot_routes = {i: [] for i in range(n_depots)}

#         if result['solution']:
#             for sol in result['solution']:
#                 depot_num = sol['depot']
#                 subset_targets = sol['subset']
#                 depot_customer_count[depot_num] += len(subset_targets)
#                 depot_subset_count[depot_num] += 1
#                 depot_routes[depot_num].append(sol)

#         out_lines.append("Assignment summary (customers per depot):")
#         for depot, count in depot_customer_count.items():
#             out_lines.append(f"  Depot {depot}: {count} customers")

#         total_distance = sum(sol['bid'] for sol in result['solution']) if result['solution'] else float('inf')
#         out_lines.append("")
#         out_lines.append("------------------------------------------------------------")

#         # For each depot, list vehicle assignments
#         for depot in range(n_depots):
#             assigned_routes = depot_routes[depot]
#             depot_coord = datasets[dataset_num]['vehicle_coords'][depot]
#             depot_load = 0
#             depot_dist = 0

#             # Number of vehicles = number of subsets assigned to this depot
#             num_vehicles_at_depot = depot_subset_count[depot]

#             out_lines.append(f"Depot {depot} at ({depot_coord[0]}, {depot_coord[1]})  -> {depot_customer_count[depot]} assigned customers")

#             # List each vehicle (subset) assigned to this depot
#             for vehicle_idx, route in enumerate(assigned_routes):
#                 route_load = sum(datasets[dataset_num]['weights'][i] for i in route['subset'])
#                 depot_load += route_load
#                 depot_dist += route['bid']
#                 customer_str = ', '.join(str(i) for i in route['subset'])
#                 out_lines.append(f"    Vehicle {vehicle_idx}: customers(global idx)=[{customer_str}], load={route_load}, distance={route['bid']:.2f}")

#             # If no routes assigned to this depot, show that no vehicles are used
#             if num_vehicles_at_depot == 0:
#                 out_lines.append("    No vehicles assigned to this depot")

#             out_lines.append(f"  Depot total distance: {depot_dist:.2f}, total load: {depot_load}")
#             out_lines.append("------------------------------------------------------------")

#         out_lines.append(f"Grand total distance across depots: {total_distance:.2f}")
#         out_lines.append("")

#     out_text = '\n'.join(out_lines)

#     with open('combinatorial_auction_results.txt', 'w') as f:
#         f.write(out_text)

#     print(f"\nResults formatted and saved to 'combinatorial_auction_results.txt'")

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
            print("works")
            bid_matrix, subsets = generate_bids_matrix(dataset)
            print("works")
            print(f"Bid matrix shape: {bid_matrix.shape}")

            # Step 2: Find best bids
            print(f"\nStep 2: Finding best bids...")
            best_bids = find_best_bids(bid_matrix, subsets)
            print(f"Best bids found for {len(best_bids)} subsets")

            # Step 3: Solve Set Partitioning Problem
            print(f"\nStep 3: Solving Set Partitioning Problem...")
            solution, total_cost, status = solve_set_partitioning(best_bids, len(dataset['target_coords']))

            # Store results
            results[dataset_num] = {
                'status': status,
                'total_cost': total_cost,
                'solution': solution if solution else [],
                'n_subsets_used': len(solution) if solution else 0,
                'feasible_subsets': 0,
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
            'Feasible_Subsets': 0,
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
# Main execution
if __name__ == "__main__":
    # Run the combinatorial auction solver for all datasets
    print("Starting Combinatorial Auction CVRP Solver...")
    print("Processing all 100 datasets...")

    results = solve_cvrp_combinatorial_auction(
        'CVRP_10Vehicles_100Targets.txt', 
        dataset_nums=list(range(1,2)),  # Process all 100 datasets
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
