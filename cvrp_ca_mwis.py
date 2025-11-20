#!/usr/bin/env python3

import numpy as np
import pandas as pd
from itertools import combinations, permutations
import math
import subprocess
import sys
import time
import csv
import networkx as nx

# -----------------------------------------------------------------------------
# Combinatorial auction pipeline that replaces the MILP stage with a
# greedy Minimum Weighted Independent Set (MWIS) heuristic:
#   * Build the usual depot→subset bids.
#   * Construct a conflict graph where vertices are subsets and edges indicate
#     overlapping customers.
#   * Iteratively pick the cheapest subset per uncovered customer count to
#     approximate the MWIS, removing conflicting vertices each iteration.
# -----------------------------------------------------------------------------

# Install required packages
try:
    import pulp
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pulp"])
    import pulp

# ============================================================================
# PART 1: DATA PARSING
# ============================================================================

def parse_cvrp_data(filename):
    """Parse CVRP dataset file."""
    datasets = {}
    with open(filename, 'r') as f:
        content = f.read()
    
    dataset_blocks = content.split('Data set #')[1:]
    
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
    """Calculate Euclidean distance."""
    return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

def generate_subsets(n_targets, max_size=3):
    """Generate all possible subsets."""
    subsets = []
    for size in range(1, min(max_size + 1, n_targets + 1)):
        for subset in combinations(range(n_targets), size):
            subsets.append(subset)
    return subsets

def solve_tsp_brute_force(depot_coord, target_coords, weights, capacity=100):
    """Solve TSP for small subset using brute force."""
    if not target_coords:
        return 0, True
    
    # Check capacity constraint
    total_weight = sum(weights)
    if total_weight > capacity:
        return float('inf'), False
    
    # Single target
    if len(target_coords) == 1:
        target_coord = target_coords[0]
        distance = 2 * euclidean_distance(depot_coord, target_coord)
        return distance, True
    
    # Multiple targets - try all permutations
    min_distance = float('inf')
    for perm in permutations(range(len(target_coords))):
        distance = euclidean_distance(depot_coord, target_coords[perm[0]])
        for i in range(len(perm) - 1):
            distance += euclidean_distance(target_coords[perm[i]], target_coords[perm[i+1]])
        distance += euclidean_distance(target_coords[perm[-1]], depot_coord)
        min_distance = min(min_distance, distance)
    
    return min_distance, True

def generate_bids_matrix(dataset, max_subset_size=3):
    """Generate bids matrix for all depot-subset pairs."""
    vehicle_coords = dataset['vehicle_coords']
    target_coords = dataset['target_coords']
    weights = dataset['weights']
    
    n_vehicles = len(vehicle_coords)
    n_targets = len(target_coords)
    
    print(f"Generating subsets for {n_targets} targets...")
    
    subsets = generate_subsets(n_targets, max_size=max_subset_size)
    print(f"Generated {len(subsets)} subsets")
    
    bid_matrix = np.full((n_vehicles, len(subsets)), float('inf'))
    feasible_subsets = []
    
    print("Calculating bids for each depot-subset pair...")
    for depot_idx, depot_coord in enumerate(vehicle_coords):
        if depot_idx % 2 == 0:
            print(f"Processing depot {depot_idx}...")
        
        for subset_idx, subset in enumerate(subsets):
            subset_coords = [target_coords[i] for i in subset]
            subset_weights = [weights[i] for i in subset]
            
            distance, is_feasible = solve_tsp_brute_force(
                depot_coord, subset_coords, subset_weights, capacity=100
            )
            
            if is_feasible:
                bid_matrix[depot_idx, subset_idx] = distance
                if subset_idx not in feasible_subsets:
                    feasible_subsets.append(subset_idx)
    
    return bid_matrix, subsets, feasible_subsets

def find_best_bids(bid_matrix, subsets, feasible_subsets):
    """Find best bid for each subset across all depots."""
    best_bids = {}
    for subset_idx in feasible_subsets:
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

# ============================================================================
# PART 2: CONFLICT GRAPH AND GREEDY MWIS
# ============================================================================

def build_conflict_graph(best_bids):
    """
    Build conflict graph where:
    - Vertices = itemsets (subsets)
    - Edges = conflicts (subsets sharing targets)
    - Weights = bids (for minimization)
    """
    print("Building conflict graph...")
    G = nx.Graph()
    
    # Add vertices with weights
    for subset_idx, bid_info in best_bids.items():
        G.add_node(subset_idx, 
                  weight=bid_info['bid'],
                  subset=bid_info['subset'], 
                  depot=bid_info['depot'],
                  bid=bid_info['bid'])
    
    # Add edges between conflicting itemsets
    subset_indices = list(best_bids.keys())
    for i in range(len(subset_indices)):
        for j in range(i + 1, len(subset_indices)):
            idx1, idx2 = subset_indices[i], subset_indices[j]
            subset1 = set(best_bids[idx1]['subset'])
            subset2 = set(best_bids[idx2]['subset'])
            
            # If subsets share any target, add edge
            if not subset1.isdisjoint(subset2):
                G.add_edge(idx1, idx2)
    
    print(f"Conflict graph: {G.number_of_nodes()} vertices, {G.number_of_edges()} edges")
    return G

def solve_mwis_greedy(G, n_targets):
    """
    Solve Minimum Weighted Independent Set using GREEDY approximation.
    
    Algorithm:
    1. While uncovered targets exist:
       a. Find vertex with minimum COST/COVERAGE ratio
          (lowest cost per new target covered)
       b. Add to solution (forms independent set)
       c. Remove this vertex and all neighbors (conflicts)
    """
    print("\nSolving using Greedy Minimum Weighted Independent Set...")
    
    selected_subsets = []
    covered_targets = set()
    total_cost = 0
    iteration = 0
    
    G_work = G.copy()
    
    while len(covered_targets) < n_targets and G_work.number_of_nodes() > 0:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}:")
        print(f"  Covered targets: {len(covered_targets)}/{n_targets}")
        print(f"  Remaining vertices: {G_work.number_of_nodes()}")
        print(f"{'='*60}")
        
        # Find vertex with BEST cost/coverage ratio
        best_vertex = None
        best_ratio = float('inf')
        best_new_targets = set()
        
        for node in G_work.nodes():
            node_data = G_work.nodes[node]
            vertex_targets = set(node_data['subset'])
            new_targets = vertex_targets - covered_targets
            
            # Only consider vertices that cover new targets
            if len(new_targets) > 0:
                # Cost per new target covered
                ratio = node_data['bid'] / len(new_targets)
                
                if ratio < best_ratio:
                    best_vertex = node
                    best_ratio = ratio
                    best_new_targets = new_targets
        
        if best_vertex is None:
            print("  No vertex found that covers new targets!")
            break
        
        # Select this vertex
        node_data = G_work.nodes[best_vertex]
        selected_subsets.append({
            'subset_idx': best_vertex,
            'subset': node_data['subset'],
            'depot': node_data['depot'],
            'bid': node_data['bid']
        })
        
        covered_targets.update(node_data['subset'])
        total_cost += node_data['bid']
        
        print(f"  Selected vertex {best_vertex}:")
        print(f"    Subset: {node_data['subset']}")
        print(f"    Bid: {node_data['bid']:.2f}")
        print(f"    New targets: {len(best_new_targets)}")
        print(f"    Cost/Coverage ratio: {best_ratio:.4f}")
        
        # Remove this vertex and all neighbors (conflicting vertices)
        neighbors = list(G_work.neighbors(best_vertex))
        G_work.remove_node(best_vertex)
        
        print(f"  Removed neighbors (conflicts): {len(neighbors)}")
        
        # Also remove vertices covering only already-covered targets
        vertices_to_remove = []
        for node in list(G_work.nodes()):
            node_subset = set(G_work.nodes[node]['subset'])
            if node_subset.issubset(covered_targets):
                vertices_to_remove.append(node)
        
        for node in vertices_to_remove:
            if G_work.has_node(node):
                G_work.remove_node(node)
        
        print(f"  Removed redundant vertices: {len(vertices_to_remove)}")
        print(f"  Total removed: {1 + len(neighbors) + len(vertices_to_remove)}")
    
    # Check if all targets covered
    if len(covered_targets) == n_targets:
        print(f"\n✓ All {n_targets} targets covered in {iteration} iterations")
        print(f"✓ Total cost: {total_cost:.2f}")
        print(f"✓ Subsets used: {len(selected_subsets)}")
        return selected_subsets, total_cost, "Complete (Greedy Min-WIS)"
    else:
        uncovered = set(range(n_targets)) - covered_targets
        print(f"\n✗ Failed to cover {len(uncovered)} targets: {uncovered}")
        return None, None, f"Incomplete: {len(uncovered)} targets uncovered"

def solve_mwis_approach(best_bids, n_targets):
    """Solve using Greedy Minimum Weighted Independent Set approach."""
    print("\n" + "="*70)
    print("SOLVING USING GREEDY MINIMUM WEIGHTED INDEPENDENT SET")
    print("="*70)
    
    G = build_conflict_graph(best_bids)
    solution, total_cost, status = solve_mwis_greedy(G, n_targets)
    
    return solution, total_cost, status

# ============================================================================
# PART 3: OUTPUT FORMATTING
# ============================================================================

def format_and_print_results(results, datasets):
    """Format results and save to file."""
    out_lines = []
    
    for dataset_num, result in results.items():
        out_lines.append("="*80)
        out_lines.append(f"Dataset: Data set #{dataset_num}")
        
        n_depots = len(datasets[dataset_num]['vehicle_coords'])
        n_targets = len(datasets[dataset_num]['target_coords'])
        out_lines.append(f"Depots: {n_depots}, Customers: {n_targets}")
        out_lines.append("")
        
        depot_customer_count = {i: 0 for i in range(n_depots)}
        depot_routes = {i: [] for i in range(n_depots)}
        
        if result['solution']:
            for sol in result['solution']:
                depot_num = sol['depot']
                subset_targets = sol['subset']
                depot_customer_count[depot_num] += len(subset_targets)
                depot_routes[depot_num].append(sol)
        
        out_lines.append("Assignment summary:")
        for depot, count in depot_customer_count.items():
            out_lines.append(f"  Depot {depot}: {count} customers")
        
        total_distance = sum(sol['bid'] for sol in result['solution']) if result['solution'] else float('inf')
        
        out_lines.append("")
        out_lines.append("-" * 60)
        
        for depot in range(n_depots):
            assigned_routes = depot_routes[depot]
            depot_coord = datasets[dataset_num]['vehicle_coords'][depot]
            depot_load = 0
            depot_dist = 0
            
            out_lines.append(f"Depot {depot} at {depot_coord}")
            
            for vehicle_idx, route in enumerate(assigned_routes):
                route_load = sum(datasets[dataset_num]['weights'][i] for i in route['subset'])
                depot_load += route_load
                depot_dist += route['bid']
                
                customer_str = ', '.join(str(i) for i in route['subset'])
                out_lines.append(f"  Vehicle {vehicle_idx}: [{customer_str}], load={route_load}, dist={route['bid']:.2f}")
            
            if not assigned_routes:
                out_lines.append("  No vehicles assigned")
            
            out_lines.append(f"  Depot total: distance={depot_dist:.2f}, load={depot_load}")
            out_lines.append("-" * 60)
        
        out_lines.append(f"Grand total distance: {total_distance:.2f}")
        out_lines.append("")
    
    out_text = '\n'.join(out_lines)
    with open('cvrp_ca_greedy_mwis_results.txt', 'w') as f:
        f.write(out_text)
    
    print(f"\nResults saved to 'cvrp_ca_greedy_mwis_results.txt'")

# ============================================================================
# PART 4: MAIN SOLVER
# ============================================================================

def solve_cvrp_combinatorial_auction_mwis(filename, dataset_nums=None, max_subset_size=3):
    """Main solver function."""
    print("="*70)
    print("COMBINATORIAL AUCTION CVRP SOLVER (GREEDY MIN-WIS)")
    print("="*70)
    
    # Parse datasets
    print("\nParsing CVRP datasets...")
    datasets = parse_cvrp_data(filename)
    print(f"Loaded {len(datasets)} datasets")
    
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
        print(f"DATASET {dataset_num}")
        print(f"{'='*50}")
        
        dataset = datasets[dataset_num]
        print(f"Depots: {len(dataset['vehicle_coords'])}")
        print(f"Targets: {len(dataset['target_coords'])}")
        
        try:
            # Generate bids
            print("\nStep 1: Generating bids matrix...")
            bid_matrix, subsets, feasible_subsets = generate_bids_matrix(dataset, max_subset_size)
            print(f"Bid matrix: {bid_matrix.shape}, Feasible subsets: {len(feasible_subsets)}")
            
            # Find best bids
            print("\nStep 2: Finding best bids...")
            best_bids = find_best_bids(bid_matrix, subsets, feasible_subsets)
            print(f"Best bids: {len(best_bids)}")
            
            # Solve using Greedy MWIS
            print("\nStep 3: Solving with Greedy Min-WIS...")
            solution, total_cost, status = solve_mwis_approach(best_bids, len(dataset['target_coords']))
            
            results[dataset_num] = {
                'status': status,
                'total_cost': total_cost,
                'solution': solution if solution else [],
                'n_subsets_used': len(solution) if solution else 0
            }
            
            elapsed = time.time() - dataset_start_time
            
            timing_results.append({
                'Dataset': f"Data set #{dataset_num}",
                'Time_Seconds': round(elapsed, 4),
                'Total_Distance': round(total_cost, 2) if total_cost else 'FAILED',
                'Status': status,
                'Subsets_Used': len(solution) if solution else 0
            })
            
            print(f"\nDataset {dataset_num}: {status}")
            print(f"Time: {elapsed:.2f}s, Cost: {total_cost if total_cost else 'N/A'}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            results[dataset_num] = {
                'status': f'Error: {str(e)}',
                'total_cost': None,
                'solution': [],
                'n_subsets_used': 0
            }
    
    # Save results
    format_and_print_results(results, datasets)
    
    if timing_results:
        df_timing = pd.DataFrame(timing_results)
        df_timing.to_csv('cvrp_ca_greedy_mwis_timing.csv', index=False)
        print(f"Timing saved to 'cvrp_ca_greedy_mwis_timing.csv'")
    
    return results

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Starting Combinatorial Auction CVRP Solver with Greedy Min-WIS...\n")
    
    results = solve_cvrp_combinatorial_auction_mwis(
        'CVRP_10Vehicles_100Targets.txt',
        dataset_nums=[1],  # Test with dataset 1
        max_subset_size=3
    )
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    for dataset_num, result in results.items():
        status = result['status']
        cost = result.get('total_cost', None)
        if cost is not None:
            print(f"Dataset {dataset_num}: {status} - Cost: {cost:.2f}")
        else:
            print(f"Dataset {dataset_num}: {status}")
    
    print("\nSolver completed!")
