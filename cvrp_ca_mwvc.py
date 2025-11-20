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
# Variant of the MWIS heuristic that converts the independent-set problem into
# a vertex-cover problem:
#   1) Build the standard subset conflict graph.
#   2) Transform node weights so that Min-WIS becomes Max-WIS, then solve a
#      Min Weighted Vertex Cover (MWVC) via a primal–dual 2-approximation.
#   3) Use the complement of the cover as the candidate independent set,
#      accept newly covered targets, and repeat until all are assigned.
# -----------------------------------------------------------------------------

# Install required packages
try:
    import pulp
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", "pulp"])
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
        best_depot = int(np.argmin(depot_bids))
        best_bid_value = float(depot_bids[best_depot])
        
        if best_bid_value != float('inf'):
            best_bids[subset_idx] = {
                'depot': best_depot,
                'bid': best_bid_value,
                'subset': tuple(subsets[subset_idx])
            }
    
    return best_bids

# ============================================================================
# PART 2: CONFLICT GRAPH AND CONVERSION-BASED MIN-WIS SOLVER
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

# ---------- Helpers for MVC primal-dual ----------
def normalize_edges(edges):
    """Return list of sorted 2-tuples (u,v) with u!=v. Removes self-loops and duplicates."""
    norm = set()
    for a,b in edges:
        if a == b:
            continue
        u, v = (a, b) if a <= b else (b, a)
        norm.add((u, v))
    return list(norm)

def min_weighted_vertex_cover_primal_dual(nodes, edges, weights):
    """
    Primal-dual 2-approximation for the minimum weighted vertex cover.

    Inputs:
    - nodes: iterable of node ids (hashable)
    - edges: iterable of 2-tuples (u,v)
    - weights: dict mapping node -> positive weight (cost)

    Returns:
    - cover: set of nodes chosen as the (approximate) vertex cover
    """
    edges = normalize_edges(edges)
    nodes = list(nodes)
    rem = {v: float(weights.get(v, 0.0)) for v in nodes}
    uncovered = set(edges)
    cover = set()

    # For fast edge removal, maintain adjacency from node -> incident edges
    incident = {v: set() for v in nodes}
    for e in edges:
        u, v = e
        if u not in incident:
            incident[u] = set()
        if v not in incident:
            incident[v] = set()
        incident[u].add(e)
        incident[v].add(e)

    # Main loop: while uncovered edges exist
    while uncovered:
        # pick an arbitrary uncovered edge
        u, v = next(iter(uncovered))
        du = rem.get(u, 0.0)
        dv = rem.get(v, 0.0)

        # If one endpoint already \"zero\", pick it
        if du <= 0.0:
            chosen = u
            cover.add(chosen)
        elif dv <= 0.0:
            chosen = v
            cover.add(chosen)
        else:
            # Raise dual until one endpoint becomes tight
            delta = min(du, dv)
            # subtract delta from both endpoints
            rem[u] = du - delta
            rem[v] = dv - delta
            # if any became tight, add to cover
            just_added = []
            if rem[u] <= 1e-12:
                cover.add(u); just_added.append(u)
            if rem[v] <= 1e-12:
                cover.add(v); just_added.append(v)
            # Remove edges incident to just_added
            if just_added:
                for x in just_added:
                    for e in list(incident.get(x, [])):
                        if e in uncovered:
                            uncovered.discard(e)
                            a,b = e
                            incident[a].discard(e)
                            incident[b].discard(e)
                continue
            # If neither became tight (delta was 0), fall back to picking min
            if delta == 0:
                chosen = u if du <= dv else v
                cover.add(chosen)

        # If a chosen vertex was added above, remove its incident edges
        if cover:
            last_chosen = chosen
            for e in list(incident.get(last_chosen, [])):
                if e in uncovered:
                    uncovered.discard(e)
                    a,b = e
                    incident[a].discard(e)
                    incident[b].discard(e)

    return cover

# ---------- Conversion-based MinWIS via MaxWIS via MinWVC ----------
def solve_minwis_via_maxwis_via_mvc(G, n_targets):
    """
    Iteratively solve MIN-WEIGHT INDEPENDENT SET by:
      - converting node weights w -> w' = K - w (K > max w in current candidate graph),
      - solving Min Weighted Vertex Cover on w' (primal-dual),
      - taking complement (an independent set under w') which is the MaxWIS candidate,
      - evaluating its cost under original weights w and selecting it for that iteration,
      - removing itemsets that intersect newly covered targets,
      - repeating until all targets covered or no candidates remain.

    Returns: selected_subsets_list, total_cost, status
    """
    print("\nSolving using ITERATIVE MinWIS via conversion to MaxWIS and MinWVC...")
    all_targets = set(range(n_targets))
    covered_targets = set()
    remaining_nodes = set(G.nodes())
    selected_subsets = []
    total_cost = 0.0
    iteration = 0

    while covered_targets != all_targets and remaining_nodes:
        iteration += 1

        # Candidate nodes: those in remaining_nodes whose subset is disjoint from covered_targets
        candidates = [n for n in remaining_nodes if set(G.nodes[n]['subset']).isdisjoint(covered_targets)]
        if not candidates:
            print(f"Iteration {iteration}: no candidates remaining that are disjoint from already-covered targets.")
            break

        G_iter = G.subgraph(candidates).copy()
        nodes_iter = list(G_iter.nodes())
        edges_iter = list(G_iter.edges())

        # Original weights (for MinWIS objective)
        orig_weights = {n: float(G_iter.nodes[n].get('bid', 0.0)) for n in nodes_iter}

        # Choose K > max(orig_weights) to ensure positive transformed weights
        max_w = max(orig_weights.values()) if orig_weights else 0.0
        # Use K = max_w + 1 to keep transformed weights positive and reasonably scaled.
        K = max_w + 1.0

        # Transform weights: w' = K - w
        transformed_weights = {n: (K - orig_weights[n]) for n in nodes_iter}

        print(f"Iteration {iteration}: candidates={len(nodes_iter)}, edges={len(edges_iter)}, covered={len(covered_targets)}/{n_targets}")
        print(f"  max_w={max_w:.2f}, K={K:.2f}")

        # Solve Min Weighted Vertex Cover with transformed weights
        cover_iter = min_weighted_vertex_cover_primal_dual(nodes_iter, edges_iter, transformed_weights)

        # Complement is MaxWIS candidate under transformed weights
        indep_candidate = set(nodes_iter) - set(cover_iter)

        # Evaluate candidate under original weights
        cost_orig = sum(orig_weights[n] for n in indep_candidate)
        covered_by_candidate = set().union(*(set(G_iter.nodes[n]['subset']) for n in indep_candidate)) if indep_candidate else set()

        print(f"  Candidate (independent from transformed MVC complement): size={len(indep_candidate)}, cost_under_original={cost_orig:.2f}, new_targets={len(covered_by_candidate)}")

        # If candidate covers no new targets, stop (no progress)
        new_targets = covered_by_candidate - covered_targets
        if not new_targets:
            print(f"Iteration {iteration}: candidate covers no NEW targets. Stopping.")
            break

        # Add chosen independent nodes to solution
        for node in sorted(indep_candidate):
            nd = G.nodes[node]
            selected_subsets.append({
                'subset_idx': node,
                'subset': nd['subset'],
                'depot': nd['depot'],
                'bid': nd['bid']
            })
        total_cost += cost_orig
        before_cov = len(covered_targets)
        covered_targets.update(new_targets)
        after_cov = len(covered_targets)

        print(f"Iteration {iteration}: selected {len(indep_candidate)} nodes, newly covered {len(new_targets)}, total covered {after_cov}/{n_targets}, iteration cost {cost_orig:.2f}")

        # Remove from remaining_nodes any node whose subset intersects covered_targets
        nodes_to_remove = [n for n in list(remaining_nodes) if not set(G.nodes[n]['subset']).isdisjoint(covered_targets)]
        for n in nodes_to_remove:
            remaining_nodes.discard(n)

        # Safety: if no progress this iteration, stop
        if after_cov == before_cov:
            print(f"Iteration {iteration}: no progress made (no new targets). Stopping to avoid infinite loop.")
            break

    # Final check
    if covered_targets == all_targets:
        print(f"\n✓ All {n_targets} targets covered in {iteration} iterations")
        print(f"✓ Total cost: {total_cost:.2f}")
        print(f"✓ Subsets used: {len(selected_subsets)}")
        return selected_subsets, total_cost, "Complete (Iterative MinWIS via MaxWIS->MinWVC)"
    else:
        uncovered = sorted(list(all_targets - covered_targets))
        print(f"\n✗ Failed to cover {len(uncovered)} targets: {uncovered}")
        return None, None, f"Incomplete: {len(uncovered)} targets uncovered"

# Keep wrapper name for compatibility
def solve_mwis_greedy(G, n_targets):
    """
    Outer entry kept — now uses conversion-based MinWIS solver.
    """
    return solve_minwis_via_maxwis_via_mvc(G, n_targets)

def solve_mwis_approach(best_bids, n_targets):
    """Solve using conversion-based approach (keeps same I/O as before)."""
    print("\n" + "="*70)
    print("SOLVING USING ITERATIVE MinWIS via MaxWIS->MinWVC")
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
    print("COMBINATORIAL AUCTION CVRP SOLVER (Iterative MinWIS via MaxWIS->MinWVC)")
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
            
            # Solve using Iterative MinWIS via conversion approach
            print("\nStep 3: Solving with Iterative MinWIS (conversion -> MinWVC)...")
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
    print("Starting Combinatorial Auction CVRP Solver (Iterative MinWIS via MaxWIS->MinWVC)...\n")
    
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
