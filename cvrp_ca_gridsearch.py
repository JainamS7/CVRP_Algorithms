#!/usr/bin/env python3
"""
cvrp_capacity_sweep.py

Wrapper that imports an existing CVRP combinatorial auction solver script as a module,
monkey-patches its TSP solver to force different vehicle capacities, runs each dataset
(1..N) for each capacity in the range, records per-dataset times and status, and saves CSVs.

Usage:
    python cvrp_capacity_sweep.py --solver-path ./your_solver.py --datafile CVRP_10Vehicles_100Targets.txt

Defaults:
    capacities = 100..300 step 20 (100,120,...,300)
    datasets = 1..100
    max_subset_size = 3
"""

import argparse
import importlib.util
import time
import csv
import os
from statistics import mean

def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def make_capacity_wrapped_solver(orig_func, capacity_override):
    """
    Returns a wrapped version of orig_func that ignores any 'capacity' passed
    and instead uses capacity_override. The original signature is:
        solve_tsp_brute_force(depot_coord, target_coords, weights, capacity=100)
    """
    def wrapped(depot_coord, target_coords, weights, capacity=100):
        return orig_func(depot_coord, target_coords, weights, capacity_override)
    # preserve name for nicer logging if needed
    wrapped.__name__ = f"wrapped_solve_tsp_cap_{capacity_override}"
    return wrapped

def run_for_capacity(module, datafile, capacity, dataset_ids, max_subset_size, output_dir):
    # Each capacity is run independently so partial failures do not block others.
    # Prepare output CSV for this capacity
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, f"timings_capacity_{capacity}.csv")

    # Monkey-patch the module's TSP solver to force the capacity value
    if not hasattr(module, 'solve_tsp_brute_force'):
        raise AttributeError("The provided solver module does not contain 'solve_tsp_brute_force'")

    original_solver = module.solve_tsp_brute_force
    module.solve_tsp_brute_force = make_capacity_wrapped_solver(original_solver, capacity)

    # Get helper functions from the module
    if not hasattr(module, 'parse_cvrp_data'):
        raise AttributeError("The module must define parse_cvrp_data(filename)")

    parse = module.parse_cvrp_data
    generate_bids_matrix = module.generate_bids_matrix
    find_best_bids = module.find_best_bids
    solve_set_partitioning = module.solve_set_partitioning

    # Parse datasets
    datasets = parse(datafile)

    rows = []
    processed_dataset_count = 0
    elapsed_times = []

    for dataset_num in dataset_ids:
        if dataset_num not in datasets:
            print(f"[cap {capacity}] Dataset {dataset_num} not found in file — skipping")
            rows.append({
                'Dataset': dataset_num,
                'Computation_Time_Seconds': '',
                'Status': 'NotFound',
                'Total_Cost': '',
                'Feasible_Subsets': '',
                'Total_Subsets': '',
                'N_Subsets_Used': ''
            })
            continue

        dataset = datasets[dataset_num]
        print(f"[cap {capacity}] Processing dataset {dataset_num} (depots={len(dataset['vehicle_coords'])}, targets={len(dataset['target_coords'])})")

        t0 = time.perf_counter()
        try:
            bid_matrix, subsets, feasible_subsets = generate_bids_matrix(dataset, max_subset_size)
            best_bids = find_best_bids(bid_matrix, subsets, feasible_subsets)
            solution, total_cost, status = solve_set_partitioning(best_bids, len(dataset['target_coords']))

            t1 = time.perf_counter()
            elapsed = t1 - t0
            processed_dataset_count += 1
            elapsed_times.append(elapsed)

            n_subsets_used = len(solution) if solution else 0

            rows.append({
                'Dataset': dataset_num,
                'Computation_Time_Seconds': round(elapsed, 6),
                'Status': status,
                'Total_Cost': round(total_cost, 6) if (total_cost is not None and isinstance(total_cost, (int, float))) else total_cost,
                'Feasible_Subsets': len(feasible_subsets) if feasible_subsets is not None else '',
                'Total_Subsets': len(subsets) if subsets is not None else '',
                'N_Subsets_Used': n_subsets_used
            })

            print(f"[cap {capacity}] Dataset {dataset_num} done: time={elapsed:.6f}s status={status} cost={total_cost}")

        except Exception as e:
            t1 = time.perf_counter()
            elapsed = t1 - t0
            rows.append({
                'Dataset': dataset_num,
                'Computation_Time_Seconds': round(elapsed, 6),
                'Status': f"Error: {str(e)}",
                'Total_Cost': '',
                'Feasible_Subsets': '',
                'Total_Subsets': '',
                'N_Subsets_Used': ''
            })
            print(f"[cap {capacity}] Dataset {dataset_num} ERROR after {elapsed:.6f}s: {e}")

    # Restore original solver to avoid side-effects if module will be reused
    module.solve_tsp_brute_force = original_solver

    # Write CSV
    with open(out_csv, 'w', newline='') as csvf:
        fieldnames = ['Dataset', 'Computation_Time_Seconds', 'Status', 'Total_Cost',
                      'Feasible_Subsets', 'Total_Subsets', 'N_Subsets_Used']
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    avg_time = mean(elapsed_times) if elapsed_times else None
    # Return structured stats so the caller can build a summary CSV.
    return out_csv, avg_time, len(elapsed_times)

def main():
    parser = argparse.ArgumentParser(description="Run capacity sweep for your CVRP solver module.")
    parser.add_argument('--solver-path', required=True, help="Path to your solver .py file (the script you pasted).")
    parser.add_argument('--datafile', required=True, help="CVRP data file path (e.g., CVRP_10Vehicles_100Targets.txt).")
    parser.add_argument('--start', type=int, default=100, help="Start capacity (inclusive). Default 100.")
    parser.add_argument('--end', type=int, default=100, help="End capacity (inclusive). Default 300.")
    parser.add_argument('--step', type=int, default=20, help="Capacity step. Default 20.")
    parser.add_argument('--datasets-from', type=int, default=1, help="Dataset numbering start (default 1).")
    parser.add_argument('--datasets-to', type=int, default=1, help="Dataset numbering end (default 100).")
    parser.add_argument('--max-subset-size', type=int, default=3, help="Max subset size used by bidder generation (default 3).")
    parser.add_argument('--output-dir', default='CA_results_LP', help="Directory to save CSV results.")
    args = parser.parse_args()

    # Load module
    print("Loading solver module from:", args.solver_path)
    mod = load_module_from_path('cvrp_solver_module', args.solver_path)

    capacities = list(range(args.start, args.end + 1, args.step))
    dataset_ids = list(range(args.datasets_from, args.datasets_to + 1))

    summary_rows = []
    for cap in capacities:
        print("="*80)
        print(f"RUNNING CAPACITY = {cap}")
        csv_path, avg_time, n_runs = run_for_capacity(
            mod, args.datafile, cap, dataset_ids, args.max_subset_size, args.output_dir
        )
        print(f"Saved per-dataset timings to: {csv_path}")
        print(f"Average time across {n_runs} runs: {avg_time:.6f}s" if avg_time is not None else "No successful runs")
        summary_rows.append({
            'Capacity': cap,
            'PerDatasetCSV': csv_path,
            'Avg_Time_Seconds': round(avg_time, 6) if avg_time is not None else '',
            'NumSuccessfulRuns': n_runs
        })

    # Write summary CSV
    summary_csv = os.path.join(args.output_dir, 'timings_summary.csv')
    with open(summary_csv, 'w', newline='') as csvf:
        fieldnames = ['Capacity', 'PerDatasetCSV', 'Avg_Time_Seconds', 'NumSuccessfulRuns']
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    print("="*80)
    print("ALL CAPACITIES DONE. Summary written to:", summary_csv)

if __name__ == '__main__':
    
    main()
