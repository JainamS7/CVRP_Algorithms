# CVRP Final Experiments

Capacitated Vehicle Routing Problem (CVRP) experiments exploring combinatorial auction formulations, OR-Tools solvers (with and without Guided Local Search), and simple nearest-neighbor heuristics. The scripts target the `CVRP_10Vehicles_100Targets.txt` benchmark (10 depots, 100 customers) and sweep vehicle capacities from 100 to 200.

## Repository Map

| Path | Description |
| --- | --- |
| `cvrp_ca.py` | Baseline combinatorial auction (CA) solver: enumerates feasible customer subsets per depot, prices them via brute-force TSP, and solves a set-partitioning MILP with PuLP. |
| `cvrp_ca_LP.py` | CA variant that keeps all subsets (infeasible ones receive a large penalty) and uses LP relaxation plus rounding instead of an integer solver. |
| `cvrp_ca_mwis.py` | CA post-processing that builds a conflict graph over priced subsets and finds a cover with a greedy Minimum Weighted Independent Set heuristic. |
| `cvrp_ca_mwvc.py` | Experimental CA heuristic that converts Min-WIS to Max-WIS via Min Weighted Vertex Cover (primal–dual 2-approximation). |
| `cvrp_ca_gridsearch.py` | Grid search harness that monkey-patches the CA solver’s subset TSP capacity to benchmark different truck capacities and record timings (`CA_results/`). |
| `cvrp_2stage_gridsearch.py` | Two-stage baseline: nearest depot assignment followed by OR-Tools CVRP per depot with Guided Local Search (GLS) enabled; results saved to `2stage_results/`. |
| `cvrp_2stage_wgls.py` | Same as above but **without** the GLS metaheuristic (first-solution only), producing `2stage_wgls_results/`. |
| `cvrp_nn_gridsearch.py` | Two-stage + nearest neighbor routing heuristic; writes CSVs to `nearest_neighbor_results/`. |
| `average_time_comparison.py` | Reads per-capacity CSVs for all techniques and plots `computation_time_comparison.png`. |
| `average_tour_comparison.png` | Example visualization comparing tour statistics (created externally). |
| `2stage_results/`, `2stage_wgls_results/`, `CA_results/`, `CA_results/timings_capacity_*.csv`, `nearest_neighbor_results/` | Aggregated metrics per capacity. |
| `CA_results/timings_capacity_*.csv` | Per-dataset timing summaries emitted by `cvrp_ca_gridsearch.py`. |

## Prerequisites

- Python ≥ 3.9
- `pip install numpy pandas pulp ortools networkx matplotlib`
- Benchmark file `CVRP_10Vehicles_100Targets.txt` placed in the repo root (or pass an explicit path via CLI flags).

## Running the Solvers

### Combinatorial Auction (exact MILP)

```bash
python cvrp_ca.py
```

- Processes datasets 1–100, enumerating customer subsets up to size 3.
- Writes human-readable assignments to `combinatorial_auction_results.txt`.
- To target a subset of datasets or change subset size, edit the `solve_cvrp_combinatorial_auction` call near the bottom of the script.

### Combinatorial Auction Variants

- **LP relaxation:** `python cvrp_ca_LP.py` runs the same pipeline but relaxes the set-partitioning MILP to an LP, rounds the fractional solution with a 1/K threshold, and resolves conflicts greedily.
- **MWIS heuristics:** `python cvrp_ca_mwis.py` and `python cvrp_ca_mwvc.py` reuse the bidding phase but replace the MILP with graph-theoretic approximations (greedy Min-WIS and Min-WIS via Min-WVC respectively). Both scripts emit summaries at `cvrp_ca_greedy_mwis_results.txt`.

### Capacity Sweeps for CA

```bash
python cvrp_ca_gridsearch.py \
  --solver-path /Users/jainam/Downloads/cvrp_final/cvrp_ca.py \
  --datafile /path/to/CVRP_10Vehicles_100Targets.txt \
  --start 100 --end 200 --step 20 \
  --datasets-from 1 --datasets-to 100 \
  --max-subset-size 3 \
  --output-dir CA_results
```

The harness temporarily overrides the CA solver’s `solve_tsp_brute_force` capacity parameter, runs each dataset for each capacity, and logs computation time, status, and subset statistics to `timings_capacity_<capacity>.csv`.

### Two-Stage OR-Tools Baselines

Both `cvrp_2stage_gridsearch.py` (with GLS) and `cvrp_2stage_wgls.py` (without GLS) perform:

1. Assign each customer to its nearest depot.
2. Build a CVRP per depot with up to one vehicle per assigned customer.
3. Run OR-Tools for at most 10 seconds per depot to extract routes.

Usage:

```bash
python cvrp_2stage_gridsearch.py   # GLS-enabled
python cvrp_2stage_wgls.py         # first-solution only (WGLS)
```

Outputs per-capacity CSVs (`results_capacity_<capacity>.csv`) containing cost, runtime, and vehicle counts in their respective result folders.

### Two-Stage + Nearest Neighbor

```bash
python cvrp_nn_gridsearch.py
```

Implements the same depot-assignment stage but replaces OR-Tools with a deterministic nearest-neighbor route builder that respects capacity. Produces per-capacity CSVs plus `nearest_neighbor_results/grid_search_summary.csv`.

### Aggregate Timing Plot

```bash
python average_time_comparison.py
```

Reads the CSVs in `CA_results/`, `nearest_neighbor_results/`, `2stage_results/`, and `2stage_wgls_results/`, averages the runtime of successful runs at capacities 100…200, prints comparison tables, and saves `computation_time_comparison.png`.

## Results & Artifacts

- **Per-capacity CSVs:** Named `results_capacity_<cap>.csv` or `timings_capacity_<cap>.csv` depending on the solver family. Each row tracks dataset ID, runtime, feasibility status, total route cost, and solver-specific counters (e.g., vehicles used or subsets selected).
- **Summaries:** `nearest_neighbor_results/grid_search_summary.csv` aggregates NN metrics across capacities. `CA_results/timings_summary.csv` summarizes CA capacity sweeps.
- **Visuals:** `computation_time_comparison.png` (runtime trends) and `average_tour_comparison.png` (tour quality comparison).

## Extending the Project

1. **New heuristics:** Reuse the parsing utilities (`parse_cvrp_data`) from any script and plug in your routing or assignment logic. Keep CSV schemas consistent to leverage the plotting scripts.
2. **Different subset sizes/capacities:** Pass `--max-subset-size` or update the solver call to explore larger bundles or different truck limits (note the combinatorial explosion when max subset size grows).
3. **Alternative datasets:** Supply another benchmark via CLI flags; ensure formatting matches the provided `CVRP_10Vehicles_100Targets.txt` template (depots, targets, weights).

## Troubleshooting

- **Slow CA runtimes:** Reduce `max_subset_size`, limit dataset IDs, or try the LP/MWIS heuristics.
- **Missing CSVs for plotting:** Verify the folder names in `TECHNIQUES` inside `average_time_comparison.py` or rerun the corresponding grid search to regenerate data.
- **OR-Tools installation issues:** Install the matching `ortools` wheel for your Python version and architecture (`pip install ortools==9.8.3296`, for example).

This README provides high-level documentation for each module; see inline comments inside the scripts for algorithm details and parameter tuning hints.

