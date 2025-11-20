#!/usr/bin/env python3

"""
create_computation_time_comparison.py

Creates a line graph comparing average computation times across different vehicle capacities
for 4 CVRP solution techniques:
1. Combinatorial Auction Method
2. Nearest Neighbor Method  
3. 2-Stage (OR-Tools)
4. 2-Stage WGLS

Reads CSV files from respective folders for capacity 100, 120, 140, 160, 180, 200.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# Configuration for each technique
TECHNIQUES = {
    'Combinatorial Auction': {
        'folder': 'CA_results',
        'file_pattern': 'timings_capacity_*.csv',
        'time_column': 'Computation_Time_Seconds',
        'status_column': 'Status',
        'status_value': 'Optimal',
        'color': 'blue',
        'linestyle': '-',
        'marker': 'o'
    },
    'Nearest Neighbor': {
        'folder': 'nearest_neighbor_results',
        'file_pattern': 'results_capacity_*.csv',
        'time_column': 'Time_Seconds',
        'status_column': 'Status',
        'status_value': 'OK',
        'color': 'orange',
        'linestyle': '-',
        'marker': 's'
    },
    '2-Stage': {
        'folder': '2stage_results',
        'file_pattern': 'results_capacity_*.csv',
        'time_column': 'TimeSeconds',
        'status_column': 'Status',
        'status_value': 'OK',
        'color': 'green',
        'linestyle': '-',
        'marker': '^'
    },
    '2-Stage WGLS': {
        'folder': '2stage_wgls_results',
        'file_pattern': 'results_capacity_*.csv',
        'time_column': 'TimeSeconds',
        'status_column': 'Status',
        'status_value': 'OK',
        'color': 'red',
        'linestyle': '-',
        'marker': 'd'
    }
}

def load_technique_results(technique_name, config, capacity_range=[100, 120, 140, 160, 180, 200]):
    """
    Load computation time results for a specific technique across all capacities.
    
    Returns:
        capacities: list of capacity values
        avg_times: list of average computation times
    """
    folder = config['folder']
    file_pattern = config['file_pattern']
    time_column = config['time_column']
    status_column = config['status_column']
    status_value = config['status_value']
    
    capacities = []
    avg_times = []
    
    for capacity in capacity_range:
        # Construct filename
        if 'timings' in file_pattern:
            filename = file_pattern.replace('*', str(capacity))
        else:
            filename = file_pattern.replace('*', str(capacity))
        
        filepath = os.path.join(folder, filename)
        
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found for {technique_name}")
            continue
        
        try:
            # Read CSV and filter only successful runs so averages aren't skewed.
            df = pd.read_csv(filepath)
            
            # Filter for successful solutions
            df_success = df[df[status_column] == status_value]
            
            if len(df_success) == 0:
                print(f"Warning: No successful solutions in {filepath}")
                continue
            
            # Calculate average computation time
            avg_time = df_success[time_column].mean()
            
            capacities.append(capacity)
            avg_times.append(avg_time)
            
            print(f"{technique_name} - Capacity {capacity}: avg={avg_time:.4f}s, n={len(df_success)}")
            
        except Exception as e:
            print(f"Error reading {filepath}: {str(e)}")
            continue
    
    return capacities, avg_times

def create_comparison_line_graph(all_results, output_file='computation_time_comparison.png'):
    """
    Create line graph comparing computation times for all techniques.
    
    Args:
        all_results: dict mapping technique_name -> (capacities, avg_times)
        output_file: output filename
    """
    plt.figure(figsize=(12, 7))
    
    # Plot each technique
    for technique_name, (capacities, avg_times) in all_results.items():
        if len(capacities) == 0:
            print(f"Warning: No data to plot for {technique_name}")
            continue
        
        config = TECHNIQUES[technique_name]
        plt.plot(capacities, avg_times,
                label=technique_name,
                color=config['color'],
                linestyle=config['linestyle'],
                marker=config['marker'],
                markersize=8,
                linewidth=2.5)
    
    # Customize plot
    plt.title('Average Computation Time Comparison Over Various Capacities',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Vehicle Capacity', fontsize=14, fontweight='bold')
    plt.ylabel('Average Computation Time (seconds)', fontsize=14, fontweight='bold')
    
    # Add legend
    plt.legend(loc='best', fontsize=12, framealpha=0.95,
              edgecolor='black', fancybox=False)
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # Set axis limits and ticks
    plt.xlim(95, 205)
    capacity_ticks = [100, 120, 140, 160, 180, 200]
    plt.xticks(capacity_ticks, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nLine graph saved as '{output_file}'")
    
    # Display plot
    plt.show()

def print_comparison_table(all_results):
    """Print a comparison table of average computation times."""
    print("\n" + "="*90)
    print("AVERAGE COMPUTATION TIME COMPARISON TABLE (seconds)")
    print("="*90)
    
    # Get all capacities
    all_capacities = set()
    for capacities, _ in all_results.values():
        all_capacities.update(capacities)
    all_capacities = sorted(list(all_capacities))
    
    # Print header
    header = f"{'Capacity':<12}"
    for technique in TECHNIQUES.keys():
        header += f"{technique:<25}"
    print(header)
    print("-" * 90)
    
    # Print data for each capacity
    for capacity in all_capacities:
        row = f"{capacity:<12}"
        for technique_name in TECHNIQUES.keys():
            if technique_name in all_results:
                capacities, avg_times = all_results[technique_name]
                if capacity in capacities:
                    idx = capacities.index(capacity)
                    row += f"{avg_times[idx]:<25.4f}"
                else:
                    row += f"{'N/A':<25}"
            else:
                row += f"{'N/A':<25}"
        print(row)
    
    print("="*90)
    
    # Print summary statistics
    print("\n" + "="*90)
    print("SUMMARY STATISTICS (across all capacities)")
    print("="*90)
    
    summary_header = f"{'Technique':<30}{'Min (s)':<15}{'Max (s)':<15}{'Mean (s)':<15}{'Std (s)':<15}"
    print(summary_header)
    print("-" * 90)
    
    for technique_name, (capacities, avg_times) in all_results.items():
        if len(avg_times) > 0:
            min_time = min(avg_times)
            max_time = max(avg_times)
            mean_time = np.mean(avg_times)
            std_time = np.std(avg_times)
            
            row = f"{technique_name:<30}{min_time:<15.4f}{max_time:<15.4f}{mean_time:<15.4f}{std_time:<15.4f}"
            print(row)
    
    print("="*90)

def main():
    """Main function to create computation time comparison line graph."""
    
    print("="*90)
    print("CVRP TECHNIQUES - AVERAGE COMPUTATION TIME COMPARISON")
    print("="*90)
    
    capacity_range = [100, 120, 140, 160, 180, 200]
    print(f"\nCapacity range: {capacity_range}")
    print(f"Techniques to compare: {list(TECHNIQUES.keys())}")
    
    # Load data for all techniques
    all_results = {}
    
    print("\n" + "-"*90)
    print("LOADING DATA FOR EACH TECHNIQUE")
    print("-"*90)
    
    for technique_name, config in TECHNIQUES.items():
        print(f"\n{technique_name}:")
        print(f"  Folder: {config['folder']}")
        print(f"  File pattern: {config['file_pattern']}")
        print(f"  Time column: {config['time_column']}")
        
        capacities, avg_times = load_technique_results(technique_name, config, capacity_range)
        
        if len(capacities) > 0:
            all_results[technique_name] = (capacities, avg_times)
            print(f"  ✓ Loaded {len(capacities)} capacity configurations")
        else:
            print(f"  ✗ No data loaded for {technique_name}")
    
    if len(all_results) == 0:
        print("\nError: No data loaded for any technique!")
        print("\nPlease ensure:")
        print("1. Folders exist: CA_results, nearest_neighbor_results, 2stage_results, 2stage_wgls_results")
        print("2. CSV files exist for each capacity: *_capacity_100.csv, *_capacity_120.csv, etc.")
        print("3. CSV files have the correct time columns:")
        print("   - CA: Computation_Time_Seconds")
        print("   - Nearest Neighbor: Time_Seconds")
        print("   - 2-Stage: TimeSeconds")
        print("   - 2-Stage WGLS: TimeSeconds")
        return
    
    # Print comparison table
    print_comparison_table(all_results)
    
    # Create line graph
    print("\n" + "-"*90)
    print("CREATING LINE GRAPH")
    print("-"*90)
    
    create_comparison_line_graph(all_results, 'computation_time_comparison.png')
    
    print("\n" + "="*90)
    print("COMPARISON COMPLETED SUCCESSFULLY")
    print("="*90)
    print("\nOutput file: computation_time_comparison.png")

if __name__ == "__main__":
    main()
