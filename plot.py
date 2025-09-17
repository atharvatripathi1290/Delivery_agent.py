import matplotlib.pyplot as plt
import numpy as np

# Data from user table
maps = ['small', 'small', 'small', 'small',
        'medium', 'medium', 'medium', 'medium',
        'large', 'large', 'large', 'large',
        'dynamic', 'dynamic', 'dynamic', 'dynamic']

algorithms = ['BFS', 'UCS', 'A*', 'Simulated Annealing',
              'BFS', 'UCS', 'A*', 'Simulated Annealing',
              'BFS', 'UCS', 'A*', 'Simulated Annealing',
              'BFS', 'UCS', 'A*', 'Simulated Annealing']

costs = [6, 6, 6, 7,
         10, 10, 10, 12,
         38, 38, 38, 45,
         float('inf'), float('inf'), float('inf'), 14]

nodes_expanded = [10, 12, 8, 80,
                  45, 52, 30, 140,
                  401, 520, 222, 1600,
                  125, 164, 67, 330]

times = [0.001, 0.002, 0.001, 0.010,
         0.005, 0.008, 0.004, 0.024,
         0.08, 0.13, 0.05, 0.40,
         0.02, 0.03, 0.01, 0.12]

# Handling inf costs: replace by a number larger than max for plotting
max_cost = max([c for c in costs if c != float('inf')])
plot_costs = [c if c != float('inf') else max_cost * 1.5 for c in costs]

# Unique maps and algorithms
unique_maps = sorted(set(maps))
unique_algos = sorted(set(algorithms), key=lambda x: ['BFS', 'UCS', 'A*', 'Simulated Annealing'].index(x))

# Create bar positions
bar_width = 0.2
indices = np.arange(len(unique_maps))
offsets = np.arange(len(unique_algos)) * bar_width - (bar_width * (len(unique_algos)-1)/2)

def plot_metric(metric_values, ylabel, title, filename, ylimit=None):
    plt.figure(figsize=(10,6))
    for i, algo in enumerate(unique_algos):
        y = []
        for m in unique_maps:
            # Find metric for this map-algo
            for j in range(len(maps)):
                if maps[j] == m and algorithms[j] == algo:
                    y.append(metric_values[j])
                    break
        plt.bar(indices + offsets[i], y, bar_width, label=algo)

    plt.xticks(indices, unique_maps)
    plt.xlabel('Map')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    if ylimit:
        plt.ylim(0, ylimit)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# Plot Cost (show capped max for inf)
plot_metric(plot_costs, 'Cost', 'Path Cost Comparison', 'cost_comparison.png', ylimit=max_cost*1.6)

# Plot Nodes Expanded
max_nodes = max(nodes_expanded)
plot_metric(nodes_expanded, 'Nodes Expanded', 'Nodes Expanded Comparison', 'nodes_comparison.png', ylimit=max_nodes*1.1)

# Plot Time
max_time = max(times)
plot_metric(times, 'Time (seconds)', 'Runtime Comparison', 'time_comparison.png', ylimit=max_time*1.1)
