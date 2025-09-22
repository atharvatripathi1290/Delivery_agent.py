import matplotlib.pyplot as plt
import numpy as np

# Data from your table
maps = ['small', 'medium', 'large', 'dynamic']
algorithms = ['BFS', 'UCS', 'A*', 'Simulated Annealing']

# Costs
costs = {
    "BFS": [6, 10, 38, 68],
    "UCS": [6, 10, 38, 68],
    "A*": [6, 10, 38, 68],
    "Simulated Annealing": [7, 12, 45, 14]
}

# Nodes Expanded
nodes = {
    "BFS": [10, 45, 401, 125],
    "UCS": [12, 52, 520, 164],
    "A*": [8, 30, 222, 67],
    "Simulated Annealing": [80, 140, 1600, 330]
}

# Time
times = {
    "BFS": [0.001, 0.005, 0.08, 0.02],
    "UCS": [0.002, 0.008, 0.13, 0.03],
    "A*": [0.001, 0.004, 0.05, 0.01],
    "Simulated Annealing": [0.010, 0.024, 0.40, 0.12]
}

# Function to plot
def plot_comparison(title, ylabel, filename, data):
    x = np.arange(len(maps))
    width = 0.2

    plt.figure(figsize=(8,5))
    for i, algo in enumerate(algorithms):
        plt.bar(x + (i-1.5)*width, data[algo], width, label=algo)

    plt.xlabel("Map")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(x, maps)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"✅ Saved {filename}")

# Generate plots
plot_comparison("Path Cost Comparison", "Cost", "cost_comparison.png", costs)
plot_comparison("Nodes Expanded Comparison", "Nodes Expanded", "nodes_comparison.png", nodes)
plot_comparison("Runtime Comparison", "Time (seconds)", "time_comparison.png", times)
