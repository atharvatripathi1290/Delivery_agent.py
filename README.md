Autonomous Delivery Agent

This project simulates an autonomous delivery agent navigating a 2D grid world. The agent must move from a start location to a goal while considering terrain costs, static obstacles, and dynamic moving obstacles. Different search algorithms are implemented and compared in terms of path cost, nodes expanded, and execution time.

The code file is named delivery_agent.py.


---

Features

GridWorld Environment

Walls (#)

Terrain costs (1-9)

Moving obstacles (cyclic paths)


Algorithms Implemented

Breadth-First Search (BFS)

Uniform Cost Search (UCS)

A* Search

Simulated Annealing with Replanning


Experiment Logging

Results saved in results/results.csv

Performance plots generated for cost, nodes expanded, and execution time


ASCII Visualization

Prints the grid map with path, start (S), and goal (G)




---

Installation

Make sure you have Python 3.8+ installed.
Install the required libraries:

pip install matplotlib pandas


---

Usage

Run the program with command-line arguments:

python delivery_agent.py [OPTIONS]

Options

--algo
Choose which algorithm to run.
Values: bfs, ucs, astar, simulated_annealing_replan, all
Default: all

--map
Choose which map to run on.
Values: small, medium, large, dynamic, all
Default: all

--plot
Generate plots from results/results.csv.



---

Examples

Run all algorithms on all maps:

python delivery_agent.py

Run A* on the medium map:

python delivery_agent.py --algo astar --map medium

Run BFS on the dynamic map with moving obstacles:

python delivery_agent.py --algo bfs --map dynamic

Generate comparison plots after running experiments:

python delivery_agent.py --plot


---

Output

Console shows algorithm performance:

--- small ---
bfs: cost=6, nodes=10, time=0.0001s
ucs: cost=6, nodes=8, time=0.0001s
astar: cost=6, nodes=6, time=0.0000s
simulated_annealing_replan: cost=6, nodes=12, time=0.0012s

Path visualization in ASCII grid:

S***
1#*1
111*
111G

Results stored in:

results/results.csv

results/cost_comparison.png

results/nodes_comparison.png

results/time_comparison.png




---

File Structure

delivery_agent.py    # Main code file
results/             # Stores results and plots
  ├── results.csv
  ├── cost_comparison.png
  ├── nodes_comparison.png
  └── time_comparison.png


---

Future Improvements

Support diagonal movement

Add probabilistic moving obstacles

Integrate real map inputs from external files

GUI visualization of the agent
