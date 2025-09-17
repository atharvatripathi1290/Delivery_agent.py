import heapq
import random
import time
import math
import argparse
import csv
import os
from collections import deque
import matplotlib.pyplot as plt

# -----------------------------
# Environment and Agent
# -----------------------------
class GridWorld:
    """
    Represents a 2D grid world with static and dynamic obstacles and terrain costs.
    """
    def __init__(self, grid, moving_obstacles=None):
        """
        grid: list of strings representing the map.
        moving_obstacles: list of MovingObstacle instances.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
        self.moving_obstacles = moving_obstacles or []

    def in_bounds(self, r, c):
        """Check if (r,c) is inside the grid."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def cost(self, r, c, t=0):
        """
        Returns the movement cost to enter cell (r,c) at time t.
        Returns math.inf if cell is blocked by static or dynamic obstacle.
        """
        if not self.in_bounds(r, c):
            return math.inf
        if self.grid[r][c] == "#":  # wall
            return math.inf
        for mob in self.moving_obstacles:
            if mob.position_at_time(t) == (r, c):
                return math.inf
        # Terrain cost: digit or default 1
        return int(self.grid[r][c]) if self.grid[r][c].isdigit() else 1

    def neighbors(self, r, c):
        """Generate 4-connected neighbors."""
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r+dr, c+dc
            if self.in_bounds(nr, nc):
                yield nr, nc

class MovingObstacle:
    """
    Represents a moving obstacle with a deterministic cyclic path.
    """
    def __init__(self, path):
        """
        path: list of (row,col) tuples representing the obstacle's path.
        """
        self.path = path

    def position_at_time(self, t):
        """Returns the obstacle's position at time t."""
        return self.path[t % len(self.path)]

# -----------------------------
# Search Algorithms
# -----------------------------
def bfs(world, start, goal):
    """
    Breadth-first search ignoring costs (assumes uniform cost).
    Returns path, cost (steps), nodes expanded.
    """
    frontier = deque([(start, [start])])
    visited = set([start])
    nodes = 0
    while frontier:
        (r,c), path = frontier.popleft()
        nodes += 1
        if (r,c) == goal:
            return path, len(path)-1, nodes
        for nr,nc in world.neighbors(r,c):
            if (nr,nc) not in visited and world.cost(nr,nc,len(path)) < math.inf:
                visited.add((nr,nc))
                frontier.append(((nr,nc), path + [(nr,nc)]))
    return None, math.inf, nodes

def ucs(world, start, goal):
    """
    Uniform Cost Search considering terrain costs and dynamic obstacles.
    Returns path, cost, nodes expanded.
    """
    frontier = [(0, start, [start])]
    visited = {}
    nodes = 0
    while frontier:
        cost, (r,c), path = heapq.heappop(frontier)
        nodes += 1
        if (r,c) == goal:
            return path, cost, nodes
        if (r,c) in visited and visited[(r,c)] <= cost:
            continue
        visited[(r,c)] = cost
        for nr,nc in world.neighbors(r,c):
            step_cost = world.cost(nr,nc,cost+1)
            if step_cost < math.inf:
                heapq.heappush(frontier, (cost + step_cost, (nr,nc), path + [(nr,nc)]))
    return None, math.inf, nodes

def astar(world, start, goal):
    """
    A* search with Manhattan distance heuristic.
    Returns path, cost, nodes expanded.
    """
    def heuristic(a,b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    frontier = [(heuristic(start, goal), 0, start, [start])]  # (f, g, node, path)
    g_scores = {start: 0}
    nodes = 0

    while frontier:
        f, g, (r,c), path = heapq.heappop(frontier)
        nodes += 1
        if (r,c) == goal:
            return path, g, nodes
        for nr,nc in world.neighbors(r,c):
            step_cost = world.cost(nr,nc,g+1)
            if step_cost == math.inf:
                continue
            new_g = g + step_cost
            if (nr,nc) not in g_scores or new_g < g_scores[(nr,nc)]:
                g_scores[(nr,nc)] = new_g
                new_f = new_g + heuristic((nr,nc), goal)
                heapq.heappush(frontier, (new_f, new_g, (nr,nc), path + [(nr,nc)]))
    return None, math.inf, nodes

# -----------------------------
# Local Search: Simulated Annealing Replan
# -----------------------------
def simulated_annealing_replan(world, start, goal, max_steps=200, initial_temp=10.0, cooling_rate=0.95):
    """
    Local search with simulated annealing to handle dynamic obstacles.
    Attempts to find a path by replanning and random moves with temperature-based acceptance.
    Returns path, cost, nodes expanded.
    """
    current = start
    path = [start]
    nodes = 0
    temp = initial_temp

    for step in range(max_steps):
        subpath, cost, expanded = astar(world, current, goal)
        nodes += expanded
        if subpath:
            # Append subpath excluding current position (already in path)
            path += subpath[1:]
            return path, cost, nodes

        # If blocked, try random neighbor moves with simulated annealing acceptance
        neighbors = list(world.neighbors(*current))
        random.shuffle(neighbors)
        accepted = False
        for n in neighbors:
            c = world.cost(*n)
            if c == math.inf:
                continue
            # Calculate delta cost (heuristic)
            h_current = abs(current[0]-goal[0]) + abs(current[1]-goal[1])
            h_neighbor = abs(n[0]-goal[0]) + abs(n[1]-goal[1])
            delta = h_neighbor - h_current
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current = n
                path.append(current)
                accepted = True
                break
        if not accepted:
            # No move accepted, random restart near start
            current = start
            path = [start]
        temp *= cooling_rate  # cool down

    return None, math.inf, nodes

# -----------------------------
# Experiments + Logging
# -----------------------------
def run_all(world, start, goal, name, save_csv=True):
    """
    Runs all algorithms on the given world and logs results.
    """
    results = []
    print(f"\n--- {name} ---")
    for algo in [bfs, ucs, astar, simulated_annealing_replan]:
        t0 = time.time()
        path, cost, nodes = algo(world, start, goal)
        dt = time.time() - t0
        print(f"{algo.__name__}: cost={cost}, nodes={nodes}, time={dt:.4f}s")
        results.append([name, algo.__name__, cost, nodes, dt])
        # Save ASCII visualization for dynamic map demo
        if path and name == "dynamic":
            ascii_grid(world, path, start, goal)
    if save_csv:
        os.makedirs("results", exist_ok=True)
        file_exists = os.path.isfile("results/results.csv")
        with open("results/results.csv", "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["map", "algo", "cost", "nodes", "time"])
            for row in results:
                writer.writerow(row)

def plot_results():
    """
    Reads results.csv and plots cost, nodes expanded, and time comparisons.
    """
    import pandas as pd
    if not os.path.exists("results/results.csv"):
        print("No results.csv found. Run experiments first.")
        return
    df = pd.read_csv("results/results.csv")
    for metric in ["cost", "nodes", "time"]:
        plt.figure(figsize=(8,5))
        for m in df["map"].unique():
            sub = df[df["map"] == m]
            plt.bar(sub["algo"] + "_" + m, sub[metric])
        plt.title(f"{metric.capitalize()} Comparison")
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"results/{metric}_comparison.png")
        plt.close()
    print("Plots saved in results/ directory.")

# -----------------------------
# ASCII Visualizer
# -----------------------------
def ascii_grid(world, path, start, goal):
    """
    Prints an ASCII visualization of the path on the grid.
    """
    grid = [list(row) for row in world.grid]
    for r, c in path:
        if (r, c) != start and (r, c) != goal:
            grid[r][c] = "*"
    grid[start[0]][start[1]] = "S"
    grid[goal[0]][goal[1]] = "G"
    print("\nPath visualization:")
    for row in grid:
        print("".join(row))

# -----------------------------
# Map Loading and Generation
# -----------------------------
def load_map_from_file(filename):
    """
    Loads a grid map from a text file.
    Each line is a row; '#' for walls, digits for terrain cost, others treated as 1.
    """
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def generate_maps():
    """
    Returns a dictionary of predefined maps and a dynamic obstacle for the dynamic map.
    """
    small = [
        "1111",
        "1#11",
        "1111",
        "1111"
    ]
    medium = [
        "111111",
        "1#1111",
        "111#11",
        "111111",
        "111111",
        "111111"
    ]
    large = ["1"*20 for _ in range(20)]
    dynamic = [
        "11111",
        "11111",
        "11111",
        "11111",
        "11111"
    ]
    # Moving obstacle moves horizontally on row 2
    mob = MovingObstacle([(2,0),(2,1),(2,2),(2,3),(2,4)])
    return {
        "small": GridWorld(small),
        "medium": GridWorld(medium),
        "large": GridWorld(large),
        "dynamic": GridWorld(dynamic, [mob])
    }

# -----------------------------
# Main with CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Delivery Agent in GridWorld")
    parser.add_argument("--algo", choices=["bfs", "ucs", "astar", "simulated_annealing_replan", "all"], default="all",
                        help="Algorithm to run")
    parser.add_argument("--map", choices=["small", "medium", "large", "dynamic", "all"], default="all",
                        help="Map to run on")
    parser.add_argument("--plot", action="store_true", help="Generate plots from results.csv")
    args = parser.parse_args()

    maps = generate_maps()
    goals = {"small": (3,3), "medium": (5,5), "large": (19,19), "dynamic": (4,4)}

    if args.plot:
        plot_results()
    else:
        selected_maps = [args.map] if args.map != "all" else maps.keys()
        for name in selected_maps:
            start = (0,0)
            goal = goals[name]
            if args.algo == "all":
                run_all(maps[name], start, goal, name)
            else:
                algo_fn = {
                    "bfs": bfs,
                    "ucs": ucs,
                    "astar": astar,
                    "simulated_annealing_replan": simulated_annealing_replan
                }[args.algo]
                t0 = time.time()
                path, cost, nodes = algo_fn(maps[name], start, goal)
                dt = time.time() - t0
                print(f"{args.algo} on {name}: cost={cost}, nodes={nodes}, time={dt:.4f}s")
                if path:
                    ascii_grid(maps[name], path, start, goal)
