# import heapq
#
# # -----------------------------
# # GRAPH INPUT FUNCTIONS
# # -----------------------------
# def get_graph_from_user():
#     print("\n=== GRAPH INPUT ===")
#     n = int(input("Enter number of nodes: "))
#
#     print("Enter node names (one per line):")
#     nodes = [input().strip() for _ in range(n)]
#
#     graph = {node: {} for node in nodes}
#
#     print("\nEnter number of edges:")
#     e = int(input())
#
#     print("\nEnter edges in format: node1 node2 cost")
#     print("Example: Kazanchis Aratkilo 125\n")
#
#     for _ in range(e):
#         a, b, w = input().split()
#         w = float(w)
#         graph[a][b] = w
#         graph[b][a] = w     # undirected graph
#
#     return graph, nodes
#
# def get_heuristics_from_user(nodes):
#     print("\n=== HEURISTIC INPUT (for A*) ===")
#     print("Enter heuristic values in format: node value")
#     print("Enter 0 for the goal node.\n")
#
#     heuristic = {}
#
#     for node in nodes:
#         h = float(input(f"h({node}) = "))
#         heuristic[node] = h
#
#     return heuristic
#
#
# # -----------------------------
# # UNIFORM COST SEARCH (UCS)
# # -----------------------------
# def uniform_cost_search(graph, start, goal):
#     pq = [(0, start, [])]  # (cost, node, path)
#     visited = set()
#
#     while pq:
#         cost, node, path = heapq.heappop(pq)
#
#         if node in visited:
#             continue
#         visited.add(node)
#
#         path = path + [node]
#
#         if node == goal:
#             return path, cost
#
#         for neigh, w in graph[node].items():
#             if neigh not in visited:
#                 heapq.heappush(pq, (cost + w, neigh, path))
#
#     return None, float("inf")
#
#
# # -----------------------------
# # A* SEARCH
# # -----------------------------
# def a_star(graph, heuristic, start, goal):
#     pq = [(heuristic[start], 0, start, [])]  # (f, g, node, path)
#     visited = set()
#
#     while pq:
#         f, g, node, path = heapq.heappop(pq)
#
#         if node in visited:
#             continue
#         visited.add(node)
#
#         path = path + [node]
#
#         if node == goal:
#             return path, g
#
#         for neigh, w in graph[node].items():
#             if neigh not in visited:
#                 g2 = g + w
#                 f2 = g2 + heuristic[neigh]
#                 heapq.heappush(pq, (f2, g2, neigh, path))
#
#     return None, float("inf")
#
#
# # -----------------------------
# # MAIN PROGRAM
# # -----------------------------
# def main():
#     print("\n===============================")
#     print("   ROUTE FINDER (UCS / A*)")
#     print("===============================")
#
#     graph, nodes = get_graph_from_user()
#
#     print("\nAvailable nodes:", nodes)
#     start = input("Enter start node: ")
#     goal = input("Enter goal node: ")
#
#     print("\nChoose Algorithm:")
#     print("1. Uniform Cost Search (UCS)")
#     print("2. A* Search")
#     choice = input("Enter choice (1 or 2): ")
#
#     if choice == "1":
#         path, cost = uniform_cost_search(graph, start, goal)
#         print("\n=== UCS RESULT ===")
#         print("Path:", " -> ".join(path))
#         print("Total Cost:", cost)
#
#     elif choice == "2":
#         heuristic = get_heuristics_from_user(nodes)
#         path, cost = a_star(graph, heuristic, start, goal)
#         print("\n=== A* RESULT ===")
#         print("Path:", " -> ".join(path))
#         print("Total Cost:", cost)
#
#     else:
#         print("Invalid choice!")
# # Run program
# main()

import heapq
# -----------------------------
# 1. DATA PREPARATION
# -----------------------------

# The weighted graph based on Figure 1 [cite: 19]
# Direct connections only.
GRAPH = {
    'CMC': {'Megenagna': 125, 'Bole': 360},
    'Megenagna': {'CMC': 125, 'Gerji': 125, 'Kazanchis': 260, 'Aratkilo': 260, 'Hayahulet':125},
    'Gerji': {'Megenagna': 125, 'Bole': 260},
    'Bole': {'CMC': 360, 'Gerji': 260, 'Hayahulet': 260},
    'Kazanchis': {'Megenagna': 260, 'Aratkilo': 125, 'Hayahulet': 125},
    'Aratkilo': {'Megenagna': 260, 'Kazanchis': 125},
    'Hayahulet': {'Bole': 260, 'Kazanchis': 125, 'Megenagna':125}
}

# Heuristic Table from Page 2
# Format: HEURISTICS[GOAL_NODE][CURRENT_NODE]
# This allows the program to look up h(n) dynamically based on the goal.
HEURISTICS = {
    'CMC': {'CMC': 0, 'Megenagna': 125, 'Gerji': 250, 'Kazanchis': 385, 'Hayahulet': 510, 'Aratkilo': 510, 'Bole': 360},
    'Megenagna': {'CMC': 125, 'Megenagna': 0, 'Gerji': 125, 'Kazanchis': 260, 'Hayahulet': 385, 'Aratkilo': 385,
                  'Bole': 385},
    'Gerji': {'CMC': 250, 'Megenagna': 125, 'Gerji': 0, 'Kazanchis': 385, 'Hayahulet': 385, 'Aratkilo': 510,
              'Bole': 260},
    'Kazanchis': {'CMC': 385, 'Megenagna': 260, 'Gerji': 385, 'Kazanchis': 0, 'Hayahulet': 125, 'Aratkilo': 125,
                  'Bole': 385},
    'Hayahulet': {'CMC': 510, 'Megenagna': 385, 'Gerji': 385, 'Kazanchis': 125, 'Hayahulet': 0, 'Aratkilo': 250,
                  'Bole': 260},
    'Aratkilo': {'CMC': 510, 'Megenagna': 385, 'Gerji': 510, 'Kazanchis': 125, 'Hayahulet': 250, 'Aratkilo': 0,
                 'Bole': 510},
    'Bole': {'CMC': 360, 'Megenagna': 385, 'Gerji': 260, 'Kazanchis': 385, 'Hayahulet': 260, 'Aratkilo': 510, 'Bole': 0}
}


# -----------------------------
# 2. SEARCH ALGORITHMS
# -----------------------------

def uniform_cost_search(start, goal):
    # Priority Queue stores tuples: (cost, current_node, path_list)
    pq = [(0, start, [start])]
    visited = set()

    while pq:
        cost, node, path = heapq.heappop(pq)

        # Apply concept of closed list to avoid repeated visits [cite: 25]
        if node in visited:
            continue
        visited.add(node)

        if node == goal:
            return path, cost

        # Get neighbors
        neighbors = GRAPH.get(node, {})
        for neighbor, weight in neighbors.items():
            if neighbor not in visited:
                new_cost = cost + weight
                heapq.heappush(pq, (new_cost, neighbor, path + [neighbor]))

    return None, float("inf")


def a_star_search(start, goal):
    # Check if heuristics exist for this specific goal
    if goal not in HEURISTICS:
        print(f"Error: No heuristic data available for goal {goal}")
        return None, 0

    # Priority Queue stores: (f_score, g_score, current_node, path_list)
    # f(n) = g(n) + h(n)
    start_h = HEURISTICS[goal][start]
    pq = [(start_h, 0, start, [start])]
    visited = set()

    while pq:
        f, g, node, path = heapq.heappop(pq)

        if node in visited:
            continue
        visited.add(node)

        if node == goal:
            return path, g

        neighbors = GRAPH.get(node, {})
        for neighbor, weight in neighbors.items():
            if neighbor not in visited:
                g_new = g + weight
                h_new = HEURISTICS[goal].get(neighbor, float('inf'))  # Look up h(n) for specific goal
                f_new = g_new + h_new
                heapq.heappush(pq, (f_new, g_new, neighbor, path + [neighbor]))

    return None, float("inf")


# -----------------------------
# 3. MAIN INTERFACE
# -----------------------------
def main():
    print("\n=== ADDIS ABABA ROUTE OPTIMIZER ===")
    print("Available Locations:", list(GRAPH.keys()))

    start = input("\nEnter Start Location: ").strip()
    goal = input("Enter Goal Location: ").strip()

    # Validation
    if start not in GRAPH or goal not in GRAPH:
        print("Invalid locations! Please use the exact names listed above.")
        return

    print(f"\nFinding path from {start} to {goal}...")

    # --- Run UCS ---
    path_ucs, cost_ucs = uniform_cost_search(start, goal)
    print("\n1. Uniform Cost Search:")
    if path_ucs:
        print(f"   Path: {' -> '.join(path_ucs)}")
        print(f"   Cost: {cost_ucs}")
    else:
        print("   No path found.")

    # --- Run A* ---
    path_astar, cost_astar = a_star_search(start, goal)
    print("\n2. A* Search:")
    if path_astar:
        print(f"   Path: {' -> '.join(path_astar)}")
        print(f"   Cost: {cost_astar}")
    else:
        print("   No path found.")


if __name__ == "__main__":
    main()

