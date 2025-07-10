import heapq
import numpy as np

def dijkstra(graph, start, end):
    """
    Efficient Dijkstra's algorithm implementation using numpy matrix.

    Args:
        graph: 2D numpy array where graph[i][j] represents the weight of edge from i to j.
               Use np.inf for no connection between nodes.
        start: Starting node index
        end: Ending node index

    Returns:
        tuple: (shortest_distance, shortest_path)
               shortest_distance: float - minimum distance from start to end
               shortest_path: list - sequence of nodes in the shortest path
    """
    n = graph.shape[0]

    # Validate inputs
    if start < 0 or start >= n or end < 0 or end >= n:
        raise ValueError("Start or end node index out of bounds")

    # Initialize distances and previous nodes
    distances = np.full(n, np.inf)
    distances[start] = 0
    previous = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)

    # Priority queue: (distance, node)
    pq = [(0, start)]

    while pq:
        current_dist, u = heapq.heappop(pq)

        # Skip if we've already found a shorter path to this node
        if visited[u]:
            continue

        # Mark as visited
        visited[u] = True

        # Early termination if we reached the target
        if u == end:
            break

        # Check all neighbors
        for v in range(n):
            # Skip if no edge exists or already visited
            if graph[u, v] == np.inf or visited[v]:
                continue

            # Calculate new distance
            new_dist = current_dist + graph[u, v]

            # Update if we found a shorter path
            if new_dist < distances[v]:
                distances[v] = new_dist
                previous[v] = u
                heapq.heappush(pq, (new_dist, v))

    # Reconstruct path
    path = []
    if distances[end] != np.inf:
        current = end
        while current != -1:
            path.append(current)
            current = previous[current]
        path.reverse()

    return path

def dijkstra_all_pairs(graph, start):
    """
    Find shortest paths from start node to all other nodes.

    Args:
        graph: 2D numpy array representing the adjacency matrix
        start: Starting node index

    Returns:
        tuple: (distances, paths)
               distances: numpy array of shortest distances to each node
               paths: dict mapping each reachable node to its shortest path from start
    """
    n = graph.shape[0]

    if start < 0 or start >= n:
        raise ValueError("Start node index out of bounds")

    distances = np.full(n, np.inf)
    distances[start] = 0
    previous = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)

    pq = [(0, start)]

    while pq:
        current_dist, u = heapq.heappop(pq)

        if visited[u]:
            continue

        visited[u] = True

        for v in range(n):
            if graph[u, v] == np.inf or visited[v]:
                continue

            new_dist = current_dist + graph[u, v]

            if new_dist < distances[v]:
                distances[v] = new_dist
                previous[v] = u
                heapq.heappush(pq, (new_dist, v))

    # Reconstruct all paths
    paths = {}
    for end in range(n):
        if distances[end] != np.inf:
            path = []
            current = end
            while current != -1:
                path.append(current)
                current = previous[current]
            path.reverse()
            paths[end] = path

    return distances, paths