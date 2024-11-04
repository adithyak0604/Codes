import heapq

class Node:
    def __init__(self, position, g=0, h=0):
        self.position = position
        self.g = g  # Cost from start to this node
        self.h = h  # Heuristic cost to goal
        self.f = g + h  # Total cost
        self.parent = None
    
    def __lt__(self, other):
        return self.f < other.f
    
    @staticmethod
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(start, goal, grid):
    open_list = []
    closed_list = set()
    start_node = Node(start, 0, Node.heuristic(start, goal))
    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)
        closed_list.add(current_node.position)

        # If the goal is reached, reconstruct the path
        if current_node.position == goal:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]  # Return reversed path

        # Check neighbors
        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for new_position in neighbors:
            node_position = (current_node.position[0] + new_position[0], 
                             current_node.position[1] + new_position[1])

            # Make sure the node is within bounds
            if (0 <= node_position[0] < len(grid)) and (0 <= node_position[1] < len(grid[0])):
                # Make sure it's walkable and not in the closed list
                if grid[node_position[0]][node_position[1]] == 0 and node_position not in closed_list:
                    g = current_node.g + 1
                    h = Node.heuristic(node_position, goal)
                    neighbor_node = Node(node_position, g, h)
                    neighbor_node.parent = current_node

                    # Check if this path is better than a previous one
                    in_open_list = False
                    for open_node in open_list:
                        if open_node.position == neighbor_node.position and open_node.g <= g:
                            in_open_list = True
                            break

                    if not in_open_list:
                        heapq.heappush(open_list, neighbor_node)

    return None  # Return None if no path is found

# Example usage:
if __name__ == "__main__":
    grid = [
        [0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0]
    ]
    start = (0, 0)
    goal = (4, 4)
    path = a_star(start, goal, grid)
    
    if path:
        print("Path found:", path)
    else:
        print("No path found.")
