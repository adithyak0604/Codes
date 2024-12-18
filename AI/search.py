graph={ 'A':['B','C'], 'B':['D','E'], 'C':['F'], 'D':[], 'E':['F'], 'F':[] }
def dfs(visited, graph, node):
    if node not in visited:
        print(node,end=" ")
        visited.add(node)
        for neighbor in graph[node]:
            dfs(visited, graph, neighbor)

visited=set()
start_node='A'
print("DFS Traversal from node",start_node)
dfs(visited, graph, start_node)
print("\n")


from collections import deque
graph={ 1:[2,3], 2:[4,5], 3:[6], 4:[], 5:[], 6:[]}
def bfs(graph, start_node):
    queue=deque([start_node])
    visited=set([start_node])
    while queue:
        current_node=queue.popleft()
        print(current_node, end=" ")
        for neighbor in graph[current_node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
print("BFS Traversal from node")
bfs(graph, 1)
print("\n")


import heapq
def best(graph, start, goal, heuristic):
    pq=[]
    heapq.heappush(pq, (heuristic[start], start))
    visited=set()
    while pq:
        current_h, current_n=heapq.heappop(pq)
        if current_n==goal:
            print(f"Goal {goal} reached")
            return True
        if current_n not in visited:
            visited.add(current_n)
            print("Route: ",current_n)
            for neighbor in graph[current_n]:
                if neighbor not in visited:
                    heapq.heappush(pq, (heuristic[neighbor],neighbor))
    print(f"Goal {goal} is not reached")
    return False

graph={ 'A':['B','C','E'], 'B':['A','D','F'], 'C':['A','D'], 'D':['B','C','H'], 'E':['A','F'], 'F':['B','E','G'], 'G':['F','H'], 'H':['D','G']}
heuristic={ 'A':7, 'B':6, 'C':9, 'D':3, 'E':8, 'F':2, 'G':4, 'H':0 }
start_node='A'
goal_node='H'
best(graph, start_node, goal_node, heuristic)
print("\n")
