from sys import maxsize
from itertools import permutations
v=4

def tsp(graph,s):
    vertex=[]
    for i in range(v):
        if i!=s:
            vertex.append(i)
    min_path=maxsize
    best_path=[]
    next_permutation=permutations(vertex)
    for i in next_permutation:
        current_pathweight=0
        k=s
        path=[s]
        for j in i:
            current_pathweight+=graph[k][j]
            path.append(j)
            k=j
        current_pathweight+=graph[k][s]
        path.append(s)
        if current_pathweight < min_path:
            min_path = current_pathweight
            best_path = path[:]
    return min_path,best_path

if __name__=="__main__":
    graph=[[0,10,15,20], [10,0,35,25], [15,35,0,30], [20,25,30,0]]
    s=0
min_cost,best_path=tsp(graph,s)
print("Minimum cost of travelling: ",min_cost)
print("Best Path: ",best_path)