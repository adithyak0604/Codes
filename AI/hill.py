class Node:
    def __init__(self,name,value):
        self.name=name
        self.value=value
        self.neighbors=[]
    
def hill_climbing(start_node, goal_node):
    current_node=start_node
    path=[current_node.name]
    while True:
        if current_node==goal_node:
            return path
        if not current_node.neighbors:
            break
        next_node=max(current_node.neighbors, key=lambda node: node.value)
        if next_node.value <= current_node.value:
            break
        current_node=next_node
        path.append(current_node.name)
    return None

if __name__=="__main__":
    node_a=Node('A',1)
    node_b=Node('B',3)
    node_c=Node('C',2)
    node_d=Node('D',4)
    node_e=Node('E',0)

    node_a.neighbors=[node_b, node_c]
    node_b.neighbors=[node_d, node_e]
    node_c.neighbors=[node_d]
    node_d.neighbors=[node_e]
    node_e.neighbors=[]

    goal_node=node_d
    result=hill_climbing(node_a, goal_node)
    if result:
        print("Goal Node Reached ",goal_node.name,"with value ",goal_node.value)
        print("Path Taken ->",result)
    else:
        print("Goal Node not reachable from the starting node\n")