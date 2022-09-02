from collections import defaultdict

def get_graph_from_text_file(filename):
    graph = defaultdict(list)
    with open(filename) as f:
        for line_idx, line in enumerate(f.readlines()):
            if line_idx == 0:
                node_num, edge_num = line.split()
                node_num = int(node_num)
            else:
                source_node, end_node, weight = line.split()
                source_node, end_node, weight = int(source_node) - 1, int(end_node) - 1, int(weight)
                #graph[source_node].append([end_node, weight])
                graph[end_node].append([source_node, weight])
    return graph, node_num

def run_one_iteration_bellman_ford(graph, node_num, dp_prev):
    dp = []
    for node in range(node_num):
        res = dp_prev[node]
        for neighbor, weight in graph[node]:
            res = min(res, dp_prev[neighbor] + weight)
        dp.append(res)
    

    return dp


def run_bellman_ford(graph, node_num, start_node):
    dp_prev = [float("inf")] * node_num
    dp_prev[start_node] = 0

    for i in range(node_num - 1):
        dp_prev = run_one_iteration_bellman_ford(graph, node_num, dp_prev)

    shortest_path_length = float("inf")
    for node in range(node_num):
        if node == start_node: continue
        shortest_path_length = min(dp_prev[node], shortest_path_length)

    # Check negative cycle
    dp_prev = run_one_iteration_bellman_ford(graph, node_num, dp_prev)
    for end_node in range(node_num):
        for source_node, weight in graph[end_node]:
            if dp_prev[source_node] + weight < dp_prev[end_node]:
                # negative cycle
                return None

    return shortest_path_length


def main():
    graph, node_num = get_graph_from_text_file("../data/test_g.txt")
    #print(run_bellman_ford(graph, node_num, 0))

    shortest_shortest_path_length = float("inf")
    for start_node in range(node_num):
        res = run_bellman_ford(graph, node_num, start_node)
        if res is None:
            print("Negative cycle detected")
        else:
            shortest_shortest_path_length = min(res, shortest_shortest_path_length)
    print(shortest_shortest_path_length)

main()



