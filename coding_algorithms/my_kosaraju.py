from collections import defaultdict
class Graph:
	def __init__(self, node_num):
		self.node_num = node_num
		self.neighbors = defaultdict(set)

	def add_edge(self, start_node, end_node):
		self.neighbors[start_node].add(end_node)

	def get_reverse(self):
		new_graph = Graph(self.node_num)
		for start_node, neighbors in self.neighbors.items():
			for end_node in neighbors:
				new_graph.add_edge(end_node, start_node)
		return new_graph

def dfs1(node, g, visited, stack):
	visited[node] = True
	for neighbor in g.neighbors[node]:
		if visited[neighbor] is False:
			dfs1(neighbor, g, visited, stack)
	stack.append(node)

def dfs2(node, visited, reversed_g, cur_scc):
	print(node)
	cur_scc.append(node)
	visited[node] = True
	for neighbor in reversed_g.neighbors[node]:
		if visited[neighbor] is False:
			dfs2(neighbor, visited, reversed_g, cur_scc)
		

def get_scc(g):
	visited = [False] * g.node_num
	stack = []
	for node in range(g.node_num):
		if visited[node] is False:
			dfs1(node, g, visited, stack)
	print(stack)

	reversed_g = g.get_reverse()
	visited = [False] * g.node_num

	scc_groups = []

	while len(stack) > 0:
		node = stack.pop()
		if visited[node] is False:
			scc_group = []
			dfs2(node, visited, reversed_g, scc_group)
			scc_groups.append(scc_group)
	return scc_groups





g = Graph(5)
g.add_edge(1, 0)
g.add_edge(0, 2)
g.add_edge(2, 1)
g.add_edge(0, 3)
g.add_edge(3, 4)

print(get_scc(g))


