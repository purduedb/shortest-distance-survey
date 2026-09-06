import networkx as nx
import numpy as np

data = "Surat"
G = nx.read_weighted_edgelist(f"./{data}_nx")
n = len(G)
print(n)

distance_matrix = np.zeros((n, n))
print("size of distance matrix:", distance_matrix.shape)
print("dtype of distance matrix:", distance_matrix.dtype, "(where float64 -> 8bytes)")
print("expected size of distance matrix:", distance_matrix.nbytes/(1024*1024), "MB")

# vertex index
vertex_index = {}

i = 0
for v in G.nodes():
    vertex_index[v] = i
    i += 1

for s in G.nodes():
    length = nx.single_source_dijkstra_path_length(G, s)
    s_i = vertex_index[s]
    for t in length:
        t_i = vertex_index[t]
        distance_matrix[s_i][t_i] = length[t]

# print(distance_matrix)
filename = f"{data}_shortest_distance_matrix.npy"
print("Saving distance matrix to file: ", filename)
np.save(filename, distance_matrix)
