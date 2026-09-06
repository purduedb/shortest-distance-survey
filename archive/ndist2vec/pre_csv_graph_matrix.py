import networkx as nx
import pickle
import numpy as np
"""
Data preprocessing: 
reading graph_name.csv, generate file 
(1)Find the largest connected subgraph file: 
graph_name_nx_edges, 
(2)node coordinate file: 
graph_name_coos.txt
(3)shortest distance matrix: 
graph_name_shortest_distance_matrix.npy
"""
### Data name ###
graph_name='Surat'


### Find the largest connected subgraph: graph_name_nx_edges ###
G = nx.Graph()
coor = {}
with open("./data/%s_Edgelist.csv"%graph_name, "r") as f:
    f.readline()
    for line in f:
        x, y, n1, n2, _, w = line.split(',')
        if n1 not in coor:
            coor[n1] = (x, y)
        G.add_weighted_edges_from([(n1, n2, float(w))])
print(f"over 0, Original graph: {len(G.nodes())} nodes, {len(G.edges())} edges")

G = G.subgraph(max(nx.connected_components(G), key=len))
print(f"Largest connected subgraph: {len(G.nodes())} nodes, {len(G.edges())} edges")
node_index = {}
for i, node in enumerate(G.nodes()):
    node_index[node] = i
new_G=nx.Graph()
for edge in G.edges():
    n1, n2 = edge
    w = G[n1][n2]['weight']
    n1 = node_index[n1]
    n2 = node_index[n2]
    new_G.add_weighted_edges_from([(n1, n2, w)])
nx.write_weighted_edgelist(new_G, "./data/%s_nx.edges"%graph_name)
print('over1, edges file is saved as %s_nx.edges'%graph_name)

### node coordinate file: graph_name_coos.txt ###
new_coor = {}
for k in coor:
    if k not in node_index:
        continue
    new_coor[node_index[k]] = coor[k]
# print("sample keys and values of coos:", list(new_coor.items())[:5])
pickle.dump(new_coor, open("./data/%s_coos.txt"%graph_name, "wb"))
print('over2, coor file is saved as %s_coos.txt'%graph_name)

### shortest distance matrix: graph_name_shortest_distance_matrix.npy ###
n = len(new_G)
# print(n)
distance_matrix = np.zeros((n, n))
print("size of distance matrix:", distance_matrix.shape)
print("dtype of distance matrix:", distance_matrix.dtype, "(where float64 -> 8bytes)")
print("expected size of distance matrix:", distance_matrix.nbytes/(1024*1024), "MB")
for s in new_G.nodes():
    length = nx.single_source_dijkstra_path_length(new_G, s)
    for t in length:
        distance_matrix[int(s)][int(t)] = length[t]
np.save("./data/%s_shortest_distance_matrix.npy"%graph_name, distance_matrix)
print("max distance:", np.max(distance_matrix))
print("min distance:", np.min(distance_matrix))
print('over3, distance matrix file is saved as %s_shortest_distance_matrix.npy'%graph_name)