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

G = G.subgraph(max(nx.connected_components(G), key=len))
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
print('over1')
### node coordinate file: graph_name_coor.txt ###
new_coor = {}
for k in coor:
    if k not in node_index:
        continue
    new_coor[node_index[k]] = coor[k]
pickle.dump(new_coor, open("./data/%s_coor.txt"%graph_name, "wb"))
print('over2')
### shortest distance matrix: graph_name_shortest_distance_matrix.npy ###
n = len(new_G)
# print(n)
distance_matrix = np.zeros((n, n))
for s in new_G.nodes():
    length = nx.single_source_dijkstra_path_length(new_G, s)
    for t in length:
        distance_matrix[int(s)][int(t)] = length[t]
np.save("./data/%s_shortest_distance_matrix.npy"%graph_name, distance_matrix)
print('over3')