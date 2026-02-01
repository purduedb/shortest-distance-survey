import networkx as nx
import pickle

data = "Surat"
G = nx.Graph()
count = 0
cos = {}
filename = f"{data}_Edgelist.csv"
print(f"Reading file: {filename}")
with open(filename, "r") as f:
    f.readline()
    for line in f:
        x, y, n1, n2, _, w = line.split(',')
        if n1 not in cos:
            cos[n1] = (x, y)
        G.add_weighted_edges_from([(n1, n2, float(w))])
        count += 1
print(f"Original graph: {len(G.nodes())} nodes, {len(G.edges())} edges")

# find the largest connected component
print("Is connected: ", nx.is_connected(G))
G = G.subgraph(max(nx.connected_components(G), key=len))
print(f"Largest connected subgraph: {len(G.nodes())} nodes, {len(G.edges())} edges")

node_index = {}
for i, node in enumerate(G.nodes()):
    node_index[node] = i

nG = nx.Graph()
for edge in G.edges():
    n1, n2 = edge
    w = G[n1][n2]['weight']
    n1 = node_index[n1]
    n2 = node_index[n2]
    nG.add_weighted_edges_from([(n1, n2, w)])

print(f"Writing to file: {data}_nx")
nx.write_weighted_edgelist(nG, f"{data}_nx")


# store coordinate file
# ncos = {}
#
# for n in cos:
#     if n not in node_index:
#         continue
#     ncos[node_index[n]] = cos[n]
#
#
# pickle.dump(ncos, open("Surat_coos", "wb"))
