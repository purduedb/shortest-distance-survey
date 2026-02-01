# New Version
* Use `myenv` to run code.
* Step 1: python pre_csv_graph_matrix.py
    * Input: Surat_Edgelist.csv
    * Output: Surat_nx.edges, Surat_coos.txt and Surat_shortest_distance_matrix.npy
* Step 2: (optional) python read_coor.py
    * pprints the cooridnates of the nodes
* Step 3: python ndist2vec.py
    * trains the model and prints the results


# Old Version
# ndist2vec
## Other requirments
pip install -r requirements.txt
## Generate data
run <br>
pre_csv_graph_matrix > name_nx.edges、name_coos.txt and name_shortest_distance_matrix.npy <br>
## Train model
run <br>
ndist2vec.py
