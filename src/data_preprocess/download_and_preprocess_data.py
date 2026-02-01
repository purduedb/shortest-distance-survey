# Imports
import os

from utils.data_utils import (
    read_figshare_data,
    download_figshare_data,
    read_dimacs_data,
    download_dimacs_data,
    preprocess_graph,
    save_graph,
    load_graph,
    get_node_attributes,
    get_edge_attributes,
    print_summary_stats,
)


# Uncomment or comment the datasets you want to use
FIGSHARE_DATASETS = [
    ## Small (<10k nodes) ##
    "Surat",  # 1843
    "Quanzhou",  # 3079
    "Dongguan",  # 4606
    "Fuzhou",  # 5955
    "Harbin",  # 6328
    "Qingdao",  # 7992
    "Dhaka",  # 8349
    "Ahmedabad",  # 8808
    "Shenyang",  # 9020
    "Chongqing",  # 9519

    ## Medium (10k-100k nodes) ##
    "Pune",  # 14542
    "Shenzhen",  # 21809
    "Mumbai",  # 25668
    "Delhi",  # 37971
    "Beijing",  # 45201
    "Jacarta",  # 50215
    "Naples",  # 63070
    "Hyderabad",  # 74357
    "Rome",  # 87739
    "Istambul",  # 98573

    ## Large (100k-1M nodes) ##
    "Atlanta",  # 113028
    "Milan",  # 119638
    "London",  # 150598
    "Boston",  # 181666
    "NewYork",  # 190368
    "Phoenix",  # 200519
    "LosAngeles",  # 228185
    "Chicago",  # 259887
    "Paris",  # 281031
    "Moscow",  # 406505
]

# Uncomment the datasets you want to use
DIMACS_DATASETS = [
    "FLA",  # 1,070,376
    "E",  # 3,598,623
]

# Default data directory to save the raw and processed data
DATA_DIR = './../data'

for i, dataset in enumerate(FIGSHARE_DATASETS):
    print(f"[{i}/{len(FIGSHARE_DATASETS)}] Processing dataset: {dataset}")
    zip_file_name = download_figshare_data(DATA_DIR, dataset)
    print(f"Zip file: {zip_file_name}")
    G = read_figshare_data(zip_file_name)

    # Preprocess the graph
    G = preprocess_graph(G)

    # Print node and edge attributes
    print("Node attributes:")
    print(get_node_attributes(G)[:5])
    print("Edge attributes:")
    print(get_edge_attributes(G)[:5])

    # Save the graph
    save_graph(G, os.path.join(DATA_DIR, dataset))

for i, dataset in enumerate(DIMACS_DATASETS):
    print(f"[{i}/{len(DIMACS_DATASETS)}] Processing dataset: {dataset}")
    graph_file_name, coordinates_file_name = download_dimacs_data(DATA_DIR, dataset)
    print(f"Graph file: {graph_file_name}")
    print(f"Coordinates file: {coordinates_file_name}")
    G = read_dimacs_data(graph_file_name, coordinates_file_name)

    # Preprocess the graph
    G = preprocess_graph(G)

    # Print node and edge attributes
    print("Node attributes:")
    print(get_node_attributes(G)[:5])
    print("Edge attributes:")
    print(get_edge_attributes(G)[:5])

    # Save the graph
    save_graph(G, os.path.join(DATA_DIR, dataset))
