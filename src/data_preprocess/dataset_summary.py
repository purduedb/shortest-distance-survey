import os
import sys
import glob
import time
import random
from tqdm import tqdm

import numpy as np
import pandas as pd
import networkx as nx

import config
from data_utils import (
    seed_everything,
    read_figshare_data,
    graph_from_edgelist,
    preprocess_graph,
    save_graph,
    FIGSHARE_DATA_DIR,
    PROCESSED_DATA_DIR,
    bcolors,
)


# Set seed for reproducibility
seed_everything(42)

# Set directories
GRAPH_DATA_DIR = os.path.join(config.PROCESSED_DATA_DIR, 'dataset_summary', 'graphs')

def get_summary_stats(data_name):
    stats = {'Dataset Name': data_name}

    # Read the edgelist data for the specified dataset
    edgelist = read_figshare_data(data_name, dir_name=config.FIGSHARE_DATA_DIR)

    # Create a graph from the edgelist
    G = graph_from_edgelist(edgelist)

    # Add metadata to the graph
    G.graph['data_name'] = data_name

    # Preprocess the graph
    G_prime = preprocess_graph(G, remove_1_degree=True, remove_2_degree=True)

    # Save the graph
    save_graph(G_prime, data_name, dir_name=GRAPH_DATA_DIR)

    stats['Nodes'] = G_prime.number_of_nodes()
    stats['Edges'] = G_prime.number_of_edges()

    degrees = [d for n, d in G_prime.degree()]
    stats['Min Degree'] = min(degrees) if degrees else np.nan
    stats['Max Degree'] = max(degrees) if degrees else np.nan
    stats['Avg Degree'] = np.mean(degrees) if degrees else np.nan

    # Pathlengths
    n_runs = 100
    dijkstra_times = []
    distances = []
    start_nodes = random.sample(list(G_prime.nodes()), n_runs)
    for start_node in start_nodes:
        # Compute shortest paths from the start node
        start_d_time = time.perf_counter()
        lengths = dict(nx.single_source_dijkstra_path_length(G_prime, start_node, weight="LENGTH"))
        d_time = time.perf_counter() - start_d_time
        dijkstra_times.append(d_time)
        # Store the distances
        distances.extend(lengths.values())
    stats['Min Distance (m)'] = min(distances) if distances else np.nan
    stats['Max Distance (m)'] = max(distances) if distances else np.nan
    stats['Avg Distance (m)'] = np.mean(distances) if distances else np.nan
    stats['Dijkstra Time (s)'] = np.mean(dijkstra_times)

    return stats

if __name__ == '__main__':
    # Extract dataset names from FIGSHARE_DATA_DIR
    all_dataset_files = glob.glob(os.path.join(FIGSHARE_DATA_DIR, '*.zip'))
    datasets = [os.path.basename(f).split('.')[0] for f in all_dataset_files]
    print("No. of datasets found: ", len(datasets))

    all_stats = []
    for data_name in tqdm(datasets):
        try:
            stats = get_summary_stats(data_name)
            all_stats.append(stats)
        except Exception as e:
            print(f"{bcolors.FAIL}Error processing dataset {data_name}: {e}{bcolors.ENDC}")
            continue
    df = pd.DataFrame(all_stats)
    df = df.sort_values(by='Nodes', ascending=True)

    print("\n\n--- Graph Dataset Summary Statistics ---")
    print(df.to_string())

    # Save summary statistics to CSV
    file_name = os.path.join(PROCESSED_DATA_DIR, f"full_dataset_summary.csv")
    # print(f"Saving datasets summary: {file_name}")
    print(f"{bcolors.OKGREEN}Saving datasets summary: {file_name}{bcolors.ENDC}")
    df.to_csv(file_name, index=False)
