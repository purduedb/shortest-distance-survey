# import libraries
import os
import sys
import json
import csv
import gzip
import glob
import random
import time
import requests
from tqdm import tqdm
from zipfile import ZipFile

import numpy as np
import pandas as pd
import igraph as ig
import networkx as nx

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min


#################################################
# General Utilities
#################################################

IS_TTY = sys.stdout.isatty()

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

## Function to print text in green color
def print_green(text):
    """Print text in green color."""
    text = f"{bcolors.OKGREEN}{text}{bcolors.ENDC}" if IS_TTY else text
    print(text)

## Function to print text in yellow color
def print_warning(text):
    """Print warning text in yellow color."""
    text = f"{bcolors.WARNING}{text}{bcolors.ENDC}" if IS_TTY else text
    print(text)

## Function to print text in red color
def print_error(text):
    """Print error text in red color."""
    text = f"{bcolors.FAIL}{text}{bcolors.ENDC}" if IS_TTY else text
    print(text)

## Function to print the percentile distribution of MAE/MRE side by side
def print_error_distribution(abs_errors, rel_errors):
    """Print distribution of MAE and MRE."""
    percentiles = [0, 25, 50, 75, 80, 90, 95, 99, 100]
    labels = ["min", "p25", "p50", "p75", "p80", "p90", "p95", "p99", "max"]
    ae_stats = np.percentile(abs_errors, percentiles)
    re_stats = np.percentile(rel_errors, percentiles) * 100
    col_width = max(10, max(len(f"{v:.2f}") for v in np.concatenate([ae_stats, re_stats])) + 2)

    print("Distribution of MAE and MRE:")
    print(" " * 10 + "".join(f"{label:>{col_width}}" for label in labels))
    print(f"  {'MAE(m)':<8}" + "".join(f"{v:{col_width}.2f}" for v in ae_stats))
    print(f"  {'MRE(%)':<8}" + "".join(f"{v:{col_width}.2f}" for v in re_stats))

## Function to detect the delimiter in CSV files
def detect_delimiter(sample_data):
    """
    Detect the delimiter used in a CSV formatted string.
    """
    try:
        dialect = csv.Sniffer().sniff(sample_data)
        print(f"Detected delimiter: `{dialect.delimiter}`")
        return dialect.delimiter
    except:
        raise ValueError("Could not detect delimiter.")

def detect_delimiter_file(file_path):
    # Detect delimiter by reading first 128 lines
    with open(file_path, 'r') as f:
        sample_data = ""
        for i, line in enumerate(f):
            if i < 128:
                sample_data += line
            else:
                break
    delimiter = detect_delimiter(sample_data)
    return delimiter

## Function to set seeds for reproducibility
def seed_everything(seed):
    """Set seeds for reproducibility.

    Args:
        seed (int): Seed value for random number generators
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Set seeds for NumPy if available
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass  # Skip if NumPy is not available

    # Set seeds for PyTorch if available
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # Skip if PyTorch is not available

    print(f"Seed set to {seed}")

## Function to get the number of available CPU cores (for thread/process counts)
def get_num_cores():
    # Get number of cores available to this process
    if hasattr(os, 'sched_getaffinity'):
        # For SLURM jobs, use sched_getaffinity to get the number of available cores
        num_cores = len(os.sched_getaffinity(0))
    else:
        # Fallback to total CPU count
        num_cores = os.cpu_count()
    # NOTE: sometimes increasing the number of workers may decrease performance, so be careful!!
    # num_cores = num_cores // 2
    return num_cores

def read_parts_file(dir_name, data_name):
    file_path = os.path.join(dir_name, f'{data_name}.parts.npz')

    # Check if parts file exists
    if not os.path.exists(file_path):
        print_green(f"Parts file not found: {file_path}. Proceeding without parts information.")
        return None

    # Read parts file
    print_green(f"Reading parts file: {file_path}")
    parts = np.load(file_path)['data']

    # (Optional) Print summary statistics of parts
    print(f"Parts shape: {parts.shape}")
    for i in range(parts.shape[1]):
        print(f"Parts column {i}: min={parts[:, i].min()}, max={parts[:, i].max()}, unique={len(np.unique(parts[:, i]))}")

    return parts

def read_embedding_file(file_name):
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"Embedding file `{file_name}` not found.")
    print_green(f"Reading embeddings: {file_name}")
    data = np.load(file_name, allow_pickle=False)['data']  # (n_nodes, 1 + n_landmarks): col 0 = node_id (1-indexed), rest = distances

    # Get node_ids from the first column
    node_ids = data[:, 0]

    # Check if node_ids are unique
    if len(node_ids) != len(np.unique(node_ids)):
        raise ValueError("Node IDs in the embedding file are not unique.")

    # Check if node_ids are dense (i.e., 1 to n_nodes)
    if node_ids.min() != 1 or node_ids.max() != len(node_ids):
        raise ValueError("Node IDs in the embedding file are not dense (1 to n_nodes).")

    # Convert node ids to 0-indexed
    print_warning("Warning: Converting node ids from 1-indexed (on disk) to 0-indexed (in memory).")
    node_ids -= 1

    # Sort data by node_ids if not sorted
    if not np.all(node_ids[:-1] <= node_ids[1:]):
        sort_idx = np.argsort(node_ids)
        data = data[sort_idx]

    # Exclude the first column (node ids) and cast to float32
    data = data[:, 1:].astype(np.float32)

    # (Optional) Print shape of embeddings
    print(f"  - Embeddings shape: {data.shape}")

    return data

def write_query_file(file_name, queries):
    print_green(f"Saving dataset: {file_name}")
    print_warning("Warning: Converting node ids from 0-indexed (in memory) to 1-indexed (on disk).")
    data = np.array(queries, copy=True)
    src = (data[:, 0] + 1).astype(np.int64)  # Convert to 1-indexed
    dst = (data[:, 1] + 1).astype(np.int64)  # Convert to 1-indexed
    dist = data[:, 2].astype(np.float32)
    np.savez_compressed(file_name, src=src, dst=dst, dist=dist)

def read_query_file(file_name, drop_duplicates=False):
    print_green(f"Reading dataset: {file_name}")
    print_warning("Warning: Converting node ids from 1-indexed (on disk) to 0-indexed (in memory).")
    raw = np.load(file_name)
    src = raw['src'].astype(np.int64) - 1  # Convert to 0-indexed
    dst = raw['dst'].astype(np.int64) - 1  # Convert to 0-indexed
    dist = raw['dist'].astype(np.float32)
    file_data = []
    for idx, (u, v, distance) in enumerate(zip(src.tolist(), dst.tolist(), dist.tolist())):
        # Exclude zero distance entries because they cause division by zero error in MRE calculation
        if distance == 0:
            print_warning(f"Warning: Zero distance found in query file (Row {idx + 1}). Skipping entry.")
            continue
        file_data.append((u, v, distance))
    print(f"  - Total number of queries: {len(file_data):,}")
    # Exclude duplicate queries
    if drop_duplicates:
        print("  - Removing duplicate queries...")
        file_data = list(set(file_data))  # Remove exact duplicates
        print(f"  - Number of queries after removing duplicates: {len(file_data)}")
    return file_data

def download_file(file_name, url, referer, max_retries=10):
    """
    Downloads a file from the given URL and saves it to the specified file name using requests.
    Retries up to max_retries times if an error is encountered.
    """
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": referer
    }
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Downloading {url} (Attempt {attempt})")
            response = requests.get(url, headers=headers, stream=True, timeout=60)
            response.raise_for_status()

            total = int(response.headers.get('content-length', 0))
            with open(file_name, 'wb') as f:
                with tqdm(total=total, unit='B', unit_scale=True, desc=os.path.basename(file_name)) as bar:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                        bar.update(len(chunk))
            print(f"Successfully downloaded: {file_name}")
            return True
        except Exception as e:
            print_warning(f"Attempt {attempt} failed: {e}")
            if attempt < max_retries + 1:
                wait_time = random.randint(1, 10)
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print_warning(f"Failed to download file from {url} after {max_retries} attempts.")
                sys.exit(1)

## Function to read workload data from files
def read_workload_data(data_name, dir_name, convert_to_int=True):
    # Define file paths
    node_file = os.path.join(dir_name, f"Workload_{data_name}", f"{data_name}.node")
    edge_file = os.path.join(dir_name, f"Workload_{data_name}", f"{data_name}.edgelist")
    groundtruth_file = os.path.join(dir_name, f"Workload_{data_name}", f"{data_name}.groundtruth")

    # Read edge data
    print_green(f"Reading edge data: {edge_file}")
    edgelist = pd.read_csv(edge_file, header=None, names=["START_NODE", "END_NODE", "LENGTH"]) \
                    .drop_duplicates(subset=['START_NODE', 'END_NODE']) \
                    .sort_values(by=['START_NODE', 'END_NODE', 'LENGTH']) \
                    .reset_index(drop=True)
    print(f"Number of edges found: {len(edgelist)}")

    # Display the first few rows of the edgelist
    print(edgelist.head())

    # Create a graph from the edge list
    G = nx.from_pandas_edgelist(edgelist,
                                source="START_NODE",
                                target="END_NODE",
                                edge_attr="LENGTH",
                                create_using=nx.Graph())  # nx.Graph() creates an undirected graph

    # Convert edge weight (LENGTH) to int type for a fair comparison with non-ML indexes
    # (Since non-ML indexes can't process floats)
    if convert_to_int:
        print_warning("Warning: Converting edge lengths to integers.")
        print("Before conversion:")
        print(edgelist.head())
        edgelist['LENGTH'] = edgelist['LENGTH'].astype(int)
        print("After conversion:")
        print(edgelist.head())

    # Check if the graph is undirected
    # TODO: removing this check for now, as we assume the graph is undirected
    # assert _has_reverse_edges(edgelist) == False, "Edgelist is has some reverse edges; We assume the graph is undirected"
    # print("Finished checking: graph is undirected")

    # Check if the graph has self loops
    assert _has_self_loops(edgelist) == False, "Edgelist has self loops; We assume the graph has no self loops"
    print("Finished checking: graph has no self loops")

    # Add metadata to the graph
    G.graph['data_name'] = data_name

    # Read node data
    print_green(f"Reading node data: {node_file}")
    node_attr = pd.read_csv(node_file, header=None, names=["START_NODE", "XCoord", "YCoord"]) \
                    .drop_duplicates(subset=["START_NODE"]) \
                    .set_index("START_NODE") \
                    .to_dict('index')
    print(f"Number of nodes attributes found: {len(node_attr)}")

    # Set node attributes in the graph
    nx.set_node_attributes(G, node_attr)

    # Read ground truth queries
    print_green(f"Reading queries data: {groundtruth_file}")
    groundtruth_df = pd.read_csv(groundtruth_file, header=None, names=["START_NODE", "END_NODE", "distance"])
    print(f"Number of ground truth queries found: {len(groundtruth_df)}")

    return G, groundtruth_df

#################################################
# DIMACS Data I/O Utilities
#################################################

## Function to download data from DIMACS
def download_dimacs_data(dir_name, data_name):
    """
    Downloads data from DIMACS to the specified directory and returns the path to the downloaded zip file.
    """
    # Create the directory if it doesn't exist
    dir_name = os.path.join(dir_name, data_name, "raw")
    os.makedirs(dir_name, exist_ok=True)

    # Load the URLs from the JSON file
    all_urls = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dimacs_urls.json'), 'r'))
    referer = "https://www.diag.uniroma1.it/challenge9/download.shtml"
    graph_file_name = os.path.join(dir_name, f"USA-road-d.{data_name}.gr.gz")
    coordinates_file_name = os.path.join(dir_name, f"USA-road-d.{data_name}.co.gz")

    # Download the graph file
    url = all_urls.get(data_name).get("graph_url")
    download_file(graph_file_name, url, referer)

    # Download the coordinates file
    url = all_urls.get(data_name).get("coordinates_url")
    download_file(coordinates_file_name, url, referer)

    return graph_file_name, coordinates_file_name

## Fast DIMACS preprocessor
def fast_preprocess_dimacs(graph_file_name, coordinates_file_name, out_dir, delimiter=',', comment='#'):
    # WGS84 -> EPSG:5070 (meters) for correct Manhattan distance computation
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)

    data_name = os.path.basename(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    edge_out = os.path.join(out_dir, f"{data_name}.edges")
    node_out = os.path.join(out_dir, f"{data_name}.nodes")

    # Read header to get declared counts for tqdm totals
    n_nodes, n_edges = 0, 0
    with gzip.open(graph_file_name, 'rt') as f:
        for line in f:
            if line.startswith('p'):
                _, _, n_nodes, n_edges = line.split()
                n_nodes, n_edges = int(n_nodes), int(n_edges)
                break

    ## Processing edges
    print_green(f"Reading edges: {graph_file_name}")
    print_green(f"Saving edges: {edge_out}")
    print(f"Format: source{delimiter}target{delimiter}weight")
    us, vs, ws = [], [], []
    with gzip.open(graph_file_name, 'rt') as fin:
        for line in tqdm(fin, total=n_edges, desc="edges", unit="edge"):
            if line.startswith('a'):
                _, u, v, w = line.split()
                us.append(int(u))
                vs.append(int(v))
                ws.append(round(int(w) / 10))  # Convert decimeters to meters

    # Some DIMACS .gr files contain exact duplicate arcs (same src,dst,weight); drop repeats
    edges_df = pd.DataFrame({'u': us, 'v': vs, 'w': ws})
    n_total = len(edges_df)
    edges_df = edges_df.drop_duplicates(subset=['u', 'v', 'w'], keep='first')
    n_dropped = n_total - len(edges_df)
    if n_dropped > 0:
        print_warning(f"Dropped {n_dropped} duplicate edge rows (exact src,dst,weight repeats) out of {n_total}.")
    edges_df.to_csv(edge_out, sep=delimiter, header=False, index=False)

    ## Processing nodes
    print_green(f"Reading nodes: {coordinates_file_name}")
    print_green(f"Saving nodes: {node_out}")
    print(f"Format: node_id{delimiter}features")
    nids, lons, lats = [], [], []
    with gzip.open(coordinates_file_name, 'rt') as fin:
        for line in tqdm(fin, total=n_nodes, desc="nodes", unit="node"):
            if line.startswith('v'):
                _, nid, lon, lat = line.split()
                nids.append(int(nid))
                lons.append(float(lon) / 1e6)  # Convert micro-degrees to degrees
                lats.append(float(lat) / 1e6)  # Convert micro-degrees to degrees

    # Vectorized coordinate transform (WGS84 lon/lat degrees → EPSG:5070 meters)
    xs, ys = transformer.transform(np.array(lons), np.array(lats))
    nodes_df = pd.DataFrame({'node_id': nids, 'x': xs, 'y': ys})
    nodes_df.to_csv(node_out, sep=delimiter, header=False, index=False)

    print("Fast DIMACS preprocessing complete.")

## Function to read dimacs data from files
def read_dimacs_data(edge_file_name, node_file_name):
    """
    Reads DIMACS .co and .gr files and returns a graph and edge DataFrame.
    """
    # Get data_name from the edge/node file name, e.g., "USA-road-d.NY.gr.gz" --> "NY"
    data_name = os.path.basename(edge_file_name).split('.')[2]

    # Read edge data
    print_green(f"Reading edge data: {edge_file_name}")
    edges = []
    with gzip.open(edge_file_name, 'rt') as f:
        for line in f:
            if line.startswith('a'):
                _, source, target, weight = line.strip().split()
                edges.append((int(source), int(target), int(weight)))  # Convert to int
    edgelist = pd.DataFrame(edges, columns=['START_NODE', 'END_NODE', 'weight'])
    print(f"Number of edges found: {len(edgelist)}")
    print(edgelist.dtypes)

    # Display the first few rows of the edgelist
    print(edgelist.head())

    # Create a graph from the edge list
    G = nx.from_pandas_edgelist(edgelist,
                                source="START_NODE",
                                target="END_NODE",
                                edge_attr="weight",
                                create_using=nx.Graph())  # nx.Graph() creates an undirected graph

    # Read node data
    print_green(f"Reading node data: {node_file_name}")
    node_attr = {}
    with gzip.open(node_file_name, 'rt') as f:
        for line in f:
            if line.startswith('v'):
                _, node_id, longitude, latitude = line.strip().split()
                node_attr[int(node_id)] = {'XCoord': float(longitude), 'YCoord': float(latitude)}
    print(f"Number of nodes attributes found: {len(node_attr)}")

    # Set node attributes in the graph
    for node in node_attr:
        data = node_attr[node]
        G.nodes[node]['feature'] = [data['XCoord'], data['YCoord']]

    # Add metadata to the graph
    G.graph['data_name'] = data_name

    return G

#################################################
# Figshare Data I/O Utilities
#################################################

## Function to download data from Figshare
def download_figshare_data(dir_name, data_name):
    """
    Downloads data from Figshare to the specified directory and returns the path to the downloaded zip file.
    """
    # Create the directory if it doesn't exist
    dir_name = os.path.join(dir_name, data_name, "raw")
    os.makedirs(dir_name, exist_ok=True)

    # Load the URLs from the JSON file
    all_urls = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figshare_urls.json'), 'r'))
    zip_file_name = os.path.join(dir_name, f"{data_name}.zip")

    # Download the zip file
    url = all_urls.get(f"{data_name}.zip")
    referer = "https://figshare.com/"
    download_file(zip_file_name, url, referer)

    return zip_file_name

## Function to read the Figshare data
def read_figshare_data(zip_file_name, convert_to_int=True):
    # Define the path to the zip file and the CSV file inside it
    data_name = os.path.basename(zip_file_name).replace('.zip', '')
    csv_file_name = f"{data_name}_Edgelist.csv"
    print_green(f"Reading data: {zip_file_name}")

    # Handle discrepancy in the data name for "Shanghai" and "Guangzhou"
    if data_name == "Shanghai":
        csv_file_name = "Shangai_Edgelist.csv"
    elif data_name == "Guangzhou":
        csv_file_name = "Guangzhou_dgelist.csv"

    # Read the zip file and extract the CSV file
    with ZipFile(zip_file_name, 'r') as zip:
        # Remove duplicates edges and keep the shortest edge
        edgelist = pd.read_csv(zip.open(csv_file_name)) \
                    .drop_duplicates(subset=['START_NODE', 'END_NODE']) \
                    .rename(columns={"LENGTH": "weight"}) \
                    .sort_values(by=['START_NODE', 'END_NODE', 'weight']) \
                    .reset_index(drop=True)
    print(f"Finished reading data: {data_name}")

    # Convert edge weight to int type for a fair comparison with non-ML indexes
    # (Since non-ML indexes can't process floats)
    if convert_to_int:
        print_warning("Warning: Converting edge weights to integers.")
        print("Before conversion:")
        print(edgelist.head())
        edgelist['weight'] = edgelist['weight'].astype(int)
        print("After conversion:")
        print(edgelist.head())

    # Set 'START_NODE' and 'END_NODE' as source and target nodes
    # Set 'weight' as edge attribute
    G = nx.from_pandas_edgelist(edgelist,
                            source='START_NODE',
                            target='END_NODE',
                            edge_attr='weight',
                            create_using=nx.Graph)  # nx.Graph() creates an undirected graph

    # Create a dictionary of node attributes (XCoord and YCoord) from the edgelist
    node_attr = edgelist[['START_NODE', 'XCoord', 'YCoord']] \
                    .drop_duplicates() \
                    .set_index('START_NODE') \
                    .to_dict('index')

    # Set node attributes in the graph
    for node in node_attr:
        data = node_attr[node]
        G.nodes[node]['feature'] = [data['XCoord'], data['YCoord']]

    # Add metadata to the graph
    G.graph['data_name'] = data_name

    return G

#################################################
# Graph I/O Utilities
#################################################

## Function to save graph data to files
def save_graph(G, dir_name, delimiter=',', comment='#'):
    """
    Saves the graph data to .edges and .nodes files in the specified directory.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(dir_name, exist_ok=True)

    # Get data_name
    data_name = os.path.basename(os.path.normpath(dir_name))

    # Save the edge list
    edge_list_path = os.path.join(dir_name, f"{data_name}.edges")
    print_green(f"Saving edges: {edge_list_path}")
    print_warning("Warning: Converting node ids from 0-indexed (in memory) to 1-indexed (on disk).")
    print(f"Format: source{delimiter}target{delimiter}weight")
    with open(edge_list_path, 'w') as f:
        # NOTE: .to_directed() ensures that we save both (u, v) and (v, u)
        for u, v, data in G.to_directed().edges(data=True):
            f.write(f"{u+1}{delimiter}{v+1}{delimiter}{data['weight']}\n")  # Convert to 1-indexed

    # Save the node attributes
    node_attr_path = os.path.join(dir_name, f"{data_name}.nodes")
    print_green(f"Saving nodes: {node_attr_path}")
    print_warning("Warning: Converting node ids from 0-indexed (in memory) to 1-indexed (on disk).")
    print(f"Format: node_id{delimiter}features")
    with open(node_attr_path, 'w') as f:
        for node, data in G.nodes(data=True):
            f.write(f"{node+1}{delimiter}{delimiter.join(map(str, data['feature']))}\n")  # Convert to 1-indexed

## Function to load graph data from files
def load_graph(dir_name, delimiter=None, comment='#'):
    """
    Loads the graph data from .edges and .nodes files using igraph (faster than NetworkX for large graphs).
    """
    # Create a graph from the edge list
    edge_list_files = glob.glob(os.path.join(dir_name, "*.edges"))
    assert len(edge_list_files) > 0, f"Expected an `.edges` file in {dir_name}, found none."
    edge_list_path = edge_list_files[0]
    print_green(f"Reading edges: {edge_list_path}")
    print_warning("Warning: Converting node ids from 1-indexed (on disk) to 0-indexed (in memory).")
    auto_delimiter = delimiter or detect_delimiter_file(edge_list_path)
    edges_df = pd.read_csv(edge_list_path, sep=auto_delimiter, header=None, comment=comment, names=['u', 'v', 'weight'])
    edges_df['u'] -= 1  # Convert to 0-indexed
    edges_df['v'] -= 1

    n_nodes = int(max(edges_df['u'].max(), edges_df['v'].max())) + 1
    G = ig.Graph(n=n_nodes,
                 edges=edges_df[['u', 'v']].values.tolist(),
                 edge_attrs={'weight': edges_df['weight'].values.tolist()},
                 directed=False)
    G.simplify(combine_edges='first')  # Deduplicate edges if multiple edges exist

    # Add node attributes
    node_attr_files = glob.glob(os.path.join(dir_name, "*.nodes"))
    assert len(node_attr_files) > 0, f"Expected a `.nodes` file in {dir_name}, found none."
    node_attr_path = node_attr_files[0]
    print_green(f"Reading nodes: {node_attr_path}")
    print_warning("Warning: Converting node ids from 1-indexed (on disk) to 0-indexed (in memory).")
    auto_delimiter = delimiter or detect_delimiter_file(node_attr_path)
    nodes_df = pd.read_csv(node_attr_path, sep=auto_delimiter, header=None, comment=comment)
    nodes_df = nodes_df.sort_values(by=nodes_df.columns[0]).reset_index(drop=True)
    G.vs['feature'] = nodes_df.iloc[:, 1:].values

    # Get data_name from direcotry name
    data_name = os.path.basename(os.path.normpath(dir_name))

    # Add metadata to the graph
    G['data_name'] = data_name

    # Ensure nodes are labeled from 0 to n-1
    n = G.vcount()
    min_label = min(G.vs.indices)
    max_label = max(G.vs.indices)
    assert (min_label == 0) and (max_label == n - 1), f"Node labels must be from 0 to n-1. However, found {n} labels from {min_label} to {max_label}."

    return G

#################################################
# Graph Preprocessing Utilities
#################################################

def igraph_single_source_shortest_path_length(G, source, cutoff):
    # BFS up to `cutoff` hops; returns {node: hop_distance} including source (d=0)
    seen = {source: 0}
    queue = [source]
    while queue:
        next_queue = []
        for v in queue:
            if seen[v] >= cutoff:
                continue
            for nb in G.neighbors(v):
                if nb not in seen:
                    seen[nb] = seen[v] + 1
                    next_queue.append(nb)
        queue = next_queue
    return seen

## Function to augment queries by perturbing source and target nodes
def augment_queries(G, queries, perturb_k=4, k_hop=1, distance_func=None):
    augmented_queries = set()
    if distance_func is None:
        # Use a dummy distance function that returns 1 for all pairs
        distance_func = lambda u, v: 1

    print(f"Augmenting queries with perturb_k={perturb_k} and k_hop={k_hop}...")
    print("Before augmentation:")
    print("  - Number of unique queries: ", len(set([(u, v) for u, v, d in queries])))
    print("  - Number of unique nodes:   ", len(set([u for u, v, d in queries] + [v for u, v, d in queries])))
    print("  - Total number of queries:  ", len(queries))

    # Compute k-hop neighbors for unique nodes in the queries
    TEMPERATURE = 5.0  # flattens the probability distribution for neighbor selection
    unique_nodes = set([u for u, v, d in queries] + [v for u, v, d in queries])
    k_hop_neighbors = {}
    k_hop_probabilities = {}
    for node in unique_nodes:
        k_hop_distances = igraph_single_source_shortest_path_length(G, node, cutoff=k_hop)
        k_hop_neighbors[node] = {n: d for n, d in k_hop_distances.items() if d > 0}
        probabilities = [1 / distance for _, distance in k_hop_neighbors[node].items()]
        total = sum(np.exp(np.array(probabilities) / TEMPERATURE))
        probabilities = [np.exp(p / TEMPERATURE) / total for p in probabilities]
        k_hop_probabilities[node] = probabilities
        k_hop_neighbors[node] = list(k_hop_neighbors[node].keys())

    for src, dst, dist in queries:
        src = int(src)
        dst = int(dst)
        # Add the original query
        augmented_queries.add((src, dst, distance_func(src, dst)))

        # For each query, perturb source and target by replacing with a random neighbor (if exists)
        for _ in range(perturb_k):
            new_src = src
            new_dst = dst
            # Perturb source
            if k_hop_neighbors[src]:
                new_src = np.random.choice(k_hop_neighbors[src], p=k_hop_probabilities[src])
            # Perturb target
            if k_hop_neighbors[dst]:
                new_dst = np.random.choice(k_hop_neighbors[dst], p=k_hop_probabilities[dst])
            # Only add if not the same as original and not self-loop
            if (new_src != src or new_dst != dst) and (new_src != new_dst):
                augmented_queries.add((new_src, new_dst, distance_func(new_src, new_dst)))

    print("After augmentation:")
    print("  - Number of unique queries: ", len(set([(u, v) for u, v, d in augmented_queries])))
    print("  - Number of unique nodes:   ", len(set([u for u, v, d in augmented_queries] + [v for u, v, d in augmented_queries])))
    print("  - Total number of queries:  ", len(augmented_queries))

    return list(augmented_queries)

## Function to augment nodelist by adding k-perturbed neighbors
def augment_nodes(G, seed_nodes, k):
    neighbor_nodes = []
    unique_set = set(seed_nodes)
    for node in seed_nodes:
        added = set()
        visited = set([node])
        queue = [node]
        depth = 0
        while len(added) < k and queue:
            next_queue = []
            for current in queue:
                neighbors = G.neighbors(current)
                for neighbor in neighbors:
                    if neighbor not in unique_set and neighbor not in added:
                        neighbor_nodes.append(neighbor)
                        added.add(neighbor)
                        if len(added) == k:
                            break
                    if neighbor not in visited:
                        next_queue.append(neighbor)
                        visited.add(neighbor)
                if len(added) == k:
                    break
            queue = next_queue
            depth += 1
    return np.unique(np.concatenate([seed_nodes, neighbor_nodes]))

## Function to fuse a node with its two neighbors by summing their edge weights
def _fuse_2_degree_node(G, node):
    # Get the neighbors of the node (there will be exactly 2 neighbors)
    neighbors = list(G.neighbors(node))
    # Check if there is a direct connection between the neighbors
    if G.has_edge(neighbors[0], neighbors[1]):
        # TODO: Handle the case where there is a direct connection between the neighbors
        # Shall we keep the shortest edge or stop here? Currently, we stop here.
        return G
    # Get the edge attributes of the neighbors
    edge1 = G.get_edge_data(neighbors[0], node)
    edge2 = G.get_edge_data(neighbors[1], node)
    # Add the edge weights
    new_edge = edge1['LENGTH'] + edge2['LENGTH']
    # Add the combined edge to the graph
    G.add_edge(neighbors[0], neighbors[1], LENGTH=new_edge)
    # Remove the node from the graph
    G.remove_node(node)
    return G

# Function to remove 2-degree nodes iteratively
def remove_2_degree_nodes(G):
    G = G.copy()
    initial_nodes = G.number_of_nodes()
    inital_node_set = set(G.nodes())
    nodes_to_remove = [node for node in G.nodes() if G.degree(node) == 2]
    print(f"Number of 2-degree nodes: {len(nodes_to_remove)}")
    for node in nodes_to_remove:
        G = _fuse_2_degree_node(G, node)
    final_nodes = G.number_of_nodes()
    final_node_set = set(G.nodes())
    print(f"Number of nodes removed: {initial_nodes - final_nodes}")
    print(f"Nodes removed: {list(inital_node_set - final_node_set)[:3]} ...")
    return G

## Function to remove 1-degree nodes iteratively
def remove_1_degree_nodes(G):
    G = G.copy()
    count = 0
    removed_nodes = []
    while True:
        nodes_to_remove = [node for node in G.nodes() if G.degree(node) == 1]
        removed_nodes.extend(nodes_to_remove)  ## Same as removed_nodes += nodes_to_remove
        count += len(nodes_to_remove)
        G.remove_nodes_from(nodes_to_remove)
        if len(nodes_to_remove) == 0:
            break
    print(f"Number of nodes removed: {count}")
    print(f"Nodes removed: {removed_nodes[:3]} ...")
    return G

## Function to check if the edgelist is directed
def _is_directed(edgelist):
    ## TODO: Need to check if this implementation is correct!
    # Create a set of tuples of (source, target) nodes
    edgelist_set = set(zip(edgelist['START_NODE'], edgelist['END_NODE']))
    # Create a set of tuples of (target, source) nodes
    edgelist_set_rev = set(zip(edgelist['END_NODE'], edgelist['START_NODE']))
    # Check if the two sets are equal
    return edgelist_set != edgelist_set_rev

## Function to check if the edgelist has reverse edges
def _has_reverse_edges(edgelist):
    # Create a set of tuples of (source, target) nodes
    edgelist_set = set(zip(edgelist['START_NODE'], edgelist['END_NODE']))
    # Create a set of tuples of (target, source) nodes
    edgelist_set_rev = set(zip(edgelist['END_NODE'], edgelist['START_NODE']))
    # Check if there are any reverse edges
    return len(edgelist_set.intersection(edgelist_set_rev)) > 0

## Function to check if the edgelist has self loops
def _has_self_loops(edgelist):
    # Check if there are any self loops
    return edgelist.query('START_NODE == END_NODE').shape[0] > 0

## Function to preprocess the graph
def preprocess_graph(G, remove_1_degree=False, remove_2_degree=False):
    G = G.copy()
    print("Original graph:")
    print_summary_stats(G)

    # Take the largest connected component
    if not nx.is_connected(G):
        print("Taking the largest connected component...")
        G = G.subgraph(max(nx.connected_components(G), key=len))
        print_summary_stats(G)

    # Iteratively remove 1-degree nodes
    if remove_1_degree:
        print("Removing 1-degree nodes...")
        G = remove_1_degree_nodes(G)
        print_summary_stats(G)

    # Iteratively remove 2-degree nodes from the graph G
    if remove_2_degree:
        print("Removing 2-degree nodes...")
        G = remove_2_degree_nodes(G)
        print_summary_stats(G)

    # Check if one-degree nodes are still present
    if len([node for node in G.nodes() if G.degree(node) == 1]) > 0:
        print_warning("Warning: One-degree nodes are still present in the graph.")

    # Check if two-degree nodes are still present
    if len([node for node in G.nodes() if G.degree(node) == 2]) > 0:
        print_warning("Warning: Two-degree nodes are still present in the graph.")

    # Re-index the nodes
    # Check if re-index is required
    if min(G.nodes()) != 0 or max(G.nodes()) != G.number_of_nodes() - 1:
        print_warning("Warning: Node labels are not in the range [0, n-1]. Re-indexing is required.")
        print("Re-indexing the nodes...")
        print(f"Min/Max node labels before re-indexing: {min(G.nodes())}/{max(G.nodes())}")
        G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default')
        print(f"Min/Max node labels after re-indexing: {min(G.nodes())}/{max(G.nodes())}")

    print("Preprocessing complete.")
    print("Processed graph:")
    print_summary_stats(G)

    return G

## Function to get the edge attributes of a graph
def get_edge_attributes(G):
    # Returns array of shape (n_edges, 3) with columns [source, target, weight]
    edgelist = np.array(G.get_edgelist())
    return np.column_stack([edgelist, G.es['weight']])

## Function to get node attributes from a graph
def get_node_attributes(G):
    # Returns array of shape (n_nodes, n_features) with columns [x1, x2, ...], indexed by node id (0..n-1)
    return np.array(G.vs['feature'])

## Function to print summary statistics of a graph
def print_summary_stats(G):
    if isinstance(G, ig.Graph):
        n, e = G.vcount(), G.ecount()
        degrees = G.degree()
        weights = G.es['weight']
        n_components = len(G.components())
    else:
        n, e = G.number_of_nodes(), G.number_of_edges()
        degrees = [d for _, d in G.degree()]
        weights = [d['weight'] for _, _, d in G.edges(data=True)]
        n_components = nx.number_connected_components(G)
    print(f"  - Node count: {n}")
    print(f"  - Edge count: {e}")
    print(f"  - Min/Max/Avg degree: {min(degrees)}/{max(degrees)}/{np.mean(degrees):.2f}")
    print(f"  - Min/Max/Avg weight: {min(weights):.2f}/{max(weights):.2f}/{np.mean(weights):.2f}")
    print(f"  - No. of connected components: {n_components}")

#################################################
# Landmark Selection Utilities
#################################################

def select_landmarks(graph, num_landmarks, strategy="random", weight_key="weight", seed=42, subset=None, node_features=None, kmeans_batch_size=None):
    """Select landmarks based on the chosen strategy."""
    print(f"Selecting {num_landmarks} landmarks using strategy: {strategy} (weight_key='{weight_key}', seed={seed})")
    num_nodes = graph.vcount()
    random.seed(seed)

    # Basic Checks for num_landmarks
    if num_landmarks is None:
        raise ValueError(f"Landmarks num_landmarks={num_landmarks} is None; it should be a positive integer")
    if num_landmarks < 0:
        raise ValueError(f"Landmarks num_landmarks={num_landmarks} is negative; it should be a positive integer")
    elif num_landmarks == 0:
        raise ValueError(f"Landmarks num_landmarks={num_landmarks} is zero; it should be a positive integer")
    elif num_landmarks < 1:
        print(f"Landmarks num_landmarks={num_landmarks} is a fraction; converting it to a positive integer")
        num_landmarks = int(num_landmarks * num_nodes)
        print(f"Setting num_landmarks={num_landmarks}")
    elif num_landmarks > num_nodes:
        print(f"Warning: Landmarks num_landmarks={num_landmarks} is greater than the number of nodes {num_nodes}")
        num_landmarks = num_nodes
        print(f"Setting num_landmarks={num_nodes}")
    if subset is not None and len(subset) < num_landmarks:
        raise ValueError(f"Subset size {len(subset)} is smaller than the number of landmarks {num_landmarks} to select.")

    # Select landmarks based on the strategy
    if strategy == "degree":
        # Sort nodes by descending degree
        landmarks_sorted = sorted(enumerate(graph.degree()), key=lambda x: x[1], reverse=True)
    elif strategy.startswith("betweenness"):
        # Sort nodes by betweenness centrality
        centrality = graph.betweenness(weights=weight_key)
        reverse = strategy.endswith("high")  # [high|low] strategy
        landmarks_sorted = sorted(enumerate(centrality), key=lambda x: x[1], reverse=reverse)
    elif strategy.startswith("closeness"):
        # Sort nodes by closeness centrality
        centrality = graph.closeness(weights=weight_key)
        reverse = strategy.endswith("high")  # [high|low] strategy
        landmarks_sorted = sorted(enumerate(centrality), key=lambda x: x[1], reverse=reverse)
    elif strategy == "eigenvector":
        # Sort nodes by descending eigenvector centrality
        centrality = graph.eigenvector_centrality(weights=weight_key)
        landmarks_sorted = sorted(enumerate(centrality), key=lambda x: x[1], reverse=True)
    elif strategy == "pagerank":
        # Sort nodes by descending PageRank score
        centrality = graph.pagerank(weights=weight_key, damping=0.85)
        landmarks_sorted = sorted(enumerate(centrality), key=lambda x: x[1], reverse=True)
    elif strategy == "harmonic":
        # Sort nodes by descending harmonic centrality
        centrality = graph.harmonic_centrality(weights=weight_key)
        landmarks_sorted = sorted(enumerate(centrality), key=lambda x: x[1], reverse=True)
    elif strategy == "random":
        # Randomly select landmarks from the nodes without replacement
        all_nodes = list(range(num_nodes))
        random.shuffle(all_nodes)
        landmarks_sorted = [(node, 0) for node in all_nodes]  # Dummy centrality value (0) for uniformity with other strategies
    elif strategy == "kmeans":
        # Use KMeans clustering to select landmarks
        if node_features is None:
            raise ValueError("Node features must be provided for kmeans strategy.")
        print(f"Node features provided with shape: {node_features.shape}")
        if subset is not None:
            node_features = np.array(node_features)[subset]
        if kmeans_batch_size is not None:
            print(f"Using MiniBatchKMeans with batch_size={kmeans_batch_size}")
            kmeans = MiniBatchKMeans(n_clusters=num_landmarks, random_state=seed, batch_size=kmeans_batch_size)
        else:
            kmeans = KMeans(n_clusters=num_landmarks, random_state=seed)
        kmeans.fit(node_features)
        closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, node_features)
        if subset is not None:
            landmarks_sorted = [(subset[i], 0) for i in closest_indices]  # Dummy centrality value (0) for uniformity with other strategies
        else:
            landmarks_sorted = [(i, 0) for i in closest_indices]  # Dummy centrality value (0) for uniformity with other strategies
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # TODO: there may be some bug with subset logic in this function
    # If subset is provided, filter landmarks to be within the subset
    if subset is not None:
        subset = set(subset)
        print(f"Filtering landmarks to be within the provided subset of size {len(subset)}")
        landmarks_sorted = [(lm, _) for lm, _ in landmarks_sorted if lm in subset]

    # Select top-K landmarks
    landmarks = [node for node, _ in landmarks_sorted[:num_landmarks]]

    return landmarks

def compute_landmark_distances(graph, landmarks, weight_key="weight", backend="networkit", n_threads=8):
    """
    Construct a matrix of shape (n_nodes, n_landmarks) such that the (i,j)-th entry is the
    shortest path distance from node i to the j-th landmark.

    Args:
        backend (str): "networkit" (default) | "igraph" | "scipy"
        n_threads (int): Used only by the networkit backend (8 is optimal on DIMACS benchmarks for 64 landmarks)
    """
    lm = list(landmarks)

    if backend not in ("scipy", "networkit", "igraph"):
        raise ValueError(f"backend must be 'igraph', 'scipy', or 'networkit'; got '{backend}'")

    if backend == "igraph":
        matrix = np.array(graph.distances(source=lm, weights=weight_key),
                          dtype=np.float32).T  # (n_landmarks, n_nodes).T → (n_nodes, n_landmarks)
        return matrix

    # Prepare the graph for scipy or networkit backends
    n = graph.vcount()
    edges   = np.array(graph.get_edgelist())
    weights = np.array(graph.es[weight_key], dtype=np.float64)

    if backend == "scipy":
        import scipy.sparse as sp
        from scipy.sparse import csgraph as scg
        r = np.concatenate([edges[:, 0], edges[:, 1]])
        c = np.concatenate([edges[:, 1], edges[:, 0]])
        w = np.concatenate([weights, weights])
        A = sp.csr_matrix((w, (r, c)), shape=(n, n))
        matrix = scg.dijkstra(A, directed=True, indices=lm,
                              return_predecessors=False).astype(np.float32).T

    else:  # networkit
        from concurrent.futures import ThreadPoolExecutor
        import networkit as nk
        nk.setNumberOfThreads(n_threads)
        G_nk = nk.Graph(n, weighted=True, directed=False)
        for (u, v), w in zip(edges, weights):
            G_nk.addEdge(u, v, w)
        def _sssp(src):
            sssp = nk.distance.Dijkstra(G_nk, int(src), storePaths=False)
            sssp.run()
            return np.array(sssp.getDistances(), dtype=np.float32)
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            rows = list(pool.map(_sssp, lm))
        matrix = np.stack(rows, axis=1)  # (n, n_landmarks)

    return matrix

#################################################
# Miscellaneous Utilities
#################################################

## Function to print the size of the distance matrix
def size_of_distance_matrix(n):
    size = n*n*4/(1024*1024)
    print(f"Size of distance matrix ({n}x{n}): {size:.2f} MB (assuming float32, i.e., 4 bytes for each element)")
