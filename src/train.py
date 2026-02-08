# %%
# import libraries
import os
import sys
import argparse
import time
from zipfile import ZipFile
import networkx as nx
import pickle
import json
from collections import Counter
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from utils.data_utils import (
    seed_everything,
    get_num_workers,
    get_edge_attributes,
    get_node_attributes,
    load_graph,
    read_embedding_file,
    print_green,
    read_parts_file,
)
from utils.plot_utils import (
    set_plot_style,
    plot_data_distribution,
    plot_learning_curves,
    plot_targets_and_predictions,
    plot_targets_and_mre_boxplots,
)
from utils.torch_utils import (
    get_available_device,
    print_device_info,
    get_optimizer,
    get_criterion,
    save_dataset,
    load_dataset,
    save_model,
    save_dictionary,
)

# %%
################
# Argument Parsing
################

## Setup argument parser
parser = argparse.ArgumentParser(description='Train a model on a dataset for shortest distance prediction.')
# Model configuration
parser.add_argument('--model_class', type=str, default='geodnn',
                    help='Class of the model')
parser.add_argument('--model_name', type=str, default=None,
                    help='Model Identifier, Version or Name for different hypeparameters')
# Data configuration
parser.add_argument('--data_dir', type=str, default='W_Jinan',
                    help='Path to directory containing `*.nodes` and `*.edges` files')
parser.add_argument('--query_dir', type=str, default='real_workload_perturb_500k',
                    help='Path to directory containing `*.queries` files')
# Training configuration
parser.add_argument('--batch_size_train', type=int, default=2**10,
                    help='Batch size for training')
parser.add_argument('--batch_size_test', type=int, default=2**20,
                    help='Batch size for testing or evaluation')
parser.add_argument('--learning_rate', type=float, default=0.01,
                    help='Learning rate for optimizer')
parser.add_argument('--optimizer_type', type=str, default='adam',
                    help='Optimizer to use for training (e.g., "adam", "sgd", etc.)')
parser.add_argument('--loss_function', type=str, default='mse',
                    help='Loss function to use (e.g., "mse", "mae", etc.)')
parser.add_argument('--epochs', type=int, default=20,
                    help='Number of epochs for training')
parser.add_argument('--time_limit', type=float, default=None,
                    help='Time limit for training in minutes')
parser.add_argument('--validate', action='store_true',
                    help='Whether to perform validation during training')
parser.add_argument('--eval_runs', type=int, default=0,
                    help='Number of evaluation runs to average query latency')
parser.add_argument('--display_step', type=int, default=5,
                    help='Logging frequency in one epoch')
parser.add_argument('--seed', type=int, default=1234,
                    help='Random seed for reproducibility')
parser.add_argument('--device', type=str, default=None,
                    help='Device to use for training and evaluation (e.g., "cuda", "cpu", "mps")')
parser.add_argument('--num_workers', type=int, default=None,
                    help='Number of worker processes for DataLoader (default: auto-detect)')
# Logging configuration
parser.add_argument('--log_dir', type=str, default='../results/default',
                    help='Directory to save logs and checkpoints')
parser.add_argument('--debug', action='store_true',
                    help='Enable debug mode for additional logging and checks')
# Model specific arguments
parser.add_argument('--embedding_dim', type=int, default=64,
                    help='Dimensionality of the embedding space')
parser.add_argument('--p_norm', type=int, default=1,
                    help='p-value for distance computation to use in the LpNorm model')
parser.add_argument('--landmark_selection', type=str, default='random',
                    help='Type of landmark selection strategy to use in the Landmark model')
parser.add_argument('--gnn_layer', type=str, default='gat',
                    help='Type of GNN layer to use in the GNN model')
parser.add_argument('--disable_edge_weight', action='store_true', default=False,
                    help='Disable edge weight in the GNN model')
parser.add_argument('--aggregation_method', type=str, default='concat',
                    help='Aggregation method to use in the EmbeddingNN model (e.g., "hadamard", "subtract", "mean", "concat")')
parser.add_argument('--embedding_filename', type=str, default=None,
                    help='Path to precomputed node embeddings file (optional)')
parser.add_argument('--select_landmarks_from_train', action='store_true', default=False,
                    help='Use only the nodes present in the training dataset for landmark selection')
parser.add_argument('--distance_measure', type=str, default='inv_dotproduct',
                    help='Distance measure to use (e.g., "inv_dotproduct", "norm", "dotproduct", etc.) in ANEDA model')
args = parser.parse_args()

# ## (Optional) Print python command used to run the program
# print(f"PYTHON_PATH: {sys.executable}")
# print(f"PYTHON_COMMAND: {os.path.basename(sys.executable)} {' '.join(sys.argv)}")
print(f"PYTHON_COMMAND: python {' '.join(sys.argv)}")

DATA_DIR = "../data"

## Resolve arguments
# Resolve model_name if not provided
if args.model_name is None:
    print(f"Resolving model_name: `None` --> `{args.model_class}`")
    args.model_name = args.model_class
# Resolve data directory if it doesn't exist
if not os.path.exists(args.data_dir):
    temp_old_value = args.data_dir
    args.data_dir = os.path.join(DATA_DIR, args.data_dir)
    print(f"Resolving data_dir: `{temp_old_value}` --> `{args.data_dir}`")
    assert os.path.exists(args.data_dir), f"Data directory `{args.data_dir}` does not exist."
# Resolve query directory if it doesn't exist
if not os.path.exists(args.query_dir):
    temp_old_value = args.query_dir
    args.query_dir = os.path.join(args.data_dir, args.query_dir)
    print(f"Resolving query_dir: `{temp_old_value}` --> `{args.query_dir}`")
    assert os.path.exists(args.query_dir), f"Query directory `{args.query_dir}` does not exist."
# Resolve num_workers if it is None
if args.num_workers is None:
    args.num_workers = get_num_workers()
    print(f"Resolving num_workers: `None` --> `{args.num_workers}`")
# Resolve embedding_filename if it is not None
if args.embedding_filename is not None and not os.path.exists(args.embedding_filename):
    temp_old_value = args.embedding_filename
    args.embedding_filename = os.path.join(args.data_dir, args.embedding_filename)
    print(f"Resolving embedding_filename: `{temp_old_value}` --> `{args.embedding_filename}`")
    assert os.path.exists(args.embedding_filename), f"Embedding file `{args.embedding_filename}` does not exist."

## Get arguments
# Model configuration
model_class = args.model_class
model_name = args.model_name
# Data configuration
data_dir = args.data_dir
data_name = os.path.basename(os.path.normpath(args.data_dir))
query_dir = args.query_dir
query_name = os.path.basename(os.path.normpath(args.query_dir))
# Training configuration
batch_size_train = args.batch_size_train
batch_size_test = args.batch_size_test
learning_rate = args.learning_rate
optimizer_type = args.optimizer_type
loss_function = args.loss_function
epochs = args.epochs
time_limit = args.time_limit
validate = args.validate
eval_runs = args.eval_runs
display_step = args.display_step
seed = args.seed
device = args.device
num_workers = args.num_workers
# Logging configuration
log_dir = args.log_dir
os.makedirs(log_dir, exist_ok=True)
debug = args.debug
# Model specific arguments
embedding_dim = args.embedding_dim
p_norm = args.p_norm
landmark_selection = args.landmark_selection
gnn_layer = args.gnn_layer
disable_edge_weight = args.disable_edge_weight
aggregation_method = args.aggregation_method
embedding_filename = args.embedding_filename
distance_measure = args.distance_measure

print("Arguments:")
for arg, value in vars(args).items():
    print(f"  - {arg:<20}: {value}")

# Set seed for reproducibility
seed_everything(seed)

# Set directories
PLOTS_DIR = os.path.join(log_dir, "plots")
set_plot_style(scale=1.25)  # Adjust scale for plotting
SAVED_MODELS_DIR = os.path.join(log_dir, "saved_models")
DEBUG_INFO_DIR = os.path.join(log_dir, "debug")

# %%
################
# Load graph
################

G = load_graph(dir_name=data_dir)
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
# (Optional) Print some graph statistics
print(f"Loaded Graph:")
print(f"  - Data name: {data_name}")
print(f"  - No. of nodes: {num_nodes}")
print(f"  - No. of edges: {num_edges}")
print(f"  - Nodes data: {list(G.nodes(data=True))[:5]}...")
print(f"  - Edges data: {list(G.edges(data=True))[:5]}...")


# Compute edge attributes
edge_attributes = get_edge_attributes(G)
print("Edgelist.shape: ", edge_attributes.shape)

# Compute node attributes
node_attributes = get_node_attributes(G)
print("Node Attributes.shape: ", node_attributes.shape)

# %%
################
# Load dataset
################

train_dataset, test_dataset = load_dataset(dir_name=query_dir, seed=seed, replicate_test=True, target_test_size=batch_size_test, drop_duplicates=False)
print("Train dataset...")
print(f"  - No. of samples: {len(train_dataset)}")
print(f"  - Min/Max distance: {train_dataset.D.min():.2f}/{train_dataset.D.max():.2f}")
print(f"  - Mean/Std distance: {train_dataset.D.mean():.2f}/{train_dataset.D.std():.2f}")
max_distance = train_dataset.D.max()
print(f"Test dataset...")
print(f"  - No. of samples: {len(test_dataset)}")
print(f"  - Min/Max distance: {test_dataset.D.min():.2f}/{test_dataset.D.max():.2f}")
print(f"  - Mean/Std distance: {test_dataset.D.mean():.2f}/{test_dataset.D.std():.2f}")

# dirty_mean = test_dataset.D.mean()
# dirty_mre = np.mean(np.abs(test_dataset.D - dirty_mean) / np.maximum(test_dataset.D, 1e-6))
# print(f"Dirty MRE (predicting mean distance={dirty_mean}): {dirty_mre:.2%}")

# dirty_median = np.median(test_dataset.D)
# dirty_mre = np.median(np.abs(test_dataset.D - dirty_median) / np.maximum(test_dataset.D, 1e-6))
# print(f"Dirty Median RE (predicting median distance={dirty_median}): {dirty_mre:.2%}")

# dirty_mode = Counter(test_dataset.D.astype(int)).most_common(1)[0][0]
# dirty_mre = np.median(np.abs(test_dataset.D - dirty_mode) / np.maximum(test_dataset.D, 1e-6))
# print(f"Dirty Mode RE (predicting mode distance={dirty_mode}): {dirty_mre:.2%}")
# exit()

# Create dataloaders
pin_memory = True
print("Creating train dataloader...")
train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
print(f"  - Batch size: {batch_size_train}")
print(f"  - No. of batches: {len(train_dataloader)}")
print(f"  - Number of workers: {num_workers}")
print(f"  - Pin memory: {pin_memory}")

# (Optional) Create a validation dataloader
val_dataloader = None
if validate:
    print("Creating val dataloader...")
    val_dataloader = DataLoader(test_dataset, batch_size=(batch_size_train),
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=pin_memory)
    print(f"  - Batch size: {batch_size_train}")
    print(f"  - No. of batches: {len(val_dataloader)}")
    print(f"  - Number of workers: {num_workers}")
    print(f"  - Pin memory: {pin_memory}")

print("Creating test dataloader...")
test_dataloader = DataLoader(test_dataset, batch_size=batch_size_test,
                             shuffle=True,  ## Shuffle to simulate streaming queries
                             num_workers=num_workers,
                             pin_memory=pin_memory)
print(f"  - Batch size: {batch_size_test}")
print(f"  - No. of batches: {len(test_dataloader)}")
print(f"  - Number of workers: {num_workers}")
print(f"  - Pin memory: {pin_memory}")

# sample_batch = next(iter(dataloader))
# i, j, d_ij = sample_batch
# print(i.shape, j.shape, d_ij.shape)
# print(i.dtype, j.dtype, d_ij.dtype)

# %%
# (Optional) Plot histogram of distances
plot_data_distribution(train_dataset.D.ravel(),
                       test_dataset.D.ravel(),
                       data_name, query_name, dir_name=PLOTS_DIR)

# %%
################
# Load model
################

if model_class == 'geodnn':
    from models.geodnn import GeoDNN

    # Initialize model
    model = GeoDNN(n_input=4,                           ## Input size (coordinates of src and dst)
                   n_hidden_1=20,                       ## Hidden layer 1 size
                   n_hidden_2=100,                      ## Hidden layer 2 size
                   n_hidden_3=20,                       ## Hidden layer 3 size
                   n_output=1,                          ## Output size
                   node_attributes=node_attributes,     ## Precomputed node attributes (coordinates)
                   max_distance=max_distance            ## Maximum distance for scaling
                   )
elif model_class == 'landmark':
    from models.landmark import Landmark

    # Find subset of nodes for landmark selection (if applicable)
    subset_nodes = None
    if args.select_landmarks_from_train:
        subset_nodes = np.unique(train_dataset.queries[:, :2]).astype(int).tolist()
        print(f"Using {len(subset_nodes)} unique nodes from training dataset for landmark selection.")

    # Initialize model
    model = Landmark(graph=G,                           ## Input graph
                     num_landmarks=embedding_dim,       ## Embedding size
                     strategy=landmark_selection,       ## Landmark selection strategy
                     weight_key="weight",               ## Edge weight key
                     node_features=node_attributes,     ## Node attributes
                     subset=subset_nodes,               ## Subset of nodes for landmark selection
                     seed=seed                          ## Random seed
                     )
elif model_class == 'ndist2vec':
    from models.ndist2vec import Ndist2vec

    # Initialize model
    model = Ndist2vec(n_input=num_nodes,                ## No. of Embddings
                      n_hidden_1=embedding_dim,         ## Embedding size
                      n_hidden_2=100,                   ## Hidden layer 1 size
                      n_hidden_3=20,                    ## Hidden layer 2 size
                      n_output=1,                       ## Output size
                      max_distance=max_distance         ## Maximum distance for scaling
                      )
elif model_class == 'lpnorm':
    from models.lpnorm import LpNorm

    # Initialize model
    model = LpNorm(p_norm, node_attributes)
elif model_class == 'vdist2vec':
    from models.vdist2vec import Vdist2vec

    # Initialize model
    model = Vdist2vec(n_input=num_nodes,                ## No. of Embddings
                      n_hidden_1=embedding_dim,         ## Embedding size
                      n_hidden_2=100,                   ## Hidden layer 1 size
                      n_hidden_3=20,                    ## Hidden layer 2 size
                      n_output=1,                       ## Output size
                      max_distance=max_distance         ## Maximum distance for scaling
                      )
elif model_class == 'rgnndist2vec':
    from models.rgnndist2vec import RGNNdist2vec

    # Initialize model
    model = RGNNdist2vec(n_input=2,                                 ## Input embedding size (coordinates)
                         n_hidden_1=512,                            ## Hidden layer 1 size
                         n_hidden_2=embedding_dim,                  ## Hidden layer 2 size
                         layer_type=gnn_layer,                      ## Type of GNN layer to use ('sage', 'gcn', 'gat', etc.)
                         node_attributes=node_attributes,           ## Precomputed node attributes (coordinates)
                         edge_attributes=edge_attributes,           ## Edge attributes (weights)
                         max_distance=max_distance,                 ## Maximum distance for scaling
                         disable_edge_weight=disable_edge_weight    ## Whether to disable edge weights
                         )
elif model_class == 'embeddingnn':
    from models.embeddingnn import EmbeddingNN

    # Load custom node embeddings
    if embedding_filename is not None:
        custom_node_embeddings = read_embedding_file(embedding_filename)
    else:
        custom_node_embeddings = None

    # Initialize model
    model = EmbeddingNN(num_nodes=num_nodes,                        ## No. of nodes
                        embed_size=embedding_dim,                   ## Embedding size
                        n_hidden_1=500,                             ## Hidden layer 1 size
                        n_output=1,                                 ## Output size
                        init_embeddings=custom_node_embeddings,     ## Precomputed node embeddings
                        aggregation_method=aggregation_method,      ## Aggregation method
                        max_distance=max_distance                   ## Maximum distance for scaling
                        )
elif model_class == 'distancenn':
    from models.distancenn import DistanceNN

    # Load custom node embeddings
    if embedding_filename is not None:
        custom_node_embeddings = read_embedding_file(embedding_filename)
    else:
        custom_node_embeddings = None

    # Initialize model
    model = DistanceNN(num_nodes=num_nodes,                         ## No. of nodes
                       embed_size=embedding_dim,                    ## Embedding size
                       init_embeddings=custom_node_embeddings,      ## Precomputed node embeddings
                       aggregation_method=aggregation_method        ## Aggregation method
                       )
elif model_class == 'aneda':
    from models.aneda import ANEDA

    # Load custom node embeddings
    if embedding_filename is not None:
        custom_node_embeddings = read_embedding_file(embedding_filename)
    else:
        custom_node_embeddings = None

    # Initialize model
    model = ANEDA(num_nodes=num_nodes,                              ## No. of nodes
                  embed_size=embedding_dim,                         ## Embedding size
                  init_embeddings=custom_node_embeddings,           ## Precomputed node embeddings
                  max_distance=max_distance,                        ## Maximum distance for scaling
                  distance_measure=distance_measure,                ## Distance measure to use
                  p=p_norm                                          ## p-value for Lp norm (if applicable)
                  )
elif model_class == 'path2vec':
    from models.path2vec import Path2vec

    # Initialize model
    model = Path2vec(G=G,                               ## Input graph
                     embed_size=embedding_dim,          ## Embedding size
                     max_distance=max_distance,         ## Maximum distance for scaling
                     regularize=True,                   ## Whether to use regularization
                     l1factor=1e-10,                    ## L1 regularization factor
                     use_neighbors=True,                ## Whether to use neighbors
                     neighbor_count=5,                  ## Number of neighbors to consider
                     alpha=0.5                          ## Alpha parameter
                     )
elif model_class == 'rne':
    from models.rne import RNE

    # Load parts information (if available)
    parts = read_parts_file(data_dir, data_name)

    # Initialize model
    model = RNE(num_nodes=num_nodes,                                ## No. of nodes
                embed_size=embedding_dim,                           ## Embedding size
                max_distance=train_dataset.D.mean(),                ## TODO: need to check if max_distance is better or mean_distance
                parts=parts)                                        ## Parts information (if available)
elif model_class == 'catboost':
    from models.catboostmodel import CatBoostModel

    # Load custom node embeddings
    custom_node_embeddings = read_embedding_file(embedding_filename)

    # Initialize model
    model = CatBoostModel(num_nodes=num_nodes,                      ## No. of nodes
                          coordinate_embs=node_attributes,          ## Precomputed node embeddings (coordinates)
                          landmark_embs=custom_node_embeddings      ## Precomputed node embeddings (landmarks distances)
                          )
elif model_class == 'catboostnn':
    from models.catboostnn import CatBoostNN

    # Load custom node embeddings
    custom_node_embeddings = read_embedding_file(embedding_filename)

    # Initialize model
    model = CatBoostNN(num_nodes=num_nodes,                         ## No. of nodes
                       coordinate_embs=node_attributes,             ## Precomputed node embeddings (coordinates)
                       landmark_embs=custom_node_embeddings,        ## Precomputed node embeddings (landmarks distances)
                       max_distance=train_dataset.D.mean()          ## Mean distance for scaling
                       )
else:
    raise ValueError(f"Unknown model class: {model_class}")

model = torch.compile(model)  # Compile the model for better performance
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Model Summary:")
print(model)
print(f"Model parameters size: {num_params}")

# %%
################
# Loss function and optimizer
################

# Initialize loss function
criterion = get_criterion(loss_function, model)
print(f"Loss function: {criterion}")

# Initialize optimizer
optimizer = get_optimizer(optimizer_type, model, learning_rate)
print(f"Optimizer: {optimizer}")

# Check available device
if args.model_class == 'catboost':
    device = 'cpu'  # Incremental training in catboost is only supported on CPU
    print(f"Setting device for catboost: `{args.device}` --> `{device}`")
elif device is None:
    print("No device specified by user, detecting available device...")
    device = get_available_device()
else:
    print(f"Using user-specified device: {device}")
print(f"Using Device: {device}")
print_device_info(device)

# %%
################
# Training Loop
################

start_time = time.perf_counter()
print("Starting training...")
train_history = model.fit(dataloader=train_dataloader,
                          criterion=criterion,
                          optimizer=optimizer,
                          learning_rate=learning_rate,
                          val_dataloader=val_dataloader,
                          epochs=epochs,
                          display_step=display_step,
                          max_distance=max_distance,
                          device=device,
                          time_limit=time_limit,
                          fast_dev_run=False)
end_time = time.perf_counter()
precomputation_time = end_time - start_time
print(f"Optimization Finished!")
print(f"Precomputation time: {precomputation_time / 60:.2f} minutes")

# Save the model
save_model(model, model_name, data_name, query_name, dir_name=SAVED_MODELS_DIR, metadata={'catboost_model': model.catboost_model if model_class == 'catboost' else None})

# %%
# (Optional) Plot epoch and iteration losses during training
plot_learning_curves(train_history, n_batches=len(train_dataloader),
                     model_name=model_name, data_name=data_name, query_name=query_name,
                     dir_name=PLOTS_DIR)

# %%
################
# Evaluate the model
################

# (Optional) Set the precision for matrix multiplication (lower precision implies higher query latency)
# Choose from: 'highest', 'high', 'medium'
# TODO: Need to check appropriate placement, before training or test or both or none
torch.set_float32_matmul_precision('high')
print(f"Set float32 matmul precision to: {torch.get_float32_matmul_precision()}")

for label_i, dataloader_i in zip(["train", "test"], [train_dataloader, test_dataloader]):
    start_time = time.perf_counter()
    predictions, targets, query_latency = model.evaluate(dataloader=dataloader_i,
                                                         max_distance=max_distance,
                                                         device=device)
    end_time = time.perf_counter()
    evaluation_time = end_time - start_time
    print(f"Evaluation on {label_i} Finished!")
    print(f"Evaluation time: {evaluation_time / 60:.2f} minutes")

    # Calculate the mae, mre
    mae = np.mean(np.abs(predictions - targets))
    mre = np.mean(np.abs(predictions - targets) / np.maximum(targets, 1e-6))  # Avoid division by zero
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Relative Error: {mre:.2%}")
    print(f"Query time per sample: {query_latency * 1_000_000:.3f} microseconds")
    print(f"Adjusted query time per sample: {evaluation_time / len(dataloader_i.dataset) * 1_000_000:.3f} microseconds")

    # (Optional) Plot targets and mre in boxplots
    plot_targets_and_mre_boxplots(
        predictions, targets,
        model_name, data_name, query_name,
        dir_name=PLOTS_DIR,
        label=label_i.capitalize(),
        num_buckets=5
    )

    # (Optional) Plot targets and predictions in sorted order
    # Randomly select 1M samples for plotting to ease plotting
    random_indices = np.random.permutation(len(predictions))[:1_000_000]
    predictions = predictions[random_indices]
    targets = targets[random_indices]
    plot_targets_and_predictions(
        predictions, targets,
        model_name, data_name, query_name,
        dir_name=PLOTS_DIR,
        label=label_i.capitalize()
    )

# %%
################
# (Optional) Eval Runs for Query Latency
################

if eval_runs > 0:
    print(f"Running evaluation {eval_runs} times to get average query time...")
    results = []
    for i in range(eval_runs):
        start_time = time.perf_counter()
        predictions, targets, query_latency = model.evaluate(dataloader=test_dataloader,
                                                             max_distance=max_distance,
                                                             device=device)
        end_time = time.perf_counter()
        evaluation_time = end_time - start_time
        results.append(query_latency)
        print(f"[Eval Run] Query time per sample: {query_latency * 1_000_000:.3f} microseconds")
        print(f"[Eval Run] Adjusted query time per sample: {evaluation_time / len(test_dataloader.dataset) * 1_000_000:.3f} microseconds")

    # We can take the average of the last 5 runs to get a more stable estimate
    avg_query_latency = np.mean(results[-5:])
    print(f"[Eval Run] Average query time per sample: {avg_query_latency * 1_000_000:.3f} microseconds")

# # %%
# ################
# # (Optional) Direct Eval Runs for Query Latency (no batching)
# ################
# if eval_runs > 0:
#     print(f"Running DIRECT (no batching) evaluation {eval_runs} times to get average query time...")
#     for i in range(eval_runs):
#         model.eval()
#         model.to(device)
#         predictions = []
#         # Shuffle dataset to simulate streaming queries
#         shuffled_indices = np.random.permutation(len(test_dataset))
#         src_indices = test_dataset.queries[shuffled_indices, 0]
#         dst_indices = test_dataset.queries[shuffled_indices, 1]

#         start_time = time.perf_counter()
#         # Move data to device
#         src_indices = torch.tensor(src_indices, dtype=torch.long).to(device)
#         dst_indices = torch.tensor(dst_indices, dtype=torch.long).to(device)
#         # Model inference
#         with torch.no_grad():
#             preds = model(src_indices, dst_indices)
#             predictions.append(preds.cpu().numpy())
#         # Synchronize if using GPU
#         if device.startswith('cuda'):
#             torch.cuda.synchronize()
#         end_time = time.perf_counter()

#         evaluation_time = end_time - start_time
#         predictions = np.concatenate(predictions, axis=0)
#         query_latency = evaluation_time / len(test_dataset)
#         print(f"[Direct Eval Run {i+1}] Query time per sample: {query_latency * 1_000_000:.3f} microseconds")

# %%
################
# (Optional) Save debug information
################

if debug:
    debug_info = {
        'args': vars(args),
        'train_history': train_history,
        'precomputation_time': precomputation_time,
        'evaluation_time': evaluation_time,
        'predictions': predictions,
        'targets': targets,
        'mre_runs': results,
        'avg_query_latency': avg_query_latency

    }
    # Save debug information to a JSON file
    save_dictionary(debug_info, model_name, data_name, query_name, dir_name=DEBUG_INFO_DIR)
