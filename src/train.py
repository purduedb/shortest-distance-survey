# %%
# import libraries
import os
import sys
import argparse
import time
import resource

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from utils.data_utils import (
    seed_everything,
    get_num_cores,
    get_edge_attributes,
    get_node_attributes,
    load_graph,
    read_embedding_file,
    print_warning,
    print_error_distribution,
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
    reset_peak_gpu_stats,
    get_optimizer,
    get_criterion,
    load_dataset,
    save_model,
    save_jit_model,
    load_model,
    save_dictionary,
    save_metrics_json,
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
                    help='Directory containing `*.nodes` and `*.edges` files')
parser.add_argument('--query_dir', type=str, default='real_workload_perturb_500k',
                    help='Directory containing `*.queries.npz` files')
# Training configuration
parser.add_argument('--batch_size_train', type=int, default=1024,
                    help='Batch size for training')
parser.add_argument('--batch_size_val', type=int, default=1024,
                    help='Batch size for subgraph-based validation during training')
parser.add_argument('--batch_size_test', type=int, default=1024,
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
parser.add_argument('--num_cores', type=int, default=None,
                    help='Number of available CPU cores (default: auto-detect all)')
parser.add_argument('--matmul_precision', type=str, default=None,
                    choices=['highest', 'high', 'medium'],
                    help='(Test only) Set torch.float32 matmul precision for evaluation '
                    '(choices: highest, high, medium). If None, precision is unchanged.')
parser.add_argument('--compile_mode', type=str, default='none',
                    choices=['none', 'default', 'reduce-overhead', 'max-autotune'],
                    help='torch.compile mode ("none" skips compilation entirely)')
parser.add_argument('--checkpoint_path', type=str, default=None,
                    help='Path to a `.pt` model checkpoint to load before training/evaluation')
parser.add_argument('--inference_only', action='store_true', default=False,
                    help='Load --checkpoint_path and run in inference mode without training')
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
parser.add_argument('--kmeans_batch_size', type=int, nargs='?', const=1024, default=None,
                    help='Use MiniBatchKMeans (batch size) for kmeans landmark selection')
parser.add_argument('--gnn_layer', type=str, default='gat',
                    help='Type of GNN layer to use in the GNN model')
parser.add_argument('--disable_edge_weight', action='store_true', default=False,
                    help='Disable edge weight in the GNN model')
parser.add_argument('--aggregation_method', type=str, default='concat',
                    help='Aggregation method to use in the EmbeddingNN model '
                    '(e.g., "hadamard", "subtract", "mean", "concat")')
parser.add_argument('--embedding_filename', type=str, default=None,
                    help='Path to precomputed node embeddings file (optional)')
parser.add_argument('--select_landmarks_from_train', action='store_true', default=False,
                    help='Use only the nodes present in the training dataset for landmark selection')
parser.add_argument('--distance_measure', type=str, default='inv_dotproduct',
                    help='Distance measure to use (e.g., "inv_dotproduct", "norm", "dotproduct", etc.) in ANEDA model')
parser.add_argument('--sparse_embedding', action='store_true', default=False,
                    help='Use sparse=True on nn.Embedding and SparseAdam optimizer for learnable embedding-based models')
args = parser.parse_args()

# ## (Optional) Print python command used to run the program
# print(f"PYTHON_PATH: {sys.executable}")
# print(f"PYTHON_COMMAND: {os.path.basename(sys.executable)} {' '.join(sys.argv)}")
print(f"PYTHON_COMMAND: python {' '.join(sys.argv)}")

DATA_DIR = "../data"

## Resolve arguments
# Handle --inference_only flag
if args.inference_only:
    assert args.checkpoint_path is not None, "--inference_only requires --checkpoint_path"
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
# Resolve num_cores if it is None
if args.num_cores is None:
    args.num_cores = get_num_cores()
    print(f"Resolving num_cores: `None` --> `{args.num_cores}`")
# Resolve embedding_filename if it is not None
if args.embedding_filename is not None and not os.path.exists(args.embedding_filename):
    temp_old_value = args.embedding_filename
    args.embedding_filename = os.path.join(args.data_dir, args.embedding_filename)
    print(f"Resolving embedding_filename: `{temp_old_value}` --> `{args.embedding_filename}`")
    assert os.path.exists(args.embedding_filename), f"Embedding file `{args.embedding_filename}` does not exist."
# Resolve batch_size_test for GNN models to prevent OOM in eval/JIT precompute
if args.model_class == 'rgnndist2vec':
    _user_provided_batch_size_test = '--batch_size_test' in sys.argv
    if not _user_provided_batch_size_test:
        print(f"Resolving batch_size_test: `{args.batch_size_test}` --> `{args.batch_size_train}` "
              f"(rgnndist2vec default: match batch_size_train to prevent OOM in eval/JIT precompute)")
        args.batch_size_test = args.batch_size_train
    elif args.batch_size_test != args.batch_size_train:
        print_warning(f"Warning [rgnndist2vec]: user-specified batch_size_test `{args.batch_size_test}` "
                      f"may cause OOM during eval and JIT precompute (recommended: {args.batch_size_train})")
# Resolve device if not provided
if args.device is None:
    if args.model_class == 'catboost':
        print("Model class is `catboost`, setting device to `cpu` since incremental training in catboost"
              " is only supported on CPU.")
        args.device = 'cpu'
    else:
        args.device = get_available_device()
    print(f"Resolving device: `None` --> `{args.device}`")
else:
    if args.model_class == 'catboost' and args.device != 'cpu':
        print(f"Model class is `catboost`, overriding user-specified device `{args.device}` to `cpu` since"
              " incremental training in catboost is only supported on CPU.")
        args.device = 'cpu'
        print(f"Resolving device: `{args.device}` --> `{args.device}`")

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
batch_size_val = args.batch_size_val
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
num_cores = args.num_cores
matmul_precision = args.matmul_precision
compile_mode = args.compile_mode
checkpoint_path = args.checkpoint_path
inference_only = args.inference_only
# Logging configuration
log_dir = args.log_dir
os.makedirs(log_dir, exist_ok=True)
debug = args.debug
# Model specific arguments
embedding_dim = args.embedding_dim
p_norm = args.p_norm
landmark_selection = args.landmark_selection
kmeans_batch_size = args.kmeans_batch_size
gnn_layer = args.gnn_layer
disable_edge_weight = args.disable_edge_weight
aggregation_method = args.aggregation_method
embedding_filename = args.embedding_filename
distance_measure = args.distance_measure
sparse_embedding = args.sparse_embedding

print("Arguments:")
for arg, value in vars(args).items():
    print(f"  - {arg:<20}: {value}")

# Initialize metrics dictionary with all args
metrics = dict(vars(args))

# Set seed for reproducibility
seed_everything(seed)

# Set directories
PLOTS_DIR = os.path.join(log_dir, "plots")
set_plot_style(scale=1.25)  # Adjust scale for plotting
SAVED_MODELS_DIR     = os.path.join(log_dir, "saved_models")
SAVED_JIT_MODELS_DIR = os.path.join(log_dir, "saved_jit_models")
SAVED_METRICS_DIR    = os.path.join(log_dir, "saved_metrics")
DEBUG_INFO_DIR = os.path.join(log_dir, "debug")

# %%
################
# Load graph
################

G = load_graph(dir_name=data_dir)
num_nodes = G.vcount()
num_edges = G.ecount()
# (Optional) Print some graph statistics
print(f"Loaded Graph:")
print(f"  - Data name: {data_name}")
print(f"  - No. of nodes: {num_nodes:,}")
print(f"  - No. of edges: {num_edges:,}")
print(f"  - Nodes data: {[(v.index, v.attributes()) for v in G.vs[:5]]}...")
print(f"  - Edges data: {[(e.source, e.target, e.attributes()) for e in G.es[:5]]}...")
metrics['num_nodes'] = num_nodes
metrics['num_edges'] = num_edges


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

train_dataset, val_dataset, test_dataset = load_dataset(dir_name=query_dir, seed=seed, drop_duplicates=False)
print("Train dataset...")
print(f"  - No. of samples: {len(train_dataset):,}")
print(f"  - Min/Max distance: {train_dataset.D.min():.2f}/{train_dataset.D.max():.2f}")
print(f"  - Mean/Std distance: {train_dataset.D.mean():.2f}/{train_dataset.D.std():.2f}")
max_distance = train_dataset.D.max()
metrics['max_distance'] = float(max_distance)  # meters
metrics['train_data_size'] = len(train_dataset)
metrics['train_data_distance_mean'] = float(train_dataset.D.mean())  # meters
print(f"Test dataset...")
print(f"  - No. of samples: {len(test_dataset):,}")
print(f"  - Min/Max distance: {test_dataset.D.min():.2f}/{test_dataset.D.max():.2f}")
print(f"  - Mean/Std distance: {test_dataset.D.mean():.2f}/{test_dataset.D.std():.2f}")
metrics['test_data_size'] = len(test_dataset)
metrics['test_data_distance_mean'] = float(test_dataset.D.mean())  # meters
print(f"Validation dataset...")
print(f"  - No. of samples: {len(val_dataset):,}")
print(f"  - Min/Max distance: {val_dataset.D.min():.2f}/{val_dataset.D.max():.2f}")
print(f"  - Mean/Std distance: {val_dataset.D.mean():.2f}/{val_dataset.D.std():.2f}")
metrics['val_data_size'] = len(val_dataset)
metrics['val_data_distance_mean'] = float(val_dataset.D.mean())  # meters

# Create dataloaders
pin_memory = True
if sys.platform == "darwin":
    num_workers = 0  # often faster on macOS
    print_warning("macOS detected: Setting num_workers=0 for DataLoader to avoid multiprocessing issues.")
else:
    num_workers = num_cores
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
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=pin_memory)
    print(f"  - Batch size: {batch_size_val}")
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
                     seed=seed,                         ## Random seed
                     kmeans_batch_size=kmeans_batch_size ## Optional MiniBatchKMeans batch size
                     )
elif model_class == 'ndist2vec':
    from models.ndist2vec import Ndist2vec

    # Initialize model
    model = Ndist2vec(n_input=num_nodes,                ## No. of Embddings
                      n_hidden_1=embedding_dim,         ## Embedding size
                      n_hidden_2=100,                   ## Hidden layer 1 size
                      n_hidden_3=20,                    ## Hidden layer 2 size
                      n_output=1,                       ## Output size
                      max_distance=max_distance,        ## Maximum distance for scaling
                      sparse=sparse_embedding           ## Sparse embedding
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
                      max_distance=max_distance,        ## Maximum distance for scaling
                      sparse=sparse_embedding           ## Sparse embedding
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
                        normalize=True,                             ## Normalize embeddings
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
                       aggregation_method=aggregation_method,       ## Aggregation method
                       normalize=True                               ## Normalize embeddings
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
                  p=p_norm,                                         ## p-value for Lp norm (if applicable)
                  sparse=sparse_embedding                           ## Sparse embedding
                  )
elif model_class == 'path2vec':
    from models.path2vec import Path2vec

    # Initialize model
    model = Path2vec(G=G,                               ## Input graph
                     embed_size=embedding_dim,          ## Embedding size
                     max_distance=max_distance,         ## Maximum distance for scaling
                     regularize=False,                  ## Whether to use regularization
                     l1factor=1e-10,                    ## L1 regularization factor
                     use_neighbors=True,                ## Whether to use neighbors
                     neighbor_count=5,                  ## Number of neighbors to consider
                     alpha=0.5,                         ## Alpha parameter
                     sparse=sparse_embedding            ## Sparse embedding
                     )
elif model_class == 'rne':
    from models.rne import RNE

    # Load parts information (if available)
    parts = read_parts_file(data_dir, data_name)

    # Initialize model
    model = RNE(num_nodes=num_nodes,                                ## No. of nodes
                embed_size=embedding_dim,                           ## Embedding size
                max_distance=train_dataset.D.mean(),                ## TODO: need to check if max_distance is better or mean_distance
                parts=parts,                                        ## Parts information (if available)
                sparse=sparse_embedding)                            ## Sparse embedding
elif model_class == 'catboost':
    from models.catboostmodel import CatBoostModel

    # Load custom node embeddings
    assert embedding_filename is not None, "Provide an embedding filename for `catboost` model."
    custom_node_embeddings = read_embedding_file(embedding_filename)


    # Initialize model
    model = CatBoostModel(num_nodes=num_nodes,                      ## No. of nodes
                          coordinate_embs=node_attributes,          ## Precomputed node embeddings (coordinates)
                          landmark_embs=custom_node_embeddings,     ## Precomputed node embeddings (landmarks distances)
                          thread_count=num_cores                    ## Threads for CatBoost training
                          )
elif model_class == 'catboostnn':
    from models.catboostnn import CatBoostNN

    # Load custom node embeddings
    assert embedding_filename is not None, "Provide an embedding filename for `catboostnn` model."
    custom_node_embeddings = read_embedding_file(embedding_filename)

    # Initialize model
    model = CatBoostNN(num_nodes=num_nodes,                         ## No. of nodes
                       coordinate_embs=node_attributes,             ## Precomputed node embeddings (coordinates)
                       landmark_embs=custom_node_embeddings,        ## Precomputed node embeddings (landmarks distances)
                       max_distance=train_dataset.D.mean()          ## Mean distance for scaling
                       )
else:
    raise ValueError(f"Unknown model class: {model_class}")

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Model Summary:")
print(model)
print(f"Model parameters size: {num_params}")
metrics['model_params'] = num_params

# %%
################
# Load model from checkpoint
################

if checkpoint_path is not None:
    checkpoint = load_model(model,
                            file_name=checkpoint_path,
                            model_class=model_class)

# %%
################
# Loss function, optimizer and device setup
################

# Compile the model for better performance
if compile_mode != 'none':
    model = torch.compile(model, mode=compile_mode)

# Initialize loss function
criterion = get_criterion(loss_function, model)
print(f"Loss function: {criterion}")
metrics['loss_function'] = str(criterion).strip("()")

# Initialize optimizer
optimizer = get_optimizer(optimizer_type, model, learning_rate)
print(f"Optimizer: {optimizer}")
metrics['optimizer'] = str(optimizer).split()[0]

# Check available device
print(f"Using Device: {device}")
print_device_info(device)
metrics['device'] = device

# %%
################
# Training Loop
################

if inference_only:
    print("Skipping training (--inference_only)")
    train_history = {
        "loss_epoch_history": [],
        "loss_iter_history": [],
        "val_mre_epoch_history": [],
        "time_history": [],
    }
    precomputation_time = 0.0
else:
    reset_peak_gpu_stats(device)
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
metrics['precomputation_time']  = precomputation_time / 60  # minutes
metrics['train_loss_history']   = train_history['loss_epoch_history']
metrics['val_mre_history']      = [float(v) * 100 for v in train_history['val_mre_epoch_history']]  # percent
metrics['time_elapsed_history'] = train_history['time_history']  # minutes
metrics['epoch_history']        = list(range(1, len(train_history['loss_epoch_history']) + 1))
metrics['last_train_epoch']     = len(train_history['loss_epoch_history']) or None
metrics['val_best_mre']         = float(min(train_history['val_mre_epoch_history'])) * 100 if train_history['val_mre_epoch_history'] else None  # percent

if device.startswith('cuda'):
    print(f"Peak GPU memory (train): {torch.cuda.max_memory_allocated(device)/1024**3:.2f} GB")
    metrics['max_gpu_memory_train'] = torch.cuda.max_memory_allocated(device)/1024**3  # GB

if inference_only:
    print(f"Skipping model save (--inference_only): using loaded checkpoint `{checkpoint_path}`")
    metrics['model_path'] = checkpoint_path
    metrics['model_size'] = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
    metrics['jit_model_path'] = None
    metrics['jit_model_size'] = None
else:
    # Save the model
    start_time = time.perf_counter()
    model_path = os.path.join(SAVED_MODELS_DIR, f"{model_name}_{data_name}_{query_name}.pt")
    model_size_bytes = save_model(model,
                                  file_name=f"{model_name}_{data_name}_{query_name}.pt",
                                  dir_name=SAVED_MODELS_DIR,
                                  model_class=model_class)
    elapsed_time = time.perf_counter() - start_time
    print(f"Model `.pt` save time: {elapsed_time / 60:.2f} minutes")
    metrics['model_path'] = model_path
    metrics['model_size'] = model_size_bytes / (1024 * 1024)  # MB

    # Save TorchScript model for C++ inference
    start_time = time.perf_counter()
    jit_model_path = os.path.join(SAVED_JIT_MODELS_DIR, f"{model_name}_{data_name}_{query_name}.jit.pt")
    jit_size_bytes = save_jit_model(model,
                                    file_name=f"{model_name}_{data_name}_{query_name}.jit.pt",
                                    dir_name=SAVED_JIT_MODELS_DIR,
                                    model_class=model_class)
    elapsed_time = time.perf_counter() - start_time
    print(f"JIT Model `.jit.pt` save time: {elapsed_time / 60:.2f} minutes")
    metrics['jit_model_path'] = jit_model_path if jit_size_bytes else None
    metrics['jit_model_size'] = (jit_size_bytes / (1024 * 1024)) if jit_size_bytes else None  # MB

    # (Optional) Plot epoch and iteration losses during training
    plot_learning_curves(train_history, n_batches=len(train_dataloader),
                         model_name=model_name, data_name=data_name, query_name=query_name,
                         dir_name=PLOTS_DIR)

# %%
################
# Evaluate the model
################

print("Starting evaluation...")
# (Optional) Set the precision for matrix multiplication (lower precision implies higher query latency)
if matmul_precision is not None:
    print(f"Setting float32 matmul precision to: {matmul_precision}")
    torch.set_float32_matmul_precision(matmul_precision)

for label_i, dataloader_i in zip(["train", "test"], [train_dataloader, test_dataloader]):
    reset_peak_gpu_stats(device)
    start_time = time.perf_counter()
    predictions, targets, query_latency = model.evaluate(dataloader=dataloader_i,
                                                         max_distance=max_distance,
                                                         device=device)
    end_time = time.perf_counter()
    evaluation_time = end_time - start_time
    print(f"Evaluation on {label_i} Finished!")
    print(f"Evaluation time: {evaluation_time / 60:.2f} minutes")
    metrics[f'{label_i}_evaluation_time'] = evaluation_time / 60  # minutes
    if device.startswith('cuda'):
        print(f"Peak GPU memory (eval_{label_i}): {torch.cuda.max_memory_allocated(device)/1024**3:.2f} GB")
        metrics[f'max_gpu_memory_eval_{label_i}'] = torch.cuda.max_memory_allocated(device)/1024**3  # GB

    # Calculate the mae, mre
    abs_errors = np.abs(predictions - targets)
    rel_errors = abs_errors / np.maximum(targets, 1e-6)  # Avoid division by zero
    mae = np.mean(abs_errors)
    mre = np.mean(rel_errors)
    print(f"Mean Absolute Error: {mae:.2f} (p{np.mean(abs_errors <= mae) * 100:.0f})")
    print(f"Mean Relative Error: {mre:.2%} (p{np.mean(rel_errors <= mre) * 100:.0f})")
    print_error_distribution(abs_errors, rel_errors)
    metrics[f'{label_i}_mae'] = float(mae)  # meters
    metrics[f'{label_i}_mre'] = float(mre) * 100  # percent
    percentiles = [0, 25, 50, 75, 80, 90, 95, 99, 100]
    labels = ["min", "p25", "p50", "p75", "p80", "p90", "p95", "p99", "max"]
    metrics[f'{label_i}_mae_percentiles'] = dict(zip(labels, np.percentile(abs_errors, percentiles).tolist()))  # meters
    metrics[f'{label_i}_mre_percentiles'] = dict(zip(labels, (np.percentile(rel_errors, percentiles) * 100).tolist()))  # percent
    print(f"Query time per sample: {query_latency * 1_000_000:.3f} microseconds")
    print(f"Adjusted query time per sample: {evaluation_time / len(dataloader_i.dataset) * 1_000_000:.3f} microseconds")
    metrics[f'{label_i}_query_time'] = query_latency * 1_000_000  # microseconds

    # Local MRE per equal-width distance bucket (mirrors plot_targets_and_mre_boxplots binning)
    num_buckets = 5
    bucket_edges = np.linspace(targets.min(), targets.max(), num_buckets + 1, dtype=float)
    for b in range(num_buckets):
        start, end = bucket_edges[b], bucket_edges[b + 1]
        bucket_indices = (targets >= start) & (targets < end)
        metrics[f'{label_i}_mre_bucket{b + 1}'] = float(rel_errors[bucket_indices].mean() * 100)  # percent

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
    # ── DataLoader inference (Batched) ───────────────────────────────
    print(f"\n[DataLoader inference, batch_size={batch_size_test}] Running {eval_runs} eval runs...")
    dataloader_latencies = []
    for i in range(eval_runs):
        predictions, targets, query_latency = model.evaluate(dataloader=test_dataloader,
                                                             max_distance=max_distance,
                                                             device=device)
        dataloader_latencies.append(query_latency)
        print(f"  [DataLoader run {i+1:2d}] {query_latency * 1_000_000:.4f} µs/query")

    dataloader_avg = np.mean(dataloader_latencies[-5:]) * 1_000_000
    dataloader_std = np.std(dataloader_latencies[-5:])  * 1_000_000
    print(f"  [DataLoader avg (last 5)] {dataloader_avg:.4f} ± {dataloader_std:.4f} µs/query")
    metrics['mean_query_time'] = float(dataloader_avg)  # microseconds
    metrics['std_query_time'] = float(dataloader_std)  # microseconds

    # # ── Direct tensor inference (Non-Batched) ──────────
    # print(f"\n[Direct tensor inference, {len(test_dataset.queries)} queries] Running {eval_runs} eval runs...")
    # direct_tensor_latencies = []
    # model.eval()
    # model.to(device)
    # queries_direct = test_dataset.queries
    # for i in range(eval_runs):
    #     shuffled_indices = np.random.permutation(len(queries_direct))
    #     src_cpu = torch.tensor(queries_direct[shuffled_indices, 0], dtype=torch.long)
    #     dst_cpu = torch.tensor(queries_direct[shuffled_indices, 1], dtype=torch.long)
    #     start_time = time.perf_counter()
    #     src_indices = src_cpu.to(device)
    #     dst_indices = dst_cpu.to(device)
    #     with torch.no_grad():
    #         preds = model.forward(src_indices, dst_indices)
    #     _ = preds.cpu().numpy()
    #     if device.startswith('cuda'):
    #         torch.cuda.synchronize()
    #     elapsed = time.perf_counter() - start_time
    #     query_latency = elapsed / len(queries_direct)
    #     direct_tensor_latencies.append(query_latency)
    #     print(f"  [Direct tensor run {i+1:2d}] {query_latency * 1_000_000:.4f} µs/query")

    # direct_tensor_avg = np.mean(direct_tensor_latencies[-5:]) * 1_000_000
    # direct_tensor_std = np.std(direct_tensor_latencies[-5:])  * 1_000_000
    # print(f"  [Direct tensor avg (last 5)] {direct_tensor_avg:.4f} ± {direct_tensor_std:.4f} µs/query")

# %%
################
# CPU memory (matches `/usr/bin/time %M` used in SLURM scripts and master logs)
################

print(f"Peak CPU memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2:.2f} GB")
metrics['max_cpu_memory'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2  # GB

# %%
################
# Save collected metrics
################

save_metrics_json(metrics,
                  file_name=f"metrics_{model_name}_{data_name}_{query_name}.json",
                  dir_name=SAVED_METRICS_DIR)
