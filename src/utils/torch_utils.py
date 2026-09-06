# import libraries
import os
import glob
import json
from tqdm import tqdm

import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, Subset

from utils.data_utils import (
    print_green,
    print_warning,
    read_query_file,
    write_query_file,
)


## Function to get the available device (MPS or GPU or CPU)
def get_available_device():
    device = None
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return device

## Function to reset peak GPU memory stats for clean per-phase profiling
def reset_peak_gpu_stats(device):
    if device and device.startswith('cuda'):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

## Function to print device information
def print_device_info(device):
    if device == 'cuda':
        print(f"Device Info: {torch.cuda.get_device_name(0)}")
    elif device == 'mps':
        print(f"Device Info: MPS (Apple Silicon GPU)")
    else:
        print(f"Device Info: CPU")

## Class to combine sparse gradient optimizer with a dense optimizer
class CombinedOptimizer:
    """Wraps SparseAdam for sparse-embedding params alongside a pre-built base optimizer for the remaining (dense) params."""
    def __init__(self, sparse_params, dense_optimizer, lr):
        self.sparse_opt = torch.optim.SparseAdam(sparse_params, lr=lr)
        self.dense_opt = dense_optimizer
        self.param_groups = self.sparse_opt.param_groups + self.dense_opt.param_groups

    def zero_grad(self):
        self.sparse_opt.zero_grad()
        self.dense_opt.zero_grad()

    def step(self):
        self.sparse_opt.step()
        self.dense_opt.step()


## Function to get optimizer
def get_optimizer(optimizer_type, model, learning_rate):
    optimizer_type = optimizer_type.lower()

    OPTIMIZER_REGISTRY = {
        'adam': optim.Adam,
        'sgd': optim.SGD,
        'rmsprop': optim.RMSprop
    }
    if optimizer_type not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    # Separate sparse and dense parameters
    sparse_params = [
        p for m in model.modules()
        if getattr(m, 'sparse', False)
        for p in m.parameters(recurse=False) if p.requires_grad
    ]
    dense_params = [
        p for p in model.parameters()
        if p.requires_grad and not any(p is sp for sp in sparse_params)
    ]

    optimizer_cls = OPTIMIZER_REGISTRY[optimizer_type]
    if sparse_params and dense_params:
        dense_optimizer = optimizer_cls(dense_params, lr=learning_rate)
        optimizer = CombinedOptimizer(sparse_params, dense_optimizer, learning_rate)
    elif sparse_params:
        optimizer = torch.optim.SparseAdam(sparse_params, lr=learning_rate)
    elif dense_params:
        optimizer = optimizer_cls(dense_params, lr=learning_rate)
    else:
        print("Model has no trainable parameters. Skipping optimizer initialization.")
        optimizer = None  # No trainable parameters

    return optimizer

## Function to get criterion
def get_criterion(loss_function, model):
    criterion = None
    has_trainable_params = any(p.requires_grad for p in model.parameters())

    if not has_trainable_params:
        print("Model has no trainable parameters. Skipping criterion initialization.")
    elif loss_function == 'mse':
        criterion = nn.MSELoss()
    elif loss_function == 'mae':
        criterion = nn.L1Loss()
    elif loss_function == 'smoothl1':
        criterion = nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unknown loss function: {loss_function}")
    return criterion


#################################################
# Torch I/O Utilities
#################################################

## Function to save a dataset to a file
def save_dataset(train_dataset, val_dataset, test_dataset, dir_name, data_name):
    """
    Save the train, validation, and test datasets to a file.
    """
    # Create the directory if it doesn't exist
    os.makedirs(dir_name, exist_ok=True)

    def _extract_queries(ds):
        # Subset is returned by torch's random_split
        if isinstance(ds, Subset):
            return ds.dataset.queries[ds.indices]
        return ds.queries

    # Save the train dataset to a file
    file_path = os.path.join(dir_name, f"{data_name}_train.queries.npz")
    write_query_file(file_path, _extract_queries(train_dataset))

    # Save the validation dataset to a file
    file_path = os.path.join(dir_name, f"{data_name}_val.queries.npz")
    write_query_file(file_path, _extract_queries(val_dataset))

    # Save the test dataset to a file
    file_path = os.path.join(dir_name, f"{data_name}_test.queries.npz")
    write_query_file(file_path, _extract_queries(test_dataset))

## Function to load a dataset from a file
def load_dataset(dir_name, test_size=0.1, val_size=0.1, seed=42, drop_duplicates=False):
    """
    Load the train, validation, and test datasets from a file.
    """
    # Load the train dataset from a file
    file_names = glob.glob(os.path.join(dir_name, "*.queries.npz"))
    file_names = [os.path.normpath(fn) for fn in file_names]
    assert len(file_names) > 0, f"No `.queries.npz` file found in {dir_name}, expected a file ending with `.queries.npz`"

    if len(file_names) == 1:
        # Single .queries.npz file: we need to do 80/10/10 split
        data_file = file_names[0]
        full_data = read_query_file(data_file, drop_duplicates=drop_duplicates)
        train_data, remainder_data = train_test_split(full_data, test_size=(test_size+val_size), random_state=seed)
        val_data, test_data = train_test_split(remainder_data, test_size=(test_size / (test_size + val_size)), random_state=seed)
    elif len(file_names) == 3:
        # Three .queries.npz files: we can use them directly as train, val, and test
        train_file = None
        val_file = None
        test_file = None
        for f in file_names:
            if f.endswith("_train.queries.npz"):
                train_file = f
            elif f.endswith("_val.queries.npz"):
                val_file = f
            elif f.endswith("_test.queries.npz"):
                test_file = f
            else:
                raise ValueError("Expected `*_train.queries.npz`, `*_val.queries.npz`, and `*_test.queries.npz` files")

        train_data = read_query_file(train_file, drop_duplicates=drop_duplicates)
        val_data = read_query_file(val_file, drop_duplicates=drop_duplicates)
        test_data = read_query_file(test_file, drop_duplicates=drop_duplicates)
    else:
        raise NotImplementedError("Loading datasets from files other than exactly one or three `.queries.npz` files is not supported.")

    # Create torch datasets
    train_dataset = WorkloadDataset(train_data)
    val_dataset = WorkloadDataset(val_data)
    test_dataset = WorkloadDataset(test_data)

    return train_dataset, val_dataset, test_dataset

## Function to save a model to a file
def save_model(model, file_name, dir_name='.', model_class=''):
    """
    Save the trained model to a `.pt` file.
    Returns the size of saved file in bytes.
    """
    # Create the directory if it doesn't exist
    os.makedirs(dir_name, exist_ok=True)

    # Save the model
    file_path = os.path.join(dir_name, file_name)
    print_green(f"Saving model: {file_path}")

    # Unwrap torch.compile wrapper if present — saves clean keys without _orig_mod. prefix
    base_model = model._orig_mod if hasattr(model, '_orig_mod') else model

    # Build the dictionary to save
    state_dict = base_model.state_dict()
    dict_to_save = {'model_state_dict': state_dict}
    if model_class.lower() == 'catboost':
        dict_to_save['catboost_model'] = base_model.catboost_model

    # Save the state dict to a file
    torch.save(dict_to_save, file_path)

    # Print size in MB of each parameter
    print("  - Model parameters:")
    total_params = 0
    for name, param in base_model.state_dict().items():
        param_size = param.numel() * param.element_size() / (1024 * 1024)  # Convert to MB
        total_params += param.numel()
        print(f"    - {name:<32}: {param_size:>8.4f} MB")
    print(f"  - Total parameters: {total_params:,}")

    # Print model size
    # Torch stores model in best possible format, so the size on disk is equivalent to the in-memory size
    file_size = os.path.getsize(file_path)
    print(f"  - Model size: {file_size / (1024 * 1024):,.2f} MB")
    return file_size

## Function to load a model from a file
def load_model(model, file_name, dir_name='.', model_class=''):
    """
    Load a trained model from a `.pt` file.
    """
    # Load the model
    file_path = os.path.join(dir_name, file_name)
    print_green(f"Reading model: {file_path}")
    checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']

    # For backward compatibility with older checkpoints where keys were saved with `_orig_mod.` prefix due to torch.compile()
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.removeprefix('_orig_mod.'): v for k, v in state_dict.items()}

    # Load the state dict into the model
    model.load_state_dict(state_dict)

    # Load CatBoost model if applicable
    if model_class.lower() == 'catboost':
        if 'catboost_model' in checkpoint:
            model.catboost_model = checkpoint['catboost_model']
        elif 'metadata' in checkpoint:
            # For backward compatibility with older checkpoints where `catboost_model` was stored under `metadata`
            model.catboost_model = checkpoint['metadata']['catboost_model']
        else:
            raise ValueError("Expected `catboost_model` in checkpoint for CatBoostNN model")
    return checkpoint


def save_jit_model(model, file_name, dir_name='.', model_class='', batch_size=1024):
    """
    Save a TorchScript (.jit.pt) model for C++ inference.
    The saved file always contains CPU tensors for portability.
    Returns the size of saved file in bytes.
    """
    # CatBoost uses a non-PyTorch forward — cannot be traced or scripted
    if model_class.lower() == 'catboost':
        return 0

    # Create the directory if it doesn't exist
    os.makedirs(dir_name, exist_ok=True)

    # Save the TorchScript model
    file_path = os.path.join(dir_name, file_name)
    print_green(f"Saving TorchScript model: {file_path}")

    # Unwrap torch.compile wrapper if present
    base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    base_model.eval()

    # Track original device
    original_device = next(base_model.parameters()).device

    # JIT compile the model; model is always saved to CPU for portability
    with torch.no_grad():
        # Get traceable model for JIT
        if model_class.lower() == 'rgnndist2vec':
            from models.rgnndist2vec import RGNNdist2vecJIT
            base_model.precompute_embeddings(batch_size=batch_size)
            trace_model = RGNNdist2vecJIT(
                base_model.cached_embeddings.cpu(),
                base_model.max_distance
            )
        else:
            base_model.to('cpu')
            trace_model = base_model
        # Construct dummy inputs
        _dummy = torch.zeros(1, dtype=torch.long)
        # Trace JIT model and save
        torch.jit.trace(trace_model, (_dummy, _dummy)).save(file_path)

    # Restore model to original device
    base_model.to(original_device)

    # Print model size
    # Torch stores model in best possible format, so the size on disk is equivalent to the in-memory size
    file_size = os.path.getsize(file_path)
    print(f"  - TorchScript Model size: {file_size / (1024 * 1024):,.2f} MB")
    return file_size

## Function to save json data to a file
def save_dictionary(data, file_name, dir_name='.'):
    """
    Save data dictionary to a torch file.
    """
    # Create the directory if it doesn't exist
    os.makedirs(dir_name, exist_ok=True)

    # Save the data to a torch file
    file_path = os.path.join(dir_name, file_name)
    print_green(f"Saving debug information: {file_path}")
    torch.save(data, file_path)

    # Print dictionary size
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
    print(f"File size: {file_size:.2f} MB")

## Function to load json data from a file
def load_dictionary(file_name, dir_name='.'):
    """
    Load data dictionary from a torch file.
    """
    # Load the data from a torch file
    file_path = os.path.join(dir_name, file_name)
    print_green(f"Reading debug information: {file_path}")
    data = torch.load(file_path, map_location='cpu')
    return data

## Function to save a flat metrics dictionary to a JSON file
def save_metrics_json(metrics, file_name, dir_name='.'):
    """
    Save a run's collected metrics/info dictionary to a JSON file.
    Values must already be native Python types (int/float/str/list/dict).
    """
    # Create the directory if it doesn't exist
    os.makedirs(dir_name, exist_ok=True)

    # Save the data to a JSON file
    file_path = os.path.join(dir_name, file_name)
    print_green(f"Saving metrics: {file_path}")
    with open(file_path, 'w') as f:
        json.dump(metrics, f, indent=2)

#################################################
# Torch Dataset Classes
#################################################

## Function to create a dataset for the all-pairs strategy
class AllPairsDataset(Dataset):
    def __init__(self, G, weight_key="weight"):
        self.n = G.vcount()
        print(f"Using all pairs strategy with n={self.n} nodes")

        # Generate all pairs (i, j) and their distances d_ij
        # TODO: use non-ml index for larger graphs
        self.queries = []
        for i, lengths in enumerate(tqdm(G.distances(weights=weight_key))):
            for j, d in enumerate(lengths):
                if i != j and d != float('inf'):
                    self.queries.append((i, j, d))
        self.queries = np.array(self.queries)
        self.D = self.queries[:, 2].reshape(-1, 1)

        # print stats
        print(f"Distance matrix of size {self.D.shape} created successfully")
        print(f"  - No. of samples: {len(self)}")
        print(f"  - Min/Max distance: {self.D.min():.2f}/{self.D.max():.2f}")
        print(f"  - Mean/Std distance: {self.D.mean():.2f}/{self.D.std():.2f}")

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        i = self.queries[idx, 0]
        j = self.queries[idx, 1]
        d_ij = self.queries[idx, 2]
        return np.int32(i), np.int32(j), np.float32(d_ij)

## Function to create a dataset for the landmark pairs strategy
class LandmarkPairsDataset(Dataset):
    def __init__(self, G, l=100, seed=42, weight_key="weight", subset_nodes=None):
        if subset_nodes is not None:
            self.n = len(subset_nodes)
            subset_nodes = [int(i) for i in subset_nodes]
            subset_nodes = set(subset_nodes)
        else:
            self.n = G.vcount()
        # checks for l
        if l < 0:
            raise ValueError(f"Landmarks l={l} is negative; it should be a positive number")
        if l == 0:
            raise ValueError(f"Landmarks l={l} is zero; it should be a positive number")
        if l < 1:
            print(f"Landmarks l={l} is a fraction; converting it to an absolute number")
            l = int(l * self.n)
        if l > self.n:
            print(f"Warning: Landmarks l={l} is greater than the number of nodes {self.n}")
            l = self.n
            print(f"Setting l to {self.n}")
        self.l = int(l)
        print(f"Using landmark-based strategy with n={self.n} nodes and l={l} landmarks")

        # select landmarks randomly
        np.random.seed(seed)
        if subset_nodes:
            self.landmarks = np.random.choice(list(subset_nodes), self.l, replace=False)
        else:
            self.landmarks = np.random.choice(self.n, self.l, replace=False)

        # create distance matrix using landmarks
        # self.D = np.zeros((self.l, self.n), dtype=np.float32)

        # self.src_nodes = []
        # self.dst_nodes = []
        # self.distances = []
        self.queries = []
        all_dists = G.distances(source=self.landmarks.tolist(), weights=weight_key)
        for i, lengths in zip(self.landmarks, all_dists):
            i = int(i)
            for j, d in enumerate(lengths):
                if (subset_nodes is None) or (j in subset_nodes):
                    if i != j and d != float('inf'):
                        self.queries.append((i, j, d))
        # self.src_nodes = np.array(self.src_nodes, dtype=np.int32)
        # self.dst_nodes = np.array(self.dst_nodes, dtype=np.int32)
        # self.distances = np.array(self.distances, dtype=np.float32)
        self.queries = np.array(self.queries, dtype=object)

        # print stats
        # print(f"Distance matrix of size {self.D.shape} created successfully")
        print(f"  - No. of samples: {len(self)}")
        print(f"  - Min/Max distance: {self.queries[:, 2].min():.2f}/{self.queries[:, 2].max():.2f}")
        print(f"  - Mean/Std distance: {self.queries[:, 2].mean():.2f}/{self.queries[:, 2].std():.2f}")

    def __len__(self):
        # return len(self.distances)
        return len(self.queries)

    # def get_indices(self, idx):
    #     i = idx // (self.n - 1)
    #     j = idx % (self.n - 1)
    #     # skip the landmark node
    #     if j >= self.landmarks[i]:
    #         j += 1
    #     return i, j

    def __getitem__(self, idx):
        # i, j = self.get_indices(idx)
        # landmark_i = self.landmarks[i]
        # range of int32 is -2B to 2B
        # return np.int32(landmark_i), np.int32(j), self.D[i, j]
        # return np.int32(self.src_nodes[idx]), np.int32(self.dst_nodes[idx]), np.float32(self.distances[idx])
        i, j, d_ij = self.queries[idx]
        return np.int32(i), np.int32(j), np.float32(d_ij)

# create a torch dataset of (i, j, d_ij) pairs
class RandomPairsDataset(Dataset):
    def __init__(self, G, k=1000, seed=42, weight_key="weight"):
        self.n = G.vcount()
        self.k = k
        if k > self.n * (self.n - 1):
            print(f"Warning: k={k} is greater than the maximum number of pairs {self.n * (self.n - 1)}")
            self.k = self.n * (self.n - 1)
            print(f"Setting k to {self.k}")
        print(f"Using random pairs strategy with n={self.n} nodes and k={k} pairs")

        # create distance matrix
        D_full = np.array(G.distances(weights=weight_key), dtype=np.float32)

        # select k random pairs
        np.random.seed(seed)
        self.random_indices = np.random.choice(self.n*(self.n-1), self.k, replace=False).astype(np.int32)

        # compute the distance for each pair
        # self.D = np.zeros(self.k, dtype=np.float32)
        self.queries = []
        for idx in range(self.k):
            i, j = self.get_indices(self.random_indices[idx])
            self.queries.append((i, j, D_full[i, j]))
        self.queries = np.array(self.queries, dtype=object)

        # print stats
        # print(f"Distance matrix of size {self.D.shape} created successfully")
        print(f"  - No. of samples: {len(self)}")
        print(f"  - Min/Max distance: {self.queries[:, 2].min():.2f}/{self.queries[:, 2].max():.2f}")
        print(f"  - Mean/Std distance: {self.queries[:, 2].mean():.2f}/{self.queries[:, 2].std():.2f}")

    def __len__(self):
        return self.k

    def get_indices(self, idx):
        i = idx // (self.n - 1)
        j = idx % (self.n - 1)
        # skip the node itself
        if j >= i:
            j += 1
        return i, j

    def __getitem__(self, idx):
        i, j, d_ij = self.queries[idx]
        return np.int32(i), np.int32(j), np.float32(d_ij)

class WorkloadDataset(Dataset):
    def __init__(self, queries, replicate=False, target_size=1_000_000):
        if replicate:
            print("Replicating queries to reach target size")
            print(f"  - Original size: {len(queries)}")
            num_copies = max(1, target_size // len(queries))
            queries = np.tile(queries, (num_copies, 1))
            print(f"  - New size:      {len(queries)}")
        self.queries = np.array(queries)
        self.D = self.queries[:, 2]

        # (Optional) Print stats
        print(f"Distance matrix of size {self.D.shape} created successfully")

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        i, j, distance = self.queries[idx]
        return np.int32(i), np.int32(j), np.float32(distance)
