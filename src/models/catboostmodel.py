"""
References:
    [1] Paper: Shortest Path Distance Prediction Based on CatBoost (WISA 2021)
    [2] Original implementation: (None found)

TODO:
    [ ] Add one more catboost model in serial fashion as given in the paper.
    [ ] Try with landmark chosen by k-Means (as suggested in the paper) and with subset of train nodes.
    [ ] Use 2 serial catboost models as mentioned in the paper.
"""

import os
import shutil
import tempfile
import time

import numpy as np

from catboost import CatBoostRegressor, Pool
from catboost.utils import get_gpu_device_count

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.basemodel import BaseModel


# Define the CatBoost model
class CatBoostModel(BaseModel):
    def __init__(self, num_nodes, coordinate_embs, landmark_embs, seed=42, thread_count=-1):
        super().__init__()

        self.seed = seed
        self.thread_count = thread_count

        # Embedding layers
        print(f"Coordinate embeddings shape: {coordinate_embs.shape}")
        print(f"Landmark embeddings shape: {landmark_embs.shape}")
        self.coordinate_embs = nn.Embedding.from_pretrained(torch.from_numpy(coordinate_embs).float(), freeze=True)
        self.landmark_embs = nn.Embedding.from_pretrained(torch.from_numpy(landmark_embs).float(), freeze=True)

        # CatBoost model
        self.catboost_model = None

    def encode(self, x1, x2):
        # Convert x1 and x2 to embeddings
        if self.coordinate_embs is not None:
            x1_coord_emb = self.coordinate_embs(x1)
            x2_coord_emb = self.coordinate_embs(x2)
        if self.landmark_embs is not None:
            x1_landmark_emb = self.landmark_embs(x1)
            x2_landmark_emb = self.landmark_embs(x2)

        # Compute cosine similarity between landmark embeddings
        if self.landmark_embs is not None:
            cosine_sim = F.cosine_similarity(x1_landmark_emb, x2_landmark_emb, dim=-1).unsqueeze(-1)
        elif self.coordinate_embs is not None:
            cosine_sim = F.cosine_similarity(x1_coord_emb, x2_coord_emb, dim=-1).unsqueeze(-1)
        else:
            cosine_sim = None

        # Compute euclidean distance between coordinate embeddings
        if self.coordinate_embs is not None:
            # Manhattan (L1) distance; abs().sum() is NVFuser-fuseable unlike torch.norm(p=1)
            euclidean_dist = torch.abs(x1_coord_emb - x2_coord_emb).sum(dim=-1, keepdim=True)
        elif self.landmark_embs is not None:
            euclidean_dist = torch.abs(x1_landmark_emb - x2_landmark_emb).sum(dim=-1, keepdim=True)
        else:
            euclidean_dist = None

        # Concatenate features, cast to CPU and convert to numpy
        features = []
        if self.landmark_embs is not None:
            features.append(x1_landmark_emb)
            features.append(x2_landmark_emb)
        if self.coordinate_embs is not None:
            features.append(x1_coord_emb)
            features.append(x2_coord_emb)
        if cosine_sim is not None:
            features.append(cosine_sim)
        if euclidean_dist is not None:
            features.append(euclidean_dist)
        features = torch.cat(features, dim=-1).numpy()

        return features

    def forward(self, x1, x2):
        # Compute features
        features = self.encode(x1, x2)

        # Use catboost model to predict distances
        predictions = self.catboost_model.predict(features)

        # Convert predictions to tensor
        predictions = torch.from_numpy(predictions).float().unsqueeze(-1)

        return predictions

    def iter_encoded_batches(self, dataloader, device, chunk_size):
        """Yield (features, targets) numpy arrays of ~chunk_size samples, encoding on the fly."""
        # Pre-allocate fixed-size buffers once and reuse across chunks
        features = None
        targets  = None
        write_idx = 0

        # Extract features and targets from dataloader
        for idx, (i, j, d_ij) in enumerate(dataloader):
            i, j = i.to(device), j.to(device)    ## Move data to device
            features_batch = self.encode(i, j)
            targets_batch = d_ij.cpu().numpy()   ## Move targets to CPU
            n = len(d_ij)

            if features is None:
                features = np.empty((chunk_size, features_batch.shape[1]), dtype=features_batch.dtype)
                targets  = np.empty((chunk_size,),                         dtype=targets_batch.dtype)

            # Flush chunk before writing if this batch would overflow
            if write_idx + n > chunk_size and write_idx > 0:
                yield features[:write_idx], targets[:write_idx]
                write_idx = 0

            # Write batch into buffer in-place
            features[write_idx:write_idx + n] = features_batch   ## In-place write, no copy
            targets[write_idx:write_idx + n]  = targets_batch    ## In-place write, no copy
            write_idx += n

        if write_idx > 0:
            yield features[:write_idx], targets[:write_idx]

    def fit(self, dataloader=None, val_dataloader=None, epochs=1, learning_rate=0.1, device="cpu", **kwargs):
        # Skip training if epochs is 0 or negative
        if epochs <= 0:
            return {
                "loss_epoch_history": [],
                "loss_iter_history": [],
                "val_mre_epoch_history": [],
                "time_history": [],
            }

        # Time limit: train in chunks of CHUNK_SIZE trees so elapsed time can be checked between chunks.
        # SIGALRM cannot be used here — CatBoost's C++ backend clears tree state on interrupt.
        # CatBoost's "remaining" estimate in verbose output restarts each chunk (cosmetic only).
        time_limit_mins = kwargs.get('time_limit')
        time_limit_secs = time_limit_mins * 60 if time_limit_mins is not None else None
        CHUNK_SIZE = 500          ## Trees per chunk; gives ~5 verbose lines and ±30s time granularity
        VERBOSE_INTERVAL = 100    ## Print progress every N trees within a chunk (matches original)
        # NOTE: This batch size is for incremental training of CatBoost model, independent of PyTorch batch size.
        # Encoding all samples upfront causes OOM on large datasets (e.g., landmark_30M); stream in chunks instead.
        DATA_CHUNK_SIZE = 1_000_000  # ~0.5 GB peak per chunk at 128 features/sample
        fit_start = time.perf_counter()

        # Validation dataloader
        val_pool = None
        if val_dataloader is not None:
            for features_val, targets_val in self.iter_encoded_batches(val_dataloader, device, DATA_CHUNK_SIZE):
                val_pool = Pool(data=features_val, label=targets_val)
                break  ## First chunk only; Pool is for eval_set monitoring, not final metrics

        # Use a per-process unique temp dir so concurrent jobs on the same node do not
        # collide on CatBoost's train_dir metadata files or the inter-batch checkpoint file.
        tmp_dir = tempfile.mkdtemp(prefix='catboost_')
        model_file = os.path.join(tmp_dir, 'catboost_model.cbm')

        current_model = None   ## Accumulates all trained trees via init_model chaining
        total_trained = 0
        time_limit_hit = False
        model_size_mb = None
        loss_epoch_history = []    ## RMSE (loss_function), one value per epoch (here, one chunk)
        loss_iter_history = []     ## RMSE, one value per iteration (here, one tree)
        val_mre_epoch_history = [] ## MAPE (eval_metric) on val_pool, one value per epoch
        time_history = []          ## Cumulative elapsed minutes, one value per epoch

        # Train catboost model in data chunks (encoding on the fly) to avoid OOM on large datasets
        print("Training CatBoost model...")

        try:
            pass_idx = 0
            while not time_limit_hit and total_trained < epochs:
                for chunk_idx, (features, targets) in enumerate(
                        self.iter_encoded_batches(dataloader, device, DATA_CHUNK_SIZE)):
                    if time_limit_hit or total_trained >= epochs:
                        break

                    elapsed = time.perf_counter() - fit_start
                    if time_limit_secs is not None and elapsed >= time_limit_secs:
                        print(f"CatBoost time limit reached: {total_trained}/{epochs} trees trained "
                              f"({elapsed / 60:.2f} min)")
                        time_limit_hit = True
                        break

                    print(f"Training on data chunk {chunk_idx} pass {pass_idx} ({features.shape[0]} samples)...")
                    batch_pool = Pool(data=features, label=targets)

                    # Train CHUNK_SIZE trees on this data chunk, checking time limit after
                    chunk = min(CHUNK_SIZE, epochs - total_trained)
                    chunk_model = CatBoostRegressor(
                        iterations=chunk,              ## Trees to train in this chunk
                        learning_rate=learning_rate,
                        random_seed=self.seed,
                        loss_function='RMSE',
                        task_type='CPU',               ## GPU does not support init_model (incremental training)
                        thread_count=self.thread_count,
                        verbose=VERBOSE_INTERVAL,
                        train_dir=tmp_dir,
                        eval_metric='MAPE',
                    )
                    chunk_model.fit(batch_pool, init_model=current_model, eval_set=val_pool,
                                    verbose=VERBOSE_INTERVAL)
                    current_model = chunk_model        ## Chain: next chunk continues from here
                    total_trained += chunk
                    elapsed = time.perf_counter() - fit_start
                    print(f"  [chunk done] total_trees={total_trained}/{epochs}, elapsed={elapsed / 60:.2f} min")

                    # If validation data is provided, evaluate model
                    evals_result = chunk_model.get_evals_result()
                    chunk_loss = evals_result['learn']['RMSE']
                    loss_iter_history.extend(chunk_loss)             # fine-grained: one per tree
                    loss_epoch_history.append(chunk_loss[-1])        # coarse: one per chunk/epoch
                    if val_pool is not None:
                        val_mre_epoch_history.append(evals_result['validation']['MAPE'][-1])
                    time_history.append(elapsed / 60)  # (in minutes)

                    # Save the updated model for the next data chunk
                    if current_model is not None:
                        current_model.save_model(model_file)

                    if time_limit_secs is not None and elapsed >= time_limit_secs:
                        time_limit_hit = True

                pass_idx += 1

            # Capture model size before temp dir is cleaned up
            if os.path.exists(model_file):
                model_size_mb = os.path.getsize(model_file) / (1024 * 1024)

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)  ## Always clean up temp dir

        self.catboost_model = current_model
        print(f"CatBoost model training completed: {total_trained} trees.")
        if model_size_mb is not None:
            print(f"CatBoost model size: {model_size_mb:.2f} MB")
        else:
            print("CatBoost model file not found (training may have been skipped).")

        # Cumulative summary line in CatBoost verbose format so the log parser reads
        # total_trained as epoch_last. learn: uses a non-numeric placeholder so the
        # parser's train_loss_history is unaffected (no scale mismatch with chunk values).
        if total_trained > 0:
            total_elapsed_secs = time.perf_counter() - fit_start
            print(f"{total_trained}:\tlearn: [cumulative]\ttotal: {total_elapsed_secs:.2f}s\tremaining: 0us")

        return {
            "loss_epoch_history": loss_epoch_history,
            "loss_iter_history": loss_iter_history,
            "val_mre_epoch_history": val_mre_epoch_history,
            "time_history": time_history,
        }
