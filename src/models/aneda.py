"""
References:
    [1] Paper: ANEDA: Adaptable Node Embeddings for Shortest Path Distance Approximation (HPEC 2023)
    [2] Original implementation: https://github.com/frankpacini/ANEDA/blob/path_search/src/aneda.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.basemodel import BaseModel


class ANEDA(BaseModel):
    def __init__(self, num_nodes, embed_size, init_embeddings=None, max_distance=1.0, distance_measure="inv_dotproduct", p=1, sparse=False):
        """
        Initializes the ANEDA model.

        Args:
            p (int or float): The order of the norm (e.g., 2 for Euclidean distance, 1 for Manhattan distance).
        """
        super().__init__()
        self.max_distance = max_distance
        self.distance_measure = distance_measure
        self.p = p
        print(f"Initializing ANEDA...")
        print(f"  - Number of nodes: {num_nodes}")
        print(f"  - Embedding size: {embed_size}")
        print(f"  - Max distance: {max_distance}")
        print(f"  - Distance measure: {distance_measure}")
        print(f"  - P: {p}")
        print(f"  - Sparse: {sparse}")

        ## Define layers
        # Embedding layer
        if init_embeddings is None:
            self.embedding = nn.Embedding(num_nodes, embed_size, sparse=sparse)
            print(f"  - Node embeddings: randomly initialized, shape {tuple(self.embedding.weight.shape)}")
        else:
            init_embeddings = torch.from_numpy(init_embeddings).float()  # Convert node attributes to tensor
            assert init_embeddings.shape[0] == num_nodes, f"Expected {num_nodes} nodes, but got {init_embeddings.shape[0]}."
            assert init_embeddings.shape[1] == embed_size, f"Expected embedding size {embed_size}, but got {init_embeddings.shape[1]}."
            self.embedding = nn.Embedding.from_pretrained(init_embeddings, freeze=False, sparse=sparse)  # Make it trainable
            print(f"  - Node embeddings: from provided attributes, shape {tuple(self.embedding.weight.shape)}")

    def forward(self, x1, x2):
        """
        Computes the Lp norm between node embeddings.

        Args:
            x1 (torch.Tensor): Node indices for the first set of nodes.
            x2 (torch.Tensor): Node indices for the second set of nodes.

        Returns:
            torch.Tensor: Lp norm between the embeddings.
        """
        # Embedding layer
        x1 = self.embedding(x1)
        x2 = self.embedding(x2)

        # Compute distance based on distance measure
        if self.distance_measure == "norm":
            # NVFuser-fuseable paths for p=1,2; torch.norm for general p
            if self.p == 1:
                x = torch.abs(x1 - x2).sum(dim=1, keepdim=True)
            elif self.p == 2:
                x = (x1 - x2).square().sum(dim=1, keepdim=True).sqrt()
            else:
                x = torch.norm(x1 - x2, p=self.p, dim=1, keepdim=True)
        elif self.distance_measure == "inv_dotproduct":
            # Equivalent to: x = (1 - dotproduct(u, v)/(norm(u)*norm(v)))*self.max_distance/2
            # Cosine similarity: square().sum().sqrt() is NVFuser-fuseable unlike F.cosine_similarity's internal norm
            dot = (x1 * x2).sum(dim=1)
            norm = (x1.square().sum(dim=1).sqrt() * x2.square().sum(dim=1).sqrt()).clamp(min=1e-8)
            x = (1 - dot / norm).unsqueeze(-1) * self.max_distance / 2  # Shape: (B, 1)
        elif self.distance_measure == "dotproduct":
            # Cosine similarity: square().sum().sqrt() is NVFuser-fuseable unlike F.cosine_similarity's internal norm
            dot = (x1 * x2).sum(dim=1)
            norm = (x1.square().sum(dim=1).sqrt() * x2.square().sum(dim=1).sqrt()).clamp(min=1e-8)
            x = (1 + dot / norm).unsqueeze(-1) * self.max_distance / 2  # Shape: (B, 1)
        return x
