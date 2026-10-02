import torch
import torch.nn as nn

from models.basemodel import BaseModel
from utils.data_utils import select_landmarks, compute_landmark_distances


class Landmark(BaseModel):
    def __init__(self, graph, num_landmarks, strategy="random", weight_key="weight", seed=19, subset=None, node_features=None, kmeans_batch_size=None):
        super().__init__()
        self.graph = graph
        self.num_nodes = graph.vcount()
        self.num_landmarks = num_landmarks
        self.strategy = strategy
        self.weight_key = weight_key
        self.seed = seed
        self.subset = subset
        # Kmeans strategy specific parameters
        self.node_features = node_features  # Optional node features
        self.kmeans_batch_size = kmeans_batch_size  # Optional batch size for MiniBatchKMeans

        ## These will be overwritten during the fit() method
        # Initializing landmarks with None
        self.landmarks = [None]*self.num_landmarks
        # Initialize the distance matrix with random values
        self.embedding = nn.Embedding.from_pretrained(torch.rand(self.num_nodes, self.num_landmarks), freeze=True)

    def forward(self, x1, x2):
        """
        For each pair of nodes in x1 and x2, look up their precomputed landmark distances,
        compute the minimum distance to any landmark

        Args:
            x1, x2 (torch.LongTensor): Tensor of node indices corresponding to graph nodes.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, 1) containing the computed distances.
        """
        # Lookup the landmark distance rows for x1 and x2
        d1 = self.embedding(x1)  # Shape: (batch_size, n_landmarks)
        d2 = self.embedding(x2)  # Shape: (batch_size, n_landmarks)

        # Return the minimum distance to any landmarks
        out, _ = torch.min(d1+d2, dim=1)  # Alternatively, could use keepdim=True
        out = out.view(-1, 1)  # Reshape to (batch_size, 1)
        return out

    def fit(self, epochs=1, **kwargs):
        """
        Preprocess the graph: selects landmarks and builds the distance matrix.
        Since no training is required, this function only computes the necessary preprocessing.
        Returns:
            torch.Tensor: The computed landmark distance matrix.
        """
        # Skip training if epochs is 0 or negative
        if epochs <= 0:
            return {
                "loss_epoch_history": [],
                "loss_iter_history": [],
                "val_mre_epoch_history": [],
                "time_history": [],
            }

        # Set the model to training mode
        self.train()

        # Select landmarks based on the chosen strategy
        self.landmarks = select_landmarks(graph=self.graph,
                                            num_landmarks=self.num_landmarks,
                                            strategy=self.strategy,
                                            weight_key=self.weight_key,
                                            seed=self.seed,
                                            subset=self.subset,
                                            node_features=self.node_features,
                                            kmeans_batch_size=self.kmeans_batch_size)

        # Compute distances to selected landmarks
        node_to_landmark_distances = compute_landmark_distances(graph=self.graph,
                                                                landmarks=self.landmarks,
                                                                weight_key=self.weight_key)

        # Update the embedding layer with the computed distances
        self.embedding.weight.data.copy_(torch.from_numpy(node_to_landmark_distances).float())

        return {
            "loss_epoch_history": [],
            "loss_iter_history": [],
            "val_mre_epoch_history": [],
            "time_history": [],
        }
