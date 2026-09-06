## Directory Structure
```
└── models/                     # Directory containing model implementations
    ├── __init__.py
    ├── basemodel.py                # Base model class
    │
    │                               ## Baselines ##
    ├── lpnorm.py                   # Manhattan/Euclidean model
    ├── landmark.py                 # Landmark model
    │
    │                               ## Neural Network (NN) models ##
    ├── geodnn.py                   # GeoDNN model
    ├── distancenn.py               # DistanceNN model
    ├── embeddingnn.py              # EmbeddingNN model
    ├── vdist2vec.py                # Vdist2vec model
    ├── ndist2vec.py                # Ndist2vec model
    ├── catboostnn.py               # LandmarkNN model
    │
    │                               ## Graph Neural Network (GNN) models ##
    ├── rgnndist2vec.py             # RGNNdist2vec model
    │
    │                               ## Functional methods ##
    ├── path2vec.py                 # Path2vec model
    ├── aneda.py                    # Aneda model
    ├── rne.py                      # RNE model
    │
    │                               ## Gradient Boosting models ##
    ├── catboostmodel.py            # CatBoost model
    │
    │                               ## Misc models ##
    ├── sparse_matrix_model.py      # Sparse Matrix models
    │
    └── README.md
```

Note: `sparse_matrix_model.py` is an optional standalone script. It simulates models that store the distance matrix as a sparse structure and retrieve exact (cached) distances at query time. No learning is involved. To benchmark these models for query latency (QT) and model size (MS), run `python sparse_matrix_model.py`.
