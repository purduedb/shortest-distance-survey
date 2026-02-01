## Directory Structure
```
└── data_preprocess/                    # Directory containing data preprocessing scripts
    ├── __init__.py
    ├── check_and_reindex_nodes.ipynb       # Notebook to check and reindex nodes in the dataset
    ├── generate_landmark_distances.py      # Script to generate distances to landmarks
    ├── generate_parts_file_rne.py          # Script to generate parts file for RNE model
    ├── generate_query_workload.py          # Script to generate query workload
    ├── download_and_preprocess_data.py     # Script to prepare data for training
    └── README.md
```

TODO: These scripts/notebooks have been moved from `src/` to `src/data_preprocess/` for better organization. Imports from or calls to them may fail and need to be updated accordingly. For notebooks, a quick fix is to add `import sys; sys.path.append("..")` in the header.
