# HCL.ipynb

## Overview

This project provides a Python implementation of **Hierarchical Contraction Labeling (HCL)** for efficient shortest path queries on large graphs, such as road networks. The main logic is contained in `HCL.ipynb`, which includes classes for parsing label files, computing distances, and running experiments.

## Features

- **FlatCutIndex**: Encodes hierarchical labels for each node.
- **ContractionLabel**: Stores label and parent information for each node.
- **ContractionIndex**: Supports efficient distance queries between nodes.
- **Experimentation**: Functions to evaluate query performance and runtime.
- **PyTorch Integration**: Dataset class for ML-based experiments.

## Usage

### Requirements

- Python 3.7+
- `torch` (for ML experiments)
- Jupyter Notebook or VS Code

### Running Experiments

1. **Prepare label files** We used the C++ preprocessing code from the following codebase: https://github.com/Jiboxiake/Hierarchical-Cut-Labelling
2. **Edit paths** in `HCL.ipynb` to point to your `file_paths.txt` which stores all paths to the precomputed labels for all road network datasets
3. **Run the notebook** or use the provided functions

### Measuring Query Performance

The notebook includes code to measure average query time and total runtime for a batch of queries.

## File Structure

- `HCL.ipynb` - Main implementation and experiments
- `readme.md` - This documentation file
- `file_paths.txt` - List of label files for batch experiments (optional)

## Example
```python
# Example: Run experiment on a label file and query file
label_list = parse_hcl_file("USA-road-d.NY.gr-label.hl")
hci = ContractionIndex(label_list)
experiment(hci, "NY_queries.txt")
```
```python
# Example: Run experiment on a single precomputed label file and use torch-generated queries
experimental_evaluation_torch(10000,"path-to-your-labels")
```
```python
# Example: Run experiments using a file storing paths to all precomputed labels
import gc
print("No-ML experiments started")
num_queries = 100000
filename = "path-to-your-path-file"  # Path to the file containing labeling files
with open(filename, 'r') as f:
    for line in f:
        file_path = line.strip()
        if file_path:
            print(f"Processing labeling file: {file_path}")
            experimental_evaluation_torch(num_queries, file_path)
            print(f"Finished processing labeling file: {file_path}")
            print("-" * 40)
            c = gc.collect()
print("No-ML experiments finished")
```

## License

MIT License (add your license here if different)

## Acknowledgements

- Based on Hierarchical Cut Labling: https://github.com/henningkoehlernz/road-networks
- Uses GitHub Copilot to aid the C++ to Python conversion
- Uses PyTorch for optional ML-based experiments.
