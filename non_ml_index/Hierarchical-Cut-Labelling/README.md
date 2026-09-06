<!-- Adapted from: https://github.com/Jiboxiake/Hierarchical-Cut-Labelling -->

# Hierarchical Cut Labelling

## Usage

All binaries are compiled into `./build/` and indexes are saved to `./saved_indexes/`.

### Compile (one-time)

```bash
make compile
```

Builds all binaries: `cut`, `index`, `query`, `topcut`, `test`.

### Build index for one dataset

```bash
make build DATA_NAME=W_Jinan
```

Reads `../../../data/W_Jinan/W_Jinan.edges`, writes `./saved_indexes/W_Jinan.hl`.
Override `DATA_DIR` if your data lives elsewhere:

```bash
make build DATA_NAME=W_Jinan DATA_DIR=/path/to/data
```

### Evaluate (1M random queries)

```bash
make evaluate DATA_NAME=W_Jinan
```

Loads the saved index and reports query latency and avg hoplinks.

### Full pipeline for one dataset

```bash
make all DATA_NAME=W_Jinan
```

### Run all datasets

```bash
bash run_all.sh
# or with a custom data dir:
bash run_all.sh /path/to/data
```

Loops over all 7 datasets (W_Jinan, W_Shenzhen, W_Chengdu, W_Beijing, W_NewYork, W_Chicago, W_Shanghai), running `build` and `evaluate` for each. Prints build time per dataset.

### Clean

```bash
make clean   # removes build/ and saved_indexes/ contents
```

---

## Source files

| File | Binary | Purpose |
|------|--------|---------|
| `index.cpp` | `build/index` | Build index from `.edges` file and write to `.hl` |
| `query.cpp` | `build/query` | Load `.hl` index, run 1M random queries, report latency |
| `main.cpp` | `build/cut` | End-to-end experiment runner (build + query in one pass) |
| `topcut.cpp` | `build/topcut` | Compute top-level cut/partition only (debug/analysis) |
| `test.cpp` | `build/test` | Correctness tests on synthetic random graphs |
| `road_network.h/cpp` | — | Core graph + index library |
| `util.h/cpp` | — | Timers and utilities |

Index files (`.hl`) are written and read in **binary format** (`write()` / `ContractionIndex(istream&)`).

---

<!-- OLD README -->

## Original README

A tool for indexing undirected edge-weighted graphs, such as road networks, to speed up distance queries.
It consists of the following main files:

* road_network.h / road_network.cpp: core library
* util.h / util.cpp: library with additional tools
* main.cpp: run experiments on one or more graphs and summarize results

Additional files are:

* index.cpp: create an index file
* query.cpp: load index from a file and evaluate random queries
* topcut.cpp: compute top-level cut & partition only
* test.cpp: run basic integration tests

### Compile & Run

Files can be compiled using `make`.
Experiments can then be run with

```
./cut [beta] graph_1.gr ... graph_n.gr
```

The balance parameter `beta` is optional, with 0.2 as default.
Graph files for testing, as well as a description of the expected file format, can be found at http://www.diag.uniroma1.it/~challenge9/download.shtml.
