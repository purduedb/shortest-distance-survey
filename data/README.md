## Directory Structure
```
└── data/                           # Data Directory containing all the datasets
    ├── README.md
    ├── Surat_subgraph/                 # Sample dataset sourced from Surat road network
    ├── W_Jinan/                        # Workload-driven dataset for Jinan road network
    ├── <data_name>/                    # Directory structure for other datasets
    │   ├── raw/                            # Raw data files (e.g., downloaded from figshare, DIMACS, etc.)
    │   ├── <data_name>.edges               # Edgelist file for the dataset
    │   ├── <data_name>.nodes               # Node attributes file for the dataset
    │   ├── <data_name>.parts               # Hierarchical partitioning file for the dataset (applicable for RNE model)
    │   ├── <data_name>.embeddings          # Precomputed embeddings file for the dataset (if any)
    │   ├── <query_dir>                     # Query directory where full dataset is given
    │   │   └── <data_name>.queries             # Train & test queries in a single file
    │   ├── <query_dir2>                    # Query directory with train-test split
    │   │   ├── <data_name>_train.queries       # Train queries
    │   │   └── <data_name>_test.queries        # Test queries
    │   ├── <query_dir3>
    │   └── ...                             # and so on for other query directories (if any)
    ├── <data_name2>/
    ├── <data_name3>/
    └── ...                             # and so on for other datasets (if any)
```

## Preprocessed Datasets
The following preprocessed workload-driven datasets are available for download, [here](https://purdue0-my.sharepoint.com/:f:/g/personal/gchoudha_purdue_edu/IgCJZ35eEvaHRY18UCM74rxxAQVi_wEO-2d34VRndJPIUvU?e=VTigxg):
* W_Jinan (already included in this repo)
    > Yu, F., Yan, H., Chen, R., Zhang, G., Liu, Y., Chen, M., & Li, Y. (2023). City-scale vehicle trajectory data from traffic camera videos. _Scientific data_, 10(1), 711.
* W_Shenzhen
    > Yu, F., Yan, H., Chen, R., Zhang, G., Liu, Y., Chen, M., & Li, Y. (2023). City-scale vehicle trajectory data from traffic camera videos. _Scientific data_, 10(1), 711.
* W_Chengdu [[GitHub](https://github.com/UrbComp/DeepTTE)]
    > Wang, D., Zhang, J., Cao, W., Li, J., & Zheng, Y. (2018, April). When will you arrive? Estimating travel time based on deep neural networks. In _Proceedings of the AAAI conference on artificial intelligence_ (Vol. 32, No. 1).
* W_Beijing [[Dataset](https://www.microsoft.com/en-us/download/details.aspx?id=52367)][[User Guide](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/)]
    > Zheng, Y., Zhang, L., Xie, X., & Ma, W. Y. (2009, April). Mining interesting locations and travel sequences from GPS trajectories. In _Proceedings of the 18th international conference on World wide web_ (pp. 791-800).

    > Zheng, Y., Li, Q., Chen, Y., Xie, X., & Ma, W. Y. (2008, September). Understanding mobility based on GPS data. In _Proceedings of the 10th international conference on Ubiquitous computing_ (pp. 312-321).

    > Zheng, Y., Xie, X., & Ma, W. Y. (2010). GeoLife: A collaborative social networking service among user, location and trajectory. _IEEE Data Eng. Bull._, 33(2), 32-39.
* W_Shanghai [[GitHub](https://github.com/chilai1996/Shanghai-Taxi-Data)]
* W_NewYork [[Dataset](https://data.cityofnewyork.us/Transportation/2016-Green-Taxi-Trip-Data/hvrh-b6nb/about_data)]
* W_Chicago [[Dataset](https://data.cityofchicago.org/Transportation/Taxi-Trips-2020/r2u4-wwk3/about_data)]

Note: We use prefix `W_` as naming convention to denote workload-driven datasets, i.e., `W_Jinan` refers to the workload-driven dataset for Jinan road network.

## Custom Datasets
You can also preprocess your own datasets following the directory structure mentioned above. At minimum, you need to provide the edgelist file (`<data_name>.edges`), node attributes file (`<data_name>.nodes`), and query files in appropriate query directories. One may also provide queries, if available, in train-test split format. Refer to the sample dataset `Surat_subgraph/` for guidance on formatting. Node ids should be 1-indexed and continuous, i.e., from 1 to N for an N node graph. The graph should be connected to avoid issues arising from unreachable nodes.

To preprocess the custom dataset:
```bash
    # Build non-ml index for ground truth distance computation
    cd ~/non_ml_index/hcl
    make build

    # Generate perturbated query workload from a given query workload (if needed)
    # Refer: slurm-jobs/generate_query_workload_v1.sh
    cd ~/src/data_preprocess
    python generate_query_workload.py --data_name W_Jinan    --query_dir real_workload --query_strategy query_perturbation --perturb_k 5   --k_hop 1 --max_queries 500000 --save_dir real_workload_perturb_500k

    # Generate parts for RNE model (if needed)
    cd ~/src/data_preprocess
    python generate_parts_file_rne.py --data_name W_Jinan

    # Generate landmark embeddings for Catboost/LandmarkNN model (if needed)
    cd ~/src/data_preprocess
    python generate_landmark_distances.py --data_name W_Jinan --num_landmarks 61

    # Generate Node2vec embeddings from OpenNE (if needed)
    # Refer: slurm-jobs/compute_embeddings.sh
    cd ~/third_party/OpenNE
    python -m openne --method node2vec --input ../../../data/W_Jinan/W_Jinan.edges --graph-format edgelist --output ../../../data/W_Jinan/landmark_dim61.embeddings --representation-size 64 --epochs 1 --workers 4
```
