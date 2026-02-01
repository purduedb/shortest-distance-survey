# Adapted from: https://github.com/Jiboxiake/HCL-Python/blob/main/HCL.ipynb
# Example Usage:
# python evaluate.py --index_path ./saved_models/W_Chicago.hl --query_path ../../data/W_Chicago/real_workload_perturb_500k/W_Chicago.queries --eval_runs 10
import csv
import time
import json
import argparse

from hcl_index import ContractionIndex, ContractionLabel, FlatCutIndex


## Function to detect delimiter in a CSV string
def detect_delimiter(sample_data):
    """
    Detect the delimiter used in a CSV formatted string.
    """
    try:
        dialect = csv.Sniffer().sniff(sample_data)
        print(f"Detected delimiter: `{dialect.delimiter}`")
        return dialect.delimiter
    except:
        raise ValueError("Could not detect delimiter.")


## Function to detect delimiter in a file
def detect_delimiter_file(file_path):
    # Detect delimiter by reading first 128 lines
    with open(file_path, 'r') as f:
        sample_data = ""
        for i, line in enumerate(f):
            if i < 128:
                sample_data += line
            else:
                break
    delimiter = detect_delimiter(sample_data)
    return delimiter


## HCL File Parsing
def parse_hcl_file(filename):
    label_list = []
    label_list.append(ContractionLabel(cut_index=FlatCutIndex(partition_bitvector=0, dist_index=[0], distances=[]), distance_offset=0, parent=None))
    with open(filename, 'r') as f:
        count = 0
        for line in f:
            if line.startswith('{'):
                count += 1
            elif line.startswith('}'):
                count += 1
            else:
                line = line.strip().rstrip(',')
                if not line:
                    continue
                key, value = line.split(':', 1)
                key = int(key)
                assert(key==count)
                # Try to parse the value as JSON

                try:
                    parsed_value = json.loads(value)
                    p = parsed_value.get('p')
                    d = parsed_value.get('d')
                    #print(f"Key: {key}, p: {p}, d: {d}")
                    label_list.append(ContractionLabel(parent=p, distance_offset=d))

                except json.JSONDecodeError:
                    # If not valid JSON, just keep as string
                    parsed_value = value
                    #print(f"Key: {key}, Value: {parsed_value}")
                    int_part, list_part = value.split(',', 1)
                    partition_vector = int(int_part.strip())
                    list_of_lists = json.loads(list_part)
                    dist_index = []
                    distances = []
                    for list in list_of_lists:
                        for entry in list:
                            distances.append(int(entry))
                        dist_index.append(len(distances))
                    #print(list_of_lists)
                    cut_index = FlatCutIndex(partition_bitvector=partition_vector, dist_index=dist_index, distances=distances)
                    label_list.append(ContractionLabel(cut_index=cut_index, distance_offset=0, parent=None))
                count += 1

    return label_list


## Evaluation loop
def evaluate(hci, dataset):
    total_time = 0
    total_queries = len(dataset)
    for i in range(len(dataset)):
        parts = dataset[i]
        source_node = parts[0]
        target_node = parts[1]
        start_time = time.perf_counter()                        ## Start timing
        distance = hci.get_distance(source_node, target_node)   ## Distance query
        end_time = time.perf_counter()                          ## End timing
        total_time += end_time - start_time

    query_latency = total_time / total_queries if total_queries > 0 else 0
    print(f"Total time for {total_queries} queries: {total_time:.4f} seconds")
    print(f"Average time per query: {query_latency * 1_000_000:.4f} microseconds")
    return query_latency


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate HCL index with query file.")
    parser.add_argument("--index_path", type=str, required=True, help="Path to the HCL index (.hl) file")
    parser.add_argument("--query_path", type=str, required=True, help="Path to the query file")
    parser.add_argument("--delimiter", type=str, default=None, help="Delimiter used in the query file")
    parser.add_argument("--comment", type=str, default="#", help="Comment character in the query file")
    parser.add_argument("--eval_runs", type=int, default=0, help="Number of evaluation runs to average over")
    args = parser.parse_args()

    index_path = args.index_path
    query_path = args.query_path
    delimiter = args.delimiter
    comment = args.comment
    eval_runs = args.eval_runs

    print("No-ML experiments with real workload started...")

    # Load index
    print(f"Processing labeling file: {index_path}")
    label_list = parse_hcl_file(index_path)
    hci = ContractionIndex(label_list)
    print("Index loaded.")

    # Load queries
    print(f"Processing query file: {query_path}")
    dataset = []
    if delimiter is None:
        delimiter = detect_delimiter_file(query_path)
    with open(query_path, "r") as f:
        for idx, line in enumerate(f):
            if line.startswith(comment):
                continue
            parts = line.strip().split(delimiter)
            u = int(parts[0])
            v = int(parts[1])
            distance = int(parts[2])

            # Exclude zero distance entries because they cause division by zero error in MRE calculation
            if distance == 0:
                print(f"Warning: Zero distance found in query file (Line {idx + 1}: {line.strip()}). Skipping entry.")
                continue
            dataset.append((u, v, distance))
    print("Query file processed. Queries loaded:", len(dataset))

    # Replicate data to get target datasize of 1M queries
    print("Replicating dataset to reach target size...")
    target_size = 1_000_000
    num_repeats = min(target_size // len(dataset), 1)
    dataset = dataset * num_repeats

    print("Starting evaluation...")
    query_latency = evaluate(hci, dataset)

    if eval_runs > 0:
        print(f"Running evaluation {eval_runs} times to get average latency...")
        results = []
        for i in range(eval_runs):
            query_latency = evaluate(hci, dataset)
            results.append(query_latency)

        observations = results[-5:]
        avg_latency = sum(observations) / len(observations)
        print(f"Average latency of last {len(observations)} runs: {avg_latency * 1_000_000:.4f} microseconds")
