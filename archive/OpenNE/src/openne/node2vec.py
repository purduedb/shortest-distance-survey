from __future__ import print_function
import time
from gensim.models import Word2Vec
from . import walker


class Node2vec(object):

    def __init__(self, graph, path_length, num_paths, dim, p=1.0, q=1.0, dw=False, **kwargs):

        kwargs["workers"] = kwargs.get("workers", 1)
        if dw:
            kwargs["hs"] = 1
            p = 1.0
            q = 1.0

        self.graph = graph
        if dw:
            self.walker = walker.BasicWalker(graph, workers=kwargs["workers"])
        else:
            self.walker = walker.Walker(graph, p=p, q=q, workers=kwargs["workers"])
            print("Preprocess transition probs...")
            self.walker.preprocess_transition_probs()
        sentences = self.walker.simulate_walks(
            num_walks=num_paths, walk_length=path_length)
        kwargs["sentences"] = sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["vector_size"] = kwargs.get("vector_size", dim)
        kwargs["sg"] = 1

        self.vector_size = kwargs["vector_size"]
        print("Learning representation...")
        print("Kwargs: ")
        for key, value in kwargs.items():
            if key == "sentences":
                print(f"  - {key}: {len(value)} sentences")
            else:
                print(f"  - {key}: {value}")
        start_time = time.perf_counter()
        word2vec = Word2Vec(**kwargs)
        end_time = time.perf_counter()
        precomputation_time = end_time - start_time
        print(f"Precomputation time: {precomputation_time / 60:.2f} minutes")
        self.vectors = {}
        for word in graph.G.nodes():
            self.vectors[word] = word2vec.wv[word]
        del word2vec

    def save_embeddings(self, filename, delimiter=',', comment='#'):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write(f"{comment}, num_nodes={node_num}, vector_size={self.vector_size}\n")
        sorted_nodes = sorted(self.vectors.keys(), key=lambda x: int(x.strip()))
        for node in sorted_nodes:
            vec = self.vectors[node]
            fout.write(f"{node.strip()}{delimiter}{delimiter.join([str(x) for x in vec])}\n")
        fout.close()
