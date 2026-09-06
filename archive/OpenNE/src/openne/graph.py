"""Graph utilities."""

# from time import time
import networkx as nx
import pickle as pkl
import numpy as np
import scipy.sparse as sp
import csv

__author__ = "Zhang Zhengyan"
__email__ = "zhangzhengyan14@mails.tsinghua.edu.cn"

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

class Graph(object):
    def __init__(self):
        self.G = None
        self.look_up_dict = {}
        self.look_back_list = []
        self.node_size = 0

    def encode_node(self):
        look_up = self.look_up_dict
        look_back = self.look_back_list
        for node in self.G.nodes():
            look_up[node] = self.node_size
            look_back.append(node)
            self.node_size += 1
            self.G.nodes[node]['status'] = ''

    def read_g(self, g):
        self.G = g
        self.encode_node()

    def read_adjlist(self, filename):
        """ Read graph from adjacency file in which the edge must be unweighted
            the format of each line: v1 n1 n2 n3 ... nk
            :param filename: the filename of input file
        """
        self.G = nx.read_adjlist(filename, create_using=nx.DiGraph())
        for i, j in self.G.edges():
            self.G[i][j]['weight'] = 1.0
        self.encode_node()

    def read_edgelist(self, filename, weighted=False, directed=False, delimiter=None, comment='#'):
        """ Read graph from edgelist file
            the format of each line: v1 v2 [weight]
            :param filename: the filename of input file
            :param weighted: whether the edge is weighted
            :param directed: whether the graph is directed
            :param delimiter: the delimiter of each line
            :param comment: the comment character, lines starting with this character will be ignored
        """
        self.G = nx.DiGraph()
        print(f"Reading graph:")
        print(f"  - filename: {filename}")
        print(f"  - weighted: {weighted}")
        print(f"  - directed: {directed}")

        if directed:
            def read_unweighted(l, delimiter):
                parts = l.split(delimiter)
                src, dst = parts[0].strip(), parts[1].strip()
                self.G.add_edge(src, dst)
                self.G[src][dst]['weight'] = 1.0

            def read_weighted(l, delimiter):
                parts = l.split(delimiter)
                src, dst, w = parts[0].strip(), parts[1].strip(), parts[2].strip()
                self.G.add_edge(src, dst)
                self.G[src][dst]['weight'] = float(w)
        else:
            def read_unweighted(l, delimiter):
                parts = l.split(delimiter)
                src, dst = parts[0].strip(), parts[1].strip()
                self.G.add_edge(src, dst)
                self.G.add_edge(dst, src)
                self.G[src][dst]['weight'] = 1.0
                self.G[dst][src]['weight'] = 1.0

            def read_weighted(l, delimiter):
                parts = l.split(delimiter)
                src, dst, w = parts[0].strip(), parts[1].strip(), parts[2].strip()
                self.G.add_edge(src, dst)
                self.G.add_edge(dst, src)
                self.G[src][dst]['weight'] = 1.0 + float(w)  # Original weight
                self.G[dst][src]['weight'] = 1.0 + float(w)  # Original weight
                # self.G[src][dst]['weight'] = 1.0/(1 + float(w))  # Inverse weight
                # self.G[dst][src]['weight'] = 1.0/(1 + float(w))  # Inverse weight

        # Detect delimiter
        if delimiter is None:
            delimiter = detect_delimiter_file(filename)

        fin = open(filename, 'r')
        func = read_unweighted
        if weighted:
            func = read_weighted
        while 1:
            l = fin.readline()
            if l == '':
                break
            # Skip comments
            if l.startswith(comment):
                continue
            func(l, delimiter)
        fin.close()
        self.encode_node()

    def read_node_label(self, filename):
        fin = open(filename, 'r')
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.split()
            self.G.nodes[vec[0]]['label'] = vec[1:]
        fin.close()

    def read_node_features(self, filename):
        fin = open(filename, 'r')
        for l in fin.readlines():
            vec = l.split()
            self.G.nodes[vec[0]]['feature'] = np.array(
                [float(x) for x in vec[1:]])
        fin.close()

    def read_node_status(self, filename):
        fin = open(filename, 'r')
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.split()
            self.G.nodes[vec[0]]['status'] = vec[1]  # train test valid
        fin.close()

    def read_edge_label(self, filename):
        fin = open(filename, 'r')
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.split()
            self.G[vec[0]][vec[1]]['label'] = vec[2:]
        fin.close()
