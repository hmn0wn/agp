import pickle as pkl
import sys
import os
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
import scipy.sparse as sp
from scipy.sparse import find, csr_matrix

import sklearn.preprocessing
import json
import time
import scipy.sparse
from mutils import train_stopping_split
from sparsegraph import SparseGraph
import struct
from sklearn.preprocessing import StandardScaler

def load_graph():
    dataset_path = '/home/user/proj/AGP/classfication-GNN/data/amazon'
    adj_matrix = scipy.sparse.load_npz('{}/adj_full.npz'.format(dataset_path))

    attr_matrix = np.load('{}/feats.npy'.format(dataset_path))
    class_map = json.load(open('{}/class_map.json'.format(dataset_path)))
    class_map = {int(k): v for k, v in class_map.items()}
    num_vertices = adj_matrix.shape[0]
    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
        class_arr = np.zeros((num_vertices, num_classes))
        for k, v in class_map.items():
            class_arr[k] = v
    else:
        num_classes = max(class_map.values()) - min(class_map.values()) + 1
        class_arr = np.zeros((num_vertices, num_classes))
        offset = min(class_map.values())
        for k, v in class_map.items():
            class_arr[k][v - offset] = 1
    # num_classes = class_arr.shape[1]
    class_arr = np.argmax(class_arr, axis=-1)
    g = SparseGraph(adj_matrix=adj_matrix, attr_matrix=attr_matrix, labels=class_arr)
    return g

def load_data(ntrain_div_classes, seed=0):
    dataset_path = '/home/user/proj/AGP/classfication-GNN/data/amazon'
    adj_matrix = scipy.sparse.load_npz('{}/adj_full.npz'.format(dataset_path))

    attr_matrix = np.load('{}/feats.npy'.format(dataset_path))
    class_map = json.load(open('{}/class_map.json'.format(dataset_path)))
    class_map = {int(k): v for k, v in class_map.items()}
    num_vertices = adj_matrix.shape[0]
    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
        class_arr = np.zeros((num_vertices, num_classes))
        for k, v in class_map.items():
            class_arr[k] = v
    else:
        num_classes = max(class_map.values()) - min(class_map.values()) + 1
        class_arr = np.zeros((num_vertices, num_classes))
        offset = min(class_map.values())
        for k, v in class_map.items():
            class_arr[k][v - offset] = 1
    #num_classes = class_arr.shape[1]
    class_arr = np.argmax(class_arr, axis=-1)
    g = SparseGraph(adj_matrix=adj_matrix, attr_matrix=attr_matrix, labels=class_arr )
    g.standardize(select_lcc=True, make_undirected=True, no_self_loops=False)
    num_classes = g.labels.max() + 1
    print(g.adj_matrix.shape)
    print(g.attr_matrix.shape)
    n = len(g.labels)
    n_train = num_classes * ntrain_div_classes
    n_val = n_train * 2
    train_idx, val_idx = train_stopping_split(labels=g.labels, ntrain_per_class=20, seed=seed, nval=n_val)
    train_val_idx = np.concatenate((train_idx, val_idx))
    test_idx = np.sort(np.setdiff1d(np.arange(n), train_val_idx))
    return g.adj_matrix, g.attr_matrix, g.labels, train_idx, val_idx, test_idx, num_classes
