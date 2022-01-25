import pickle as pkl
import sys
import os
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
import scipy.sparse as sp
import sklearn.preprocessing
import json
import time
from scipy.sparse import find, csr_matrix
import mutils as utils

import scipy.sparse
import struct
from sklearn.preprocessing import StandardScaler

import os, errno

def mkdir(path):
	try:
		os.makedirs(path)
	except OSError as exc:  # Python ≥ 2.5
		if exc.errno == errno.EEXIST and os.path.isdir(path):
			pass
		# possibly handle other errno cases here, otherwise finally:
		else:
			raise

def graphsave(adj, dir):
    if (sp.isspmatrix_csr(adj)):
        el = adj.indices
        pl = adj.indptr

        EL = np.array(el, dtype=np.uint32)
        PL = np.array(pl, dtype=np.uint32)

        EL_re = []

        for i in range(1, PL.shape[0]):
            EL_re += sorted(EL[PL[i - 1]:PL[i]], key=lambda x: PL[x + 1] - PL[x])

        EL_re = np.asarray(EL_re, dtype=np.uint32)

        print("EL:", EL_re.shape)
        f1 = open(dir + 'el.txt', 'wb')
        for i in EL_re:
            m = struct.pack('I', i)
            f1.write(m)
        f1.close()

        print("PL:", PL.shape)
        f2 = open(dir + 'pl.txt', 'wb')
        for i in PL:
            m = struct.pack('I', i)
            f2.write(m)
        f2.close()
    else:
        print("Format Error!")


def load_data(dataset_path, prefix, normalize=True):
    '''
    adj_full = scipy.sparse.load_npz('{}/{}/adj_full.npz'.format(dataset_path, prefix)).astype(np.bool)
    adj_train = scipy.sparse.load_npz('{}/{}/adj_train.npz'.format(dataset_path, prefix)).astype(np.bool)
    role = json.load(open('{}/{}/role.json'.format(dataset_path, prefix)))
    feats = np.load('{}/{}/feats.npy'.format(dataset_path, prefix))
    class_map = json.load(open('{}/{}/class_map.json'.format(dataset_path, prefix)))
    class_map = {int(k): v for k, v in class_map.items()}
    assert len(class_map) == feats.shape[0]
    # ---- normalize feats ----
    train_nodes = np.array(list(set(adj_train.nonzero()[0])))
    train_feats = feats[train_nodes]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)
    # -------------------------
    num_vertices = adj_full.shape[0]
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
    # print(class_arr)
    node_train = np.array(role['tr'])
    node_val = np.array(role['va'])
    node_test = np.array(role['te'])
    train_feats = feats[node_train]
    adj_train = adj_train[node_train, :][:, node_train]
    labels = class_arr

    adj_train = adj_train + sp.eye(adj_train.shape[0])
    adj_full = adj_full + sp.eye(adj_full.shape[0])
    '''
    adj_full, attr_matrix, labels, train_idx, val_idx, test_idx = \
        utils.get_data(
            dataset_path,
            seed=0,
            ntrain_div_classes=20,
            normalize_attr=None)
    adj_full = csr_matrix((adj_full), dtype=np.bool)
    print(adj_full)
    train_feats = attr_matrix[train_idx]
    adj_train = adj_full[train_idx, :][:, train_idx]
    return adj_full, adj_train, attr_matrix.toarray(), train_feats.toarray(), labels, train_idx, val_idx, test_idx


def graphsaint(datastr, dataset_name):
    
    if dataset_name == 'pubmed':
        adj_full, adj_train, feats, train_feats, labels, idx_train, idx_val, idx_test = load_data(datastr, 'pubmed')
        graphsave(adj_full, dir='../data/pubmed_full_adj_')
        graphsave(adj_train, dir='../data/pubmed_train_adj_')
        feats = np.array(feats, dtype=np.float64)
        train_feats = np.array(train_feats , dtype=np.float64)
        np.save('../data/pubmed_feat.npy', feats)
        np.save('../data/pubmed_train_feat.npy', train_feats)
        np.savez('../data/pubmed_labels.npz', labels=labels, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)


if __name__ == "__main__":
    # Your file storage path. For example, this is shown below.
    import pathlib
    path = pathlib.Path(__file__).parent.resolve()

    mkdir(f"{path}/../pretrained")
    mkdir(f"{path}/dataset")

    datastr = f"{path}/../data/pubmed.npz"

    # dataset name, pubmed
    dataset_name = 'pubmed'
    graphsaint(datastr, dataset_name)