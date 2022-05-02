import pickle as pkl
import sys
import os
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
import scipy.sparse as sp
from scipy.sparse import find, csr_matrix
import mutils as utils
from read_amazon import load_data as ld
import sklearn.preprocessing
import json
import time
import scipy.sparse
import struct
from sklearn.preprocessing import StandardScaler


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

    adj_full, attr_matrix, labels, train_idx, val_idx, test_idx, _ = ld(ntrain_div_classes=20, name=prefix, seed=0)
    adj_full = csr_matrix((adj_full), dtype=np.bool_)
    train_feats = attr_matrix[train_idx]
    adj_train = adj_full[train_idx, :][:, train_idx]
    return adj_full, adj_train, attr_matrix , train_feats, labels, train_idx, val_idx, test_idx


def graphsaint(datastr, dataset_name):
    adj_full, adj_train, feats, train_feats, labels, idx_train, idx_val, idx_test = load_data(datastr, dataset_name)
    graphsave(adj_full, dir=f'../data/{dataset_name}_full_adj_')
    graphsave(adj_train, dir=f'../data/{dataset_name}_train_adj_')
    if dataset_name == 'pubmed' or dataset_name == 'cora_full':
        feats = feats.toarray()
        train_feats = train_feats.toarray()
    feats = np.array(feats, dtype=np.float64)
    train_feats = np.array(train_feats, dtype=np.float64)
    np.save(f'../data/{dataset_name}_feat.npy', feats)
    np.save(f'../data/{dataset_name}_train_feat.npy', train_feats)
    np.savez(f'../data/{dataset_name}_labels.npz', labels=labels, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)


if __name__ == "__main__":
    # Your file storage path. For example, this is shown below.
    datastr = "../data"

    # dataset name, yelp or reddit
    dataset_name = 'yelp'
    #EL: (263793649,)
    #PL: (1066628,)
    #EL: (1950,)
    #PL: (1501,)
    graphsaint(datastr, dataset_name)
