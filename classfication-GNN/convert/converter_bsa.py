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
    '''
    adj_full = scipy.sparse.load_npz('{}/{}/adj_full.npz'.format(dataset_path, prefix)).astype(np.bool_)
    adj_train = scipy.sparse.load_npz('{}/{}/adj_train.npz'.format(dataset_path, prefix)).astype(np.bool_)
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
    node_train = np.array(role['tr'])
    node_val = np.array(role['va'])
    node_test = np.array(role['te'])
    train_feats = feats[node_train]
    adj_train = adj_train[node_train, :][:, node_train]
    labels = class_arr

    adj_train = adj_train + sp.eye(adj_train.shape[0])
    adj_full = adj_full + sp.eye(adj_full.shape[0])
    '''
    adj_full, attr_matrix, labels, train_idx, val_idx, test_idx, _ = ld(ntrain_div_classes=20)
    '''
    adj_full, attr_matrix, labels, train_idx, val_idx, test_idx = \
        utils.get_data(
            dataset_path,
            seed=0,
            ntrain_div_classes=20,
            normalize_attr=None)
    '''
    adj_full = csr_matrix((adj_full), dtype=np.bool_)
    train_feats = attr_matrix[train_idx]
    adj_train = adj_full[train_idx, :][:, train_idx]
    return adj_full, adj_train, attr_matrix , train_feats, labels, train_idx, val_idx, test_idx


def graphsaint(datastr, dataset_name):
    if dataset_name == 'amazon':
        adj_full, adj_train, feats, train_feats, labels, idx_train, idx_val, idx_test = load_data(datastr, 'amazon')
        graphsave(adj_full, dir='../data/amazon_full_adj_')
        graphsave(adj_train, dir='../data/amazon_train_adj_')
        feats = np.array(feats, dtype=np.float64)
        train_feats = np.array(train_feats, dtype=np.float64)
        np.save('../data/amazon_feat.npy', feats)
        np.save('../data/amazon_train_feat.npy', train_feats)
        np.savez('../data/amazon_labels.npz', labels=labels, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)
    if dataset_name == 'yelp':
        adj_full, adj_train, feats, train_feats, labels, idx_train, idx_val, idx_test = load_data(datastr, 'yelp')
        graphsave(adj_full, dir='../data/amazon_full_adj_')
        graphsave(adj_train, dir='../data/amazon_train_adj_')
        feats = np.array(feats, dtype=np.float64)
        train_feats = np.array(train_feats, dtype=np.float64)
        np.save('../data/yelp_feat.npy', feats)
        np.save('../data/yelp_train_feat.npy', train_feats)
        np.savez('../data/yelp_labels.npz', labels=labels, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)
    if dataset_name == 'pubmed':
        adj_full, adj_train, feats, train_feats, labels, idx_train, idx_val, idx_test = load_data(datastr, 'pubmed')
        graphsave(adj_full, dir='../data/pubmed_full_adj_')
        graphsave(adj_train, dir='../data/pubmed_train_adj_')
        feats = feats.toarray()
        train_feats = train_feats.toarray()
        feats = np.array(feats, dtype=np.float64)
        train_feats = np.array(train_feats, dtype=np.float64)
        np.save('../data/pubmed_feat.npy', feats)
        np.save('../data/pubmed_train_feat.npy', train_feats)
        np.savez('../data/pubmed_labels.npz', labels=labels, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)
    if dataset_name == 'cora_full':
        adj_full, adj_train, feats, train_feats, labels, idx_train, idx_val, idx_test = load_data(datastr, 'cora_full')
        graphsave(adj_full, dir='../data/cora_full_full_adj_')
        graphsave(adj_train, dir='../data/cora_full_train_adj_')
        feats = feats.toarray()
        train_feats = train_feats.toarray()
        feats = np.array(feats, dtype=np.float64)
        train_feats = np.array(train_feats, dtype=np.float64)
        np.save('../data/cora_full_feat.npy', feats)
        np.save('../data/cora_full_train_feat.npy', train_feats)
        np.savez('../data/cora_full_labels.npz', labels=labels, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)
    if dataset_name == 'mag_coarse':
        adj_full, adj_train, feats, train_feats, labels, idx_train, idx_val, idx_test = load_data(datastr, 'mag_coarse')
        graphsave(adj_full, dir='../data/mag_coarse_full_adj_')
        graphsave(adj_train, dir='../data/mag_coarse_train_adj_')
        #feats = np.array(feats, dtype=np.float64)
        #train_feats = np.array(train_feats, dtype=np.float64)
        np.save('../data/mag_coarse_feat.npy', feats)
        np.save('../data/mag_coarse_train_feat.npy', train_feats)
        np.savez('../data/mag_coarse_labels.npz', labels=labels, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)
    if dataset_name == 'reddit':
        adj_full, adj_train, feats, train_feats, labels, idx_train, idx_val, idx_test = load_data(datastr, 'reddit')
        graphsave(adj_full, dir='../data/reddit_full_adj_')
        graphsave(adj_train, dir='../data/reddit_train_adj_')
        feats = np.array(feats, dtype=np.float64)
        train_feats = np.array(train_feats, dtype=np.float64)
        np.save('../data/reddit_feat.npy', feats)
        np.save('../data/reddit_train_feat.npy', train_feats)
        np.savez('../data/reddit_labels.npz', labels=labels, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)


if __name__ == "__main__":
    # Your file storage path. For example, this is shown below.
    datastr = "../data"

    # dataset name, yelp or reddit
    dataset_name = 'yelp'
    graphsaint(datastr, dataset_name)
