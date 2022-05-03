from cProfile import label
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
from sparsegraph import SparseGraph

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


def load_data_semi(dataset_path, dataset_name, normalize=True, from_graphsaint=False, seed=0):
    ntrain_div_classes = 20 
    normalize_feats_agp = False
    if from_graphsaint:
        adj_full = scipy.sparse.load_npz('{}/{}/adj_full.npz'.format(dataset_path, dataset_name)).astype(np.bool_)
      
        feats = np.load('{}/{}/feats.npy'.format(dataset_path, dataset_name))
        class_map = json.load(open('{}/{}/class_map.json'.format(dataset_path, dataset_name)))
        class_map = {int(k): v for k, v in class_map.items()}
        assert len(class_map) == feats.shape[0]
        
        num_vertices = adj_full.shape[0]
        if isinstance(list(class_map.values())[0], list):
            num_classes = len(list(class_map.values())[0])
            labels = np.zeros((num_vertices, num_classes))
            for k, v in class_map.items():
                labels[k] = v
        else:
            num_classes = max(class_map.values()) - min(class_map.values()) + 1
            labels = np.zeros((num_vertices, num_classes))
            offset = min(class_map.values())
            for k, v in class_map.items():
                labels[k][v - offset] = 1
        
        labels = np.argmax(labels, axis=-1)
        g = SparseGraph(adj_matrix=adj_full, attr_matrix=feats, labels=labels)
        g.standardize(select_lcc=True, make_undirected=True, no_self_loops=False)

        num_classes = g.labels.max() + 1
        n_train = num_classes * ntrain_div_classes
        n_val = n_train * 2
        train_idx, val_idx = utils.train_stopping_split(labels=g.labels, ntrain_per_class=ntrain_div_classes,  seed=seed, nval=n_val)
        train_val_idx = np.concatenate((train_idx, val_idx))
        test_idx = np.sort(np.setdiff1d(np.arange(len(g.labels)), train_val_idx))
        adj_full, feats, labels = g.adj_matrix, g.attr_matrix, g.labels
    else:
        adj_full, feats, labels, train_idx, val_idx, test_idx = \
            utils.get_data(
                f"{dataset_path}/{dataset_name}.npz",
                seed=seed,
                ntrain_div_classes=ntrain_div_classes,
                normalize_attr=None)

    adj_full = csr_matrix((adj_full), dtype=np.bool_)
    print(f"adj_full:{adj_full}")
    
    adj_train = adj_full[train_idx, :][:, train_idx]
    train_feats = feats[train_idx]
    
    if normalize_feats_agp:
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)

    return adj_full, adj_train, feats, train_feats, labels, train_idx, val_idx, test_idx


def graphsaint(dataset_path, dataset_name, from_graphsaint):
    adj_full, adj_train, feats, train_feats, labels, idx_train, idx_val, idx_test = load_data_semi(dataset_path, dataset_name, from_graphsaint=from_graphsaint)
    semi_name = f"{dataset_name}_semi"
    graphsave(adj_full, dir=f'{dataset_path}/{semi_name}_full_adj_')
    graphsave(adj_train, dir=f'{dataset_path}/{semi_name}_train_adj_')
    #print(feats)
    
    if type(feats)!=np.ndarray:
        feats = feats.toarray()
    if type(train_feats)!=np.ndarray:
        train_feats = train_feats.toarray()
    feats = np.array(feats, dtype=np.float64)
    train_feats = np.array(train_feats , dtype=np.float64)
    np.save(f'{dataset_path}/{semi_name}_feat.npy', feats)
    np.save(f'{dataset_path}/{semi_name}_train_feat.npy', train_feats)
    np.savez(f'{dataset_path}/{semi_name}_labels.npz', labels=labels, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)


if __name__ == "__main__":
    # Your file storage path. For example, this is shown below.
    import pathlib
    path = pathlib.Path(__file__).parent.resolve()

    mkdir(f"{path}/../pretrained")
    mkdir(f"{path}/dataset")
    
    dataset_path = f"{path}/../data"
    #dataset_path = f"{path}/data"
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
        from_graphsaint = sys.argv[2]=="True"

    graphsaint(dataset_path, dataset_name, from_graphsaint)
