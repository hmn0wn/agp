import resource
import numpy as np
import sklearn
import scipy.sparse as sp
import tensorflow as tf
from typing import List, Tuple, Union
import tracemalloc
from sparsegraph import load_from_npz
import linecache
import os
import scipy.sparse.linalg as spla
def sparse_feeder(M):
    """Convert a sparse matrix to the format suitable for feeding as a tf.SparseTensor.
    Parameters
    ----------
    M : sp.spmatrix
        Matrix to convert.
    Returns
    -------
    indices : array-like, shape [num_edges, 2]
        Indices of the nonzero elements.
    values : array-like, shape [num_edges]
        Values of the nonzero elements.
    shape : tuple
        Shape of the matrix.
    """
    M = M.tocoo()
    return np.vstack((M.row, M.col)).T, M.data, M.shape


class SparseRowIndexer:
    def __init__(self, csr_matrix):
        data = []
        indices = []
        indptr = []

        # Iterating over the rows this way is significantly more efficient
        # than csr_matrix[row_index,:] and csr_matrix.getrow(row_index)
        for row_start, row_end in zip(csr_matrix.indptr[:-1], csr_matrix.indptr[1:]):
            data.append(csr_matrix.data[row_start:row_end])
            indices.append(csr_matrix.indices[row_start:row_end])
            indptr.append(row_end - row_start)  # nnz of the row

        self.data = np.array(data)
        self.indices = np.array(indices)
        self.indptr = np.array(indptr)
        self.n_columns = csr_matrix.shape[1]

    def __getitem__(self, row_selector):
        data = np.concatenate(self.data[row_selector])
        indices = np.concatenate(self.indices[row_selector])
        indptr = np.append(0, np.cumsum(self.indptr[row_selector]))

        shape = [indptr.shape[0] - 1, self.n_columns]

        return sp.csr_matrix((data, indices, indptr), shape=shape)

def sparse_matrix_to_tensor(X: sp.spmatrix) -> tf.SparseTensor:
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(
            indices,
            np.array(coo.data, dtype=np.float32),
            coo.shape)


def matrix_to_tensor(
        X: Union[np.ndarray, sp.spmatrix]) -> Union[tf.Tensor, tf.SparseTensor]:
    if sp.issparse(X):
        return sparse_matrix_to_tensor(X)
    else:
        return tf.constant(X, dtype=tf.float32)

def split_random(seed, n, n_train, n_val, n_test):
    np.random.seed(seed)
    rnd = np.random.permutation(n)

    train_idx = np.sort(rnd[:n_train])
    val_idx = np.sort(rnd[n_train:n_train + n_val])

    train_val_idx = np.concatenate((train_idx, val_idx))
    test_idx = np.sort(np.setdiff1d(np.arange(n_test), train_val_idx))

    return train_idx, val_idx, test_idx
def normalize_attributes(attr_matrix):
    attr_norms = sp.linalg.norm(attr_matrix, ord=1, axis=1)
    attr_invnorms = 1 / np.maximum(attr_norms, 1e-12)
    attr_matrix = attr_matrix.multiply(attr_invnorms[:, np.newaxis]).tocsr()
    return attr_matrix

def get_data(dataset_path, seed, ntrain_div_classes, normalize_attr=None):
    '''
    Get data from a .npz-file.
    Parameters
    ----------
    dataset_path
        path to dataset .npz file
    seed
        Random seed for dataset splitting
    ntrain_div_classes
        Number of training nodes divided by number of classes
    normalize_attr
        Normalization scheme for attributes. By default (and in the paper) no normalization is used.
    '''
    g = load_from_npz(dataset_path)

    g.standardize(select_lcc=True, make_undirected=True, no_self_loops=False)
    # number of nodes and attributes
    n, d = g.attr_matrix.shape

    # optional attribute normalization
    if normalize_attr == 'per_feature':
        if sp.issparse(g.attr_matrix):
            scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        else:
            scaler = sklearn.preprocessing.StandardScaler()
        attr_matrix = scaler.fit_transform(g.attr_matrix)
    elif normalize_attr == 'per_node':
        if sp.issparse(g.attr_matrix):
            attr_norms = sp.linalg.norm(g.attr_matrix, ord=1, axis=1)
            attr_invnorms = 1 / np.maximum(attr_norms, 1e-12)
            attr_matrix = g.attr_matrix.multiply(attr_invnorms[:, np.newaxis]).tocsr()
        else:
            attr_norms = np.linalg.norm(g.attr_matrix, ord=1, axis=1)
            attr_invnorms = 1 / np.maximum(attr_norms, 1e-12)
            attr_matrix = g.attr_matrix * attr_invnorms[:, np.newaxis]
    else:
        attr_matrix = g.attr_matrix

    num_classes = g.labels.max() + 1
    n_train = num_classes * ntrain_div_classes
    n_val = n_train * 2
    train_idx, val_idx = train_stopping_split(labels=g.labels, ntrain_per_class=ntrain_div_classes,  seed=seed, nval=n_val)
    train_val_idx = np.concatenate((train_idx, val_idx))
    test_idx = np.sort(np.setdiff1d(np.arange(n), train_val_idx))
    return g.adj_matrix, attr_matrix, g.labels, train_idx, val_idx, test_idx

def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))
def train_stopping_split(labels: np.ndarray,   ntrain_per_class: int = 20,
                          nval: int = 500, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rnd_state = np.random.RandomState(seed)
    train_idx_split = []
    idx = np.arange(len(labels))
    print(len(np.unique(labels)))
    print(labels[0])
    print(max(labels))
    for i in range(np.max(labels) + 1):
        if len(idx[labels == i]) < ntrain_per_class:
            train_idx_split.append(idx[labels == i])
        else:
            train_idx_split.append(rnd_state.choice( idx[labels == i], ntrain_per_class, replace=False))

    train_idx = np.concatenate(train_idx_split)
    val_idx = rnd_state.choice(
        exclude_idx(idx, [train_idx]),
        nval, replace=False)

    return train_idx, val_idx

def exclude_idx(idx: np.ndarray, idx_exclude_list: List[np.ndarray]) -> np.ndarray:
    idx_exclude = np.concatenate(idx_exclude_list)
    return np.array([i for i in idx if i not in idx_exclude])

def top_degree_nodes_in_class(adj, labels, nclasses, num_nodes=20, top=True):
    if top:
        labels_high_degree = np.argsort(np.sum(adj, axis=-1))[::-1]
    else:
        labels_high_degree = np.argsort(np.sum(adj, axis=-1))
    dict_labels = dict()
    all_train = []
    for c in range(nclasses):
        dict_labels[c] = []
        for i in labels_high_degree:
            if labels[i] == c:
                if len(dict_labels[c]) < num_nodes:
                    dict_labels[c].append(int(i))
                else:
                    all_train += dict_labels[c]
                    break
    #print(len(all_train), len(all_train))
    return all_train

def get_max_memory_bytes():
    return 1024 * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss