import errno
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
import scipy.sparse
import struct
from sklearn.preprocessing import StandardScaler
import mutils as utils

def graphsave(adj,dir):
  if(sp.isspmatrix_csr(adj)):
    el=adj.indices
    pl=adj.indptr
    EL=np.array(el,dtype=np.uint32)
    PL=np.array(pl,dtype=np.uint32)
    EL_re=[]
    for i in range(1,PL.shape[0]):
      EL_re+=sorted(EL[PL[i-1]:PL[i]],key=lambda x:PL[x+1]-PL[x])
    EL_re=np.asarray(EL_re,dtype=np.uint32)

    print("EL:",EL_re.shape)
    f1=open(dir+'el.txt','wb')
    for i in EL_re:
      m=struct.pack('I',i)
      f1.write(m)
    f1.close()

    print("PL:",PL.shape)
    f2=open(dir+'pl.txt','wb')
    for i in PL:
      m=struct.pack('I',i)
      f2.write(m)
    f2.close()
  else:
    print("Format Error!")

def load_graphsage_data(dataset_path, dataset_str, normalize=True, gp_load=True):
  """Load GraphSAGE data.""" 
  ntrain_div_classes = 20
  seed = 0
  normalize_feats_agp = False

  start_time = time.time()
  graph_json = json.load(open('{}/{}/{}-G.json'.format(dataset_path, dataset_str,dataset_str)))
  graph_nx = json_graph.node_link_graph(graph_json)

  id_map = json.load(open('{}/{}/{}-id_map.json'.format(dataset_path, dataset_str,dataset_str)))
  is_digit = list(id_map.keys())[0].isdigit()
  id_map = {(int(k) if is_digit else k): int(v) for k, v in id_map.items()}
  class_map = json.load(open('{}/{}/{}-class_map.json'.format(dataset_path, dataset_str,dataset_str)))

  is_instance = isinstance(list(class_map.values())[0], list)
  class_map = {(int(k) if is_digit else k): (v if is_instance else int(v))
               for k, v in class_map.items()}

  broken_count = 0
  to_remove = []
  for node in graph_nx.nodes():
    if node not in id_map:
      to_remove.append(node)
      broken_count += 1
  for node in to_remove:
    graph_nx.remove_node(node)
  print('Removed {} nodes that lacked proper annotations due to networkx versioning issues'.format(broken_count))

  feats = np.load('{}/{}/{}-feats.npy'.format(dataset_path, dataset_str, dataset_str))

  print('Loaded data ({} seconds).. now preprocessing..'.format(time.time() - start_time))
  start_time = time.time()
  edges = []
  for edge in graph_nx.edges():
    if edge[0] in id_map and edge[1] in id_map:
      edges.append((id_map[edge[0]], id_map[edge[1]]))
  num_data = len(id_map)
  edges = np.array(edges, dtype=np.int32)

  # Process labels
  if isinstance(list(class_map.values())[0], list):
    num_classes = len(list(class_map.values())[0])
    labels = np.zeros((num_data, num_classes), dtype=np.float32)
    for k in class_map.keys():
      labels[id_map[k], :] = np.array(class_map[k])
  else:
    num_classes = len(set(class_map.values()))
    labels = np.zeros((num_data, num_classes), dtype=np.float32)
    for k in class_map.keys():
      labels[id_map[k], class_map[k]] = 1

  def _construct_adj(edges):
    adj = sp.csr_matrix((np.ones(
        (edges.shape[0]), dtype=np.float32), (edges[:, 0], edges[:, 1])),
                        shape=(num_data, num_data))
    # adj += adj.transpose()
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj=adj+sp.eye(adj.shape[0])
    return adj

  full_adj = _construct_adj(edges)

  if True:
    num_classes = len(class_map) + 1
    n_train = num_classes * ntrain_div_classes
    n_val = n_train * 2
    train_idx, val_idx = utils.train_stopping_split(labels=labels, ntrain_per_class=ntrain_div_classes,  seed=seed, nval=n_val)
    train_val_idx = np.concatenate((train_idx, val_idx))
    num_vertices = full_adj.shape[0]
    test_idx = np.sort(np.setdiff1d(np.arange(num_vertices), train_val_idx))

  else:
    val_idx = np.array(
        [id_map[n] for n in graph_nx.nodes() if graph_nx.node[n]['val']],
        dtype=np.int32)
    test_idx = np.array(
        [id_map[n] for n in graph_nx.nodes() if graph_nx.node[n]['test']],
        dtype=np.int32)
    is_train = np.ones((num_idx), dtype=np.bool_)
    is_train[val_idx] = False
    is_train[test_idx] = False
    train_idx = np.array([n for n in range(num_idx) if is_train[n]],
                          dtype=np.int32)

    train_edges = [
        (e[0], e[1]) for e in edges if is_train[e[0]] and is_train[e[1]]
    ]
    train_edges = np.array(train_edges, dtype=np.int32)
  
    if dataset_str=='Amazon2M':
      test_idx = val_idx


  train_feats = feats[train_idx]
  if normalize:
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)

  adj_train = full_adj[train_idx, :][:, train_idx]

  print('Data loaded, {} seconds.'.format(time.time() - start_time))
  return adj_train, full_adj, feats, train_feats, labels, train_idx, val_idx, test_idx

def Amazon2M(datastr, gp_load):
  train_adj, full_adj, feats, train_feats, labels, idx_train, idx_val, idx_test = load_graphsage_data(datastr, 'Amazon2M', normalize=True, gp_load=gp_load)
  
  labels = np.where(labels>0.5)[1]
  graphsave(full_adj,dir='../data/Amazon2M_full_adj_')
  graphsave(train_adj,dir='../data/Amazon2M_train_adj_')
  np.save('../data/Amazon2M_feat.npy',feats)
  np.save('../data/Amazon2M_train_feat.npy',train_feats)
  np.savez('../data/Amazon2M_labels.npz',labels=labels,idx_train=idx_train,idx_val=idx_val,idx_test=idx_test)

def mkdir(path):
	try:
		os.makedirs(path)
	except OSError as exc:  # Python ≥ 2.5
		if exc.errno == errno.EEXIST and os.path.isdir(path):
			pass
		# possibly handle other errno cases here, otherwise finally:
		else:
			raise

if __name__ == "__main__":

  import pathlib
  path = pathlib.Path(__file__).parent.resolve()

  mkdir(f"{path}/../pretrained")
  mkdir(f"{path}/dataset")

  agp_load = sys.argv[1]=="True"

  Amazon2M(path, agp_load)
