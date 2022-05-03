import torch
import gc
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from propagation import AGP
import os, psutil
from memory_profiler import profile

from inspect import currentframe

def get_linenumber():
    cf = currentframe()
    return cf.f_back.f_lineno

#@profile(precision=4)
def load_inductive(datastr,agp_alg,alpha,t,rmax,L):
	if datastr=="citeseer_semi":
		train_m = 27; train_n = 120
		full_m = 7388; full_n = 2110
	if datastr=="cora_ml_semi":
		train_m = 64; train_n = 140
		full_m = 15962; full_n = 2810
	if datastr=="pubmed_semi":
		train_m = 0; train_n = 60
		full_m = 88648; full_n = 19717
		#EL: (88648,)  full_m
		#PL: (19718,)  full_n + 1
		#EL: (0,)      train_m
		#PL: (61,)     train_n + 1
	if datastr=="cora_full_semi":
		train_m = 854; train_n = 1392
		full_m = 125370; full_n = 18800
	if datastr=='reddit_semi':
		train_m=1144; train_n=820
		full_m=114615892; full_n=232965

	if datastr=='yelp_semi':
		train_m = 420; train_n = 420
		full_m=13927667; full_n=703655
	if datastr=="ogbn-products_semi":
		train_m=284; train_n=874
		full_m=123612734; full_n=2385903
	if datastr=="amazon_semi":
		train_m = 1950; train_n = 1500
		full_m = 263793649; full_n = 1066627

	if datastr=='reddit':
		train_m=10907170; train_n=153932
		full_m=23446803; full_n=232965
		
	py_agp=AGP()
	process = psutil.Process(os.getpid())
	ram_prev = process.memory_info().rss
	print("--------------------------", flush=True)
	print("For train features propagation:", flush=True)
	features_train=np.load('data/'+datastr+'_train_feat.npy')
	train_prop_t = np.array([0],dtype=np.double)
	train_prop_clock_t = np.array([0],dtype=np.double)
	ram_delta = process.memory_info().rss - ram_prev
	ram_prev = process.memory_info().rss
	print(f"RAM{get_linenumber()}: {ram_delta}")
	_=py_agp.agp_operation(datastr+'_train',agp_alg,train_m,train_n,L,rmax,alpha,t,features_train, train_prop_t, train_prop_clock_t )

	features =np.load('data/'+datastr+'_feat.npy')
	print("--------------------------", flush=True)
	print("For full features propagation:", flush=True)
	full_prop_t, full_prop_clock_t = np.array([0],dtype=np.double), np.array([0],dtype=np.double)

	memory_dataset=py_agp.agp_operation(datastr+'_full',agp_alg,full_m,full_n,L,rmax,alpha,t,features,full_prop_t, full_prop_clock_t)

	features_train = torch.FloatTensor(features_train)
	features = torch.FloatTensor(features)
	data = np.load("data/"+datastr+"_labels.npz")
	labels = data['labels']
	idx_train = data['idx_train']
	idx_val = data['idx_val']
	idx_test = data['idx_test']
	labels = torch.LongTensor(labels)
	idx_train = torch.LongTensor(idx_train)
	idx_val = torch.LongTensor(idx_val)
	idx_test = torch.LongTensor(idx_test)
	ram_delta = process.memory_info().rss - ram_prev
	ram_prev = process.memory_info().rss
	print(f"RAM{get_linenumber()}: {ram_delta}")
	return features_train,features,labels,idx_train,idx_val,idx_test,memory_dataset, train_prop_t[0], train_prop_clock_t[0] , full_prop_t[0], full_prop_clock_t[0]

def load_transductive(datastr,agp_alg,alpha,t,rmax,L):
	if(datastr=="papers100M"):
		m=3339184668; n=111059956
		#m=2484941; n=169343 ogbn-arxiv
	
	py_agp=AGP()
	print("Load graph and initialize! It could take a few minutes...")
	features =np.load('data/'+datastr+'_feat.npy')
	memory_dataset= py_agp.agp_operation(datastr,agp_alg,m,n,L,rmax,alpha,t,features)
	features = torch.FloatTensor(features)
	#print(features.shape)

	data = np.load("data/"+datastr+"_labels.npz")
	train_idx = torch.LongTensor(data['train_idx'])
	val_idx = torch.LongTensor(data['val_idx'])
	test_idx =torch.LongTensor(data['test_idx'])
	train_labels = torch.LongTensor(data['train_labels'])
	val_labels = torch.LongTensor(data['val_labels'])
	test_labels = torch.LongTensor(data['test_labels'])
	train_labels=train_labels.reshape(train_labels.size(0),1)
	val_labels=val_labels.reshape(val_labels.size(0),1)
	test_labels=test_labels.reshape(test_labels.size(0),1)
	return features,train_labels,val_labels,test_labels,train_idx,val_idx,test_idx,memory_dataset

class SimpleDataset(Dataset):
	def __init__(self,x,y):
		self.x=x
		self.y=y
		assert self.x.size(0)==self.y.size(0)

	def __len__(self):
		return self.x.size(0)

	def __getitem__(self,idx):
		return self.x[idx],self.y[idx]

def muticlass_f1(output, labels):
	preds = output.max(1)[1]
	preds = preds.cpu().detach().numpy()
	labels = labels.cpu().detach().numpy()
	micro = f1_score(labels, preds, average='micro')
	return micro

def mutilabel_f1(y_true, y_pred):
	y_pred[y_pred > 0] = 1
	y_pred[y_pred <= 0] = 0
	return f1_score(y_true, y_pred, average="micro")

def acc_f1(output, labels):
    preds = np.argmax(output, axis=1) #output.max(1)[1]
    #preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    micro = f1_score(labels, preds, average='micro')
    return micro
