import time
import random
import argparse
import uuid
import resource
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from model import GnnAGP
from utils import load_inductive,mutilabel_f1
from timer_perf import TimerPerf
from memory_profiler import profile

BR_LEN = 100
BR_LENH = BR_LEN // 2
results = {"accur" : [], 
"memGB" : [], 
"epochs_time" : [],
"total_train_time" : [],
#"time_predict_test" : [],
#"time_prep" : [],
"train_prop_t" : [],
"train_prop_clock_t" : [],
"full_prop_t" : [],
"full_prop_clock_t" : [],
}

global_timer = TimerPerf()
features_train,features,labels,idx_train,idx_val,idx_test,memory_dataset = None, None, None, None, None, None, None
args = None
model = None
loader = None
optimizer = None
loss_fn = None
checkpt_file = None

def evaluate(model,feats_val,labels_val):
    model.eval()
    with torch.no_grad():
        logits = model(feats_val)
        f1_mic = mutilabel_f1(labels_val.cpu().numpy(),logits.cpu().numpy())
        return f1_mic
        

def train():
    timer = TimerPerf()
    timer.lap()
    model.train()
    timer.lap("train.train")
    loss_list = []
    en = enumerate(loader)
    timer.lap("enum")
    time_ep = 0
    for step, (batch_x, batch_y) in en:
        batch_x = batch_x.cuda(args.dev)
        batch_y = batch_y.cuda(args.dev)
        timer.lap("cuda")
        t0 = time.perf_counter()
        optimizer.zero_grad()
        output = model(batch_x)
        loss_train = loss_fn(output, batch_y)
        loss_train.backward()
        optimizer.step()
        time_ep = time.perf_counter() - t0
        timer.lap("epoch")
        loss_list.append(loss_train.item())
        mean = np.mean(loss_list)
        timer.lap("train_etc")
    torch_mem_gb = torch.cuda.memory_allocated(0)/1024/1024/1024
    print(f" torch mem {torch_mem_gb} ")
    return mean, time_ep, timer, torch_mem_gb

#@profile(precision=4)
def validate():
    return evaluate(model,features[idx_val].cuda(args.dev),labels[idx_val])

def test():
    model.load_state_dict(torch.load(checkpt_file))
    return evaluate(model,features[idx_test].cuda(args.dev),labels[idx_test])
#@profile(precision=4)
def run(seed):
    bad_counter = 0
    best = 0
    best_epoch = 0
    geval_timer = TimerPerf()
    time_eps = 0
    print("-"*BR_LENH)
    print("Training...")
    torch_mem_gb_list = []
    for epoch in range(args.epochs):
        loss_tra, time_ep, timer, torch_mem_gb = train()
        torch_mem_gb_list.append(torch_mem_gb)
        time_eps += time_ep
        geval_timer.merge(timer)
        #timer.print()
        print(f"ep:{epoch} : {timer.get('epoch'):.4f}", end=" | ")
        
        val_acc=validate()
        if(epoch+1)%(args.epochs/4)== 0:
            print(f'Epoch:{epoch+1:02d},'
                f'Train_loss:{loss_tra:.3f}',
                f'Valid_acc:{100*val_acc:.2f}% ',
                f"Train_timer:{geval_timer.get('epoch'):.3f}")
            geval_timer.print()
        if val_acc > best:
            best = val_acc
            best_epoch = epoch+1
            torch.save(model.state_dict(), checkpt_file)
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

    geval_timer.lap()
    test_acc = test()
    geval_timer.lap("test")
    print(f"Train cost: {time_eps:>10.2f}s")
    print(f'Load {best_epoch:>10}th epoch')
    print(f"Test accuracy:{100*test_acc:>10.2f}%")

    memory_main = 1024 * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30
    memory=memory_main-memory_dataset
    print(f"Memory overhead:{memory:>10.2f}GB")
    print("-"*BR_LENH)
    print(f"train_time_all: {geval_timer.total():>10}")
    global_timer.merge(geval_timer)

    
    results["accur"].append(test_acc)
    results["memGB"].append(memory)
    results["epochs_time"].append(geval_timer.get('epoch'))

    total_train_time = geval_timer.get('train.train') + geval_timer.get('enum')+ geval_timer.get('cuda')+ geval_timer.get('epoch')+ geval_timer.get('train_etc')
    results["total_train_time"].append(total_train_time)
    results["torch_mem_gb"].append(torch_mem_gb)

#@profile(precision=4)
def evaluate_all():
    # Training settings
    parser = argparse.ArgumentParser()
    # Dataset and Algorithom
    #parser.add_argument('--seed', type=int, default=20159, help='random seed..')
    parser.add_argument('--dataset', default='yelp', help='dateset.')
    parser.add_argument('--agp_alg',default='appnp_agp',help='APG algorithm.')
    # Algorithm parameters
    parser.add_argument('--alpha', type=float, default=0.2, help='alpha for APPNP_AGP.')
    parser.add_argument('--ti',type=float,default=4,help='t for GDC_AGP.')
    parser.add_argument('--rmax', type=float, default=1e-7, help='threshold.')
    parser.add_argument('--L', type=int, default=3,help='propagation levels.')
    # Learining parameters
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay.')
    parser.add_argument('--layer', type=int, default=1, help='number of layers.')
    parser.add_argument('--hidden', type=int, default=256, help='hidden dimensions.')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate.')
    parser.add_argument('--bias', default='bn', help='bias.')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs.')
    parser.add_argument('--batch', type=int, default=100000, help='batch size.')
    parser.add_argument('--patience', type=int, default=200, help='patience.')
    parser.add_argument('--dev', type=int, default=0, help='device id.')
    parser.add_argument('--rep_num', type=int, default=1, help='repeat num and calc avg')
    global args
    args = parser.parse_args()

    global_timer.lap()
    for seed in range(0,args.rep_num):
        
        print("#"*BR_LEN)
        print("#"*BR_LEN)
        print(f"Seed: {seed}")
        print("-"*BR_LENH)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        print("-"*BR_LENH, flush=True)
        print(args)
        global_timer.lap("seed")
        global features_train,features,labels,idx_train,idx_val,idx_test,memory_dataset,train_prop_t, train_prop_clock_t ,full_prop_t, full_prop_clock_t 
        features_train,features,labels,idx_train,idx_val,idx_test,memory_dataset,train_prop_t, train_prop_clock_t ,full_prop_t, full_prop_clock_t = load_inductive(args.dataset,args.agp_alg, args.alpha,args.ti,args.rmax,args.L)
        global_timer.lap("load_inductive")
        results["train_prop_t"].append(train_prop_t)
        results["train_prop_clock_t"].append(train_prop_clock_t)
        results["full_prop_t"].append(full_prop_t)
        results["full_prop_clock_t"].append(full_prop_clock_t)

        global checkpt_file 
        checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'

        global model
        model = GnnAGP(nfeat=features_train.shape[1],nlayers=args.layer,nhidden=args.hidden,nclass=labels.shape[1],dropout=args.dropout,bias = args.bias).cuda(args.dev)
        global_timer.lap("GnnAGP")
        global optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        global_timer.lap("optim.Adam")
        global loss_fn
        loss_fn = nn.BCEWithLogitsLoss()
        global_timer.lap("BCEWithLogitsLoss")
        labels = labels.float()

        torch_dataset = Data.TensorDataset(features_train, labels[idx_train])
        global loader
        loader = Data.DataLoader(dataset=torch_dataset,batch_size=args.batch,shuffle=True,num_workers=12)
        global_timer.lap("Data")
        if False:
            import cProfile
            cProfile.run(run())
        else:        
            run(seed)
            global_timer.lap("run")
    print("Global timer:")
    global_timer.print()

    print("="*BR_LEN)
    for key, value in results.items():
        vals = ' '.join([f"{v:.4}" for v in value])
        print(f"{key:>20}: avg: {sum(value)/len(value):<5.4f} of num: {len(value)} in {vals}")
    print("="*BR_LEN)


evaluate_all()