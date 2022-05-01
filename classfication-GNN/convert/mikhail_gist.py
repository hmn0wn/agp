def load_data(dataset_path, prefix, normalize=True):
     
    adj_full, attr_matrix, labels, train_idx, val_idx, test_idx = \
        utils.get_data(
            dataset_path,
            seed=0,
            ntrain_div_classes=20,
            normalize_attr=None)
    train_feats = attr_matrix[train_idx]
    adj_train = adj_full[train_idx, :][:, train_idx]
    #lab_ = np.zeros((len(labels), labels.max()+1))
    #for i in range(len(labels)):
    #    lab_[i][labels[i]] = 1

    #print(labels[0])
    return adj_full, adj_train, attr_matrix, train_feats, labels, train_idx, val_idx, test_idx
   
   
def graphsaint(datastr, dataset_name):
    if dataset_name == 'yelp':
        adj_full, adj_train, feats, train_feats, labels, idx_train, idx_val, idx_test = load_data(datastr, 'yelp')
        graphsave(adj_full, dir='../data/yelp_full_adj_')
        graphsave(adj_train, dir='../data/yelp_train_adj_')
        feats = np.array(feats, dtype=np.float64)
        train_feats = np.array(train_feats, dtype=np.float64)
        np.save('../data/yelp_feat.npy', feats)
        np.save('../data/yelp_train_feat.npy', train_feats)
        np.savez('../data/yelp_labels.npz', labels=labels, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)
    if dataset_name == 'reddit':
        adj_full, adj_train, feats, train_feats, labels, idx_train, idx_val, idx_test = load_data(datastr, 'reddit')
        graphsave(adj_full, dir='../data/reddit_full_adj_')
        graphsave(adj_train, dir='../data/reddit_train_adj_')
        feats = np.array(feats, dtype=np.float64)
        train_feats = np.array(train_feats, dtype=np.float64)

        #labels = np.where(labels > 0.5)[1]
        print(labels[0])
        np.save('../data/reddit_feat.npy', feats)
        np.save('../data/reddit_train_feat.npy', train_feats)
        np.savez('../data/reddit_labels.npz', labels=labels, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)


if __name__ == "__main__":
    # Your file storage path. For example, this is shown below.
    datastr = "../data"

    # dataset name, yelp or reddit
    dataset_name = 'yelp'
    graphsaint(datastr, dataset_name) 