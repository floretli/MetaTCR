from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

class BagDataset(Dataset):
    def __init__(self, bags, subbag_size = 500):  ## bags is a list of bags
        self.bags = bags
        self.subbag_size = subbag_size

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        bag1 = self.bags[idx]
        bag2_idx = np.random.choice(len(self.bags))
        bag2 = self.bags[bag2_idx]

        num_features = bag1.shape[1]

        pos_sample1_padded = np.zeros((self.subbag_size, num_features))
        pos_sample2_padded = np.zeros((self.subbag_size, num_features))

        pos_idx1 = np.random.choice(len(bag1), self.subbag_size, replace=True)
        pos_sample1_padded[:len(pos_idx1)] = bag1[pos_idx1]

        pos_idx2_padded = np.random.choice(len(bag1), self.subbag_size, replace=True)
        pos_sample2_padded[:len(pos_idx2_padded)] = bag1[pos_idx2_padded]

        pos_sample1 = pos_sample1_padded
        pos_sample2 = pos_sample2_padded

        neg_sample_padded = np.zeros((self.subbag_size, num_features))

        neg_idx = np.random.choice(len(bag2), self.subbag_size, replace=True)  ## sample M instances from bag2
        neg_sample_padded[:len(neg_idx)] = bag2[neg_idx]

        neg_sample = neg_sample_padded

        return pos_sample1, pos_sample2, neg_sample
    
## Unsupervised training for contrast learning
class ClusterBagDataset(Dataset):  ## contains clusters after mapping to centroids

    def __init__(self, superbags, use_simu_data = False, subbag_size = 30, data_repeat = 1):
        self.bags = []  ## one bag is a tensor matrix
        self.sample_ids = []
        self.bag_orders = []
        self.subbag_size = subbag_size
        self.num_features = 0
        self.sample_num = 0
        self.data_repeat = data_repeat

        if use_simu_data:
            self.simu_data()
        else:
            ## check if features is a dict
            assert isinstance(superbags, list), "superbags should be a 2-dim list"
            assert superbags[0][0].shape[1] is not None, "superbags should be a 2-dim list, contains tensors"
            self.split_bags(superbags)

    def split_bags(self, superbags):
        cluster_num = len(superbags[0])
        self.num_features = superbags[0][0].shape[1]
        self.sample_num = len(superbags)

        for smp_id in range(self.sample_num):
            self.sample_ids += [smp_id] * cluster_num
            self.bag_orders += [i for i in range(cluster_num)]
            self.bags += superbags[smp_id]  ## a list of tensors

    def simu_data(self):
        num_samples = 50
        num_bags = 10
        min_instances = 20
        max_instances = 100
        num_features = 96

        self.sample_num = num_samples

        print("Simulate data for testing ...")

        for smp_id in range(num_samples):
            X, y = make_blobs(n_samples=max_instances * num_bags, centers=1, n_features=num_features)
            gmm = GaussianMixture(n_components=3)
            gmm.fit(X)

            for bag_order in range(num_bags):
                num_instances = np.random.randint(min_instances, max_instances)
                X_new, _ = gmm.sample(num_instances)
                X_new = X_new + bag_order * 0.1
                self.bags.append(torch.from_numpy(X_new))
            self.sample_ids += [smp_id] * num_bags
        self.bag_orders = [i for i in range(num_bags)] * num_samples
        self.num_features = num_features

        print("Done. ({} bags, {} samples, {}-dim features)".format(len(self.bags), self.sample_num, self.num_features))
        # ### show the data
        # print("Sample 1 bag 0: ", self.bags[0].shape)
        # print("bag 0's feature: ", self.bags[0])
        # print("Sample 1 , last bag: ", self.bags[num_bags - 1].shape)
        # print("last bag's feature: ", self.bags[num_bags - 1])
        # print("size of self.bags: ", len(self.bags))
        # print("size of self.sample_ids: ", len(self.sample_ids))
        # print("size of self.bag_orders: ", len(self.bag_orders))

    def __len__(self):
        return len(self.bags) * self.data_repeat

    def __getitem__(self, idx):

        idx = idx % len(self.bags)

        bag1 = self.bags[idx]
        sample_id = self.sample_ids[idx]
        bag_order = self.bag_orders[idx]

        ## sample a negative bag with the same bag_order and different sample_id

        neg_sample_id = np.random.choice([i for i in range(self.sample_num) if i != sample_id])

        neg_idx = self.sample_ids.index(neg_sample_id) + bag_order
        bag2 = self.bags[neg_idx]  ## negative bag

        # ## test
        # print(" ### ")
        # print("testing sample subbags ...")
        # print("this idx: ", idx)
        # print("sample_id: ", sample_id)
        # print("bag_order: ", bag_order)
        # print("neg_sample_id: ", neg_sample_id)
        # print("neg_idx: ", neg_idx)

        ## sample subbags
        pos_sample1_padded = np.zeros((self.subbag_size, self.num_features))
        pos_sample2_padded = np.zeros((self.subbag_size, self.num_features))

        pos_idx1 = np.random.choice(len(bag1), self.subbag_size, replace=True)
        pos_sample1_padded[:len(pos_idx1)] = bag1[pos_idx1]

        pos_idx2 = np.random.choice(len(bag1), self.subbag_size, replace=True)
        pos_sample2_padded[:len(pos_idx2)] = bag1[pos_idx2]

        neg_sample_padded = np.zeros((self.subbag_size, self.num_features))
        neg_idx = np.random.choice(len(bag2), self.subbag_size, replace=True)  ## sample M instances from bag2
        neg_sample_padded[:len(neg_idx)] = bag2[neg_idx]

        return pos_sample1_padded, pos_sample2_padded, neg_sample_padded

class ClusterBagMultiDataset(Dataset):  ## contains clusters after mapping to centroids; merges multi datasets

    def __init__(self, superbags, dataset_ids, subbag_size = 30, data_repeat = 1):
        self.bags = []  ## one bag is a tensor matrix
        self.sample_ids = []
        self.bag_orders = []
        self.bag_setids = []
        self.subbag_size = subbag_size
        self.num_features = 0
        self.sample_num = 0
        self.data_repeat = data_repeat
        self.dataset_ids = dataset_ids  ## len(dataset_ids) = len(superbags)

        ## check if features is a dict
        assert isinstance(superbags, list), "superbags should be a 2-dim list"
        assert superbags[0][0].shape[1] is not None, "superbags should be a 2-dim list, contains tensors"
        self.split_bags(superbags)

    def split_bags(self, superbags):
        cluster_num = len(superbags[0])
        self.num_features = superbags[0][0].shape[1]
        self.sample_num = len(superbags)

        for smp_id in range(self.sample_num):
            self.sample_ids += [smp_id] * cluster_num
            self.bag_orders += [i for i in range(cluster_num)]
            self.bags += superbags[smp_id]  ## a list of tensors, each tensor is a cluster
            self.bag_setids += [self.dataset_ids[smp_id]] * cluster_num
        ## check the length of each list
        assert len(self.bags) == len(self.sample_ids) == len(self.bag_orders) == len(self.bag_setids), "length of bags, sample_ids, bag_orders, sample_setids should be the same"

    def __len__(self):
        return len(self.bags) * self.data_repeat

    def __getitem__(self, idx):

        idx = idx % len(self.bags)

        bag1 = self.bags[idx]
        sample_id = self.sample_ids[idx]
        bag_order = self.bag_orders[idx]
        set_id = self.bag_setids[idx]

        ## sample a negative bag with the same bag_order and different sample_id
        candidate_neg_ids = [i for i in range(self.sample_num) if self.dataset_ids[i] == set_id]
        candidate_neg_ids.remove(sample_id)
        neg_sample_id = np.random.choice(candidate_neg_ids)

        neg_bag_idx = self.sample_ids.index(neg_sample_id) + bag_order
        bag2 = self.bags[neg_bag_idx]  ## negative bag

        ## sample subbags
        pos_sample1_padded = np.zeros((self.subbag_size, self.num_features))
        pos_sample2_padded = np.zeros((self.subbag_size, self.num_features))

        pos_idx1 = np.random.choice(len(bag1), self.subbag_size, replace=True)
        pos_sample1_padded[:len(pos_idx1)] = bag1[pos_idx1]

        pos_idx2 = np.random.choice(len(bag1), self.subbag_size, replace=True)
        pos_sample2_padded[:len(pos_idx2)] = bag1[pos_idx2]

        neg_sample_padded = np.zeros((self.subbag_size, self.num_features))
        neg_idx = np.random.choice(len(bag2), self.subbag_size, replace=True)  ## sample M instances from bag2
        neg_sample_padded[:len(neg_idx)] = bag2[neg_idx]

        return pos_sample1_padded, pos_sample2_padded, neg_sample_padded


import pickle
def load_pkfile(filename):
    with open(filename, "rb") as fp:
        data_dict = pickle.load(fp)
    return data_dict

if __name__ == "__main__":

    ## use one dataset
    # dataset = ClusterBagDataset(use_simu_data = True)
    # dataset.__getitem__(105)

    ## use multiple datasets

    pretrain_bags = load_pkfile('../data/example_pretrain_bags.pk')
    dataset_ids = load_pkfile('../data/example_dataset_ids.pk')

    print("pretrain_bags ", len(pretrain_bags))
    print("pretrain_bags 0, len: ", len(pretrain_bags[0]))
    # print("pretrain_bags 0, 0: ", pretrain_bags[0][0])
    print("dataset_ids ", dataset_ids)
    dataset =ClusterBagMultiDataset(pretrain_bags, dataset_ids, subbag_size=30, data_repeat=1)
    print("dataset length: ", len(dataset))  ## sample_num * cluster_num
    dataset.__getitem__(105)



