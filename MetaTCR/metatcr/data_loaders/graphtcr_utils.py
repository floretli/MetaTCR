import numpy as np
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import random
import os
import glob
from sklearn.model_selection import KFold

class GraphTcrUtil:
    @staticmethod
    def add_args(parser):
        parser.set_defaults(gnn_dropout=0.5)
        parser.set_defaults(gnn_emb_dim=128)

    # @staticmethod
    # def loss_fn():
    #     def calc_loss(pred, batch):
    #         loss = F.cross_entropy(pred, batch.y)
    #         return loss
    #     return calc_loss
    #
    # @staticmethod
    # @torch.no_grad()
    # def eval(model, device, loader):
    #     model.eval()
    #
    #     correct = 0
    #     pred_probs = []
    #     true_labels = []
    #
    #     for step, batch in enumerate(tqdm(loader, desc="Eval")):
    #         batch = batch.to(device)
    #         pred = model(batch)
    #         pred_probs += F.softmax(pred, dim = 1)[:, 1].cpu().tolist()
    #         true_labels += batch.y.cpu().tolist()
    #         pred = pred.max(dim=1)[1]
    #         correct += pred.eq(batch.y).sum().item()
    #
    #     return {"acc": correct / len(loader.dataset), "auc": roc_auc_score(true_labels, pred_probs)}

    @staticmethod
    def make_adj_list(N, edge_index_transposed):  ## N = number of nodes
        A = np.eye(N)
        for edge in edge_index_transposed:
            A[edge[0], edge[1]] = 1
        adj_list = A != 0
        return adj_list

    @staticmethod
    def split_dataset_n_folds(sample_names, n_folds=5, shuffle=True, random_state=None, save_split_result=False,
                              directory="./results/nfold_smpid", output_prefix="split"):
        ## sample_names: list of sample names

        kf = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
        fold_splits = []

        for i, (train_indices, valid_indices) in enumerate(kf.split(sample_names)):
            train_sample_names = [sample_names[i] for i in train_indices]
            valid_sample_names = [sample_names[i] for i in valid_indices]

            if save_split_result:
                os.makedirs(directory, exist_ok=True)
                with open(directory + f"/{output_prefix}_train_fold_{i + 1}.txt", "w") as train_file:
                    train_file.write("\n".join(train_sample_names))

                with open(directory + f"/{output_prefix}_valid_fold_{i + 1}.txt", "w") as valid_file:
                    valid_file.write("\n".join(valid_sample_names))

            fold_splits.append((train_indices, valid_indices))

        return fold_splits

    @staticmethod
    def load_n_fold_splits_from_files(sample_names, directory="./results/nfold_smpid", output_prefix="split"):
        fold_splits = []
        train_files_pattern = os.path.join(directory, f"{output_prefix}_train_fold_*.txt")

        train_files = sorted(glob.glob(train_files_pattern))
        n_folds = len(train_files)

        for i in range(n_folds):
            train_sample_names = GraphTcrUtil.load_sample_names(
                os.path.join(directory, f"{output_prefix}_train_fold_{i + 1}.txt"))
            valid_sample_names = GraphTcrUtil.load_sample_names(
                os.path.join(directory, f"{output_prefix}_valid_fold_{i + 1}.txt"))

            # Convert sample names to indices
            train_sample_indices = [sample_names.index(name) for name in train_sample_names]
            valid_sample_indices = [sample_names.index(name) for name in valid_sample_names]

            fold_splits.append((train_sample_indices, valid_sample_indices))

        return fold_splits

    @staticmethod
    def load_sample_names(filename):
        with open(filename, "r") as file:
            sample_names = file.read().splitlines()
        return sample_names

    @staticmethod
    def get_fold_datasets(dataset, splits):
        fold_datasets = []

        for train_indices, valid_indices in splits:
            train_dataset = [dataset[i] for i in train_indices]
            valid_dataset = [dataset[i] for i in valid_indices]
            fold_datasets.append((train_dataset, valid_dataset))

        return fold_datasets

    @staticmethod
    def get_id_and_label(df, smpid_col="case_id", label_col="label"):
        sample_list = df[smpid_col].tolist()
        label_list = df[label_col].tolist()

        return sample_list, label_list

    @staticmethod
    def create_dataset(smp2features, edge_index, num_nodes, sample_list, label_list, shuffle_edge=False):  ##create dataset from Data objects
        dataset = []

        if shuffle_edge == True:
            r = torch.randperm(edge_index.shape[1])
            edge_index[0] = edge_index[0][r]

        adj_list = GraphTcrUtil.make_adj_list(num_nodes, edge_index.T)

        for i in range(len(sample_list)):
            smp = sample_list[i]
            label = torch.as_tensor([label_list[i]])  ## tensor ([0])
            feature = smp2features[smp]  ## datatype: tensor mtx

            x1 = Data(x=feature, edge_index=edge_index, y=label)
            x1["num_nodes"] = num_nodes
            x1["adj_list"] = adj_list
            x1["name"] = smp
            dataset.append(x1)
        return dataset

    @staticmethod
    def create_loaders(smp2features, edge_index, args, maindata_df, testdata_df = None):

        if args.use_test_data and testdata_df is None:
            raise ValueError("use_test_data is set to True, but testdata_df is not available.")

        main_sample_list, main_label_list = GraphTcrUtil.get_id_and_label(maindata_df, smpid_col="case_id", label_col="label")
        main_dataset = GraphTcrUtil.create_dataset(smp2features, edge_index, args.num_nodes, main_sample_list, main_label_list, args.num_nodes)
        eval_bs = args.batch_size if args.eval_batch_size is None else args.eval_batch_size

        num_classes = len(set(main_label_list))
        feature_dim = main_dataset[0].x.shape[1]

        if args.use_test_data:
            test_sample_list, test_label_list = GraphTcrUtil.get_id_and_label(testdata_df, smpid_col="case_id", label_col="label")
            test_dataset = GraphTcrUtil.create_dataset(smp2features, edge_index, args.num_nodes, test_sample_list, test_label_list, args.num_nodes)
            test_loader = DataLoader(test_dataset, batch_size=eval_bs, shuffle=False, num_workers=args.num_workers,
                                     pin_memory=True)
        else:
            test_loader = None

        if args.fold_num == 1:
            train_dataset, valid_dataset = GraphTcrUtil.split_dataset_2parts(main_dataset, split_rate=0.8, shuffle=True, addnoise=False)
            train_loader = DataLoader(main_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
            train_loader_eval = DataLoader(main_dataset, batch_size=eval_bs, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            valid_loader = DataLoader(valid_dataset, batch_size=eval_bs, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            return [(train_loader, train_loader_eval, valid_loader, test_loader)], num_classes, feature_dim

        splits = GraphTcrUtil.split_dataset_n_folds(main_sample_list, n_folds=args.fold_num, random_state=args.seed)
        fold_datasets = GraphTcrUtil.get_fold_datasets(main_dataset, splits)

        loaders = []
        for fold_index, (train_dataset, valid_dataset) in enumerate(fold_datasets):
            print(f"Processing data for fold {fold_index + 1}/{len(fold_datasets)}...")

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers, pin_memory=True)
            train_loader_eval = DataLoader(train_dataset, batch_size=eval_bs, shuffle=False,
                                           num_workers=args.num_workers, pin_memory=True)
            valid_loader = DataLoader(valid_dataset, batch_size=eval_bs, shuffle=False, num_workers=args.num_workers,
                                      pin_memory=True)
            loaders.append((train_loader, train_loader_eval, valid_loader, test_loader))

            ## show the length of each dataset
            print(f"...size of train_dataset: {len(train_dataset)};", f"size of valid_dataset: {len(valid_dataset)} ... " )
            # ## show the case name of valid dataset
            # print(f"...valid_dataset: {valid_dataset[0].name} ...")


        return loaders, num_classes, feature_dim ## loaders: list of tuples

    @staticmethod
    def split_dataset_3parts(dataset, shuffle = True):
        num_train = int(len(dataset) * 0.8)
        num_val = int(len(dataset) * 0.1)
        num_test = len(dataset) - (num_train + num_val)

        if shuffle == True:
            train_set, val_set, test_set = random_split(dataset, [num_train, num_val, num_test])
        else:
            train_set, val_set, test_set = dataset[:num_train], dataset[num_train:num_train+num_val], dataset[:-num_test]

        return train_set, val_set, test_set

    @staticmethod
    def split_dataset_2parts(dataset, split_rate = 0.8, shuffle = True, addnoise = False):
        num_train = int(len(dataset) *split_rate)
        num_val = len(dataset) - num_train

        if shuffle == True:
            train_set, val_set = random_split(dataset, [num_train, num_val])
        else:
            train_set, val_set = dataset[:num_train], dataset[num_train:]

        if addnoise == True:
            train_set = GraphTcrUtil.data_augment(train_set)

        return train_set, val_set

    @staticmethod
    def data_augment(dataset, repeat = 3, add_rate = 0.1):
        ## test
        new_dataset = []
        for i in range(repeat):
            for data in dataset:

                new_data = data.clone()
                ## add noise with randomly dropout some nodes
                noised_nodes = random.sample(range(data.x.shape[0]), int(data.x.shape[0] * add_rate))

                for node in noised_nodes:
                    new_data.x[node] = data.x[node] + (torch.rand_like(data.x[node]) - 0.5) * 0.02

                new_dataset.append(new_data)

        ## shuffle
        new_dataset = random.sample(new_dataset, len(new_dataset))
        return new_dataset
