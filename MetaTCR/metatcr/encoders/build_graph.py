import numpy as np
import os
import pandas as pd
import time
import torch
import random
import pickle
import tqdm
# import mkl
# mkl.get_max_threads()
import faiss
from .tcr2vec_tcr_encoder import seqlist2ebd, load_tcr2vec


random.seed(0)
file_cut_size = None  ## set None for all data, use how many files from the filelist to run script
db_cut_size = 1000  ## how many tcrs in each repertoire to process pre-clustering
ge_x_cut_size = None
pretrain_cut_size = None

tcr_type = "full_length"  ## full_length or cdr3
freq_col = 'frequencyCount (%)'  ## col name of frequency or counts
tcr_col = 'aminoAcid'  ## col name of full length full_seq or aminoAcid
emb_model_path = None
emb_model = None

def update_graph_config(args):
    global file_cut_size, db_cut_size, tcr_type, freq_col, tcr_col, emb_model_path
    file_cut_size = getattr(args, 'file_cut_size', file_cut_size)
    db_cut_size = getattr(args, 'db_cut_size', db_cut_size)
    freq_col = getattr(args, 'freq_col', freq_col)
    tcr_col = getattr(args, 'tcr_col', tcr_col)
    emb_model_path = getattr(args, 'emb_model_path', emb_model_path)
    if emb_model_path != None:
        emb_model = load_tcr2vec(emb_model_path)


class count_graph:
    def __init__(self, clst_num = 10):
        self.shape = [clst_num, clst_num]
        self.nodes = [i for i in range(clst_num)]
        self.count_mtx = np.zeros(self.shape)  ## init with 0

    # def add_edge_weight(self, clst_ids):  ## just co exist
    #
    #     for m in clst_ids:
    #         for n in clst_ids:
    #             if m == n :
    #                 continue
    #             self.count_mtx[m,n] += 1

    def add_unequal_weight(self, clst_ids, weight):  ## just co exist
        for m in clst_ids:
            for n in clst_ids:
                if m == n :
                    continue
                self.count_mtx[m,n] += weight

    def get_count_mtx(self):
        return self.count_mtx

def load_pkfile(filename):
    with open(filename, "rb") as fp:
        data_dict = pickle.load(fp)
    return data_dict

def save_pk(file_savepath, data):
    with open(file_savepath, "wb") as fp:
        pickle.dump(data, fp)

def read_filelist(filepath_txt, return_smplist = False):

    if isinstance(filepath_txt, list):
        total_filelist = []
        for f in filepath_txt:
            filelist = open(f, "r").read().split("\n")
            filelist.remove("")
            total_filelist += filelist
    else:
        total_filelist = open(filepath_txt, "r").read().split("\n")
        total_filelist.remove("")

    for filepath in total_filelist:
        if not (os.path.isfile(filepath)):
            total_filelist.remove(filepath)
            print(f"Warning: file {filepath} does not exist. Removing from list.")

    if file_cut_size:
        ## random select files from the filelist
        random.shuffle(total_filelist)
        total_filelist = total_filelist[:file_cut_size]
    if return_smplist:
        ## get basename of each file
        smp_list = [os.path.basename(f).replace('_filt_full.tsv', '') for f in total_filelist]
        return total_filelist, smp_list
    return total_filelist


def read_filelist_from_dir(directory, format = ".tsv", suffix="_filt_full.tsv"):
    # Initialize empty lists for the file paths and case ids
    file_list = []
    case_list = []

    # Iterate over all files in the given directory
    for filename in os.listdir(directory):
        # Check if the file has the given suffix
        if filename.endswith(format):
            # Add the full file path to the file list
            file_list.append(os.path.join(directory, filename))
            # Remove the suffix to get the case id, and add it to the case list
            case_id = filename[:-len(suffix)]
            case_list.append(case_id)
    if file_cut_size:
        ## random select files from the filelist
        random.shuffle(file_list)
        file_list = file_list[:file_cut_size]
        case_list = [os.path.basename(f).replace(suffix, '') for f in file_list]

    return file_list, case_list

## calcu clst label after training
def compute_cluster_assignment(centroids, x):
    assert centroids is not None, "should train before assigning"
    d = centroids.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(centroids)
    distances, labels = index.search(x, 1)
    return labels.ravel()

def kmeans_clustering(X, cluster_num = 50):  ## X is a 2-dim np.array
    assert np.all(~np.isnan(X)), 'x contains NaN'
    assert np.all(np.isfinite(X)), 'x contains Inf'
    d = X.shape[1]
    kmeans = faiss.Clustering(d, cluster_num)
    kmeans.verbose = bool(0)
    kmeans.niter = 100
    kmeans.nredo = 1

    # otherwise the kmeans implementation sub-samples the training set
    kmeans.max_points_per_centroid = 10000000

    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)
    kmeans.train(X, index)
    # seq_labels = km.labels_
    centroids = faiss.vector_float_to_array(kmeans.centroids).reshape(cluster_num, d)
    labels = compute_cluster_assignment(centroids, X)

    ## get the obj value
    iteration_stats = kmeans.iteration_stats
    # Get the number of iterations
    num_iterations = iteration_stats.size()

    # Get the stat for the last iteration
    last_iteration_stat = iteration_stats.at(num_iterations - 1)

    return labels, centroids, last_iteration_stat

def seqlist2clst(seq_list, cluster_num = 50):
    
    tx1 = time.time()
    X = seqlist2ebd(seq_list, emb_model, keep_pbar=True)
    tx2 = time.time()
    print('[INFO]TCR2vec: Encoding total elapsed time: %.3f m' % ((tx2 - tx1) / 60.0))

    ### # perform the training
    t0 = time.time()
    labels, centroids, _ = kmeans_clustering(X, cluster_num)
    t1 = time.time()
    print('[INFO]FAISS: Clustering total elapsed time: %.3f m' % ((t1 - t0) / 60.0))

    seq2label = dict(zip(seq_list, labels))
    # print("mertring clusters...")
    # si = silhouette_score(X, labels)
    # ch = calinski_harabasz_score(X, labels)
    # print("silhouette_score: ", si)
    # print("calinski_harabasz_score: ", ch)
    return seq2label, centroids  ## dict, list

def kmeans_traverse_k(X, cluster_range):  ## X is a 2-dim np.array, cluster_range is a list

    all_cluster_labels = []
    all_centers = []
    all_errs = []
    all_imbalances = []
    length = X.shape[0]

    for cluster_num in cluster_range:
        print(f"try clustering with {cluster_num} clusters...")
        labels, centroids, last_stat = kmeans_clustering(X, cluster_num)

        all_cluster_labels.append(labels)
        all_centers.append(centroids)
        all_errs.append(last_stat.obj/length)
        all_imbalances.append(last_stat.imbalance_factor)

    return all_cluster_labels, all_centers, all_errs, all_imbalances

# def check_clst_exist(seq_list, total_clst_num, seq2clst, clst_thres):
#
#     total_clst = [i for i in range(total_clst_num)]
#     clst_contents = [0] * total_clst_num
#
#     for seq in seq_list:
#         clst_id = seq2clst[seq]
#         clst_contents[clst_id] += 1
#
#     exist_clsts = []
#     for clst_id in total_clst:
#         if clst_contents[clst_id] > clst_thres:
#             exist_clsts.append(clst_id)
#     return exist_clsts

def file2seqlist(f, cut_size = db_cut_size):
    df = pd.read_csv(f, sep="\t", engine='c')
    df.sort_values(by=freq_col, ascending=False, inplace=True)
    df = df[df[tcr_col] != 'Failure']
    df = df[:cut_size]
    return df[tcr_col].to_list()

def merge_seq_list(filelist, multi_process=True):
    all_seqs = []

    if not multi_process:
        for file in filelist:
            all_seqs += file2seqlist(file, cut_size=db_cut_size)
    else:
        import multiprocessing as mp
        from functools import partial

        pool = mp.Pool(processes=mp.cpu_count())
        partial_func = partial(file2seqlist, cut_size=db_cut_size)
        results = pool.map(partial_func, filelist)
        pool.close()
        pool.join()
        all_seqs = sum(results, [])
    all_seqs = list(set(all_seqs))
    return all_seqs

def adj2edge_index(adj_mtx):  ## input: nd array (node_num x node_num), output: tensor matrix ([in1,in2, ..], [out1,out2 ..])

    node_num = adj_mtx.shape[0]
    in_nodes = []
    out_nodes = []

    for m in range(node_num):
        for n in range(node_num):
            if adj_mtx[m, n] > 0:
                in_nodes.append(m)
                out_nodes.append(n)
    return torch.as_tensor([in_nodes, out_nodes])

# def db2count_graph(filepath_txt, seq2clst, clst_num, count_graph):  ##count mtx is a 2-dim numpy array, contains co exp count number
#
#     filelist = read_filelist(filepath_txt)
#     for file in filelist:
#
#         seqlist = file2seqlist(file, db_cut_size)  ## aminoAcid or full_seq
#
#         clst_thres = int(db_cut_size / clst_num * 0.5)  ## > 50% mean clst size
#
#         exist_clsts = check_clst_exist(seqlist, clst_num, seq2clst, clst_thres)
#         count_graph.add_edge_weight(exist_clsts)
#
#     return count_graph

## get embedding from CL model. input data is a set of instances (bag)
def get_bag_ebd(data_tensor, model):  ## data_tensor: tensor with size=(N, ebd_dim), model: ContrastiveModel

    with torch.no_grad():
        embeddings = []
        for d in data_tensor:
            out = model(d.unsqueeze(0))
            embeddings.append(out.squeeze(0))
        ## list to tensor
        embeddings = torch.stack(embeddings, dim=0)
        return embeddings

# def get_bag_ebd(data_tensor, model):  ## data_tensor: tensor with size=(N, ebd_dim), model: ContrastiveModel
#
#     with torch.no_grad():
#         embeddings = []
#
#         for d in data_tensor:
#             out = model(d.unsqueeze(0), d.unsqueeze(0))
#             out1_final = out[-1]
#             embeddings.append(out1_final.squeeze(0))
#
#         ## list to tensor
#         embeddings = torch.stack(embeddings, dim=0)
#
#         return embeddings

def transfer_node_feature(seq_list, centroids, agg_type = "mean", model = None):  ## ebd_mtx: nd.array, output mtx: tensor

    total_clst_num = len(centroids)
    bag_features, ebd_dim = assign_clsts(seq_list, centroids)

    node_features = torch.empty(total_clst_num, ebd_dim)  ## init
    total_clst = [i for i in range(total_clst_num)]

    if agg_type == "mean":
        for clst_id in total_clst:
            # clst_ebd_mtx = torch.stack((bag_features[clst_id]), dim=0) ## list to tensor
            clst_ebd_mtx = bag_features[clst_id]
            node_features[clst_id] = torch.mean(clst_ebd_mtx, dim=0)

    elif agg_type == "max":
        for clst_id in total_clst:
            # clst_ebd_mtx = torch.stack(tuple(bag_features[clst_id]), dim=0)
            clst_ebd_mtx = bag_features[clst_id]
            node_features[clst_id] = torch.max(clst_ebd_mtx, dim=0)

    elif agg_type == "bag":
        node_features = get_bag_ebd(bag_features, model)


    return node_features

def assign_clsts(seq_list, centroids):  ## ebd_mtx: nd.array, output mtx: tensor

    total_clst_num = len(centroids)

    X = seqlist2ebd(seq_list, emb_model, keep_pbar = False)
    assert np.all(~np.isnan(X)), 'x contains NaN'
    assert np.all(np.isfinite(X)), 'x contains Inf'

    labels = compute_cluster_assignment(centroids, X)
    # ebd_dim = X.shape[1]

    total_clst = [i for i in range(total_clst_num)]

    ## get sub mtx when label == clst_id
    bag_features = []
    for clst_id in total_clst:
        clst_ebd_mtx = X[labels == clst_id]
        if clst_ebd_mtx.shape[0] == 0:
            clst_ebd_mtx = np.zeros((1, X.shape[1]))
        clst_ebd_mtx = torch.from_numpy(clst_ebd_mtx)
        bag_features.append(clst_ebd_mtx)

    return bag_features, X.shape[1]


def files2features(filepath_txt, centroids, sample_ids, sample_labels, sample_sets, all_features, all_rawebds, label = "0", setname = "main", agg_type = "mean", model = None):

    filelist = read_filelist(filepath_txt)

    print("processing  data from :", filepath_txt)

    for file in filelist:

        smp_id = (file.split("/")[-1]).split(".tsv")[0]  ## 016_surgery_tumor.tsv or TestReal-Keck0086_MC1.tsv
        print("processing sample : ", smp_id)

        seqlist = file2seqlist(file, ge_x_cut_size)  ## aminoAcid or full_seq

        ## need a seq 2 weight dict for each sample
        all_features[smp_id] = transfer_node_feature(seqlist, centroids, agg_type, model)  ## node features

        # all_rawebds[smp_id] = seqlist2ebd(seqlist)

        sample_ids.append(smp_id)
        sample_labels.append(label)
        sample_sets.append(setname)

def count_to_frequency(embeddings):
    row_sums = np.sum(embeddings, axis=1, keepdims=True)
    frequency_embeddings = embeddings / row_sums

    return frequency_embeddings


def files2clstbag(filepath_txt, centroids, return_smplist = False):
    all_features = {}
    filelist = read_filelist(filepath_txt)
    smp_ids = []

    for file in filelist:
        smp_id = (file.split("/")[-1]).split(".tsv")[0]  ## 016_surgery_tumor.tsv or TestReal-Keck0086_MC1.tsv
        seqlist = file2seqlist(file, pretrain_cut_size)  ## aminoAcid or full_seq
        smp_ids.append(smp_id)

        ## need a seq 2 weight dict for each sample
        all_features[smp_id], _ = assign_clsts(seqlist, centroids)  ## node features

    bags = [ all_features[smp_id]  for smp_id in all_features.keys()]

    if return_smplist:
        return bags, smp_ids
    return bags  ## list of list of tensors

def files2clstfreq(filelist, centroids, cutoff = None):

    freq_features = np.zeros((len(filelist), len(centroids)))
    freqsum_features = np.zeros((len(filelist), len(centroids)))

    with tqdm.tqdm(total=len(filelist)) as pbar:
        for idx, file in enumerate(filelist):
            pbar.update(1)
            time.sleep(0.001)

            df = pd.read_csv(file, sep = "\t")
            df.sort_values(by = freq_col, ascending = False, inplace =True)
            df = df[df[tcr_col] != 'Failure']
            df = df[:cutoff]
            seqlist = df[tcr_col].to_list()  ## aminoAcid or full_seq
            X = seqlist2ebd(seqlist, emb_model, keep_pbar = False)
            labels = compute_cluster_assignment(centroids, X)

            # Combine labels and count values into a new DataFrame
            label_counts_df = pd.DataFrame({"label": labels, "freq": df[freq_col]})

            # Compute the sum of count values for each clst_id
            label_counts_sum = label_counts_df.groupby("label")["freq"].sum()

            for clst_id in range(len(centroids)):
                freqsum_features[idx, clst_id] = label_counts_sum.get(clst_id, 0)
                freq_features[idx, clst_id] = np.sum(labels == clst_id)

    return count_to_frequency(freq_features), count_to_frequency(freqsum_features)


def db2count_graph(diversity_mtx, keep_edge_rate=0.4):  ##count mtx is a 2-dim numpy array, contains co exp count number

    (sample_num, clst_num) = diversity_mtx.shape
    occur_thres = (1 / clst_num) * 0.5 ## 50% mean clst size
    weights = np.arange(1, sample_num + 1) / sample_num

    global_graph = count_graph(clst_num)
    for sample_id in range(sample_num):
        meta_vec = diversity_mtx[sample_id, :]
        weight = weights[sample_id]
        valid_ids = np.where(meta_vec > occur_thres)[1]
        global_graph.add_unequal_weight(valid_ids, weight)
    count_mtx = global_graph.get_count_mtx()

    positive_edge_count = np.sum(count_mtx > 0)
    topk = min(int(keep_edge_rate * clst_num * clst_num), positive_edge_count)

    adj_mtx = count_mtx.copy()
    ## get top k value in adj_mtx
    topk_value = np.sort(adj_mtx, axis=None)[-topk]
    adj_mtx[adj_mtx < topk_value] = 0
    adj_mtx[adj_mtx > 0] = 1
    edge_tensor = adj2edge_index(adj_mtx)
    return edge_tensor

def build_co_occurr_graph(diversity_mtx, centroids, save_dir, keep_edge_rate=0.4):

    clst_num = len(centroids)
    occur_thres = (1 / clst_num) * 0.5

    occur_mtx = diversity_mtx.copy()
    occur_mtx[occur_mtx < occur_thres] = 0
    occur_mtx[occur_mtx > 0] = 1

    global_graph = count_graph(clst_num)
    for sample in occur_mtx:
        cluster_ids = np.where(sample == 1)[0]
        global_graph.add_edge_weight(cluster_ids)
    count_mtx = global_graph.get_count_mtx()

    zero_rows = np.sum(np.any(count_mtx, axis=1) == False)
    print(f'count_mtx has {zero_rows} rows with all zeros.')

    adj_mtx = count_mtx.copy()
    positive_edge_count = np.sum(count_mtx > 0)
    topk = min(int(keep_edge_rate * clst_num * clst_num), positive_edge_count)
    ## get top k value in adj_mtx
    topk_value = np.sort(adj_mtx, axis=None)[-topk]

    adj_mtx[adj_mtx < topk_value] = 0
    adj_mtx[adj_mtx > 0] = 1
    edge_tensor = adj2edge_index(adj_mtx)

    ## save global graph
    adj_path = os.path.join(save_dir, "adj_mtx_" + str(clst_num) + ".csv")
    edge_path = os.path.join(save_dir, "edge_tensor_" + str(clst_num) + ".pk")
    freq_mtx_path = os.path.join(save_dir, "freq_mtx_" + str(clst_num) + ".pk")

    print("freq_mtx_path , ", freq_mtx_path )
    np.savetxt(adj_path, adj_mtx, delimiter=",", fmt='%d')
    save_pk(edge_path, edge_tensor)
    # if save_freq:
    #     save_pk(freq_mtx_path, freq_mtx)
    # print("global graph saved.")

    return edge_tensor

# def build_global_graph(global_g_dbs, clst_num, save_dir, keep_edge_rate = 0.2):
#
#     print("merging TCR database")
#
#     ref_seqs = merge_seq_list(global_g_dbs)
#     print(len(ref_seqs), "unique seqs after merging.")
#
#     print("encoding and clustering TCRs from database")
#     seq2clst, clst_centers = seqlist2clst(ref_seqs, cluster_num=clst_num)
#
#     print("counting the edges in database")
#     global_graph = count_graph(clst_num)  ## init with 0
#     global_graph = db2count_graph(global_g_dbs, seq2clst, clst_num, global_graph)  ## updata count mtx
#     count_mtx = global_graph.get_count_mtx()
#
#     print("building and saving global graph")
#     adj_mtx = count_mtx.copy()
#     topk = int(keep_edge_rate * clst_num * clst_num)
#     ## get top k value in adj_mtx
#     topk_value = np.sort(adj_mtx, axis=None)[-topk]
#
#     adj_mtx[adj_mtx < topk_value] = 0
#     adj_mtx[adj_mtx > 0] = 1
#
#     edge_tensor = adj2edge_index(adj_mtx)
#
#     ## save global graph
#     centroid_path = os.path.join(save_dir, "cluster_centroids.pk")
#     adj_path = os.path.join(save_dir, "adj_mtx_" + str(clst_num) + ".csv")
#     edge_path = os.path.join(save_dir, "edge_tensor_" + str(clst_num) + ".pk")
#
#     np.savetxt(adj_path, adj_mtx, delimiter=",", fmt='%d')
#     save_pk(edge_path, edge_tensor)
#     with open(centroid_path, 'wb') as f:
#         np.save(f, clst_centers)
#
#     print("global graph saved.")
#
#     return clst_centers, edge_tensor
#
# def build_co_occurr_graph(filepath_txt, centroids, save_dir, keep_edge_rate=0.4, freq_mtx = None):
#
#     clst_num = len(centroids)
#     occur_thres = (1 / clst_num) * 0.5
#     save_freq = False
#     if freq_mtx is None:
#         print("counting the edges in database")
#         freq_mtx, _ = files2clstfreq(filepath_txt, centroids, db_cut_size) ## freq_mtx: sample_num x cluster_num
#         save_freq = True
#
#     occur_mtx = freq_mtx.copy()
#     occur_mtx[occur_mtx < occur_thres] = 0
#     occur_mtx[occur_mtx > 0] = 1
#
#     global_graph = count_graph(clst_num)
#     for sample in occur_mtx:
#         cluster_ids = np.where(sample == 1)[0]
#         global_graph.add_edge_weight(cluster_ids)
#     count_mtx = global_graph.get_count_mtx()
#
#     zero_rows = np.sum(np.any(count_mtx, axis=1) == False)
#     print(f'count_mtx has {zero_rows} rows with all zeros.')
#
#     adj_mtx = count_mtx.copy()
#     positive_edge_count = np.sum(count_mtx > 0)
#     topk = min(int(keep_edge_rate * clst_num * clst_num), positive_edge_count)
#     ## get top k value in adj_mtx
#     topk_value = np.sort(adj_mtx, axis=None)[-topk]
#
#     adj_mtx[adj_mtx < topk_value] = 0
#     adj_mtx[adj_mtx > 0] = 1
#
#     edge_tensor = adj2edge_index(adj_mtx)
#
#     ## save global graph
#     adj_path = os.path.join(save_dir, "adj_mtx_" + str(clst_num) + ".csv")
#     edge_path = os.path.join(save_dir, "edge_tensor_" + str(clst_num) + ".pk")
#     freq_mtx_path = os.path.join(save_dir, "freq_mtx_" + str(clst_num) + ".pk")
#
#     print("freq_mtx_path , ", freq_mtx_path )
#     np.savetxt(adj_path, adj_mtx, delimiter=",", fmt='%d')
#     save_pk(edge_path, edge_tensor)
#     if save_freq:
#         save_pk(freq_mtx_path, freq_mtx)
#     print("global graph saved.")
#
#     return edge_tensor
#
# def build_corr_graph(filepath_txt, centroids, save_dir, keep_edge_rate=0.4, freq_mtx = None):
#     clst_num = len(centroids)
#     save_freq = False
#     if freq_mtx is None:
#         print("counting the edges in database")
#         freq_mtx, _ = files2clstfreq(filepath_txt, centroids, db_cut_size) ## freq_mtx: sample_num x cluster_num
#         save_freq = True
#
#     print("building and saving global graph")
#     weight_mtx = np.corrcoef(freq_mtx.T)
#     weight_mtx = np.nan_to_num(weight_mtx)
#     np.fill_diagonal(weight_mtx, 0)
#     print(weight_mtx)
#     print(weight_mtx.shape)
#
#     # topk = int(keep_edge_rate * clst_num * clst_num)
#     positive_edge_count = np.sum(weight_mtx > 0)
#     topk = min(int(keep_edge_rate * clst_num * clst_num), positive_edge_count)
#     topk_value = np.sort(weight_mtx, axis=None)[-topk]
#
#     print("topk_value:", topk_value)
#     print("topk:", topk)
#
#     adj_mtx = np.zeros((clst_num, clst_num))
#     adj_mtx[weight_mtx < topk_value] = 0
#     adj_mtx[weight_mtx >= topk_value] = 1
#
#     np.fill_diagonal(adj_mtx, 0)
#
#     edge_tensor = adj2edge_index(adj_mtx)
#
#     print("topk_value:", topk_value)
#     adj_path = os.path.join(save_dir, "adj_mtx_" + str(clst_num) + ".csv")
#     edge_path = os.path.join(save_dir, "edge_tensor_" + str(clst_num) + ".pk")
#     freq_mtx_path = os.path.join(save_dir, "freq_mtx_" + str(clst_num) + ".pk")
#     np.savetxt(adj_path, adj_mtx, delimiter=",", fmt='%d')
#     save_pk(edge_path, edge_tensor)
#
#     if save_freq:
#         save_pk(freq_mtx_path , freq_mtx)
#     print("global graph saved.")
#
#     return edge_tensor
