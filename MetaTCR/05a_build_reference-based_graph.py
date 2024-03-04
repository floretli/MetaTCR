from metatcr.encoders.build_graph import db2count_graph, load_pkfile
import configargparse
import os
import numpy as np
import matplotlib.colors as colors
import networkx as nx
import community as community_louvain
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
from community import community_louvain


parser = configargparse.ArgumentParser()
parser.add_argument('--mtx_dir', type=str, default='./results/data_analysis/datasets_meta_mtx', help='')
parser.add_argument('--clst_num', type=int, default=96, help='Number of clusters for building global graph')
parser.add_argument('--out_dir', type=str, default='./results/co_occurr_graph', help='Output directory for processed data')
parser.add_argument('--centroids_dir', type=str, default='./results/data_analysis', help='Output directory for processed data')
parser.add_argument('--keep_edge_rate', type=float, default=0.5, help='Generated edge rate for building global graph')

args = parser.parse_args()
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

print("Running with the following parameters:")
for key, value in vars(args).items():
    print(f"{key}: {value}")
print("#######################################")

def merge_dataset_mtx(filelist):
    # Initialize an empty list to store the numpy arrays
    diversity_mtx_list = []
    dataset_name_list = []

    for filename in filelist:
        # Check if the file is a .pk file
        if filename.endswith('.pk'):
            # Get the dataset name
            dataset_name = filename[:-3]
            # Open the .pk file and load the dictionary
            # dataset_dict = load_pkfile(os.path.join(args.mtx_dir, filename))
            try:
                dataset_dict = load_pkfile(os.path.join(args.mtx_dir, filename))
            except FileNotFoundError:
                # If the file is not found in the first directory, try the second directory
                dataset_dict = load_pkfile(os.path.join(args.add_dir, filename))

            # Add the "diversity_mtx" element of the dictionary to the list
            diversity_mtx_list.append(dataset_dict["diversity_mtx"])
            dataset_name_list += [dataset_name] * len(dataset_dict["sample_list"])

    # Stack all of the "diversity_mtx" arrays vertically to form a new numpy array
    combined_diversity_mtx = np.vstack(diversity_mtx_list)

    return combined_diversity_mtx, dataset_name_list

def get_metavectors(filelist):
    # Initialize an empty list to store the numpy arrays
    diversity_mtx_list = []
    dataset_name_list = []
    for filename in filelist:
        # Check if the file is a .pk file
        if filename.endswith('.pk'):
            try :
                dataset_name = filename[:-3]
                dataset_dict = load_pkfile(os.path.join(args.mtx_dir, filename))
            except FileNotFoundError:
                # If the file is not found in the first directory, try the second directory
                dataset_dict = load_pkfile(os.path.join(args.add_dir, filename))
            except:
                print("Error, file does not exist: " + filename)
                exit()
        diversity_mtx_list.append(dataset_dict["diversity_mtx"])
        dataset_name_list.append(dataset_name)

    # Add the "diversity_mtx" element of the dictionary to the list
    return diversity_mtx_list, dataset_name_list

def get_metavector(filename):
    try :
        dataset_name = filename[:-3]
        dataset_dict = load_pkfile(os.path.join(args.mtx_dir, filename))
    except FileNotFoundError:
        # If the file is not found in the first directory, try the second directory
        dataset_dict = load_pkfile(os.path.join(args.add_dir, filename))
    except:
        print("Error, file does not exist: " + filename)
        exit()

    # Add the "diversity_mtx" element of the dictionary to the list
    return dataset_dict["diversity_mtx"], dataset_name

def louvain_clusters(diversity_mtx):
    edge_tensor, global_adj_mtx = db2count_graph(diversity_mtx, keep_edge_rate=args.keep_edge_rate, get_topk=False)
    valid_indices = np.where(np.sum(global_adj_mtx, axis=1) != 0)[0]
    if len(valid_indices) != diversity_mtx.shape[1]:
        print("Warning: some nodes have no edges in the global graph. This may cause problems in the Louvain algorithm.")

    G = nx.from_numpy_array(global_adj_mtx)
    louvain_partition = community_louvain.best_partition(G, resolution=1)
    print("louvain_partition", louvain_partition)

    num_superclusters = len(set(louvain_partition.values()))
    supercluster_ids = np.full(diversity_mtx.shape[1], num_superclusters)

    for k, v in louvain_partition.items():
        supercluster_ids[k] = v

    return G, louvain_partition, supercluster_ids, num_superclusters


all_file = os.listdir(args.mtx_dir)
combined_diversity_mtx, dataset_name_list = merge_dataset_mtx(all_file)
G, partition, supercluster_ids, num_superclusters = louvain_clusters(combined_diversity_mtx)

exit()
plot_graph_entropies(all_file, supercluster_ids, n_superclusters = num_superclusters)
## 有-inf说明va_parent = 0, 负值说明va_parent < va。 基于欧氏距离的层次聚类不行。要用lovain算法对global graph层次聚类（这样更能代表cluster之间的关系）

# negative_file = ["PMID28422742_liver_PBMC.pk", "PMID28422742_liver_Tissue.pk"]
# combined_diversity_mtx, combined_abundance_mtx, dataset_name_list = merge_dataset_mtx(negative_file)
# supercluster_ids, num_superclusters = louvain_clusters(combined_diversity_mtx)
# print("supercluster_ids",supercluster_ids)
# plot_graph_entropies(negative_file, supercluster_ids, n_superclusters = num_superclusters)

exit()

covid_files = ["Covid_ADAPT.pk","Covid_ADAPT_MIRA.pk","Covid_IRST.pk","Covid_HU.pk","Covid_NIAID.pk"]
one_covid_files = ["Covid_ADAPT.pk"]
one_covid_files = ["Covid_ADAPT_MIRA.pk"]
cmv_files = ["Emerson2017_HIP.pk","Emerson2017_Keck.pk"]
bgi_files = ["ZhangSLE.pk", "ZhangControl.pk"]

plot_graph_entropies(covid_files, supercluster_ids, n_superclusters = 6)
