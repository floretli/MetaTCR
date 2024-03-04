from metatcr.encoders.build_graph import db2count_graph, load_pkfile
import configargparse
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import community as community_louvain
import seaborn as sns
import pandas as pd

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

def train_linkage(X, method='ward', metric='euclidean'):
    from scipy.cluster.hierarchy import linkage
    l_model = linkage(X, method=method, metric=metric)
    return l_model

def linkage_merge_clusters(l_model, n_superclusters=10, criterion='maxclust'):
    from scipy.cluster.hierarchy import fcluster
    ref_cluster_labels = fcluster(l_model, n_superclusters, criterion=criterion)
    return ref_cluster_labels


def cluster_entropy(degrees, cluster_id, Va_parent, VG, ga):
    # ∑ - gα/V(G) log(Vα/Va_parent)
    Va = degrees[cluster_id]
    return - ga / VG * np.log2(Va / Va_parent)

def supercluster_entropy(adjacency_matrix, supercluster_ids, supercluster_id):
    # ∑ - gα/V(G) log(Vα/Va_parent)

    ## get nodes in the same supercluster and nodes in the different superclusters
    same_sc_ids = np.where(supercluster_ids == supercluster_id)[0]
    diff_sc_ids = np.where(supercluster_ids != supercluster_id)[0]
    degrees = np.sum(adjacency_matrix, axis=1)
    VG = np.sum(degrees)  # total volume of graph G
    Va_parent = np.sum(degrees[same_sc_ids]) + 1e-2  ## to avoid 0


    total_entropy = 0
    for cluster_id in same_sc_ids:
        ga = np.sum(adjacency_matrix[np.ix_(same_sc_ids, diff_sc_ids)])  # adjacency matrix of node a
        entropy = cluster_entropy(degrees, cluster_id, Va_parent, VG, ga)
        total_entropy += entropy
        # print("gα", ga, "VG", VG, "Va_parent", Va_parent,  "Va", degrees[cluster_id], "entropy", entropy)

    return total_entropy

def dataset_entropy(adjacency_matrix, supercluster_ids, supercluster_nums):

    degrees = np.sum(adjacency_matrix, axis=1)
    valid_nodes = np.where(degrees > 0)[0]
    valid_adj_mtx = adjacency_matrix[np.ix_(valid_nodes, valid_nodes)]
    supercluster_ids = supercluster_ids[valid_nodes]

    entropys = []
    total_entropy = 0
    for i in range(1, supercluster_nums + 1):
        entropy = supercluster_entropy(valid_adj_mtx, supercluster_ids, i)
        entropys.append(entropy)

    total_entropy = np.sum(entropys)
    # entropys.append(total_entropy)
    return entropys, total_entropy

def calculate_entropies(adjacency_matrices, supercluster_ids, supercluster_nums):
    all_entropies = []
    graph_entropies = []
    for i in range(len(adjacency_matrices)):
        supercluster_entropies, graph_entropy = dataset_entropy(adjacency_matrices[i], supercluster_ids, supercluster_nums)
        all_entropies.append(supercluster_entropies)
        graph_entropies.append(graph_entropy)
    return all_entropies, graph_entropies

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

def plot_graph_entropies(filelist, supercluster_ids, n_superclusters = 6):
    adjacency_matrices = []
    diversity_mtx_list, dataset_name_list = get_metavectors(filelist)
    num_graph = len(diversity_mtx_list)

    for i in range(num_graph):
        diversity_mtx = diversity_mtx_list[i]
        _, adj_mtx = db2count_graph(diversity_mtx, keep_edge_rate=args.keep_edge_rate, get_topk=True)
        adjacency_matrices += [adj_mtx]

    all_entropies, graph_entropies = calculate_entropies(adjacency_matrices, supercluster_ids, n_superclusters)
    all_entropies_array = np.array(all_entropies)
    print("all_entropies.shape", all_entropies_array.shape)

    column_names = ['c'+str(i) for i in range(1, n_superclusters + 1)]


    # Create figure with sub-plots
    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [5, 1]}, figsize=(10, 6))

    df = pd.DataFrame(all_entropies_array, columns=column_names, index=dataset_name_list)

    pd.options.display.max_columns = 99
    print(df)
    # mask = df.isin([-np.inf])
    # df.replace(-np.inf, 0, inplace=True)
    ## 打印全部值

    print(df)
    # no normalization
    # sns.heatmap(df, cmap='YlGnBu', ax=ax1,  yticklabels=True)

    sns.clustermap(df, cmap='Blues', metric="euclidean", yticklabels=True)
    plt.show()
    exit()

    sns.barplot(y=dataset_name_list, x=graph_entropies, ax=ax2, orient='h')

    # Draw barplot in the second sub-plot
    ax1.set_xlabel('Supercluster')
    ax1.set_ylabel('Sample')
    ax2.set_xlabel('Total')
    ax2.get_yaxis().set_visible(False)  # Hide the y-axis on the barplot

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_communities_subplot(G, partition, ax, title):
    pos = nx.spring_layout(G, seed=0, k=2)
    cmap = plt.get_cmap("Accent")
    # 获取社区数量和颜色映射
    node_colors = [cmap(partition[node] % 10) for node in G.nodes()]
    # 绘制网络图形和节点颜色
    nx.draw(G, pos, with_labels=True, node_color=node_colors,
            edge_color="gray", width=0.5, ax=ax)
    ax.set_title(title)


n_superclusters = 6
centroids = load_pkfile(os.path.join(args.centroids_dir, str(args.clst_num) + '_best_centers.pk'))
l_model = train_linkage(centroids)
supercluster_ids = linkage_merge_clusters(l_model, n_superclusters)
print("supercluster_ids", supercluster_ids, len(supercluster_ids))

all_file = os.listdir(args.mtx_dir)
combined_diversity_mtx, dataset_name_list = merge_dataset_mtx(all_file)
_, adj_mtx = db2count_graph(combined_diversity_mtx, keep_edge_rate=args.keep_edge_rate)
## np. set printoptions(threshold=np.inf)  ## 打印全部
np.set_printoptions(threshold=100)

plot_graph_entropies(all_file, supercluster_ids, n_superclusters = n_superclusters)
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

exit()


sle_files = ["ZhangSLE.pk"]
combined_diversity_mtx, dataset_name_list = merge_dataset_mtx(sle_files)
_, adj_mtx = db2count_graph(combined_diversity_mtx, keep_edge_rate=args.keep_edge_rate)
en = structure_entropy(adj_mtx)
print("sle ", en)

control_files = ["ZhangControl.pk"]
combined_diversity_mtx, dataset_name_list = merge_dataset_mtx(control_files)
_, adj_mtx = db2count_graph(combined_diversity_mtx, keep_edge_rate=args.keep_edge_rate)
en = structure_entropy(adj_mtx)
print("control ", en)

# print('Time for building global graph: ', t1-t0)

#
# freq_mtx = load_pkfile(os.path.join("./results06/co_occurr_graph", "freq_mtx_" + str(args.clst_num) + ".pk"))
#
# build_corr_graph( args.file_list, centroids, "./results06/corr_graph02" , 0.2, freq_mtx)