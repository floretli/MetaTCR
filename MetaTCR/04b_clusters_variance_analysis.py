import os
from metatcr.utils.utils import load_pkfile
from metatcr.visualization.dataset_vis import visualize_metavec
import configargparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import random
random.seed(0)

parser = configargparse.ArgumentParser()
parser.add_argument('--mtx_dir', type=str, default='./results/data_analysis/datasets_mtx', help='')
parser.add_argument('--out_dir', type=str, default='./results/data_analysis/clusters_identification', help='Output directory for processed data')
parser.add_argument('--dataset_type_file', type=str, default='./data/datasets_type.csv', help='Output directory for processed data')

args = parser.parse_args()
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

print("Running with the following parameters:")
for key, value in vars(args).items():
    print(f"{key}: {value}")
print("#######################################")

def get_prefix(filename):
    return filename.split('_')[0]

def merge_dataset_mtx(filelist, merge_subset = False):
    # Initialize an empty list to store the numpy arrays
    diversity_mtx_list = []
    abundance_mtx_list = []
    dataset_name_list = []

    for filename in filelist:
        # Check if the file is a .pk file
        if filename.endswith('.pk'):
            # Get the dataset name
            dataset_name = filename[:-3]
            dataset_dict = load_pkfile(os.path.join(args.mtx_dir, filename))

            # Add the "diversity_mtx" element of the dictionary to the list
            diversity_mtx_list.append(dataset_dict["diversity_mtx"])
            abundance_mtx_list.append(dataset_dict["abundance_mtx"])
            if merge_subset:
                studyname = dataset_name.split("_")[0]
                dataset_name_list += [studyname] * len(dataset_dict["sample_list"])
            else:
                dataset_name_list += [dataset_name] * len(dataset_dict["sample_list"])

    # Stack all of the "diversity_mtx" arrays vertically to form a new numpy array
    combined_diversity_mtx = np.vstack(diversity_mtx_list)
    combined_abundance_mtx = np.vstack(abundance_mtx_list)

    return combined_diversity_mtx, combined_abundance_mtx, dataset_name_list


def get_mean_vec(data):
    return np.mean(data, axis=0)

def get_dataset_vectors(filelist, merge_subset = False):
    # Initialize an empty list to store the numpy arrays
    diversity_mtx_list = []
    dataset_name_list = []

    for filename in filelist:
        # Check if the file is a .pk file
        if filename.endswith('.pk'):
            # Get the dataset name
            dataset_name = filename[:-3]
            dataset_dict = load_pkfile(os.path.join(args.mtx_dir, filename))

            # Add the "diversity_mtx" element of the dictionary to the list
            diversity_mtx_list.append(dataset_dict["diversity_mtx"])
            if merge_subset:
                studyname = dataset_name.split("_")[0]
                dataset_name_list += [studyname] * len(dataset_dict["sample_list"])
            else:
                dataset_name_list += [dataset_name] * len(dataset_dict["sample_list"])

    # Stack all of the "diversity_mtx" arrays vertically to form a new numpy array
    combined_diversity_mtx = np.vstack(diversity_mtx_list)

    ## get the mean vector for each dataset
    unique_datasets = np.unique(dataset_name_list)
    mean_vec_list = []
    for dataset in unique_datasets:
        idx = [i for i, x in enumerate(dataset_name_list) if x == dataset]
        mean_vec = get_mean_vec(combined_diversity_mtx[idx, :])
        mean_vec_list.append(mean_vec)

    combined_diversity_mtx = np.vstack(mean_vec_list)
    dataset_name_list = unique_datasets.tolist()

    return combined_diversity_mtx, dataset_name_list


def variability_in_diversity(combined_diversity_mtx):
    data = combined_diversity_mtx.T  ## cluster x samples

    cluster_num = data.shape[0]

    vars = np.var(data, axis=1)
    means = np.mean(data, axis=1)

    sorted_vars = np.sort(vars)
    sorted_means = np.sort(means)

    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(12, 5)
    # fig.suptitle('Variance and mean curve plot')

    ax0.plot(sorted_vars)
    ax0.set_xlabel('Clusters')
    ax0.set_ylabel('Variance')

    ax1.plot(sorted_means)
    ax1.set_xlabel('Clusters')
    ax1.set_ylabel('Mean')

    # plt.show()
    plt.savefig(os.path.join(args.out_dir, 'clusters_variance_mean_curve.svg'))

    ## find max and min variance clusters
    show_num = 10
    sorted_idx = np.argsort(vars)
    condition_idx = np.where(means > 1 / cluster_num)[0]
    valid_data = data[condition_idx, :]
    filtered_idx = [idx for idx in sorted_idx if idx in condition_idx]

    max_idx = sorted_idx[-show_num:]
    min_idx = sorted_idx[:show_num]

    # set fig size
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(15, 5)

    max_y = np.max(valid_data)
    flierprops = dict(marker='.', markersize=2)

    for i, idx in enumerate(min_idx):
        ax1.boxplot([data[idx]], positions=[i], widths=0.5, patch_artist=True, boxprops={'facecolor': 'lightskyblue'},
                    flierprops=flierprops)
    ax1.set_title('Min variance')
    ax2.set_xlabel('Cluster id')
    ax1.set_xticks(range(show_num))
    ax1.set_xticklabels(min_idx)
    ax1.set_ylim(0, max_y)
    ax1.set_ylim(0, 0.05)

    for i, idx in enumerate(max_idx):
        ax2.boxplot([data[idx]], positions=[i], widths=0.5, patch_artist=True, boxprops={'facecolor': 'lightskyblue'},
                    flierprops=flierprops)
    ax2.set_title('Max variance')
    ax2.set_xticks(range(show_num))
    ax2.set_xticklabels(max_idx)
    ax2.set_xlabel('Cluster id')
    ax2.set_ylim(0, max_y)

    plt.savefig(os.path.join(args.out_dir, 'clusters_variance_boxplot.svg'))

    return min_idx, max_idx

def plot_cluster_composition(min_idx, max_idx, label_col = "Pathology"):

    ## label_col = "Pathology" or "Epitope.peptide"
    df = pd.read_csv(os.path.join(args.out_dir, 'McPAS_cluster.csv'))
    print(df)
    p_num = len(df[label_col].unique())

    labels = df[label_col].values
    labels_series = pd.Series(labels)
    # Count the occurrences of each label
    label_counts = labels_series.value_counts()
    valid_labels = label_counts[label_counts >= 50].index
    df = df[df[label_col].isin(valid_labels)]

    print(label_counts)
    max_samples_per_class = 1000
    new_labels_series = pd.Series(df[label_col].values)
    new_metadata = pd.DataFrame()
    for label in new_labels_series.unique():
        print("label", label)
        class_samples = df[df[label_col] == label]
        # if len(class_samples) > max_samples_per_class:
        #     class_samples = class_samples.sample(n=max_samples_per_class, random_state=42)
        class_samples = class_samples.sample(n=max_samples_per_class, random_state=42, replace=True)
        new_metadata = pd.concat([new_metadata, class_samples])

    new_indices = new_metadata.index
    labels = new_metadata[label_col].values
    # print(new_indices)
    print(new_metadata)

    df = new_metadata
    p_num = len(df[label_col].unique())

    print("p_num:", p_num)

    # cmap = cm.get_cmap('jet', p_num)
    cmap = cm.get_cmap('tab20', p_num)

    # print(df.groupby(['cluster','Pathology'])['CDR3.beta.aa'])
    df['cluster'] = df['cluster'].astype('category')

    # count for each cluster
    g = df.groupby('cluster')
    pathology_dist = g[label_col].value_counts().unstack(0)

    pathology_dist = pathology_dist.T
    pathology_dist = pathology_dist.div(pathology_dist.sum(axis=1), axis=0)
    print(pathology_dist)

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(11, 6))
    lines, labels = [], []
    ax_idx = 0
    for idx in [min_idx, max_idx]:
        ax = axs[ax_idx]
        ax_idx += 1
        pathology_dist_slice = pathology_dist.iloc[idx, :]
        print(pathology_dist_slice)
        p = pathology_dist_slice.plot(kind='bar', stacked=True, ax=ax, colormap=cmap)
        line, label = ax.get_legend_handles_labels()
        lines.extend(line)
        labels.extend(label)
        ax.set_title(f'Cluster id')
        # ax.set_title(f'Cluster id {i} - {i+23}')

    for ax in axs:
        ax.get_legend().remove()
    unique = {label: line for line, label in zip(lines, labels)}
    labels, lines = zip(*unique.items())
    fig.legend(lines, labels, loc='center right', bbox_to_anchor=(1.00, 0.5), fontsize=8)
    fig.suptitle('Pathology Distribution by Cluster')
    # plt.tight_layout()
    plt.subplots_adjust(right=0.75, hspace=0.5)
    plt.savefig(os.path.join(args.out_dir, 'pathology_dist_min_max.png'), dpi=600, bbox_inches='tight')

def identify_public_clusters(diversity_mtx):
    ## public clusters: clusters that diversity is high (>q3) and variance is low (<q1)
    mean = diversity_mtx.mean(axis=0)
    std = diversity_mtx.std(axis=0)
    cv = std / mean
    # cv_threshold = np.nanpercentile(cv, 10)  ## Q1
    # mean_median = np.nanpercentile(mean, 50)
    cv_threshold = np.nanpercentile(cv, 50)  ## Q1
    mean_median = np.nanpercentile(mean, 10)
    # low_var_elements = np.where((~np.isnan(cv)) & (cv < threshold))
    high_diversity_low_cv_clusters = np.where((~np.isnan(cv)) & (cv < cv_threshold) & (mean > mean_median))[0]

    print("Public clusters:", high_diversity_low_cv_clusters)
    return high_diversity_low_cv_clusters


# def identify_variance_clusters(diversity_mtx):
#     mean = diversity_mtx.mean(axis=0)
#     std = diversity_mtx.std(axis=0)
#     cv = std / mean
#     cv_threshold = np.nanpercentile(cv, 25)  ## Q1
#     mean_median = np.nanpercentile(mean, 10)
#     # low_var_elements = np.where((~np.isnan(cv)) & (cv < threshold))
#     high_diversity_high_cv_clusters = np.where((~np.isnan(cv)) & (cv > cv_threshold) & (mean > mean_median))[0]
#
#     print("variance clusters:", high_diversity_high_cv_clusters)
#     return high_diversity_high_cv_clusters

def calculate_weighted_stats(diversity_mtx, weights):
    """
    Calculate the weighted mean and standard deviation for the diversity matrix.
    """
    # Calculate weighted mean
    weighted_mean = np.average(diversity_mtx, axis=0, weights=weights)

    # Calculate weighted standard deviation
    # Note that we need to normalize the weights for the standard deviation calculation
    sum_weights = np.sum(weights)
    normalized_weights = weights * len(weights) / sum_weights
    variance = np.average((diversity_mtx - weighted_mean) ** 2, axis=0, weights=weights)
    weighted_std = np.sqrt(variance)

    return weighted_mean, weighted_std


def identify_variance_clusters(diversity_mtx, label_list):
    # Calculate label frequencies
    label_frequencies = {label: label_list.count(label) / len(label_list) for label in set(label_list)}

    # Set weights as the inverse frequency
    weights = np.array([1.0 / label_frequencies[label] for label in label_list])

    # Calculate weighted mean and std deviation
    mean, std = calculate_weighted_stats(diversity_mtx, weights)

    # Calculate weighted coefficient of variation
    cv = std / mean

    # Define thresholds
    cv_threshold = np.nanpercentile(cv, 25)  # Q1
    mean_median = np.nanpercentile(mean, 50)

    # Identify high diversity high CV clusters
    high_diversity_high_cv_clusters = np.where((~np.isnan(cv)) & (cv > cv_threshold) & (mean > mean_median))[0]

    print("Variance clusters:", high_diversity_high_cv_clusters)
    return high_diversity_high_cv_clusters

def get_sub_matrix(data, sample_idx, cluster_idx):
    subdata = data[np.ix_(sample_idx, cluster_idx)]
    subdata = subdata / np.sum(subdata, axis=1).reshape(-1,1)
    return subdata



# def identify_class_specific_clusters(diversity_mtx):
#     ## class specific clusters: clusters that diversity is high (>q3) and variance is high (>q3)
#     mean = diversity_mtx.mean(axis=0)
#     std = diversity_mtx.std(axis=0)
#     cv = std / mean
#     cv_threshold = np.nanpercentile(cv, 75)  ## Q3
#     high_diversity_high_cv_clusters = np.where((~np.isnan(cv)) & (cv > cv_threshold) & (mean > mean_threshold))[0]


filelist = os.listdir(args.mtx_dir)
combined_diversity_mtx, _, dataset_name_list = merge_dataset_mtx(filelist, merge_subset=True)
diversity_vecs, vec_dataset_list = get_dataset_vectors(filelist, merge_subset=True)
# print(combined_diversity_mtx.shape)
print(diversity_vecs.shape)
print(vec_dataset_list)


# exit()
# ## plot variance and mean curve; plot boxplot for max and min variance clusters
# min_idx, max_idx = variability_in_diversity(combined_diversity_mtx)
#
# ## plot cluster composition for max and min variance clusters
# plot_cluster_composition(min_idx, max_idx, label_col = "Pathology")


data_info = pd.read_csv(args.dataset_type_file)
## transfer the dataset name to the dataset type
study2pip = dict(zip(data_info['Study'], data_info['Data process pipeline']))
dataset_pipelines = [study2pip[name] for name in dataset_name_list]

vec_pipelines = [study2pip[name] for name in vec_dataset_list]

## 为data_info['Data process pipeline']的每种分类建立一个index,numpy 格式
MiXCR_idx = np.array([i for i, x in enumerate(dataset_pipelines) if x == "MiXCR"])
ImmunoSEQ_idx = np.array([i for i, x in enumerate(dataset_pipelines) if x == "ImmunoSEQ"])

MiXCR_idx_vec = np.array([i for i, x in enumerate(vec_pipelines) if x == "MiXCR"])
ImmunoSEQ_idx_vec = np.array([i for i, x in enumerate(vec_pipelines) if x == "ImmunoSEQ"])

# print("MiXCR datasets:", MiXCR_idx)
# print("ImmunoSEQ datasets:", ImmunoSEQ_idx)

public_cluster_idx_MiXCR = identify_public_clusters(diversity_vecs[MiXCR_idx_vec])
public_cluster_idx_ImmunoSEQ = identify_public_clusters(diversity_vecs[ImmunoSEQ_idx_vec])
high_variance_cluster_idx = identify_variance_clusters(diversity_vecs, vec_pipelines)

print("Public clusters MiXCR:", public_cluster_idx_MiXCR)
print("Public clusters ImmunoSEQ:", public_cluster_idx_ImmunoSEQ)
print("High variance clusters:", high_variance_cluster_idx)

## 平台之间差异大  平台内cv小的cluster： 对High variance clusters和public_cluster_idx_MiXCR和public_cluster_idx_ImmunoSEQ找交集
batch_cluster_idx = np.intersect1d(public_cluster_idx_MiXCR, public_cluster_idx_ImmunoSEQ)
print("batch_cluster_idx:", batch_cluster_idx)
batch_cluster_idx= np.intersect1d(batch_cluster_idx, high_variance_cluster_idx)

print("batch_cluster_idx:", batch_cluster_idx)


# public_cluster_idx_MiXCR = identify_public_clusters(combined_diversity_mtx[MiXCR_idx])
# visualize_metavec(combined_diversity_mtx[np.ix_(MiXCR_idx, public_cluster_idx_MiXCR)],
#                   [dataset_name_list[i] for i in MiXCR_idx],
#                   min_dist=0.5, n_neighbors=50, type = "MiXCR datasets (public clusters)",
#                   out_dir = args.out_dir)

# public_cluster_idx_ImmunoSEQ = identify_public_clusters(combined_diversity_mtx[ImmunoSEQ_idx])
# visualize_metavec(combined_diversity_mtx[np.ix_(ImmunoSEQ_idx, public_cluster_idx_ImmunoSEQ)],
#                   [dataset_name_list[i] for i in ImmunoSEQ_idx],
#                   min_dist=0.5, n_neighbors=50, type = "ImmunoSEQ datasets (public clusters)",
#                   out_dir = args.out_dir)


## 两个array的交集
public_cluster_idx = np.intersect1d(public_cluster_idx_MiXCR, public_cluster_idx_ImmunoSEQ)
mix_imm_idx = np.concatenate((MiXCR_idx, ImmunoSEQ_idx))

print(mix_imm_idx)

# print("Public clusters:", public_cluster_idx)  ##  [18 57 82 84]
# visualize_metavec(combined_diversity_mtx[np.ix_(mix_imm_idx, public_cluster_idx)],
#                   [dataset_name_list[i] for i in mix_imm_idx],
#                   min_dist=0.5, n_neighbors=50, type = "MiXCR and ImmuneSEQ datasets (shared clusters)",
#                   out_dir = args.out_dir)

## 差集
MiXCR_specific_cluster_idx = np.setdiff1d(public_cluster_idx_MiXCR, public_cluster_idx)
ImmunoSEQ_specific_cluster_idx = np.setdiff1d(public_cluster_idx_ImmunoSEQ, public_cluster_idx)

print("MiXCR specific clusters:", MiXCR_specific_cluster_idx)
print("ImmunoSEQ specific clusters:", ImmunoSEQ_specific_cluster_idx)


all_cluster_idx = np.arange(combined_diversity_mtx.shape[1])

# class_specific_cluster_idx = np.setdiff1d(all_cluster_idx, np.concatenate((MiXCR_specific_cluster_idx, ImmunoSEQ_specific_cluster_idx)))
class_specific_cluster_idx = np.setdiff1d(all_cluster_idx, batch_cluster_idx)

print("Class specific clusters:", class_specific_cluster_idx)
visualize_metavec(combined_diversity_mtx[np.ix_(mix_imm_idx, class_specific_cluster_idx)],
                  [dataset_name_list[i] for i in mix_imm_idx],
                  min_dist=0.5, n_neighbors=50, type = "MiXCR and ImmuneSEQ datasets (rm seq batch effect)",
                  out_dir = args.out_dir)

## 以pipeline为标签
visualize_metavec(get_sub_matrix(combined_diversity_mtx,mix_imm_idx, class_specific_cluster_idx),
                  [dataset_pipelines[i] for i in mix_imm_idx],
                  min_dist=0.5, n_neighbors=50, type = "Pilelines(rm seq batch effect)",
                  out_dir = args.out_dir)

visualize_metavec(get_sub_matrix(combined_diversity_mtx,mix_imm_idx, batch_cluster_idx),
                  [dataset_pipelines[i] for i in mix_imm_idx],
                  min_dist=0.5, n_neighbors=50, type = "Pilelines(only show batch effect clusters)",
                  out_dir = args.out_dir)