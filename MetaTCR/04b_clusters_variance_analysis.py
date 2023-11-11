import os
from metatcr.utils.utils import load_pkfile
import configargparse
# from encoders.build_graph import seqlist2ebd, compute_cluster_assignment
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import random
random.seed(0)

parser = configargparse.ArgumentParser()
parser.add_argument('--mtx_dir', type=str, default='./results_50/data_analysis/datasets_meta_mtx', help='')
parser.add_argument('--out_dir', type=str, default='./results_50/data_analysis/clusters_identification', help='Output directory for processed data')

args = parser.parse_args()
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

print("Running with the following parameters:")
for key, value in vars(args).items():
    print(f"{key}: {value}")
print("#######################################")

def merge_diversity_mtx(filelist):
    # Initialize an empty list to store the numpy arrays
    diversity_mtx_list = []
    dataset_name_list = []

    for filename in filelist:
        # Check if the file is a .pk file
        if filename.endswith('.pk'):
            # Get the dataset name
            dataset_name = filename[:-3]
            # Open the .pk file and load the dictionary
            try:
                dataset_dict = load_pkfile(os.path.join(args.mtx_dir, filename))
            except FileNotFoundError:
                # If the file is not found in the first directory, try the second directory
                dataset_dict = load_pkfile(os.path.join(args.add_dir, filename))

            ## random sample 50 data from each dataset
            total_num = len(dataset_dict['diversity_mtx'])
            if total_num > 50:
                valid_idx = random.sample(range(len(dataset_dict['diversity_mtx'])), 50)
                diversity_mtx = dataset_dict['diversity_mtx'][valid_idx, :]
            else:
                diversity_mtx = dataset_dict['diversity_mtx']

            diversity_mtx_list.append(diversity_mtx)

    # Stack all of the "diversity_mtx" arrays vertically to form a new numpy array
    combined_diversity_mtx = np.vstack(diversity_mtx_list)

    return combined_diversity_mtx


filelist =  os.listdir(args.mtx_dir)
combined_diversity_mtx = merge_diversity_mtx(filelist)
print(combined_diversity_mtx.shape)

data = combined_diversity_mtx.T  ## cluster x samples

cluster_num = data.shape[0]

vars = np.var(data, axis=1)
means = np.mean(data, axis=1)

sorted_vars = np.sort(vars)
sorted_means = np.sort(means)

fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(12, 5)
fig.suptitle('Variance and Mean Curve Plot')

ax0.plot(sorted_vars)
ax0.set_xlabel('Clusters')
ax0.set_ylabel('Variance')

ax1.plot(sorted_means)
ax1.set_xlabel('Clusters')
ax1.set_ylabel('Mean')

# plt.show()
plt.savefig(os.path.join(args.out_dir, 'clusters_variance_mean_curve.png'))


## find max and min variance clusters
show_num = 10
sorted_idx = np.argsort(vars)
condition_idx = np.where(means > 1/cluster_num)[0]
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
    ax1.boxplot([data[idx]], positions=[i], widths=0.5, patch_artist=True, boxprops = {'facecolor':'lightskyblue'}, flierprops=flierprops)
ax1.set_title('Min Variance')
ax2.set_xlabel('Cluster id')
ax1.set_xticks(range(show_num))
ax1.set_xticklabels(min_idx)
ax1.set_ylim(0, max_y)

for i, idx in enumerate(max_idx):
    ax2.boxplot([data[idx]], positions=[i], widths=0.5, patch_artist=True, boxprops = {'facecolor':'lightskyblue'}, flierprops=flierprops)
ax2.set_title('Max Variance')
ax2.set_xticks(range(show_num))
ax2.set_xticklabels(max_idx)
ax2.set_xlabel('Cluster id')
ax2.set_ylim(0, max_y)

# plt.show()
plt.savefig(os.path.join(args.out_dir, 'clusters_variance_boxplot.png'))

label_col = "Pathology"
# label_col = "Epitope.peptide"
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
    print("label",label)
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

# 为每个cluster-Pathology pair计数
g = df.groupby('cluster')
pathology_dist = g[label_col].value_counts().unstack(0)
# counts = df.groupby(['cluster'])['Pathology'].count().unstack()

pathology_dist = pathology_dist.T
pathology_dist = pathology_dist.div(pathology_dist.sum(axis=1), axis=0)
print(pathology_dist)

fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(11,6))
lines, labels = [], []
ax_idx = 0
for idx in [min_idx, max_idx]:
    ax = axs[ax_idx]
    ax_idx += 1
    pathology_dist_slice = pathology_dist.iloc[idx,:]
    print(pathology_dist_slice)
    p = pathology_dist_slice.plot(kind='bar',stacked=True, ax=ax, colormap=cmap)
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
plt.savefig(os.path.join(args.out_dir, 'pathology_dist_min_max.png'), dpi=600)

# plt.show()