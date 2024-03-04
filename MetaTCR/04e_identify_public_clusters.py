import os
from metatcr.utils.utils import load_pkfile
import configargparse
# from encoders.build_graph import seqlist2ebd, compute_cluster_assignment
import numpy as np
import random
random.seed(0)

parser = configargparse.ArgumentParser()
parser.add_argument('--mtx_dir', type=str, default='./results_50/data_analysis/datasets_meta_mtx', help='')
parser.add_argument('--out_dir', type=str, default='./results_50/data_analysis/clusters_identification', help='Output directory for processed data')
parser.add_argument('--add_dir', type=str, default='./results_50/data_analysis/datasets_meta_mtx_add', help='')

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

def dataset_cv(filelist):
    # Initialize an empty list to store the numpy arrays
    diversity_mtx_list = []
    dataset_name_list = []
    dataset_reprentation = {}

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
            # total_num = len(dataset_dict['diversity_mtx'])
            # if total_num > 50:
            #     valid_idx = random.sample(range(len(dataset_dict['diversity_mtx'])), 50)
            #     diversity_mtx = dataset_dict['diversity_mtx'][valid_idx, :]
            # else:
            #     diversity_mtx = dataset_dict['diversity_mtx']

            diversity_mtx = dataset_dict['diversity_mtx']

            mean = diversity_mtx.mean(axis=0)
            std = diversity_mtx.std(axis=0)
            cv = std / mean
            cv_threshold = np.nanpercentile(cv, 25)  ## Q1
            mean_median = np.nanmedian(mean)
            # low_var_elements = np.where((~np.isnan(cv)) & (cv < threshold))
            high_mean_low_cv_elements = np.where((~np.isnan(cv)) & (cv < cv_threshold) & (mean > mean_median))[0]

            print("dataset_name:", dataset_name)
            # print(np.where((~np.isnan(cv)) & (cv < cv_threshold)))
            # print(np.where((~np.isnan(cv)) & (cv < cv_threshold) & (mean > mean_median)))
            # print("len,", len(diversity_mtx))
            # print("mean:", mean)
            # print("mean:", mean.shape)
            # ## sum of mean
            # print("sum of mean:", np.sum(mean))
            # print("std:", std)
            # print("cv:", cv)

            print(f"Low variation elements for dataset: {high_mean_low_cv_elements}, len: {len(high_mean_low_cv_elements)}")
            print("#######################################")

            dataset_reprentation[dataset_name] = high_mean_low_cv_elements
            diversity_mtx_list.append(diversity_mtx)


    dataset_reprentation_keys = list(dataset_reprentation.keys())
    print(dataset_reprentation_keys)


    from collections import Counter
    counter = Counter()
    for key in dataset_reprentation_keys:
        counter.update(dataset_reprentation[key])

    print("counter:", counter)
    min_count = len(dataset_reprentation_keys) * 0.8
    ## elements that appear in 80% or more datasets
    elements = [element for element, count in counter.items() if count >= min_count]

    print("Elements that appear in 90% or more datasets:", elements)


def public_cv(diversity_mtx):
    mean = diversity_mtx.mean(axis=0)
    std = diversity_mtx.std(axis=0)
    cv = std / mean
    cv_threshold = np.nanpercentile(cv, 25)  ## Q1
    mean_median = np.nanpercentile(mean, 75)
    # low_var_elements = np.where((~np.isnan(cv)) & (cv < threshold))
    high_mean_low_cv_elements = np.where((~np.isnan(cv)) & (cv < cv_threshold) & (mean > mean_median))[0]

    # print("dataset_name:", dataset_name)
    # print(np.where((~np.isnan(cv)) & (cv < cv_threshold)))
    # print(np.where((~np.isnan(cv)) & (cv < cv_threshold) & (mean > mean_median)))
    # print("len,", len(diversity_mtx))
    # print("mean:", mean)
    # print("mean:", mean.shape)
    # ## sum of mean
    # print("sum of mean:", np.sum(mean))
    # print("std:", std)
    # print("cv:", cv)
    print("Elements that appear in 90% or more datasets:", high_mean_low_cv_elements)

def oneset_cv(filelist):
    # Initialize an empty list to store the numpy arrays
    diversity_mtx_list = []
    dataset_name_list = []
    dataset_reprentation = {}

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

            diversity_mtx = dataset_dict['diversity_mtx']

            mean = diversity_mtx.mean(axis=0)
            std = diversity_mtx.std(axis=0)
            cv = std / mean
            cv_threshold = np.nanpercentile(cv, 25)  ## Q1
            mean_median = np.nanmedian(mean)
            # low_var_elements = np.where((~np.isnan(cv)) & (cv < threshold))
            high_mean_low_cv_elements = np.where((~np.isnan(cv)) & (cv < cv_threshold) & (mean > mean_median))[0]

            print("dataset_name:", dataset_name)
            print(f"Low variation elements for dataset: {high_mean_low_cv_elements}, len: {len(high_mean_low_cv_elements)}")
            print("#######################################")

# filelist = os.listdir(args.mtx_dir)[:5]
# print(filelist)
# _, dataset_reprentation = dataset_cv(filelist)
#
# ## 对所有dataset_reprentation取交集
# dataset_reprentation_keys = list(dataset_reprentation.keys())
# print(dataset_reprentation_keys)
# intersection = dataset_reprentation[dataset_reprentation_keys[0]]
# for key in dataset_reprentation_keys[1:]:
#     intersection = np.intersect1d(intersection, dataset_reprentation[key])
#
# print("intersection:", intersection)
# cmv_files = ["Emerson2017_HIP.pk","Emerson2017_Keck.pk"]
# oneset_cv(cmv_files)
#
# sx_files = ["Sx_gastric_Normal.pk", "Sx_gastric_Tumor.pk"]
# oneset_cv(sx_files)

# covid_files = os.listdir(args.mtx_dir)[:5]
# dataset_cv(covid_files)
#
# sx_files = ["Sx_gastric_Normal.pk", "Sx_gastric_Tumor.pk"]
# dataset_cv(sx_files)
#
# healthy_files = ["Sx_gastric_Normal.pk","ZhangControl.pk","Emerson2017_HIP.pk","Emerson2017_Keck.pk","tcrbv4_control_Normal.pk"]
# dataset_cv(healthy_files)
#
# cancer_files = ["Sx_gastric_Tumor.pk", "ESCC_multi_region_PBMC.pk", "PMID28422742_liver_PBMC.pk", "PMID33317041_AML.pk",
#               "TRACERx_lung_PBMC.pk", "Formenti2018.pk", "MDanderson2019_PBMC.pk",
#                 "robert2014_CCR.pk", "valpione2020_nm.pk", "weber2018_cir.pk", "huuhtanen2022_nc.pk"]
# dataset_cv(cancer_files)

# all_files = os.listdir(args.mtx_dir)
# dataset_cv(all_files)
filelist = os.listdir(args.mtx_dir)
combined_diversity_mtx = merge_diversity_mtx(filelist)
print(combined_diversity_mtx.shape)
public_cv(combined_diversity_mtx)
exit()





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