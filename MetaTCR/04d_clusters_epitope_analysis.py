import os
import configargparse
# from encoders.build_graph import seqlist2ebd, compute_cluster_assignment
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import random
random.seed(0)


parser = configargparse.ArgumentParser()
parser.add_argument('--out_dir', type=str, default='./results/data_analysis/clusters_identification', help='Output directory for processed data')
parser.add_argument('--centroids', type=str, default='./results/data_analysis/96_best_centers.pk', help='centroids file')
parser.add_argument('--ept_file', type=str, default='./data/McPAS-TCR_filt_ept_full_deduplicated.tsv', help='input epitope file')
parser.add_argument('--tcr_col', type=str, default='full_seq',
                    help='Column name for full length TCR sequence or amino acid')


args = parser.parse_args()
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

print("Running with the following parameters:")
for key, value in vars(args).items():
    print(f"{key}: {value}")
print("#######################################")

# centroids = load_pkfile(args.centroids)
#
# df = pd.read_csv(args.ept_file, sep = "\t")
# seqlist = df[args.tcr_col].to_list()  ## full_seq
# X = seqlist2ebd(seqlist, keep_pbar = False)
# labels = compute_cluster_assignment(centroids, X)
# df['cluster'] = labels
# print(df)
#
# g = df.groupby('cluster')
#
# num_seqs = g['CDR3.beta.aa'].count()
#
# num_trbj = g['TRBJ'].nunique()
#
# num_trbv = g['TRBV'].nunique()
#
# num_pathology =g['Pathology'].nunique()
#
# num_epitope = g['Epitope.peptide'].nunique()
#
# result = pd.DataFrame({'num_seqs':num_seqs,
#                     'num_trbj':num_trbj,
#                     'num_trbv':num_trbv,
#                     'num_pathology':num_pathology,
#                     'num_epitope':num_epitope})
# result.to_csv(os.path.join(args.out_dir, 'cluster_info.csv'))
# df.to_csv(os.path.join(args.out_dir, 'McPAS_cluster.csv'))


# label_col = "Pathology"
label_col = "Epitope.peptide"
df = pd.read_csv(os.path.join(args.out_dir, 'McPAS_cluster.csv'))
print(df)

## 二次过滤

labels = df[label_col].values
labels_series = pd.Series(labels)
# Count the occurrences of each label
label_counts = labels_series.value_counts()
valid_labels = label_counts[label_counts >= 200].index
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

fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(11,12))
lines, labels = [], []

for i in range(0,96,24):
    ax = axs[i//24]
    pathology_dist_slice = pathology_dist.iloc[i:i+24,:]
    p = pathology_dist_slice.plot(kind='bar',stacked=True, ax=ax, colormap=cmap)

    # 收集每个子图的图例信息
    line, label = ax.get_legend_handles_labels()
    lines.extend(line)
    labels.extend(label)

    ax.set_title(f'Cluster id {i} - {i+23}')

for ax in axs:
    ax.get_legend().remove()
unique = {label: line for line, label in zip(lines, labels)}
labels, lines = zip(*unique.items())
fig.legend(lines, labels, loc='center right', bbox_to_anchor=(1.00, 0.5), fontsize=8)
fig.suptitle('Pathology Distribution by Cluster')
# plt.tight_layout()
plt.subplots_adjust(right=0.75, hspace=0.5)
plt.savefig(os.path.join(args.out_dir, 'epitope_dist_balance.png'), dpi=600)

# plt.show()