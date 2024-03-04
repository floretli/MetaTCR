import os
from metatcr.utils.utils import load_pkfile
import configargparse
from metatcr.visualization.dataset_vis import calc_metrics, diff_score, visualize_metavec_inset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
import random
random.seed(0)

import warnings
from numba import NumbaDeprecationWarning
warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)

parser = configargparse.ArgumentParser()
parser.add_argument('--out_dir', type=str, default='./results/data_analysis/Emerson_intra', help='Output directory for processed data')
parser.add_argument('--centroids', type=str, default='./results/data_analysis/96_best_centers.pk', help='centroids file')
parser.add_argument('--meta_data', type=str, default='./data/Emerson2017.csv', help='input metadata file')

args = parser.parse_args()
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

print("Running with the following parameters:")
for key, value in vars(args).items():
    print(f"{key}: {value}")
print("#######################################")

def show_TCR_freq_in_batches(mtx, id_col = "sample_id", label_col = "umapbatch"):
    maxy = 0.065

    labels = df_meta[label_col].unique()
    sids1 = df_meta[df_meta[label_col]==labels[0]][id_col].tolist()
    sids2 = df_meta[df_meta[label_col]==labels[1]][id_col].tolist()
    sids1 = list(set(sids1).intersection(set(smp_list)))
    sids2 = list(set(sids2).intersection(set(smp_list)))

    idx1 = [smp_list.index(sid) for sid in sids1]
    idx2 = [smp_list.index(sid) for sid in sids2]

    plt.figure(figsize=(10, 12))

    ax1 = plt.subplot(3, 1, 1)
    df = pd.DataFrame(mtx.T[:,idx1]/ len(sids1))
    df.plot.bar(stacked=True, ax=ax1)

    ax2 = plt.subplot(3, 1, 2)
    df = pd.DataFrame(mtx.T[:,idx2]/ len(sids2))

    df.plot.bar(stacked=True, ax=ax2)

    ax1.legend().set_visible(False)
    ax1.set_xticklabels([])
    ax1.set_ylabel('Average TCR frequency')
    ax1.set_title(labels[0])
    ax1.set_ylim([0, maxy])

    ax2.legend().set_visible(False)
    ax2.set_xticklabels([])
    # ax2.set_xticklabels(np.arange(1, mtx.shape[1] + 1))
    # ax2.set_xlabel('Cluster id')
    ax2.set_ylabel('Average TCR frequency')
    ax2.set_title(labels[1])
    ax2.set_ylim([0, maxy])

    ax3 = plt.subplot(3, 1, 3)
    sum1 = np.sum(mtx.T[:, idx1] / len(sids1), axis=1)
    sum2 = np.sum(mtx.T[:, idx2] / len(sids2), axis=1)
    df1 = pd.DataFrame({labels[0]: sum1})
    df2 = pd.DataFrame({labels[1]: sum2})
    df1.plot.bar(color='royalblue', alpha=0.7, ax=ax3, label=labels[0])
    df2.plot.bar(color='orange', alpha=0.5, ax=ax3, label=labels[1])


    # ax3.bar(np.arange(len(sum1)), sum1, color='skyblue', label=labels[0], alpha=0.6)
    # ax3.bar(np.arange(len(sum2)), sum2, color='salmon', label=labels[1], alpha=0.6)
    ax3.set_xticklabels(np.arange(1, sum1.shape[0] + 1))
    ax3.set_xlabel('Cluster id')
    ax3.set_ylabel('Comparison of TCR frequency')
    ax3.set_title(labels[0] + " vs " + labels[1])
    ax3.set_ylim([0, maxy])
    ax3.tick_params(axis='x', labelsize=6)
    ax3.legend()
    plt.subplots_adjust(hspace=0.3)

    return plt

def show_TCR_freq_in_batches_topk(mtx, id_col = "sample_id", label_col = "umapbatch"):
    maxy = 0.065

    labels = df_meta[label_col].unique()
    sids1 = df_meta[df_meta[label_col]==labels[0]][id_col].tolist()
    sids2 = df_meta[df_meta[label_col]==labels[1]][id_col].tolist()
    sids1 = list(set(sids1).intersection(set(smp_list)))
    sids2 = list(set(sids2).intersection(set(smp_list)))

    idx1 = [smp_list.index(sid) for sid in sids1]
    idx2 = [smp_list.index(sid) for sid in sids2]

    plt.figure(figsize=(6, 4))
    ax3 = plt.subplot()
    sum1 = np.sum(mtx.T[:, idx1] / len(sids1), axis=1)
    sum2 = np.sum(mtx.T[:, idx2] / len(sids2), axis=1)

    diff = np.abs(sum1 - sum2) / np.maximum(sum1, sum2)
    top10_index = np.argsort(diff)[-10:]

    df1 = pd.DataFrame({labels[0]: sum1[top10_index]})
    df2 = pd.DataFrame({labels[1]: sum2[top10_index]})

    df1.plot.bar(color='royalblue', alpha=0.7, ax=ax3, label=labels[0])
    df2.plot.bar(color='orange', alpha=0.5, ax=ax3, label=labels[1])

    cluster_ids = np.array(["c_" + str(i) for i in range(len(diff))])
    top10_ids = cluster_ids[top10_index]
    ax3.set_xticklabels(top10_ids)
    ax3.set_xlabel('Cluster id')
    ax3.set_ylabel('Comparison of TCR frequency')
    ax3.set_title(labels[0] + " vs " + labels[1])
    ax3.set_ylim([0, maxy])
    ax3.tick_params(axis='x', labelsize=8)
    ax3.legend()
    plt.subplots_adjust(hspace=0.3, bottom=0.3)

    return plt

### load data
batch1_path = "results/data_analysis/datasets_mtx/Emerson2017_HIP.pk"
batch2_path = "results/data_analysis/datasets_mtx/Emerson2017_Keck.pk"

dataset_dict1 = load_pkfile(batch1_path)
dataset_dict2 = load_pkfile(batch2_path)

## merge two batches
representations = np.concatenate((dataset_dict1['diversity_mtx'], dataset_dict2['diversity_mtx']), axis=0)
smp_list = dataset_dict1['sample_list'] + dataset_dict2['sample_list']

## load metadata
df_meta = pd.read_csv(args.meta_data)
df_meta['Set'] = df_meta['Set'].replace({'HIP': 'Batch HIP', 'Keck': 'Batch Keck'})
df_meta['Sample label'] = df_meta['Sample label'].replace({'0': 'CMV-', '1': 'CMV+'})

# ### umap
visualize_metavec_inset(representations, smp_list, df_metadata=df_meta, refdata=None,
                      id_col="smpid", label_col=["Sample label", "Set"],
                      min_dist=0.05, n_neighbors=20, dim=2, type="Emerson datasets", out_dir=args.out_dir)

## calculate dataset distance score
umap_embedding = umap.UMAP(min_dist=0.05, n_neighbors=50, n_components=3, random_state=0).fit_transform(representations)
# split mtx by simulated batch label

df_meta = df_meta.reset_index()
batchA_list = df_meta[df_meta["set"]=='Batch HIP']["smpid"].tolist()
batchB_list = df_meta[df_meta["set"]=='Batch Keck']['smpid'].tolist()
mtx1 = umap_embedding[[smp_list.index(sid) for sid in smp_list if sid in batchA_list]]
mtx2 = umap_embedding[[smp_list.index(sid) for sid in smp_list if sid in batchB_list]]
score = diff_score(*calc_metrics(mtx1, mtx2))
thres = 1.245

print("Score: ", score)
if score > thres:
    print("Significant difference batch effect detected!")
else:
    print("No significant difference batch effect detected.")

# show TCR frequency in different batches
plt1 = show_TCR_freq_in_batches(mtx = representations, id_col = "smpid", label_col = "Sample label")
plt.savefig(os.path.join(args.out_dir,"TCR frequency in different labels.svg"), dpi=600, bbox_inches='tight')

plt2 = show_TCR_freq_in_batches(mtx = representations, id_col = "smpid", label_col = "Set")
plt.savefig(os.path.join(args.out_dir,"TCR frequency in different batches.svg"), dpi=600, bbox_inches='tight')

plt3 = show_TCR_freq_in_batches_topk(mtx = representations, id_col = "smpid", label_col = "Set")
plt.savefig(os.path.join(args.out_dir,"TCR frequency in Emerson2017 batches top10.svg"), dpi=600, bbox_inches='tight')