import os
from metatcr.utils.utils import load_pkfile
import configargparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import random
random.seed(0)

parser = configargparse.ArgumentParser()
parser.add_argument('--mtx_dir', type=str, default='./results_50/data_analysis/datasets_meta_mtx', help='')
parser.add_argument('--add_dir', type=str, default='./results_50/data_analysis/datasets_meta_mtx_add', help='')
parser.add_argument('--out_dir', type=str, default='./results_50/data_analysis/time_series', help='Output directory for processed data')

args = parser.parse_args()
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

print("Running with the following parameters:")
for key, value in vars(args).items():
    print(f"{key}: {value}")
print("#######################################")


def get_dataset_mtx(filename, key):
    # Initialize an empty list to store the numpy arrays

    try:
        dataset_dict = load_pkfile(os.path.join(args.mtx_dir, filename))
    except FileNotFoundError:
        # If the file is not found in the first directory, try the second directory
        dataset_dict = load_pkfile(os.path.join(args.add_dir, filename))
    return dataset_dict[key]

def visualize_one_dataset_time(mtx, smplist, metadata, refdata, id_col, label_col, min_dist=0.2, n_neighbors=20, dim=2, type="TCR diversity", out_dir = "./results/data_analysis"):
    """
    Use UMAP to visualize all datasets in mtxs. Color each dataset differently.
    dim: dimension of UMAP embedding
    label_col: one or multiple column names of the label in metadata
    """
    # Load metadata
    df_metadata = pd.read_csv(metadata)
    df_metadata.set_index(id_col, inplace=True)
    print(df_metadata)

    common_samples = list(set(smplist) & set(df_metadata.index))
    common_indices = [smplist.index(sample) for sample in common_samples]
    mtx = mtx[common_indices, :]
    smplist = [smplist[i] for i in common_indices]

    if isinstance(label_col, str):
        label_col = [label_col]

    labels_mtx = []
    for col in label_col:
        labels_mtx.append([df_metadata.at[sample_id, col] for sample_id in smplist])

        # Check if refdata is None, if so only use mtx for embedding and plotting
    if refdata is None:
        data = mtx
        labels = labels_mtx
        s_size = 30
    else:
        # Concatenate mtx and refdata
        data = np.vstack((mtx, refdata))
        labels_ref = ["Reference"] * refdata.shape[0]
        # Combine labels for mtx data and refdata
        labels = [labels_m + labels_ref for labels_m in labels_mtx]
        s_size = 20
        smplist = smplist + ["Reference"] * refdata.shape[0]

    # UMAP embed all data
    embedding = umap.UMAP(min_dist=min_dist,n_neighbors=n_neighbors, n_components=dim, random_state=1).fit_transform(data)

    # Create a dataframe with the embedding and dataset labels
    df = pd.DataFrame(embedding, columns=["umap1", "umap2"])
    df['sample'] = smplist

    if len(label_col) > 1:
        fig, axs = plt.subplots(1, len(label_col), figsize=(13, 5))

    else:
        fig, ax = plt.subplots(figsize=(5,5))
        axs = [ax]

    for i, label in enumerate(labels):
        # ax = axs[i]
        plt.sca(axs[i])
        df["label"] = label
        if refdata is not None:
            ax = sns.scatterplot(
                x="umap1",
                y="umap2",
                hue="label",
                data=df[df["label"] == "Reference"],
                palette=["lightgray"],
                legend="full",
                alpha=0.5,
                s=s_size,
            )
        # Plot mtx data colored by label
        ax = sns.scatterplot(
            x="umap1",
            y="umap2",
            hue="label",
            data=df[df["label"] != "Reference"],
            legend="full",
            alpha=1,
            s=s_size,
            palette="rainbow"
        )
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles=handles, labels=labels, bbox_to_anchor=(1.00, 1), fontsize=8)

        ax.set_title(str(label_col[i]))

    plt.suptitle("UMAP - datasets: " + type)
    plt.subplots_adjust(wspace=0.4)
    plt.savefig(os.path.join(out_dir, "UMAP_visualization_of_datasets_{}.png".format(type)), dpi=600)
    # plt.show()


## Plot time series samples from Snyder2017 dataset
snyder2017_mtx = get_dataset_mtx("snyder2017_plos.pk", "diversity_mtx")
snyder2017_smp = get_dataset_mtx("snyder2017_plos.pk", "sample_list")
snyder2017_meta = "./data/snyder2017.csv"

visualize_one_dataset_time(snyder2017_mtx, snyder2017_smp, metadata=snyder2017_meta, refdata=None,
                      id_col="Sample ID", label_col=["patient_id", "Clinical Response"],
                      min_dist=0.05, n_neighbors=10, dim=2, type="snyder2017", out_dir=args.out_dir)


## Plot time series samples from healthy-time-course dataset
tc_mtx = get_dataset_mtx("healthy-time-course.pk", "diversity_mtx")
tc_smp = get_dataset_mtx("healthy-time-course.pk", "sample_list")
tc_meta = "./data/healthy-time-course.csv"

visualize_one_dataset_time(tc_mtx, tc_smp, metadata=tc_meta, refdata=None,
                      id_col="sample_id", label_col=["patient_id", "data_source"],
                      min_dist=0.05, n_neighbors=10, dim=2, type="healthy-time-course", out_dir=args.out_dir)