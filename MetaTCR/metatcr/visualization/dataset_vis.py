import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import os

from scipy.stats import entropy
def jensen_shannon_divergence(p, q):  ## smaller JSD means more similar
    m = 0.5 * (p + q)
    jsd = 0.5 * (entropy(p, m) + entropy(q, m))
    return jsd

def visualize_all_datasets(mtx, setnames, min_dist=0.1, n_neighbors=50, dim=2, type = "cluster TCR diversity", out_dir = "./results/data_analysis"):

    """
    Use UMAP to visualize all datasets in mtxs. Color each dataset differently.
    mtxs: a numpy array of shape (n_samples, n_features)
    setnames: a list of dataset names, len(setnames) == mtxs.shape[0]
    dim: dimension of UMAP embedding
    """

    # UMAP embed all data
    embedding = umap.UMAP(min_dist=min_dist, n_neighbors=n_neighbors, n_components=dim, random_state=0).fit_transform(mtx)

    # Make a color palette with a color for each dataset
    # palette = sns.color_palette("tab20", len(setnames))
    unique_setnames = set(setnames)
    num_label = len(unique_setnames)
    if num_label <= 10:
        palette = sns.color_palette("tab10", num_label)
    else:
        palette = sns.color_palette("hls", num_label)

    # Create a dataframe with the embedding and dataset labels
    df = pd.DataFrame(embedding, columns=['umap1', 'umap2'])

    df['dataset'] = setnames
    if mtx.shape[0] > 1000:
        s_size = 10
    else:
        s_size = 30

    # Plot the UMAP embedding colored by dataset

    fig, ax = plt.subplots(figsize=(8, 6))
    ax = sns.scatterplot(x='umap1', y='umap2', hue='dataset', palette=palette, data=df, legend='full', alpha=0.7, s=s_size)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('UMAP visualization: ' + type)
    file_path = os.path.join(out_dir, 'UMAP_visualization_{}.png'.format(type))
    plt.savefig(file_path, dpi = 900, bbox_inches='tight')

def visualize_one_dataset_valid(mtx, smplist, df_metadata, refdata, id_col, label_col, min_dist=0.2, n_neighbors=20, dim=2, type="TCR diversity", out_dir = "./results/data_analysis"):
    """
    Use UMAP to visualize all datasets in mtxs. Color each dataset differently.
    dim: dimension of UMAP embedding
    label_col: one or multiple column names of the label in metadata
    """
    df_metadata.set_index(id_col, inplace=True)

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
        fig, axs = plt.subplots(1, len(label_col), figsize=(15, 5))

    else:
        fig, ax = plt.subplots(figsize=(10,10))
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
                palette=["gray"],
                legend="full",
                alpha=0.5,
                s=s_size,
            )
        # Plot control data
        ax = sns.scatterplot(
            x="umap1", y="umap2",
            hue="label",
            data=df[df["label"] == "Control"],
            legend="full",
            alpha=1,
            palette=["lightgray"],
            s=s_size,
        )
        # Plot mtx data colored by label
        ax = sns.scatterplot(
            x="umap1",
            y="umap2",
            hue="label",
            data=df[np.logical_and(df["label"] != "Reference", df["label"] != "Control")],
            legend="full",
            alpha=0.8,
            s=s_size,
            palette="Set2"
        )

        ax.set_title(str(label_col[i]))
    plt.suptitle("UMAP - datasets: " + type)
    plt.subplots_adjust(top=0.8, bottom=0.1)
    plt.savefig(os.path.join(out_dir, "UMAP_visualization_of_datasets_{}.png".format(type)))




def inner_dist_mtx(mtx):
    smp_num = mtx.shape[0]
    dist_matrix = np.zeros((smp_num, smp_num))
    for i in range(smp_num):
        for j in range(i+1, smp_num):
            ## euclidean distance
            dist = np.linalg.norm(mtx[i] - mtx[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix

def outer_dist_mtx(mtx1, mtx2):
    smp_num1 = mtx1.shape[0]
    smp_num2 = mtx2.shape[0]
    dist_matrix = np.zeros((smp_num1, smp_num2))
    for i in range(smp_num1):
        for j in range(smp_num2):
            ## euclidean distance
            dist = np.linalg.norm(mtx1[i] - mtx2[j])
            dist_matrix[i, j] = dist
    return dist_matrix

def get_upper_triangle(dist_matrix):
    upper_triangle = np.triu(dist_matrix, k=1)
    result = upper_triangle[upper_triangle != 0].tolist()
    return result

def calc_metrics(mtx1, mtx2):
    ### get the summary of the distance between two matrix
    mtx1_dist = inner_dist_mtx(mtx1)
    mtx1_dist = get_upper_triangle(mtx1_dist)
    mtx2_dist = inner_dist_mtx(mtx2)
    mtx2_dist = get_upper_triangle(mtx2_dist)
    mtx1_mtx2_dist = outer_dist_mtx(mtx1, mtx2)

    ## mean
    mean_mtx1 = np.mean(mtx1_dist)
    mean_mtx2 = np.mean(mtx2_dist)
    mean_mtx1_mtx2 = np.mean(mtx1_mtx2_dist)

    ## Variance
    var_mtx1 = np.var(mtx1_dist)
    var_mtx2 = np.var(mtx2_dist)

    ## Coefficient of Variation
    cv_mtx1 = np.std(mtx1_dist) / mean_mtx1 if mean_mtx1 else 0
    cv_mtx2 = np.std(mtx2_dist) / mean_mtx2 if mean_mtx2 else 0

    return mean_mtx1, mean_mtx2, mean_mtx1_mtx2, var_mtx1, var_mtx2, cv_mtx1, cv_mtx2

def diff_score(mean_mtx1, mean_mtx2, mean_mtx1_mtx2, var_mtx1, var_mtx2, cv_mtx1, cv_mtx2):
    alpha = 0.4
    beta = 0.4
    gamma = 0.2
    score = alpha * (mean_mtx1_mtx2 / max(mean_mtx1, mean_mtx2)) + beta * (max(var_mtx1, var_mtx2) / min(var_mtx1, var_mtx2)) + gamma * (max(cv_mtx1, cv_mtx2) / min(cv_mtx1, cv_mtx2))
    return score