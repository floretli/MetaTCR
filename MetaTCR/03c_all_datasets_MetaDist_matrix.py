import os
from metatcr.utils.utils import save_pk, load_pkfile
import configargparse
import numpy as np
from metatcr.visualization.dataset_vis import jensen_shannon_divergence, calc_metrics, diff_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import umap
import random
random.seed(0)


parser = configargparse.ArgumentParser()
parser.add_argument('--mtx_dir', type=str, default='./results/data_analysis/datasets_mtx', help='')
parser.add_argument('--out_dir', type=str, default='./results/data_analysis/datasets', help='Output directory for processed data')

args = parser.parse_args()
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

print("Running with the following parameters:")
for key, value in vars(args).items():
    print(f"{key}: {value}")
print("#######################################")

def merge_dataset_mtx(filelist, max_sample = None):
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
                if max_sample is not None and len(dataset_dict["sample_list"]) > max_sample:
                    indices = random.sample(range(len(dataset_dict["sample_list"])), max_sample)
                    dataset_dict["sample_list"] = [dataset_dict["sample_list"][i] for i in indices]
                    dataset_dict["diversity_mtx"] = dataset_dict["diversity_mtx"][indices]
            except FileNotFoundError:
                # If the file is not found in the first directory, try the second directory
                dataset_dict = load_pkfile(os.path.join(args.add_dir, filename))

            # Add the "diversity_mtx" element of the dictionary to the list
            diversity_mtx_list.append(dataset_dict["diversity_mtx"])
            dataset_name_list += [dataset_name] * len(dataset_dict["sample_list"])

    # Stack all of the "diversity_mtx" arrays vertically to form a new numpy array
    file1_size = len(diversity_mtx_list[0])
    combined_diversity_mtx = np.vstack(diversity_mtx_list)
    return combined_diversity_mtx, dataset_name_list, file1_size

def get_dataset_mtx(filename, key):
    # Initialize an empty list to store the numpy arrays

    try:
        dataset_dict = load_pkfile(os.path.join(args.mtx_dir, filename))
    except FileNotFoundError:
        # If the file is not found in the first directory, try the second directory
        dataset_dict = load_pkfile(os.path.join(args.add_dir, filename))
    return dataset_dict[key]

def draw_JSD(melanoma_files,keyword ="melanoma"):
    # mtxs_a = [np.mean(get_dataset_mtx(file, "abundance_mtx"), axis= 0 ) for file in melanoma_files]
    mtxs_f = [np.mean(get_dataset_mtx(file, "diversity_mtx"), axis= 0) for file in melanoma_files]
    names = [file[:-3] for file in melanoma_files]

    average_jsd_values = np.zeros((len(names), len(names)))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            average_jsd = jensen_shannon_divergence(mtxs_f[i], mtxs_f[j])
            average_jsd_values[i, j] = average_jsd
            average_jsd_values[j, i] = average_jsd

    plt.figure()
    # ax = sns.heatmap(average_jsd_values, annot=False, xticklabels=names, yticklabels=names, cmap="coolwarm")
    cg = sns.clustermap(average_jsd_values, cmap="coolwarm", annot=False, yticklabels=names, xticklabels=names)
    plt.title("Average Jensen-Shannon Divergence diversity_mtx: " + keyword)
    file_path = os.path.join(args.out_dir, 'diversity_mtx {}.png'.format(keyword))
    cg.savefig(file_path, dpi=600,  bbox_inches='tight')
    # plt.show()
    return average_jsd_values

def process_data_pair(file1, file2):
    print("Identifying data pair: ", file1, file2)
    # Load and merge the datasets
    combined_diversity_mtx, _, file1_size = merge_dataset_mtx([file1, file2], max_sample=None)
    # Apply UMAP
    embedding = umap.UMAP(min_dist=0.05, n_neighbors=50, n_components=3, random_state=0).fit_transform(combined_diversity_mtx)
    # Split the embedding back into the individual datasets
    mtx1 = embedding[:file1_size]
    mtx2 = embedding[file1_size:]
    print("file1 size: ", file1_size, "file2 size: ", len(embedding) - file1_size)
    # Calculate the metrics
    return calc_metrics(mtx1, mtx2)

filelist = os.listdir(args.mtx_dir)
filenames = [name[:-3] for name in filelist]

label_col = "Data process pipeline"
# label_col = "Sequencing Platform"
info_df = pd.read_csv("./data/datasets_type.csv")
## class label: pipeline
name2label = dict(zip(info_df['Dataset'], info_df[label_col]))
seqs = [name2label[name] for name in filenames]
unique_labels = list(set(seqs))
platform_palette = sns.color_palette("hls", len(info_df[label_col].unique()))
label2color = {platform: color for platform, color in zip(info_df[label_col].unique(), platform_palette)}


## write the data distance score to a csv file, index and columns are the dataset names
pair_dist_df = pd.DataFrame(index=filenames, columns=filenames)

# Process each data pair
# for pair_idx, (file1, file2) in enumerate(all_pairs):
for i in range(len(filelist)):
    file1 = filelist[i]
    print("Processing data pair - file1 idx: ", i+1, "/", len(filelist))
    for j in range(i + 1, len(filelist)):
        file2 = filelist[j]
        mean_mtx1, mean_mtx2, mean_mtx1_mtx2, var_mtx1, var_mtx2, cv_mtx1, cv_mtx2 = process_data_pair(file1, file2)
        score = diff_score(mean_mtx1, mean_mtx2, mean_mtx1_mtx2, var_mtx1, var_mtx2, cv_mtx1, cv_mtx2)
        pair_dist_df.loc[file1[:-3], file2[:-3]] = score
        pair_dist_df.loc[file2[:-3], file1[:-3]] = score

pair_dist_df.fillna(1, inplace=True)
pair_dist_df.to_csv(os.path.join(args.out_dir, "dataset_metadist.csv"), float_format='%.4f')


pair_dist_df = pd.read_csv(os.path.join(args.out_dir, "dataset_metadist.csv"), index_col=0)
filenames = pair_dist_df.index
pair_dist_df = np.log(pair_dist_df)
row_colors = pair_dist_df.index.map(lambda x: label2color[name2label[x]] if x in name2label else "black")
cg = sns.clustermap(pair_dist_df, cmap="Blues", annot=False, row_colors=row_colors,
                    cbar_kws={'label': 'log(distance)', 'shrink': 0.5, 'fraction': 0.025})

# adjust the subplot margins
cg.fig.subplots_adjust(left=0.2, right=0.8, top=0.85, bottom=0.25)

# set the x and y tick label font size
cg.ax_heatmap.tick_params(labelsize=8)
# move the colorbar to the top right corner
cg.cax.set_position([.78, .8, .02, .1])
# plt.title("Dataset Meta Distance")
plt_path = os.path.join(args.out_dir, 'dataset_metadist_pipeline.svg')
cg.savefig(plt_path,  bbox_inches='tight', dpi=600)

# mds for the distance matrix
from sklearn.manifold import MDS

embedding = MDS(n_components=2, dissimilarity='precomputed').fit_transform(pair_dist_df)
plt.figure()
# plt.scatter(embedding[:, 0], embedding[:, 1], c=row_colors)
for label in unique_labels:
    indices = [i for i, x in enumerate(pair_dist_df.index) if name2label[x] == label]
    plt.scatter(embedding[indices, 0], embedding[indices, 1], color=label2color[label], label=label)
plt.legend(title=label_col, loc='upper left', labels=unique_labels)
plt.savefig(os.path.join(args.out_dir, "dataset_metadist_mds_pipeline.svg"), dpi=600)

