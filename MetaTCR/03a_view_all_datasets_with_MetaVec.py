import os
from metatcr.utils.utils import save_pk, load_pkfile
import configargparse
import numpy as np
from metatcr.visualization.dataset_vis import visualize_metavec, jensen_shannon_divergence, calc_metrics, diff_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import umap
import random
random.seed(0)
import itertools

parser = configargparse.ArgumentParser()
parser.add_argument('--mtx_dir', type=str, default='./results/data_analysis/datasets_mtx', help='')
parser.add_argument('--add_dir', type=str, default='./results/data_analysis/datasets_mtx_subset', help='')
parser.add_argument('--out_dir', type=str, default='./results/data_analysis/datasets', help='Output directory for processed data')
parser.add_argument('--dataset_type_file', type=str, default='./data/datasets_type.csv', help='Output directory for processed data')


args = parser.parse_args()
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)
if not os.path.exists(args.add_dir):
    os.makedirs(args.add_dir)

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
            dataset_prefix = get_prefix(dataset_name)
            # Open the .pk file and load the dictionary
            # dataset_dict = load_pkfile(os.path.join(args.mtx_dir, filename))
            try:
                dataset_dict = load_pkfile(os.path.join(args.mtx_dir, filename))
            except FileNotFoundError:
                # If the file is not found in the first directory, try the second directory
                dataset_dict = load_pkfile(os.path.join(args.add_dir, filename))

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
    # plt.title("Average Jensen-Shannon Divergence diversity_mtx: " + keyword)
    file_path = os.path.join(args.out_dir, 'diversity_mtx {}.svg'.format(keyword))
    cg.savefig(file_path, dpi=600)
    # plt.show()
    return average_jsd_values

def process_data_pair(file1, file2):
    print("Identifying data pair: ", file1, file2)
    # Load and merge the datasets
    combined_diversity_mtx, combined_abundance_mtx, dataset_name_list = merge_dataset_mtx([file1, file2])
    # Apply UMAP
    embedding = umap.UMAP(min_dist=0.05, n_neighbors=50, n_components=3, random_state=0).fit_transform(combined_diversity_mtx)
    # Split the embedding back into the individual datasets
    ori_mtx1 = get_dataset_mtx(file1, "diversity_mtx")
    mtx1 = embedding[:ori_mtx1.shape[0]]
    mtx2 = embedding[ori_mtx1.shape[0]:]
    # Calculate the metrics
    return calc_metrics(mtx1, mtx2)


pos_pair_nums = 30
similar_pair_nums = 25

main_filelist = os.listdir(args.mtx_dir)
sub_filelist = os.listdir(args.add_dir)

## get the negative pairs
sub_prefixes = set(get_prefix(name) for name in sub_filelist)
neg_pairs = set()
for prefix in sub_prefixes:
    files_with_prefix = [file for file in sub_filelist if get_prefix(file) == prefix]
    for pair in itertools.combinations(files_with_prefix, 2):
        neg_pairs.add(pair)
neg_pairs = list(neg_pairs)

similar_pairs =set()  ## in one study but not the same batch or sample type
for file in main_filelist:
    prefix = get_prefix(file)
    files_with_prefix = [file for file in main_filelist if get_prefix(file) == prefix]
    for pair in itertools.combinations(files_with_prefix, 2):
        similar_pairs.add(pair)
similar_pairs = list(similar_pairs)
similar_pairs = random.sample(similar_pairs, similar_pair_nums)

main_pairs = set(itertools.combinations(main_filelist, 2))
pos_pairs = set()
for pair in main_pairs:
    if (get_prefix(pair[0]) != get_prefix(pair[1])):
        pos_pairs.add(pair)
pos_pairs = list(pos_pairs)
# Randomly select x pairs
pos_pairs = random.sample(pos_pairs, pos_pair_nums)

selected_pairs = pos_pairs + similar_pairs + neg_pairs

print("Number of selected pairs: ", len(selected_pairs))
print("pos_pairs: ", pos_pairs)
print("similar_pairs: ", similar_pairs)
print("neg_pairs: ", neg_pairs)


pair_labels = ['different'] * pos_pair_nums + ['similar'] * len(similar_pairs) + ['same'] * len(neg_pairs)
batch_paras = pd.DataFrame(columns=["File1", "File2", "Mean_mtx1", "Mean_mtx2", "Mean_mtx1_mtx2", "Var_mtx1", "Var_mtx2", "CV_mtx1", "CV_mtx2"])

# Process each data pair
for pair_idx, (file1, file2) in enumerate(selected_pairs):
    mean_mtx1, mean_mtx2, mean_mtx1_mtx2, var_mtx1, var_mtx2, cv_mtx1, cv_mtx2 = process_data_pair(file1, file2)
    score = diff_score(mean_mtx1, mean_mtx2, mean_mtx1_mtx2, var_mtx1, var_mtx2, cv_mtx1, cv_mtx2)
    batch_paras = batch_paras.append({
        "File1": file1,
        "File2": file2,
        "Mean_mtx1": mean_mtx1,
        "Mean_mtx2": mean_mtx2,
        "Mean_mtx1_mtx2": mean_mtx1_mtx2,
        "Var_mtx1": var_mtx1,
        "Var_mtx2": var_mtx2,
        "CV_mtx1": cv_mtx1,
        "CV_mtx2": cv_mtx2,
        "Score": score,
        "Label": pair_labels[pair_idx]  ## int(pair_labels[pair_idx])
    }, ignore_index=True)

batch_paras.sort_values(by="Score",inplace=True,ascending=False)
batch_paras.to_csv(os.path.join(args.out_dir, "data_pair_summary.csv"), index=False)


## load the data pair summary
batch_paras = pd.read_csv(os.path.join(args.out_dir, "data_pair_summary.csv"))
## plot auc curve for the data pairs score and label

fig, axs = plt.subplots(1, 2, figsize=(9, 4))
# Plot the data pair summary score and the threshold
ax1 = axs[0]
ax1.set_yscale('log')
break_from = 50
break_to = 150
colors = {'different': 'orchid', 'similar': 'lightskyblue', 'same': 'dodgerblue'}

for label, color in colors.items():
    ax1.bar(batch_paras[batch_paras['Label'] == label].index,
            batch_paras[batch_paras['Label'] == label]['Score'],
            color=color,
            label=label.capitalize(),
            width=0.7)

ax1.legend()
# ax1.title.set_text("Distance score of dataset pairs")
ax1.set_ylabel("Dissimilarity score")
ax1.set_xlabel("Dataset pairs")
# ax1.get_xaxis().set_visible(False)

batch_paras["Label"] = batch_paras["Label"].apply(lambda x: 1 if x == "different" else 0)

ax2 = axs[1]

max_auc = 0
best_threshold = None
y_true = batch_paras["Label"]
y_scores = batch_paras["Score"]
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
auc_score = auc(fpr, tpr)

ax2.plot(1 - fpr, tpr, lw=1.5, alpha=0.7, label=f'(AUC = {auc_score:.2f})')
ax2.plot([0, 1], [1, 0],'r--', lw=0.5)
ax2.set_xlim([1.05, -0.05])
ax2.set_ylim([-0.05, 1.05])
# ax2.set_title("AUC curve for evaluating dataset distance score")
ax2.legend(loc="lower right", frameon=False)
ax2.set_xlabel("Specificity")
ax2.set_ylabel("Sensitivity")
ratio = 1
xleft, xright = ax2.get_xlim()
ybottom, ytop = ax2.get_ylim()
ax2.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)
plt.tight_layout()
plt.savefig(os.path.join(args.out_dir, "data_pair_summary.svg"), dpi=600)
plt.show()

## Iterate over all .pk files in the specified directory
filelist = os.listdir(args.mtx_dir)
combined_diversity_mtx, combined_abundance_mtx, dataset_name_list = merge_dataset_mtx(filelist, merge_subset=True)
visualize_metavec(combined_diversity_mtx, dataset_name_list, min_dist=0.5, n_neighbors=50, type = "All datasets", out_dir = args.out_dir)

## relabel the datasets
data_info = pd.read_csv(args.dataset_type_file)
## transfer the dataset name to the dataset type
study2seq = dict(zip(data_info['Study'], data_info['Sequencing Platform']))
dataset_seqtypes = [study2seq[name] for name in dataset_name_list]
visualize_metavec(combined_diversity_mtx, dataset_seqtypes, min_dist=0.5, n_neighbors=50, type = "All datasets (sequencing)", out_dir = args.out_dir)

study2pip = dict(zip(data_info['Study'], data_info['Data process pipeline']))
dataset_pipelines = [study2pip[name] for name in dataset_name_list]
visualize_metavec(combined_diversity_mtx, dataset_pipelines, min_dist=0.5, n_neighbors=50, type = "All datasets (pipeline)", out_dir = args.out_dir)

dataset_types = [study2seq[name] + " + " + study2pip[name] for name in dataset_name_list]
visualize_metavec(combined_diversity_mtx, dataset_types, min_dist=0.5, n_neighbors=50, type = "All datasets (sequencing + pipeline)", out_dir = args.out_dir)

# ## melanoma
melanoma_files = ["Robert2014.pk", "Valpione2020.pk", "Weber2018.pk", "Huuhtanen2022.pk"]
combined_diversity_mtx, combined_abundance_mtx, dataset_name_list = merge_dataset_mtx(melanoma_files)
visualize_metavec(combined_diversity_mtx, dataset_name_list, min_dist=0.1, n_neighbors=15, type = "Melanoma datasets", out_dir = args.out_dir)

# ## covid
covid_files = [file for file in main_filelist if get_prefix(file) == 'ImmuneCODE' ]
combined_diversity_mtx, combined_abundance_mtx, dataset_name_list = merge_dataset_mtx(covid_files)
visualize_metavec(combined_diversity_mtx, dataset_name_list, min_dist=0.05, n_neighbors=50, type = "ImmuneCODE datasets", out_dir = args.out_dir)

# ## cmv
cmv_files = ["Emerson2017_HIP.pk","Emerson2017_Keck.pk"]
combined_diversity_mtx, combined_abundance_mtx, dataset_name_list = merge_dataset_mtx(cmv_files)
visualize_metavec(combined_diversity_mtx, dataset_name_list, min_dist=0.05, n_neighbors=50, type = "Emerson2017 datasets", out_dir = args.out_dir)

# ## lung cancer
lung_files = ["Jia2018.pk", "TRACERx_Tissue.pk", "TRACERx_PBMC.pk", "Formenti2018.pk", "MDanderson2019.pk", "Houghton2017.pk"]
combined_diversity_mtx, combined_abundance_mtx, dataset_name_list = merge_dataset_mtx(lung_files)
visualize_metavec(combined_diversity_mtx, dataset_name_list, min_dist=0.05, n_neighbors=50, type = "Lung cancer datasets", out_dir = args.out_dir)

# ## multi cancer PBMC
cancer_files = ["Wang2022_Tumor.pk", "Yan2019_PBMC.pk", "Shi2017_PBMC.pk", "Lee2020.pk",
              "TRACERx_PBMC.pk", "Formenti2018.pk", "MDanderson2019.pk",
              "Robert2014.pk", "Valpione2020.pk", "Weber2018.pk", "Huuhtanen2022.pk"]
combined_diversity_mtx, combined_abundance_mtx, dataset_name_list = merge_dataset_mtx(cancer_files)
visualize_metavec(combined_diversity_mtx, dataset_name_list, min_dist=0.5, n_neighbors=50, type = "Multi-cancer datasets (PBMC)", out_dir = args.out_dir)

# ## healthy control
healthy_files = ["Wang2022_Normal.pk", "Liu2019_Normal.pk", "Emerson2017-Keck_Normal.pk", "Emerson2017-HIP_Normal.pk", "Tcrbv4-control_Normal.pk", "Dewitt2015.pk"]
combined_diversity_mtx, combined_abundance_mtx, dataset_name_list = merge_dataset_mtx(healthy_files)
visualize_metavec(combined_diversity_mtx, dataset_name_list, min_dist=0.5, n_neighbors=50, type = "Healthy control datasets", out_dir = args.out_dir)

# ## wang2022 gastric two labels
sx_files = ["Wang2022_Normal.pk", "Wang2022_Tumor.pk"]
combined_diversity_mtx, combined_abundance_mtx, dataset_name_list = merge_dataset_mtx(sx_files)
visualize_metavec(combined_diversity_mtx, dataset_name_list, min_dist=0.1, n_neighbors=15, type = "Wang2022 sub-datasets", out_dir = args.out_dir)

## plot the JSD bewteen the datasets
draw_JSD(cancer_files,keyword ="cancer_files")
draw_JSD(healthy_files,keyword ="healthy_files")
draw_JSD(melanoma_files,keyword ="melanoma")
draw_JSD(covid_files,keyword ="covid")
draw_JSD(lung_files,keyword ="lung")
