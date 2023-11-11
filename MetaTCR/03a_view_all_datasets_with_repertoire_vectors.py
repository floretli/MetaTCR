import os
from metatcr.utils.utils import save_pk, load_pkfile
import configargparse
import numpy as np
from metatcr.visualization.dataset_vis import visualize_all_datasets, jensen_shannon_divergence, calc_metrics, diff_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import umap
import random
random.seed(0)
import itertools

parser = configargparse.ArgumentParser()
parser.add_argument('--mtx_dir', type=str, default='./results_50/data_analysis/datasets_meta_mtx', help='')
parser.add_argument('--add_dir', type=str, default='./results_50/data_analysis/datasets_meta_mtx_add', help='')
parser.add_argument('--out_dir', type=str, default='./results_50/data_analysis/datasets', help='Output directory for processed data')

args = parser.parse_args()
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)
if not os.path.exists(args.add_dir):
    os.makedirs(args.add_dir)

print("Running with the following parameters:")
for key, value in vars(args).items():
    print(f"{key}: {value}")
print("#######################################")

def merge_dataset_mtx(filelist):
    # Initialize an empty list to store the numpy arrays
    diversity_mtx_list = []
    abundance_mtx_list = []
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
            abundance_mtx_list.append(dataset_dict["abundance_mtx"])
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
    plt.title("Average Jensen-Shannon Divergence diversity_mtx: " + keyword)
    file_path = os.path.join(args.out_dir, 'diversity_mtx {}.png'.format(keyword))
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

def split_sx_dataset(dataset_dict):
    # Initialize empty dictionaries
    nor_dict = {"sample_list": [], "diversity_mtx": [], "abundance_mtx": []}
    tu_dict = {"sample_list": [], "diversity_mtx": [], "abundance_mtx": []}
    # Iterate over each sample
    for i, sample in enumerate(dataset_dict["sample_list"]):
        if "Cr" in sample:  ## control data
            nor_dict["sample_list"].append(sample)
            nor_dict["diversity_mtx"].append(dataset_dict["diversity_mtx"][i])
            nor_dict["abundance_mtx"].append(dataset_dict["abundance_mtx"][i])
        elif "Gr" in sample:  ## gastric cancer data
            tu_dict["sample_list"].append(sample)
            tu_dict["diversity_mtx"].append(dataset_dict["diversity_mtx"][i])
            tu_dict["abundance_mtx"].append(dataset_dict["abundance_mtx"][i])

    nor_dict["diversity_mtx"] = np.vstack(nor_dict["diversity_mtx"])
    nor_dict["abundance_mtx"] = np.vstack(nor_dict["abundance_mtx"])
    tu_dict["diversity_mtx"] = np.vstack(tu_dict["diversity_mtx"])
    tu_dict["abundance_mtx"] = np.vstack(tu_dict["abundance_mtx"])
    save_pk(os.path.join(args.add_dir, "Sx_gastric_Normal.pk"), nor_dict)
    save_pk(os.path.join(args.add_dir, "Sx_gastric_Tumor.pk"), tu_dict)

def split_tcrbv4_control_dataset(dataset_dict):

    ## PBMC and normal samples
    normal_list = ["Subject_143","Subject_111","Subject_118","Subject_119","Subject_112","Subject_123","Subject_110","Subject_137","Subject_115","Subject_114","Subject_28","Subject_124","Subject_138","Subject_116","Subject_37","Subject_120","Subject_122","Subject_25","Subject_117"]
    # Initialize empty dictionaries
    normal_dict = {"sample_list": [], "diversity_mtx": [], "abundance_mtx": []}
    tumor_dict = {"sample_list": [], "diversity_mtx": [], "abundance_mtx": []}

    # Iterate over each sample
    for i, sample in enumerate(dataset_dict["sample_list"]):
        if sample in normal_list:  ## control data
            normal_dict["sample_list"].append(sample)
            normal_dict["diversity_mtx"].append(dataset_dict["diversity_mtx"][i])
            normal_dict["abundance_mtx"].append(dataset_dict["abundance_mtx"][i])
        else:
            tumor_dict["sample_list"].append(sample)
            tumor_dict["diversity_mtx"].append(dataset_dict["diversity_mtx"][i])
            tumor_dict["abundance_mtx"].append(dataset_dict["abundance_mtx"][i])


    normal_dict["diversity_mtx"] = np.vstack(normal_dict["diversity_mtx"])
    normal_dict["abundance_mtx"] = np.vstack(normal_dict["abundance_mtx"])
    tumor_dict["diversity_mtx"] = np.vstack(tumor_dict["diversity_mtx"])
    tumor_dict["abundance_mtx"] = np.vstack(tumor_dict["abundance_mtx"])

    save_pk(os.path.join(args.add_dir, "tcrbv4_control_Normal.pk"), normal_dict)
    save_pk(os.path.join(args.add_dir, "tcrbv4_control_Tumor.pk"), tumor_dict)

def split_cmv_dataset(hip_dict, keck_dict):
    metadf = pd.read_csv('./data/CMVlabels.csv')
    pos_list = metadf[metadf['label'] == "1"]['smpid'].tolist()
    neg_list = metadf[metadf['label'] == "0"]['smpid'].tolist()

    hip_pos_dict = {"sample_list": [], "diversity_mtx": [], "abundance_mtx": []}
    hip_neg_dict = {"sample_list": [], "diversity_mtx": [], "abundance_mtx": []}
    keck_pos_dict = {"sample_list": [], "diversity_mtx": [], "abundance_mtx": []}
    keck_neg_dict = {"sample_list": [], "diversity_mtx": [], "abundance_mtx": []}

    for i, sample in enumerate(hip_dict["sample_list"]):
        if sample in pos_list:
            hip_pos_dict["sample_list"].append(sample)
            hip_pos_dict["diversity_mtx"].append(hip_dict["diversity_mtx"][i])
            hip_pos_dict["abundance_mtx"].append(hip_dict["abundance_mtx"][i])
        elif sample in neg_list:
            hip_neg_dict["sample_list"].append(sample)
            hip_neg_dict["diversity_mtx"].append(hip_dict["diversity_mtx"][i])
            hip_neg_dict["abundance_mtx"].append(hip_dict["abundance_mtx"][i])
    for i, sample in enumerate(keck_dict["sample_list"]):
        if sample in pos_list:
            keck_pos_dict["sample_list"].append(sample)
            keck_pos_dict["diversity_mtx"].append(keck_dict["diversity_mtx"][i])
            keck_pos_dict["abundance_mtx"].append(keck_dict["abundance_mtx"][i])
        elif sample in neg_list:
            keck_neg_dict["sample_list"].append(sample)
            keck_neg_dict["diversity_mtx"].append(keck_dict["diversity_mtx"][i])
            keck_neg_dict["abundance_mtx"].append(keck_dict["abundance_mtx"][i])

    hip_pos_dict["diversity_mtx"] = np.vstack(hip_pos_dict["diversity_mtx"])
    hip_pos_dict["abundance_mtx"] = np.vstack(hip_pos_dict["abundance_mtx"])
    hip_neg_dict["diversity_mtx"] = np.vstack(hip_neg_dict["diversity_mtx"])
    hip_neg_dict["abundance_mtx"] = np.vstack(hip_neg_dict["abundance_mtx"])
    keck_pos_dict["diversity_mtx"] = np.vstack(keck_pos_dict["diversity_mtx"])
    keck_pos_dict["abundance_mtx"] = np.vstack(keck_pos_dict["abundance_mtx"])
    keck_neg_dict["diversity_mtx"] = np.vstack(keck_neg_dict["diversity_mtx"])
    keck_neg_dict["abundance_mtx"] = np.vstack(keck_neg_dict["abundance_mtx"])

    save_pk(os.path.join(args.add_dir, "Emerson2017_HIP_pos.pk"), hip_pos_dict)
    save_pk(os.path.join(args.add_dir, "Emerson2017_HIP_neg.pk"), hip_neg_dict)
    save_pk(os.path.join(args.add_dir, "Emerson2017_Keck_pos.pk"), keck_pos_dict)
    save_pk(os.path.join(args.add_dir, "Emerson2017_Keck_neg.pk"), keck_neg_dict)

def split_MDAtissue_dataset(dataset_dict):
    # Initialize empty dictionaries
    tu_dict = {"sample_list": [], "diversity_mtx": [], "abundance_mtx": []}
    lu_dict = {"sample_list": [], "diversity_mtx": [], "abundance_mtx": []}
    # Iterate over each sample
    for i, sample in enumerate(dataset_dict["sample_list"]):
        if "-T" in sample:  ## tumor data
            tu_dict["sample_list"].append(sample)
            tu_dict["diversity_mtx"].append(dataset_dict["diversity_mtx"][i])
            tu_dict["abundance_mtx"].append(dataset_dict["abundance_mtx"][i])
        else:
            lu_dict["sample_list"].append(sample)
            lu_dict["diversity_mtx"].append(dataset_dict["diversity_mtx"][i])
            lu_dict["abundance_mtx"].append(dataset_dict["abundance_mtx"][i])

    tu_dict["diversity_mtx"] = np.vstack(tu_dict["diversity_mtx"])
    tu_dict["abundance_mtx"] = np.vstack(tu_dict["abundance_mtx"])
    lu_dict["diversity_mtx"] = np.vstack(lu_dict["diversity_mtx"])
    lu_dict["abundance_mtx"] = np.vstack(lu_dict["abundance_mtx"])

    save_pk(os.path.join(args.add_dir, "MDanderson2019_Tissue_Tumor.pk"), tu_dict)
    save_pk(os.path.join(args.add_dir, "MDanderson2019_Tissue_Lung.pk"), lu_dict)

def split_houghton_dataset(dataset_dict):
    tu_dict = {"sample_list": [], "diversity_mtx": [], "abundance_mtx": []}
    lu_dict = {"sample_list": [], "diversity_mtx": [], "abundance_mtx": []}
    # Iterate over each sample
    for i, sample in enumerate(dataset_dict["sample_list"]):
        if "_Tu" in sample:  ## tumor data
            tu_dict["sample_list"].append(sample)
            tu_dict["diversity_mtx"].append(dataset_dict["diversity_mtx"][i])
            tu_dict["abundance_mtx"].append(dataset_dict["abundance_mtx"][i])
        else:
            lu_dict["sample_list"].append(sample)
            lu_dict["diversity_mtx"].append(dataset_dict["diversity_mtx"][i])
            lu_dict["abundance_mtx"].append(dataset_dict["abundance_mtx"][i])
    tu_dict["diversity_mtx"] = np.vstack(tu_dict["diversity_mtx"])
    tu_dict["abundance_mtx"] = np.vstack(tu_dict["abundance_mtx"])
    lu_dict["diversity_mtx"] = np.vstack(lu_dict["diversity_mtx"])
    lu_dict["abundance_mtx"] = np.vstack(lu_dict["abundance_mtx"])

    save_pk(os.path.join(args.add_dir, "houghton_2017_ncomms_Tumor.pk"), tu_dict)
    save_pk(os.path.join(args.add_dir, "houghton_2017_ncomms_Lung.pk"), lu_dict)

## split the dataset with label
sx_dict = load_pkfile(os.path.join(args.mtx_dir, "Sx_gastric.pk"))
split_sx_dataset(sx_dict)

tcrbv4_control_dict = load_pkfile(os.path.join(args.mtx_dir, "tcrbv4_control.pk"))
split_tcrbv4_control_dataset(tcrbv4_control_dict)

hip_dict = load_pkfile(os.path.join(args.mtx_dir, "Emerson2017_HIP.pk"))
keck_dict = load_pkfile(os.path.join(args.mtx_dir, "Emerson2017_Keck.pk"))
split_cmv_dataset(hip_dict, keck_dict)

MDAtissue_dict = load_pkfile(os.path.join(args.mtx_dir, "MDanderson2019_Tissue.pk"))
split_MDAtissue_dataset(MDAtissue_dict)

houghton_dict = load_pkfile(os.path.join(args.mtx_dir, "houghton_2017_ncomms.pk"))
split_houghton_dataset(houghton_dict)


pos_pair_nums = 3
filelist =  os.listdir(args.mtx_dir)
all_pairs = list(itertools.combinations(filelist, 2))
# Randomly select 20 pairs
selected_pairs = random.sample(all_pairs, pos_pair_nums)

# Add the negative pairs (two subsets from the same batch) to the list

neg_pairs = [('Sx_gastric_Tumor.pk', 'Sx_gastric_Normal.pk'),
            ('tcrbv4_control_Tumor.pk', 'tcrbv4_control_Normal.pk'),
            ('Emerson2017_HIP_pos.pk', 'Emerson2017_HIP_neg.pk'),
            ('Emerson2017_Keck_pos.pk', 'Emerson2017_Keck_neg.pk'),
            ('MDanderson2019_Tissue_Tumor.pk', 'MDanderson2019_Tissue_Lung.pk'),
            ('houghton_2017_ncomms_Tumor.pk', 'houghton_2017_ncomms_Lung.pk')]
neg_pairs = neg_pairs

for neg_pair in neg_pairs:
    selected_pairs.append(neg_pair)

pair_labels = [1] * pos_pair_nums + [0] * len(neg_pairs)
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
        "Label": int(pair_labels[pair_idx])
    }, ignore_index=True)

batch_paras.sort_values(by="Score",inplace=True,ascending=False)
batch_paras.to_csv(os.path.join(args.out_dir, "data_pair_summary.csv"), index=False)


## load the data pair summary
batch_paras = pd.read_csv(os.path.join(args.out_dir, "data_pair_summary.csv"))
## plot auc curve for the data pairs score and label

fig, axs = plt.subplots(1, 2, figsize=(9, 4))

# Plot the data pair summary score and the threshold
ax1 = axs[0]
colors = {'1': 'orchid', '0': 'skyblue'}
batch_paras['Score'].plot(kind='bar', width=0.5, color=[colors[str(int(i))] for i in batch_paras['Label']], ax=ax1)

thres = 1.3
ax1.axhline(y=thres, color='grey', linestyle='--', lw=1)
ax1.title.set_text("Distance score of dataset pairs")
ax1.text(25, thres + 1, "threshold = {:.1f}".format(thres), size=8, color="k")
ax1.set_ylabel("Distance Score")
ax1.get_xaxis().set_visible(False)

# plot the auc curve
ax2 = axs[1]
fpr, tpr, _ = roc_curve(batch_paras["Label"], batch_paras["Score"])
auc_score = auc(fpr, tpr)

ax2.plot(1 - fpr, tpr, lw=1, alpha=0.3, label=f'(AUC = {auc_score:.2f})')
ax2.plot([0, 1], [1, 0],'r--', lw=0.5)
ax2.set_xlim([1.05, -0.05])
ax2.set_ylim([-0.05, 1.05])
ax2.set_title("AUC curve for evaluating dataset distance score")
ax2.legend(loc="lower right", frameon=False)
ax2.set_xlabel("Specificity")
ax2.set_ylabel("Sensitivity")
ratio = 1
xleft, xright = ax2.get_xlim()
ybottom, ytop = ax2.get_ylim()
ax2.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)
plt.tight_layout()
plt.savefig(os.path.join(args.out_dir, "data_pair_summary.png"), dpi=600)
# plt.show()

## Iterate over all .pk files in the specified directory
filelist =  os.listdir(args.mtx_dir)
combined_diversity_mtx, combined_abundance_mtx, dataset_name_list = merge_dataset_mtx(filelist)
visualize_all_datasets(combined_diversity_mtx, dataset_name_list, min_dist=0.5, n_neighbors=50, type = "all datasets", out_dir = args.out_dir)
visualize_all_datasets(combined_abundance_mtx, dataset_name_list, min_dist=0.5, n_neighbors=50, type = "all datasets (abundance)", out_dir = args.out_dir)

# ## melanoma
melanoma_files = ["robert2014_CCR.pk", "valpione2020_nm.pk", "weber2018_cir.pk", "huuhtanen2022_nc.pk"]
combined_diversity_mtx, combined_abundance_mtx, dataset_name_list = merge_dataset_mtx(melanoma_files)
visualize_all_datasets(combined_diversity_mtx, dataset_name_list, min_dist=0.1, n_neighbors=15, type = "melanoma datasets", out_dir = args.out_dir)
visualize_all_datasets(combined_abundance_mtx, dataset_name_list, min_dist=0.1, n_neighbors=15, type = "melanoma datasets (abundance)", out_dir = args.out_dir)

# ## covid
covid_files = ["Covid_ADAPT.pk","Covid_ADAPT_MIRA.pk","Covid_IRST.pk","Covid_HU.pk","Covid_NIAID.pk"]
combined_diversity_mtx, combined_abundance_mtx, dataset_name_list = merge_dataset_mtx(covid_files)
visualize_all_datasets(combined_diversity_mtx, dataset_name_list, min_dist=0.05, n_neighbors=50, type = "ImmuneCODE datasets", out_dir = args.out_dir)
visualize_all_datasets(combined_abundance_mtx, dataset_name_list, min_dist=0.05, n_neighbors=50, type = "ImmuneCODE datasets (abundance)", out_dir = args.out_dir)

# ## cmv
cmv_files = ["Emerson2017_HIP.pk","Emerson2017_Keck.pk"]
combined_diversity_mtx, combined_abundance_mtx, dataset_name_list = merge_dataset_mtx(cmv_files)
visualize_all_datasets(combined_diversity_mtx, dataset_name_list, min_dist=0.05, n_neighbors=50, type = "cmv datasets", out_dir = args.out_dir)
visualize_all_datasets(combined_abundance_mtx, dataset_name_list, min_dist=0.05, n_neighbors=50, type = "cmv datasets (abundance)", out_dir = args.out_dir)

# ## lung cancer
lung_files = ["PMID30560866_lung.pk", "TRACERx_lung_Tissue.pk", "TRACERx_lung_PBMC.pk", "Formenti2018.pk", "MDanderson2019_PBMC.pk", "MDanderson2019_Tissue.pk", "houghton_2017_ncomms.pk"]
combined_diversity_mtx, combined_abundance_mtx, dataset_name_list = merge_dataset_mtx(lung_files)
visualize_all_datasets(combined_diversity_mtx, dataset_name_list, min_dist=0.05, n_neighbors=50, type = "lung cancer datasets", out_dir = args.out_dir)
visualize_all_datasets(combined_abundance_mtx, dataset_name_list, min_dist=0.05, n_neighbors=10, type = "lung cancer datasets (abundance)", out_dir = args.out_dir)

# ## multi cancer PBMC
cancer_files = ["Sx_gastric_Tumor.pk", "ESCC_multi_region_PBMC.pk", "PMID28422742_liver_PBMC.pk", "PMID33317041_AML.pk",
              "TRACERx_lung_PBMC.pk", "Formenti2018.pk", "MDanderson2019_PBMC.pk",
                "robert2014_CCR.pk", "valpione2020_nm.pk", "weber2018_cir.pk", "huuhtanen2022_nc.pk"]
combined_diversity_mtx, combined_abundance_mtx, dataset_name_list = merge_dataset_mtx(cancer_files)
visualize_all_datasets(combined_diversity_mtx, dataset_name_list, min_dist=0.5, n_neighbors=50, type = "multi-cancer(PBMC) datasets", out_dir = args.out_dir)
visualize_all_datasets(combined_abundance_mtx, dataset_name_list, min_dist=0.5, n_neighbors=50, type = "multi-cancer(PBMC) datasets (abundance)", out_dir = args.out_dir)

# ## healthy control
healthy_files = ["Sx_gastric_Normal.pk","ZhangControl.pk","Emerson2017_HIP.pk","Emerson2017_Keck.pk","tcrbv4_control_Normal.pk"]
combined_diversity_mtx, combined_abundance_mtx, dataset_name_list = merge_dataset_mtx(healthy_files)
visualize_all_datasets(combined_diversity_mtx, dataset_name_list, min_dist=0.5, n_neighbors=50, type = "healthy control datasets", out_dir = args.out_dir)
visualize_all_datasets(combined_abundance_mtx, dataset_name_list, min_dist=0.5, n_neighbors=50, type = "healthy control datasets (abundance)", out_dir = args.out_dir)

# ## Sx gastric two labels
sx_files = ["Sx_gastric_Normal.pk", "Sx_gastric_Tumor.pk"]
combined_diversity_mtx, combined_abundance_mtx, dataset_name_list = merge_dataset_mtx(sx_files)
visualize_all_datasets(combined_diversity_mtx, dataset_name_list, min_dist=0.1, n_neighbors=15, type = "Sx gastric two label datasets", out_dir = args.out_dir)
visualize_all_datasets(combined_abundance_mtx, dataset_name_list, min_dist=0.1, n_neighbors=15, type = "Sx gastric two label (abundance)", out_dir = args.out_dir)

# ## PmID 28 331
pm_files = ["PMID33317041_AML.pk", "TRACERx_lung_PBMC.pk"]
combined_diversity_mtx, combined_abundance_mtx, dataset_name_list = merge_dataset_mtx(pm_files)
visualize_all_datasets(combined_diversity_mtx, dataset_name_list, min_dist=0.1, n_neighbors=15, type = "PM two label datasets", out_dir = args.out_dir)
visualize_all_datasets(combined_abundance_mtx, dataset_name_list, min_dist=0.1, n_neighbors=15, type = "PM two label (abundance)", out_dir = args.out_dir)


## plot the JSD bewteen the datasets
draw_JSD(cancer_files,keyword ="cancer_files")
draw_JSD(healthy_files,keyword ="healthy_files")
draw_JSD(melanoma_files,keyword ="melanoma")
draw_JSD(covid_files,keyword ="covid")
draw_JSD(cmv_files,keyword ="cmv")
draw_JSD(lung_files,keyword ="lung")
