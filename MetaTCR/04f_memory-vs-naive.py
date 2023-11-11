import os
from metatcr.utils.utils import load_pkfile
import configargparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

def shannon_index(arr):
    """Compute the Shannon diversity index for a row of data."""
    arr = np.where(arr == 0, 1e-10, arr)
    shannon_index = -np.sum(arr * np.log2(arr))
    return shannon_index

def plot_mem_naive(df_merged):
    cols = list(range(96))
    groups = df_merged.groupby('patient_id')
    # Create a figure for each group
    for name, group in groups:
        # Create a subplot for each time point
        subgroups = group.groupby('time_point')
        fig, axs = plt.subplots(len(subgroups), 1, figsize=(10, 2 * len(subgroups)))
        for i, (time_point, subgroup) in enumerate(subgroups):
            # Select columns for heatmap
            df_subset = subgroup.sort_values('sample_id')
            y_labels = ['Naive' if 'Naive' in id else 'Memory' for id in df_subset['sample_id']]
            df_subset = df_subset[cols]

            sns.heatmap(df_subset, cmap='viridis', ax=axs[i], vmin=0, vmax=0.2)

            axs[i].set_title(f'Time point: {time_point}')
            axs[i].set_yticklabels(y_labels, rotation=0)

        # Set the figure title
        fig.suptitle(f'MetaTCR diversity for {name}')

        # Adjust layout
        plt.subplots_adjust(hspace=0.5, top=0.8, bottom=0.1)
        plt.savefig(os.path.join(args.out_dir, f"memory-vs-naive_{name}.png"), dpi=600)
        # plt.show()

def plot_time_series(df):
    cols = list(range(96))
    subjects = df['patient_id'].unique()

    # Create a figure for each subject
    for subject in subjects:
        # Filter the dataframe for the current subject
        df_subject = df[df['patient_id'] == subject]
        df_subject = df_subject.sort_values('time_point', ascending=True)
        y_labels = df_subject['time_point'].tolist()
        df_subject = df_subject[cols]


        # Create a new figure
        plt.figure(figsize=(10, 4))

        # Create a heatmap
        sns.heatmap(df_subject, cmap='viridis', vmin=0, vmax=0.2, yticklabels=y_labels)

        # Rotate the y-axis labels
        plt.yticks(rotation=0)
        #
        # plt.ylabel(y_labels, rotation=0)
        plt.title(f"Heatmap for {subject}")
        plt.savefig(os.path.join(args.out_dir, f"heatmap_{subject}.png"), dpi=600)

        # Show the plot
        # plt.show()
        # exit()

## Plot time series samples from healthy-time-course dataset
# tc_mtx = get_dataset_mtx("healthy-time-course.pk", "abundance_mtx")
tc_mtx = get_dataset_mtx("healthy-time-course.pk", "diversity_mtx")
tc_smp = get_dataset_mtx("healthy-time-course.pk", "sample_list")
tc_meta = "./data/healthy-time-course.csv"

# Compute the Shannon index for each row in tc_mtx
shannon_indices = np.apply_along_axis(shannon_index, axis=1, arr=tc_mtx)

# Convert your matrix and sample list into a DataFrame
df_mtx = pd.DataFrame(tc_mtx, index=tc_smp)

# Load your clinical metadata into a DataFrame
df_meta = pd.read_csv(tc_meta)

# Merge your data based on the 'sample_id' column
df_merged = df_meta.merge(df_mtx, left_on='sample_id', right_index=True)
# Add the Shannon indices to df_merged
df_merged['Shannon_index'] = shannon_indices

# Remove rows where 'sample_id' contains 'PBMC'
df_pbmc = df_merged[df_merged['sample_id'].str.contains('PBMC')]
df_merged = df_merged[~df_merged['sample_id'].str.contains('PBMC')]
df_merged.insert(1, 'sample_name', df_merged['sample_id'].str.replace('_Memory', '').str.replace('_Naive', ''))

# Pivot the DataFrame to get 'Memory' and 'Naive' as separate columns
df_merged = df_merged.sort_values(by=['sample_name'])
df_merged = df_merged.groupby('sample_name').filter(lambda x: len(x) >= 2)
df_merged.to_csv(os.path.join(args.out_dir, "healthy-time-course-dataset_Memory-vs-Naive.csv"), index=False)


# Select columns '0' to '95'
# cols = list(range(96))
# df_subset = df_merged[cols]

# Group by 'patient_id'


plot_mem_naive(df_merged)
plot_time_series(df_pbmc)
# print(df_pbmc)