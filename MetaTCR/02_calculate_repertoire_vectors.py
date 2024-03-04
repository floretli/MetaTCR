from metatcr.utils.utils import save_pk, load_pkfile
from metatcr.encoders.build_graph import update_graph_config, files2clstfreq, read_filelist_from_dir
import configargparse
import os
import glob
import numpy as np
from collections import defaultdict

parser = configargparse.ArgumentParser()
parser.add_argument('--dataset_dirs', type=str, default='./data/example_repertoire.datalist', help='dataset directories for processing. Each line stores a directory path')
parser.add_argument('--centroids_file', type=str, default='./results/data_analysis/96_best_centers.pk', help='Centroids file from clustering')
parser.add_argument('--out_dir', type=str, default='./results/data_analysis/datasets_mtx', help='Output directory for processed data')
parser.add_argument('--file_cut_size', type=int, default=None, help='How many files should be selected from the filelist')
parser.add_argument('--freq_col', type=str, default='frequencyCount (%)', help='Column name for frequency count')
parser.add_argument('--tcr_col', type=str, default='full_seq',
                    help='Column name for full length TCR sequence or amino acid')
parser.add_argument('--batch_size', type=int, default=200, help='Batch size for processing files')

args = parser.parse_args()
update_graph_config(args)
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

print("Running with the following parameters:")
for key, value in vars(args).items():
    print(f"{key}: {value}")
print("#######################################")


def read_dataset_dirs(file_path):
    with open(file_path, 'r') as file:
        return file.read().splitlines()

def process_directory(directory, centroids, batch_size=200):

    base_name = os.path.basename(directory)
    print("Processing dataset: ", base_name)

    filelist, all_sample_list = read_filelist_from_dir(directory)

    for i in range(0, len(filelist), batch_size):
        batch_files = filelist[i:i+batch_size]
        sample_list = all_sample_list[i:i+batch_size]

        diversity_mtx, abundance_mtx = files2clstfreq(batch_files, centroids, cutoff = None)

        result_dict = {
            "diversity_mtx": diversity_mtx,
            "abundance_mtx": abundance_mtx,
            "sample_list": sample_list,
        }
        print("Batch: ", i, " - ", i+batch_size)
        print("Diversity matrix shape: ", diversity_mtx.shape)
        print("Abundance matrix shape: ", abundance_mtx.shape)
        print("Sample list length: ", len(sample_list))

        suffix = '' if len(filelist) <= batch_size else f'_part{i//batch_size+1}'
        output_file = os.path.join(args.out_dir, base_name + suffix + '.pk')
        print("Saving results to: ", output_file)
        save_pk(output_file, result_dict)



def merge_dicts(directory):
    files = glob.glob(os.path.join(directory, '*.pk'))
    if len(files) <= 1:
        return

    prefix_dict = defaultdict(list)
    for file in files:
        prefix = os.path.basename(file).split('_part')[0]
        prefix_dict[prefix].append(file)

    for prefix, files in prefix_dict.items():
        if len(files) <= 1:
            continue

        print(f"Merging files for prefix: {prefix}")
        merged_dict = {
            "diversity_mtx": [],
            "abundance_mtx": [],
            "sample_list": [],
        }

        for file in sorted(files):
            result_dict = load_pkfile(file)
            merged_dict["diversity_mtx"].append(result_dict["diversity_mtx"])
            merged_dict["abundance_mtx"].append(result_dict["abundance_mtx"])
            merged_dict["sample_list"].extend(result_dict["sample_list"])

        merged_dict["diversity_mtx"] = np.concatenate(merged_dict["diversity_mtx"], axis=0)
        merged_dict["abundance_mtx"] = np.concatenate(merged_dict["abundance_mtx"], axis=0)

        output_file = os.path.join(directory, prefix + '.pk')
        print("Merged diversity matrix shape: ", merged_dict["diversity_mtx"].shape)
        print("Saving merged results to: ", output_file)
        save_pk(output_file, merged_dict)

        # Delete all part files after merging
        for file in files:
            print(f"Removing file: {file}")
            os.remove(file)

centroids = load_pkfile(args.centroids_file)
dirs = read_dataset_dirs(args.dataset_dirs)
for directory in dirs:
    process_directory(directory, centroids, args.batch_size)
merge_dicts(args.out_dir)

