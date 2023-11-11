from metatcr.encoders.build_graph import merge_seq_list, read_filelist_from_dir, update_graph_config
import configargparse
import random
random.seed(0)

parser = configargparse.ArgumentParser()
parser.add_argument('--db_dir_list', type=str, default='./data/dataset_dir.list', help='Database file directory')
parser.add_argument('--db_cut_size', type=int, default=5000, help='How many sequence in each data file to process pre-clustering')
parser.add_argument('--file_cut_size', type=int, default=50, help='How many files should be selected from one dataset')
parser.add_argument('--out_path', type=str, default='./data/reference_top5k.full.tcr', help='Output file path for merged database')
parser.add_argument('--freq_col', type=str, default='frequencyCount (%)', help='Column name for frequency count')
parser.add_argument('--tcr_col', type=str, default='full_seq',
                    help='Column name for full length TCR sequence or amino acid')

args = parser.parse_args()
update_graph_config(args)

print("Running with the following parameters:")
for key, value in vars(args).items():
    print(f"{key}: {value}")
print("#######################################")

with open(args.db_dir_list, 'r') as file:
    db_dirs = file.read().splitlines()

all_filelist = []
for db_dir in db_dirs:
    filelist, _ = read_filelist_from_dir(db_dir)
    all_filelist += filelist

print("merging TCR database from", len(all_filelist), "files...")

ref_seqs = merge_seq_list(all_filelist)
print("collected", len(ref_seqs), "unique seqs.")

## list to txt
with open(args.out_path, 'w') as file:
    for seq in ref_seqs:
        file.write(seq + '\n')

print("reference tcrs are saved to ", args.out_path)

