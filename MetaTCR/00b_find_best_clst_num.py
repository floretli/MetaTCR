import time
from metatcr.encoders.build_graph import  kmeans_traverse_k
import configargparse
import os
import numpy as np
from metatcr.utils.utils import load_pkfile
from metatcr.visualization.cluster_vis import ch_analysis, plot_ch_curve
import random
random.seed(1)


clst_num_list = [16, 32, 64, 96, 128, 256]
parser = configargparse.ArgumentParser()
parser.add_argument('--database_file', type=str, default='./data/all_dbfile_top5k.full.tcr', help='Tcr list as reference database')
parser.add_argument('--out_dir', type=str, default='./results/data_analysis', help='Output directory for processed data')

args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

# print("Running with the following parameters:")
# for key, value in vars(args).items():
#     print(f"{key}: {value}")
# print("#######################################")

# ## read database and transfer to list

# with open(args.database_file, 'r') as f:
#     lines = f.readlines()
# tcrs = [line.rstrip('\n') for line in lines]
# print("number of TCRs in database:", len(lines))

# print("encoding TCRs from database...")
# X = seqlist2ebd(tcrs)
# print("X shape:", X.shape)

# save_pk(os.path.join(args.out_dir, 'all_ref_embedding.pk'), X)

print("loading ref embeddings.pk")
X = load_pkfile(os.path.join(args.out_dir, 'all_ref_embedding.pk'))
print("X shape:", X.shape)

# print("clustering TCRs")
# all_cluster_labels, all_centers, all_errors, all_imbalances = kmeans_traverse_k(X, clst_num_list)

# te0 = time.time()
# all_dist = elbow_analysis(X, clst_num_list, all_cluster_labels, all_centers)
# te1 = time.time()
# print("[INFO] elbow analysis time:", te1-te0)

# plt_eb, best_k = plot_elbow_curve(clst_num_list, all_dist)
# plt_eb.savefig(os.path.join(args.out_dir, 'elbow_curve.png'), dpi=300)

# print("According to the elbow experiment, it is recommended that k >=", best_k)

ts0 = time.time()
n_sample = min(100000, len(X))
indices = np.random.choice(len(X), size=n_sample, replace=False)
X_subset = X[indices]
all_cluster_labels, _, _, all_imbalances = kmeans_traverse_k(X_subset, clst_num_list)
all_ch = ch_analysis(X_subset, clst_num_list, all_cluster_labels)
ts1 = time.time()
print("[INFO] calinski-harabasz analysis time:", ts1-ts0)

plt_ch = plot_ch_curve(clst_num_list, all_ch, all_imbalances)
plt_ch.savefig(os.path.join(args.out_dir, 'ch_curve.png'), dpi=300)

