from metatcr.encoders.build_graph import kmeans_clustering
import configargparse
import os
from metatcr.utils.utils import save_pk, load_pkfile

import random
random.seed(1)


parser = configargparse.ArgumentParser()
parser.add_argument('--out_dir', type=str, default='./results/data_analysis', help='Output directory for processed data')
parser.add_argument('--selected_k', type=int, default=96, help='Number of clusters')
parser.add_argument('--n_superclusters', type=int, default=6, help='Number of superclusters')

args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

print("Running with the following parameters:")
for key, value in vars(args).items():
    print(f"{key}: {value}")
print("#######################################")

best_k = args.selected_k
# ### re clustering with the best k
centroids = load_pkfile(os.path.join(args.out_dir, str(best_k) + '_best_centers.pk'))
labels = load_pkfile(os.path.join(args.out_dir, str(best_k) + '_best_labels.pk'))


### hiraechy clustering
from metatcr.visualization.cluster_vis import train_linkage, plot_supercluster, plot_hierarchy
n_superclusters = args.n_superclusters
linkage_model = train_linkage(centroids)
plt_hierarchy = plot_hierarchy(linkage_model,n_superclusters=n_superclusters, n_data=best_k)
plt_hierarchy.savefig(os.path.join(args.out_dir, 'hierarchy_tree.svg'), dpi=300, format="svg")

plt_sub = plot_supercluster(X, linkage_model, centroids, subset_n = 5000, umap_n_neighbors=100, umap_min_dist=0.5, n_superlusters=n_superclusters)
plt_sub.savefig(os.path.join(args.out_dir, 'linkage_supercluster.svg'), dpi=300, format="svg")
