import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import calinski_harabasz_score


def elbow_analysis(X, range_n_clusters, all_cluster_labels, all_centers):
    all_dist = []
    for n, n_clusters in enumerate(range_n_clusters):
        cluster_labels = all_cluster_labels[n]
        centers = all_centers[n]
        dist = 0
        for i in range(n_clusters):
            x_data = X[cluster_labels == i]
            wss = np.sum((x_data - centers[i]) ** 2)
            dist += wss
        all_dist.append(dist)
    return all_dist

def find_elbow_point(range_n_clusters, all_dist):
    coords = np.column_stack((range_n_clusters, all_dist))
    start = coords[0]
    end = coords[-1]
    vec_line = end - start
    vec_line_norm = vec_line / np.sqrt(np.sum(vec_line**2))

    vec_from_start = coords - start
    scalar_proj = np.dot(vec_from_start, vec_line_norm)
    vec_proj = np.outer(scalar_proj, vec_line_norm)
    vec_perp = vec_from_start - vec_proj
    dist = np.sqrt(np.sum(vec_perp ** 2, axis=1))

    elbow_index = np.argmax(dist)
    elbow_point = range_n_clusters[elbow_index]
    return elbow_point, all_dist[elbow_index]

def plot_elbow_curve(range_n_clusters, all_dist):
    elbow_point, elbow_value = find_elbow_point(range_n_clusters, all_dist)

    plt.figure()
    plt.plot(range_n_clusters, all_dist, linestyle='-', marker='o', color='deepskyblue')
    # plt.scatter(elbow_point, elbow_value, marker='o', color='darkorchid', s=80, label=f'Elbow: k={elbow_point}')

    for i in range(len(range_n_clusters)):
        plt.annotate(f"k = {range_n_clusters[i]}",
                     xy=(range_n_clusters[i], all_dist[i]), fontsize=10,
                     xytext=(range_n_clusters[i] + 5, all_dist[i] - 5))

    # Define a custom formatter for the y-axis
    def millions_formatter(x, pos):
        return f'{x / 1e6:.1f}M'

    def thousands_formatter(x, pos):
        return '%1.0fk' % (x*1e-3)

    y_formatter = FuncFormatter(thousands_formatter)
    plt.gca().yaxis.set_major_formatter(y_formatter)

    plt.xlim(0, range_n_clusters[-1] + 10)
    plt.ylim(all_dist[-1] * 0.9, all_dist[0] + all_dist[-1])
    plt.xlabel("Number of clusters", fontsize=13)
    plt.ylabel("WSS", fontsize=13)
    plt.title("Elbow method for k", fontsize=16)
    plt.tight_layout()

    return plt, elbow_point

def ch_analysis(X, range_n_clusters, all_cluster_labels):
    # Calculate calinski-harabasz scores for each cluster number
    all_ch = []
    for n, n_clusters in enumerate(range_n_clusters):
        labels= all_cluster_labels[n]
        ch = calinski_harabasz_score(X, labels)
        all_ch.append(ch)

    return all_ch

def plot_ch_curve(range_n_clusters, all_ch, all_imbalances):
    fig, ax1 = plt.subplots()

    # Plot imbalance factors on the left y-axis
    l1, = ax1.plot(range_n_clusters, all_imbalances, linestyle='-', marker='o', color='lightcoral',
                   label='Imbalance factors')
    ax1.set_xlabel("Number of clusters", fontsize=13)
    ax1.set_ylabel("Imbalance factors", fontsize=13, color='lightcoral')
    ax1.tick_params(axis='y', labelcolor='lightcoral')

    # Set the x-axis ticks and labels
    ax1.set_xticks(range_n_clusters)
    ax1.set_xlim(0, range_n_clusters[-1] + 10)

    # Create a second y-axis for the Calinski-Harabasz score
    ax2 = ax1.twinx()
    l2, = ax2.plot(range_n_clusters, all_ch, linestyle='-', marker='o', color='slateblue',
                   label='Calinski-Harabasz score')
    ax2.set_ylabel("Calinski-Harabasz score", fontsize=13, color='slateblue')
    ax2.tick_params(axis='y', labelcolor='slateblue')

    # Add a title and set the layout
    plt.title("Imbalance factors and Calinski-Harabasz scores for k", fontsize=16)
    fig.tight_layout()

    # Combine legends for both curves
    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    return plt

def train_birch(X, n_superlusters=10, birch_threshold=0.5, birch_branching_factor=12):
    from sklearn.cluster import Birch
    birch_model = Birch(threshold=birch_threshold, branching_factor=birch_branching_factor, n_clusters=n_superlusters)
    birch_model.fit(X)
    return birch_model

def train_linkage(X, method='ward', metric='euclidean'):
    from scipy.cluster.hierarchy import linkage
    l_model = linkage(X, method=method, metric=metric)
    return l_model

def linkage_merge_clusters(l_model, n_superclusters=10, criterion='maxclust'):
    from scipy.cluster.hierarchy import fcluster
    ref_cluster_labels = fcluster(l_model, n_superclusters, criterion=criterion)
    return ref_cluster_labels
def linkage_predict(X, refdata, l_model, n_superclusters=10):
    from scipy.spatial.distance import cdist

    ref_cluster_labels = linkage_merge_clusters(l_model, n_superclusters)

    distances = cdist(X, refdata)
    ## find the closest cluster for each point
    closest_indices = np.argmin(distances, axis=1)
    return ref_cluster_labels[closest_indices], ref_cluster_labels

def plot_hierarchy(model, n_superclusters=10, n_data=96):
    from scipy.cluster.hierarchy import dendrogram, set_link_color_palette
    from matplotlib.colors import ListedColormap, to_hex

    ref_cluster_labels =linkage_merge_clusters(model, n_superclusters)
    cmap = ListedColormap(plt.get_cmap('tab10').colors[:n_superclusters])

    def find_direct_link(data_idx, model, n_data):
        for link_idx in range(model.shape[0]):
            if data_idx in model[link_idx, :2]:
                return link_idx + n_data
        return None

    def get_target_links_for_cluster(cluster_label, ref_cluster_labels, model, n_data):
        target_cluster_data_indices = np.where(ref_cluster_labels == cluster_label)[0]
        target_links = []
        for idx in target_cluster_data_indices:
            link = find_direct_link(idx, model, n_data)
            if link is not None:
                target_links.append(link)
        return np.array(target_links)

    def link_color_func(link_index, target_links_list, cmap):
        for cluster_label, target_links in enumerate(target_links_list, start=1):
            if link_index in target_links:
                return to_hex(cmap(cluster_label - 1))
        return 'gray'

    target_links_list = []
    for cluster_label in range(1, n_superclusters + 1):
        target_links = get_target_links_for_cluster(cluster_label, ref_cluster_labels, model, n_data)
        target_links_list.append(target_links)

    plt.figure(figsize=(5, 15))
    dendrogram(model, orientation='left',
               link_color_func=lambda link_index: link_color_func(link_index, target_links_list, cmap),
               labels=np.array([str(i + 1) for i in range(n_data)]))

    # plt.axvline(x=color_threshold, color='gray', linestyle='--')
    plt.title('Hierarchical Clustering Dendrogram')
    plt.ylabel('Sample Index')
    plt.xlabel('Distance')
    return plt

def plot_supercluster(X, model, centroids, subset_n = 2000, n_superlusters=10, umap_n_neighbors=50, umap_min_dist=1):
    from sklearn.metrics import silhouette_samples
    from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
    import warnings
    warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
    warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
    import umap
    from matplotlib.colors import ListedColormap, BoundaryNorm

    np.random.seed(1)
    random_indices = np.random.choice(X.shape[0], size=subset_n, replace=False)
    X_sampled = X[random_indices]

    ## model use linkage
    cluster_labels,  ref_cluster_labels = linkage_predict(X_sampled, centroids, model, n_superlusters)

    sample_silhouette_values = silhouette_samples(X_sampled, cluster_labels)
    silhouette_avg = np.mean(sample_silhouette_values)

    X_combined = np.vstack((X_sampled, centroids))
    umap_model = umap.UMAP(n_neighbors=umap_n_neighbors, min_dist=umap_min_dist, n_components=2, random_state=0)
    X_combined_2d = umap_model.fit_transform(X_combined)


    X_sampled_2d = X_combined_2d[:X_sampled.shape[0], :]
    centroids_2d = X_combined_2d[X_sampled.shape[0]:, :]

    cmap = ListedColormap(plt.get_cmap('tab10').colors[:n_superlusters])
    bounds = np.linspace(0, n_superlusters, n_superlusters + 1)
    norm = BoundaryNorm(bounds, cmap.N)

    plt.figure()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(12, 5)
    gap_size = 50
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X_sampled) + (n_superlusters + 1) * gap_size])

    ## show in silhouette order
    mean_s_values = []
    s_values_clusters = []
    for i in range( 1, 1 + n_superlusters):
        s_values = sample_silhouette_values[cluster_labels == i]
        s_values_clusters.append(s_values)
        if s_values.size == 0:
            mean_s_values.append(-1)
        else:
            mean_s_values.append(np.mean(s_values))
    sorted_indices = np.argsort(mean_s_values)

    y_lower = 10
    for idx in sorted_indices:
        s_values = s_values_clusters[idx]
        s_values.sort()
        size_cluster_i = s_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cmap(norm(idx))
        ax1.fill_betweenx(y=np.arange(y_lower, y_upper), x1=0, x2=s_values, color=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(idx + 1))
        y_lower = y_upper + gap_size

    ax1.set_title("Silhouette analysis of super-clusters", fontsize=16)
    ax1.set_xlabel("Silhouette value", fontsize=12)
    ax1.set_ylabel("Super-clusters index", fontsize=12)
    ax1.axvline(x=silhouette_avg, color="grey", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xlim(0, 0.6)
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6])

    scatter = ax2.scatter(X_sampled_2d[:, 0], X_sampled_2d[:, 1], c=cluster_labels -1, cmap=cmap,norm=norm, s=0.5, alpha = 0.5)
    ax2.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c=ref_cluster_labels -1, marker='^', s=20, cmap=cmap,
               edgecolors='k', linewidths=0.5)
    num_clusters = len(np.unique(cluster_labels))


    tick_positions = np.arange(num_clusters) + 0.5 # 0.5, 1.5, 1.5, 2.5, ..., 9.5
    tick_labels = np.arange(1, num_clusters + 1)  ## from 1 to num_clusters

    cbar = plt.colorbar(scatter, ticks=tick_positions, ax=ax2)
    cbar.set_ticklabels(tick_labels)
    ax2.set_title("UMAP projection of the super-clusters", fontsize=16)
    return fig