import networkx as nx
import matplotlib.pyplot as plt
import torch
import numpy as np

class GraphVisualizer:
    def __init__(self, node_num, feature, edge_index):
        self.G = nx.Graph()
        self.node_num = node_num

        for i in range(self.node_num):
            self.G.add_node(i)

        for i in range(edge_index.shape[1]):
            self.G.add_edge(edge_index[0][i].item(), edge_index[1][i].item())

        self.features = feature
        for i in range(self.node_num):
            self.G.nodes[i]['feature'] = self.features[i].tolist()

    def get_cosine_similarity_matrix(self, other):
        cosine_sim_matrix = np.zeros(self.node_num)
        for i in range(self.node_num):
            feature1 = self.features[i].unsqueeze(0)
            feature2 = other.features[i].unsqueeze(0)
            cosine_sim_matrix[i] = torch.nn.functional.cosine_similarity(feature1, feature2)[0]
        return cosine_sim_matrix

    def draw_distance_graph(self, other=None):
        if other is None:
            other = self
        cosine_sim_matrix = self.get_cosine_similarity_matrix(other)

        D = self.G.copy()
        node_colors = []
        node_sizes = []
        sim_threshold = 0.5
        for i in range(self.node_num):
            feature = cosine_sim_matrix[i]
            D.nodes[i]['feature'] = feature
            if feature > sim_threshold:
                node_colors.append('blue')
                node_sizes.append(100)
            else:
                node_colors.append('red')
                node_sizes.append(500)
        fig, ax = plt.subplots()
        pos = nx.kamada_kawai_layout(D)
        nx.draw(D, pos, node_color=node_colors, node_size=node_sizes, ax=ax)
        nx.draw(D, pos, node_color=node_colors, node_size=node_sizes, ax=ax)

        node_labels = {i: f"{[i]} {D.nodes[i]['feature']:.2f}" for i in D.nodes}
        node_labels = {k: v for k, v in node_labels.items() if v != ''}
        nx.draw_networkx_labels(D, pos, labels=node_labels, font_size=12,
                                font_color='black', ax=ax)
        ax.set_axis_off()
        return ax
