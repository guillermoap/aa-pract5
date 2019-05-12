import random
import numpy as np
from copy import deepcopy
from src.math import mean
from sklearn.metrics.pairwise import euclidean_distances

class KMeans:
    @staticmethod
    def centroid(vector=None):
        if vector is None:
            vector = [random.randrange(1, 6, 1) for _ in range(26)] # los valores de las respuestas ∈ {1..5}
        return np.array(vector)

    def __init__(self, data, cluster_count=2, dimension=26, epsilon=0.001):
        self.data = data
        self.data_cluster_idx = [0 for idx in range(len(data))]
        self.centroids = [self.centroid() for _ in range(cluster_count)]
        self.cluster_count = cluster_count
        self.dimension = dimension
        self.clusters = [set() for _ in range(cluster_count)]
        self.cost_function = 0
        self.epsilon = epsilon

    def _train(self):
        self.fill_clusters()
        self.update_centroids()

    def update_centroids(self):
        for idx, cluster in enumerate(self.clusters):
            distances = [np.linalg.norm(np.array(value) - self.centroids[idx]) for value in cluster]
            self.cost_function += sum(distances)
            if cluster:
                mean = np.mean([np.array(item) for item in cluster], axis=0)
                self.centroids[idx] = self.centroid(mean)

    def fill_clusters(self):
        for idx, value in enumerate(self.data.reset_index().values):
            value = value[1:]
            distances = [np.linalg.norm(value - self.centroids[i]) for i in range(self.cluster_count)]
            current_cluster_idx = self.data_cluster_idx[idx]
            new_cluster_idx = np.argmin(distances)
            if current_cluster_idx != new_cluster_idx:
                value = tuple(value)
                self.clusters[current_cluster_idx] -= set(value)
                self.data_cluster_idx[idx] = new_cluster_idx
                self.clusters[new_cluster_idx].add(value)

    def train(self, K=10):
        runs = 0
        while True:
            last_clusters = deepcopy(self.clusters)
            self._train()
            if last_clusters == self.clusters:
                break
            runs += 1
