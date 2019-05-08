import random
import numpy as np
from src.math import mean
from sklearn.metrics.pairwise import euclidean_distances

class KMeans:
    @staticmethod
    def centroid():
        vector = [random.randrange(1, 6, 1) for _ in range(26)]
        return np.array(vector)

    def __init__(self, data, cluster_count=2, dimension=26):
        self.data = data
        self.data_cluster_idx = [idx for idx in range(len(data))]
        self.centroids = [self.centroid() for _ in range(cluster_count)]
        self.cluster_count = cluster_count
        self.dimension = dimension
        self.clusters = [set() for _ in range(cluster_count)]
        # dist = np.linalg.norm(a-b)

    def _train(self):
        for idx, value in enumerate(self.data):
            distances = [np.linalg.norm(value - self.centroids[i]) for i in range(self.cluster_count)]
            current_cluster_idx = self.data_cluster_idx[idx]
            new_cluster_idx = np.argmin(distances)
            if current_cluster_idx != new_cluster_idx:
                self.clusters[current_cluster_idx] -= set(value)
                self.data_cluster_idx[idx] = new_cluster_idx
                self.clusters[new_cluster_idx].add(value)

