import datetime
import pandas
from time import time
from src.kmeans import KMeans

if __name__ == "__main__":
    start_time = time()
    data = pandas.read_csv('./data.csv')
    data.drop(['id', 'candidatoId', 'fecha'], axis=1, inplace=True)
    kmeans = KMeans(data[:100])
    clusters, centroids, data_cluster_idx = kmeans.train()
    elapsed_time = time() - start_time

    print(f'TOTAL TIME: {datetime.timedelta(seconds=elapsed_time)}')
    print(clusters)
    print(centroids)
    print(data_cluster_idx)
