import datetime
import pandas
from time import time
from src.kmeans import KMeans
from sklearn import metrics
from sklearn.metrics import pairwise_distances

if __name__ == "__main__":
    start_time = time()
    data = pandas.read_csv('./data.csv')
    predicted = data['candidatoId']
    data.drop(['id', 'candidatoId', 'fecha'], axis=1, inplace=True)
    cluster_size = [2,3,5,10]
    for size in cluster_size:
        kmeans = KMeans(data)
        kmeans.train()
        result = metrics.silhouette_score(data, kmeans.data_cluster_idx, metric='euclidean')
        elapsed_time = time() - start_time
        print(f'----------------------------------------')
        print(f'###### CLUSTER SIZE: {size} ######')
        print(f'###### SILHOUETTE SCORE: {result} ######')
        print(f'----------------------------------------')

    print(f'TOTAL TIME: {datetime.timedelta(seconds=elapsed_time)}')
