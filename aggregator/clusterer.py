from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

vectors = pd.read_csv('../data2/aggregated_vectors.csv', header=None)
X = vectors.values.tolist()

kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
