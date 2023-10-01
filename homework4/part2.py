import numpy as np, collections
import matplotlib.pyplot as plt
from numpy.core.numeric import indices
import pandas as pd
from scipy.io import arff
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Variables
k = [2,3]

data = arff.loadarff('breast.w.arff')
df = pd.DataFrame(data[0])
df = df.dropna()
data = df.drop(columns=["Class"]).values
results = df[df.keys()[-1]].astype('string').values

def ECR(results, pred_labels):
    clusters = np.unique(pred_labels)
    ecr=0
    for i in clusters:
        points = results[pred_labels == i]
        _, counts = np.unique(points, return_counts=True)
        ecr += 1/len(clusters)*(np.sum(counts) - np.max(counts))
    return ecr

# 4
for i in k:
    kmeans = KMeans(n_clusters=i).fit(data)
    pred_labels = kmeans.labels_

    print(f'## {i}-Means: ##')
    print(f'\tECR: {ECR(results, pred_labels)}')
    
    silh_score = silhouette_score(data, pred_labels, metric='euclidean')
    print(f'\tSilhouette score: {silh_score}')

# 5
kmeansX = KMeans(n_clusters=3)
kmeans4 = kmeans.fit(data)
selector = SelectKBest(score_func=mutual_info_classif, k=2)
data_best2 = selector.fit_transform(data, results)
kmeans5 = kmeans.fit(data_best2)
centroids = kmeans5.cluster_centers_
labels4 = kmeans4.labels_

for i in np.unique(labels4):
    plt.scatter(data_best2[labels4 == i,0], data_best2[labels4 == i,1], label=f'Cluster {i} points',
                    alpha=0.25, s=50+15*i)
plt.scatter(centroids[:,0], centroids[:,1], s=75, c='black', label='Centroids')
cols = selector.get_support(indices=True)
features = df.iloc[:,cols].columns
plt.xlabel(features[0]) ; plt.ylabel(features[1])
plt.legend()
plt.show()

#6
print(f'\nWith top-2 features:\n\tECR: {ECR(results, labels4)}')
silh_score5 = silhouette_score(data_best2, labels4, metric='euclidean')
print(f'\tSilhouette score: {silh_score5}')