import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Wholesale Customers data.csv')

from sklearn.preprocessing import normalize
dataset=pd.DataFrame(normalize(dataset),columns=dataset.columns)

import scipy.cluster.hierarchy as hrc
plt.figure(figsize=(10,7))
plt.title('Dendograms')
dend=hrc.dendrogram(hrc.linkage(dataset,method='ward'))
plt.axhline(y=6,c='magenta',linestyle='--')

from sklearn.cluster import AgglomerativeClustering
cluster=AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='ward')
cluster.fit_predict(dataset)

plt.figure(figsize=(10,7))
plt.scatter(dataset['Milk'],dataset['Grocery'],c=cluster.labels_)
