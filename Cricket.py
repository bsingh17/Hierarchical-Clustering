import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv('cricket.csv')
dataset=dataset.drop(['PLAYER'],axis='columns')
dataset=dataset.replace(to_replace='-',value='0')

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(dataset)
dataset=scaler.fit_transform(dataset)

from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(dataset)
x=pca.transform(dataset)
x=pd.DataFrame(x)
x.columns=['X1','X2']

import scipy.cluster.hierarchy as hrc
plt.figure(figsize=(10,7))
dend=hrc.dendrogram(hrc.linkage(dataset,method='ward'))
plt.axhline(y=26,c='pink',linestyle='--')

from sklearn.cluster import AgglomerativeClustering
cluster=AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='ward')
cluster.fit_predict(x)

plt.figure(figsize=(10,7))
plt.scatter(x['X1'],x['X2'],c=cluster.labels_)
