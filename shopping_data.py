import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('shopping_data.csv')
dataset=dataset.drop(['CustomerID'],axis='columns')

from sklearn.preprocessing import LabelEncoder
lbl_gender=LabelEncoder()
dataset['Genre']=lbl_gender.fit_transform(dataset['Genre'])

from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(dataset)
x_pca=pca.transform(dataset)

import scipy.cluster.hierarchy as hrc
plt.figure(figsize=(10,7))
plt.title('Dendogram')
dend=hrc.dendrogram(hrc.linkage(x_pca,method='ward'))
plt.axhline(y=150,c='magenta',linestyle='--')


from sklearn.cluster import AgglomerativeClustering
model=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
model.fit_predict(x_pca)

plt.figure(figsize=(10,7))
plt.scatter(x_pca[:,0],x_pca[:,1],c=model.labels_,cmap='rainbow')