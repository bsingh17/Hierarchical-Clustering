import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('musteriler.csv')

from sklearn.preprocessing import LabelEncoder,normalize
lbl_cinsiyet=LabelEncoder()
dataset['Cinsiyet']=lbl_cinsiyet.fit_transform(dataset['Cinsiyet'])


import scipy.cluster.hierarchy as hrc
plt.figure(figsize=(10,7))
plt.title('Dendrograms')
dend=hrc.dendrogram(hrc.linkage(dataset,method='ward'))
plt.axhline(y=2,linestyle='--',c='magenta')

from sklearn.cluster import AgglomerativeClustering
cluster=AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='ward')
cluster.fit_predict(dataset)

plt.figure(figsize=(10,7))
plt.scatter(dataset['Hacim'],dataset['Maas'],c=cluster.labels_)