# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 11:50:19 2020

@author: Varun
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

wine= pd.read_csv("wine.csv")
wine
wine.describe()
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


wine_norm= scale(wine.iloc[:,1:])
wine_norm

pca=PCA()
pca_values = pca.fit_transform(wine_norm)
pca_values.shape
var = pca.explained_variance_ratio_
var
pca.components_[0]


var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1
plt.plot(var1,color="red")


x = np.array(pca_values[:,0])
y = np.array(pca_values[:,1])
z = np.array(pca_values[:,2])
plt.plot(x,y,"bo")

###clustering
from	sklearn.cluster	import	KMeans
from scipy.spatial.distance import cdist
wine=pd.read_csv("wine.csv")
wine_norm=pd.DataFrame(scale(wine.iloc[:,1:]))
k=list(range(2,15))
k    
TWSS=[]
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(wine_norm)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(wine_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,wine_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
    plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)
####from plot, k value is 8
kmodel=KMeans(n_clusters=8)
kmodel.fit(wine_norm)
kmodel.labels_
kd=pd.Series(kmodel.labels_)
wine['clust']=kd
wine
winess=wine.iloc[:,[14,0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
winess.iloc[:,1:15].groupby(wine.clust).mean()
winess.to_csv("winespca.csv")
