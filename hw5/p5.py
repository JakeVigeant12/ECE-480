import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as skl
from scipy.stats import multivariate_normal

import sklearn.mixture as sklm
import seaborn as sns

from matplotlib.colors import Colormap, ListedColormap


def calcMeans(data, labels):
    cluster1 = []
    cluster2 = []
    cluster3 = []
    cluster4 = []
    for l in range(len(labels)):
        if labels[l] == 0:
            cluster1.append(data[l].tolist())
        if labels[l] == 1:
            cluster2.append(data[l].tolist())
        if labels[l] == 2:
            cluster3.append(data[l].tolist())
        if labels[l] == 3:
            cluster4.append(data[l].tolist())
    mean1 = np.array(cluster1).mean(axis=0)
    mean2 = np.array(cluster2).mean(axis=0)
    mean3 = np.array(cluster3).mean(axis=0)
    mean4 = np.array(cluster4).mean(axis=0)
    cov1 = np.cov(np.array(cluster1).T)
    cov2 = np.cov(np.array(cluster2).T)
    cov3 = np.cov(np.array(cluster3).T)
    cov4 = np.cov(np.array(cluster4).T)
    w1 = (len(cluster1)/len(data))
    w2 = (len(cluster2)/len(data))
    w3 = (len(cluster3)/len(data))
    w4 = (len(cluster4)/len(data))
    return mean1, mean2, mean3, mean4, cov1, cov2, cov3, cov4, w1, w2, w3, w4

def largeClust(data,mean1,mean2,mean3,mean4, labels):
    cluster1 = []
    cluster2 = []
    cluster3 = []
    cluster4 = []
    final = []
    for l in range(len(labels)):
        if labels[l] == 0:
            cluster1.append(data[l].tolist())
        if labels[l] == 1:
            cluster2.append(data[l].tolist())
        if labels[l] == 2:
            cluster3.append(data[l].tolist())
        if labels[l] == 3:
            cluster4.append(data[l].tolist())
    for point in cluster1:
        final.append(point-mean1)
    for point in cluster2:
        final.append(point-mean2)
    for point in cluster3:
        final.append(point-mean3)
    for point in cluster4:
        final.append(point-mean4)
    return final
def genCluster(numObservations, mean, cov):
    return np.random.multivariate_normal(mean, cov, numObservations)


data = np.concatenate((genCluster(100, [-3, 3], [[2, -0.6], [-0.6, 2]]),
                       genCluster(100, [-2, -2], [[4, 2], [2, 4]]),
                       genCluster(100, [2,2], [[5, 2], [2, 5]]),
                       genCluster(100, [3, -3], [[2, -1.8], [-1.8, 2]])))

datax, datay = zip(*data)
starting_centroids = np.array([[-4, 0], [5, 8], [3, 4], [7, 0]])
kMeans = skl.KMeans(n_clusters=4, init=starting_centroids)
model1 = kMeans.fit(data)

mean1,mean2,mean3,mean4, cov1, cov2, cov3, cov4, w1, w2, w3, w4 = calcMeans(data, model1.labels_)
combinedClust = largeClust(data,mean1,mean2,mean3,mean4,model1.labels_)

tieddiagcov = np.cov(np.array(combinedClust).T)
tieddiagcov[0][1] = 0
tieddiagcov[1][0] = 0

diagcov1 = cov1
diagcov1[0][1] = 0
diagcov1[1][0] = 0

diagcov2 = cov2
diagcov2[0][1] = 0
diagcov2[1][0] = 0

diagcov3 = cov3
diagcov3[0][1] = 0
diagcov3[1][0] = 0

diagcov4 = cov4
diagcov4[0][1] = 0
diagcov4[1][0] = 0

tiedfull = np.cov(np.array(combinedClust).T)

bigclustx, bigclusty = zip(*combinedClust)
res = np.concatenate((bigclustx, bigclusty))
varval = np.var(res)

sphertied = [[varval,0],[0,varval]]

model3 = sklm.GaussianMixture(n_components = 4, covariance_type = "full")
model3.fit(data)
labels = model3.predict(data)

gmmmean1,gmmmean2,gmmmean3,gmmmean4 = model3.means_
gmmcov1, gmmcov2, gmmcov3, gmmcov4 = model3.covariances_
gmmw1, gmmw2, gmmw3, gmmw4 = model3.weights_

matplotlib.use('TkAgg')

x, y = np.mgrid[-8.2:10.2:.01, -8.2:10.2:.01]
pos = np.dstack((x, y))
rv = multivariate_normal(mean1, sphertied)
rv2 = multivariate_normal(mean2, sphertied)
rv3 = multivariate_normal(mean3, sphertied)
rv4 = multivariate_normal(mean4, sphertied)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.contourf(x, y, w1*rv.pdf(pos))
ax2.contour(x,y,w2*rv2.pdf(pos))
ax2.contour(x,y,w3*rv3.pdf(pos))
ax2.contour(x,y,w4*rv4.pdf(pos))
plt.show()





