import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as skl
import sklearn.mixture as sklm
from matplotlib.colors import Colormap, ListedColormap


def genCluster(numObservations, mean, cov):
    return np.random.multivariate_normal(mean, cov, numObservations)


data = np.concatenate((genCluster(100, [-5, 5], [[1, 0], [0, 1]]),
                       genCluster(100, [-5, -5], [[2, 0], [0, 2]]),
                       genCluster(100, [5, 5], [[3, 0], [0, 3]]),
                       genCluster(100, [3, -5], [[0.5, 0], [0, 0.5]])))

datax, datay = zip(*data)
starting_centroids = np.array([[-4, 0], [5, 8], [3, 4], [7, 0]])
kMeans = skl.KMeans(n_clusters=4, init=starting_centroids)
model = kMeans.fit(data)

beter_centroids = np.array([[-7,7],[-7,-7],[7,7],[7,-7]])
model2 = skl.KMeans(n_clusters=4,init=beter_centroids).fit(data)

model3 = sklm.GaussianMixture(n_components = 4, covariance_type = "full")
model3.fit(data)
labels = model3.predict(data)




matplotlib.use('TkAgg')

plt.figure(1)
plt.scatter(data[:, 0], data[:, 1], c=model.labels_.astype(float))
plt.title("Poor starting choice")
plt.show()

plt.figure(2)
plt.scatter(data[:, 0], data[:, 1], c=model2.labels_.astype(float))
plt.title("Better starting choice")
plt.show()

plt.figure(3)
plt.scatter(datax, datay, color="b")
plt.title("Clustered Observations")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

plt.figure(4)
plt.title("Gaussian Mixture Model Clustering")
plt.scatter(datax, datay, c=labels, s=40, cmap='viridis', zorder=2)
plt.show()