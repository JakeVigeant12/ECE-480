import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as skl
import sklearn.mixture as sklm
from matplotlib.patches import Ellipse
from matplotlib.colors import Colormap, ListedColormap


def genCluster(numObservations, mean, cov):
    return np.random.multivariate_normal(mean, cov, numObservations)


data = np.concatenate((genCluster(100, [-3, 3], [[2, -0.6], [-0.6, 2]]),
                       genCluster(100, [-2, -2], [[4, 2], [2, 4]]),
                       genCluster(100, [2,2], [[5, 2], [2, 5]]),
                       genCluster(100, [3, -3], [[2, -1.8], [-1.8, 2]])))

datax, datay = zip(*data)
starting_centroids = np.array([[-4, 0], [5, 8], [3, 4], [7, 0]])
kMeans = skl.KMeans(n_clusters=4, init=starting_centroids)
model = kMeans.fit(data)

model3 = sklm.GaussianMixture(n_components = 4, covariance_type = "full")
model3.fit(data)
labels = model3.predict(data)




matplotlib.use('TkAgg')

plt.figure(1)
plt.scatter(data[:, 0], data[:, 1], c=model.labels_.astype(float))
plt.title("KMeans")
plt.show()



plt.figure(3)
plt.scatter(datax, datay, color="b")
plt.title("Clustered Observations")
plt.xlabel("x")
plt.ylabel("y")
plt.show()




def draw_ellipse(position, covariance, ax=None, **kwargs):
    matplotlib.use('TkAgg')
    ax = ax or plt.gca()
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))


def plot_gmm(gmm, data, x,y, label=True, ax=None):
    matplotlib.use('TkAgg')
    ax = ax or plt.gca()
    labels = gmm.fit(data).predict(data)
    if label:
        ax.scatter(x,y, c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(x, y, s=40, zorder=2)
    ax.axis('equal')
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)
    plt.show()



plt.figure(4)
plot_gmm(model3,data,datax,datay)