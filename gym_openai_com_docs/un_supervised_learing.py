import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

#数据处理
X, Y = make_blobs(n_samples=500, centers=7, cluster_std=0.8, random_state=0)

#模型构建、预测
kmeans = KMeans(n_clusters=9).fit(X)
prebs = kmeans.predict(X)
centers = kmeans.cluster_centers_#得到聚类中心

#绘图
plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
#c:color;s:radius;alpha:颜色深浅
plt.show()



# 均值偏移算法
# 从分配给它们自己的集群的数据点开始。
# 现在，它计算质心并更新新质心的位置。
# 通过重复这个过程，向簇的顶点靠近，即朝向更高密度的区域移动。
# 该算法停止在质心不再移动的阶段。
import numpy as np
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

#数据处理
centers = [[2, 2], [4, 5], [3, 10]]
X, _ = make_blobs(n_samples=500, centers=centers, cluster_std=1)

#模型训练
ms = MeanShift().fit(X)
labels = ms.labels_ #给X点做标记
cluster_centers = ms.cluster_centers_#得到聚点中心坐标
n_clusters_ = len(np.unique(labels))#聚点种类数

#绘图
colors = 10*['r.', 'g.', 'b.', 'c.', 'k.', 'y.', 'm.']#颜色表
for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker="x", color='k', s=150, linewidths=5, zorder=10)
plt.show()