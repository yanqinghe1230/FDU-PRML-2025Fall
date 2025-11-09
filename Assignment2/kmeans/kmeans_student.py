import numpy as np
import pandas as pd
from viz_kmeans import plot_kmeans
from test_kmeans import evaluate_kmeans

class KMeans:
    def __init__(self, k=3, max_iter=300, epsilon=1e-4, random_state=None, distance_metric='euclidean', p=1):
        """
        KMeans聚类算法实现
        k: 聚类中心个数
        max_iter: 最大迭代次数
        epsilon: 收敛阈值
        random_state: 随机数种子
        distance_metric: 距离度量方式('euclidean', 'chebyshev', 'minkowski')
        p: minkowski距离的p值，默认p=1（曼哈顿距离）
        """
        self.k = k
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.random_state = random_state
        self.distance_metric = distance_metric
        self.p = p

    def _compute_distances(self, X, centroids):
        """
        根据给定的距离度量计算所有样本与各中心的距离矩阵
        输入X: (n_samples, n_features)
        centroids: (k, n_features)
        输出: (n_samples, k) 距离矩阵
        """
        # =============== TODO ===============
        if self.distance_metric == 'euclidean':
            pass
        elif self.distance_metric == 'chebyshev':
            pass
        elif self.distance_metric == 'minkowski':
            pass
        else:
            raise ValueError('Unsupported distance metric!')
        # 返回结果
        pass
        # =============== END TODO ===============

    def fit(self, X):
        """
        按指定距离度量进行KMeans聚类
        X为输入特征矩阵, shape=(n_samples, n_features)
        """
        # =============== TODO ===============
        # 随机选取K个样本作为初始聚类中心  self.centroids

        # =============== END TODO ===============

        for i in range(self.max_iter):
            distances = self._compute_distances(X, self.centroids) 
            # =============== TODO ===============
            # 判断每个样本所属的簇
            # 计算新的聚类中心
            # 根据epsilon判断是否收敛，更新中心
            pass
            # =============== END TODO ===============


# 获取数据
df = pd.read_csv("data/iris_dataset.csv")
X = df.iloc[:, :4].values
y = df['target'].values

# 1. 多距离度量 k=3 检验
dist_metrics = [ ('euclidean', None), ('chebyshev', None), ('minkowski', 1) ]

for dist, p in dist_metrics:
    method_name = dist if p is None else f"{dist}_p{p}"
    kmeans = KMeans(k=3, random_state=42, distance_metric=dist, p=p if p is not None else 1)
    kmeans.fit(X)
    labels = kmeans.labels
    centers = kmeans.centroids
    img_name = f"kmeans_k3_{method_name}.png"
    plot_kmeans(X, labels, centers, f"k=3, {method_name}", save_path=img_name)
    metrics = evaluate_kmeans(X, labels)
    print(f"==== k=3, {method_name} ====")
    print("Davies-Bouldin(越小越好): {:.4f}".format(metrics['davies_bouldin']))
    print("Silhouette(越大越好): {:.4f}".format(metrics['silhouette']))

# 2. 欧氏距离多k检验
for k in [2, 3, 4, 5]:
    kmeans = KMeans(k=k, random_state=42, distance_metric='euclidean')
    kmeans.fit(X)
    labels = kmeans.labels
    centers = kmeans.centroids
    img_name = f"kmeans_k{k}_euclidean.png"
    plot_kmeans(X, labels, centers, f"k={k}, euclidean", save_path=img_name)
    metrics = evaluate_kmeans(X, labels)
    print(f"==== k={k}, euclidean ====")
    print("Davies-Bouldin(越小越好): {:.4f}".format(metrics['davies_bouldin']))
    print("Silhouette(越大越好): {:.4f}".format(metrics['silhouette']))
