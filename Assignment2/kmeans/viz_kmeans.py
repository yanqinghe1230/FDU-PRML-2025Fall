import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_kmeans(X, labels, centers, title, save_path="kmeans_pca.png"):
    """
    X: shape(n_samples, n_features)
    labels: shape(n_samples,)
    centers: shape(k, n_features)
    title: str, 用作图片标题
    save_path: 图片保存路径
    """
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    centers_2d = pca.transform(centers)

    plt.figure(figsize=(8,6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7, label='Sample')
    plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', s=200, marker='o', label='Centers')
    plt.title(f"K-Means Clustering on Iris Dataset (PCA 2D, {title})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.savefig(save_path, dpi=300)
    # plt.show()
