from sklearn.metrics import davies_bouldin_score, silhouette_score

def evaluate_kmeans(X, labels, y_true=None):
    """
    X: shape(n_samples, n_features)
    labels: shape(n_samples,)
    """
    metrics = {}
    metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
    metrics['silhouette'] = silhouette_score(X, labels)
    return metrics
