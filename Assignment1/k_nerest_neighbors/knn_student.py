import os
import numpy as np
from typing import List, Tuple
from data_generate import load_prepared_dataset
from viz_knn import plot_k_curve, plot_decision_boundary_multi

# 输出目录
OUT_DIR = "./output"
DATA_DIR = "./input_knn"
FIG_K_CURVE   = os.path.join(OUT_DIR, "knn_k_curve.png")
FIG_BOUNDARY  = os.path.join(OUT_DIR, "knn_boundary.png")

# ============ TODO 1：pairwise_dist ============
def pairwise_dist(X_test, X_train, metric, mode):
    """
    Compute pairwise distances between X_test (Nte,D) and X_train (Ntr,D).

    Required:
      - L2 distance 'l2' with modes:
          * 'two_loops'  
          * 'no_loops' 
      - 'cosine' distance (distance = 1 - cosine_similarity)
    """
    X_test = np.asarray(X_test, dtype=np.float64)
    X_train = np.asarray(X_train, dtype=np.float64)
    Nte, D  = X_test.shape
    Ntr, D2 = X_train.shape
    assert D == D2, "Dim mismatch between test and train."

    if metric == "l2":
        if mode == "two_loops":
            # =============== TODO (students, REQUIRED) ===============
            dist = np.zeros((Nte, Ntr))
            for i in range(Nte):
                for j in range(Ntr):
                    dist[i,j]=np.sqrt(np.sum((X_test[i]-X_train[j])**2))
            return dist                    
            # =========================================================

        elif mode == "no_loops":
            # =============== TODO (students, REQUIRED) ===============
            test_sq=np.sum(X_test**2,axis=1,keepdims=True)
            train_sq=np.sum(X_train**2,axis=1)
            cross=X_test.dot(X_train.T)
            dist=np.sqrt(test_sq - 2*cross + train_sq)
            return dist
            # =========================================================

        else:
            raise ValueError("Unknown mode for L2.")

    elif metric == "cosine":
        # =============== TODO (students, REQUIRED) ===============
        dot=X_test.dot(X_train.T)
        test_norm=np.linalg.norm(X_test,axis=1,keepdims=True)
        train_norm=np.linalg.norm(X_train,axis=1)
        cosine_sim=dot/(test_norm*train_norm)
        return 1-cosine_sim
        # ================================================
    else:
        raise ValueError("metric must be 'l2' or 'cosine'.")


# ============ TODO 2：knn_predict（多数表决） ============
def knn_predict(X_test, X_train, y_train, k, metric, mode):
    """
    kNN prediction.
    Required: majority vote with L2 distance.

    Returns
    -------
    y_pred : (Nte,) int
    """
    dists = pairwise_dist(X_test, X_train, metric=metric, mode=mode)
    y_train = np.asarray(y_train).reshape(-1).astype(int)
    Nte = dists.shape[0]
    y_pred = np.zeros(Nte, dtype=int)

    for i in range(Nte):
        idx = np.argsort(dists[i])[:k]
        neighbors = y_train[idx]

        # =============== TODO (students, REQUIRED) ===============
        n_dist=dists[i][idx]

        weights=1/(n_dist+1e-8)
        vote={}
        for label,w in zip(neighbors,weights):
            vote[label]=vote.get(label,0)+w

        max_vote = max(vote.values())
        candidates = [label for label, v in vote.items() if v == max_vote]
        y_pred[i] = min(candidates)
        # =========================================

    return y_pred


# ============ TODO 3：select_k_by_validation ============
def select_k_by_validation(X_train, y_train, X_val, y_val, ks: List[int], metric, mode) -> Tuple[int, List[float]]:
    """
    Grid-search K on validation set.

    Returns
    -------
    best_k : int
    accs   : list of validation accuracies aligned with ks
    """
    # =============== TODO (students, REQUIRED) ===============
    accs=[]
    for k in ks:
        y_pred=knn_predict(X_val,X_train,y_train,k,metric,mode)
        acc=(y_pred==y_val).mean()
        accs.append(acc)
    best_k=ks[np.argmax(accs)]
    return best_k,accs
    # =========================================================


def run_with_visualization():
    X_train, y_train, X_val, y_val, X_test, y_test = load_prepared_dataset(DATA_DIR)

    ks = [1, 3, 5, 7, 9, 11, 13]
    metric = "cosine"            # ["l2", "cosine"]
    mode   = "no_loops"     # ["two_loops", "no_loops", "one_loop"]

    best_k, accs = select_k_by_validation(X_train, y_train, X_val, y_val,
                                          ks, metric=metric, mode=mode)
    print(f"[ModelSelect] best k={best_k} (val acc={max(accs):.4f})")
    plot_k_curve(ks, accs, os.path.join(OUT_DIR, f"knn_k_curve_{metric}2.png"))

    X_trv = np.vstack([X_train, X_val]); y_trv = np.hstack([y_train, y_val])
    def predict_fn_for_k(k):
        return lambda Xq: knn_predict(Xq, X_trv, y_trv, k, metric=metric, mode=mode)

    ks_panel = sorted(set(ks + [best_k]))
    plot_decision_boundary_multi(predict_fn_for_k, X_train, y_train, X_test, y_test,
                                 ks=ks_panel,
                                 out_path=os.path.join(OUT_DIR, f"knn_boundary_grid_{metric}2.png"),
                                 grid_n=200, batch_size=4096)


if __name__ == "__main__":
    run_with_visualization()
