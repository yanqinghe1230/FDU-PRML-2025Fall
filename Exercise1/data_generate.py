# -*- coding: utf-8 -*-
"""
generate_data.py
生成线性回归训练/测试数据，并按作业格式保存：
  ./input/train.txt   # 每行: x1 x2 ... xD y
  ./input/test_X.txt  # 每行: x1 x2 ... xD
  ./input/test_y.txt  # 每行: y

使用方式：
  直接运行：python generate_data.py
  修改脚本顶部“参数区”即可调整数据维度、规模、噪声、真值参数等。
"""

from __future__ import annotations
import os
import numpy as np


# ===================== 参数区（按需修改） =====================
D = 3                # 特征维度 (>=1)
N_TRAIN = 300        # 训练样本数
N_TEST = 200         # 测试样本数

# 线性真值： y = X @ W + B + 噪声
W = [[1.0, 0.5, -0.8]]             # 若为 None，则随机生成长度为 D 的权重；否则给出长度为 D 的列表/数组，如 [1.0, 0.5, -0.8]
B = 1              # 偏置

# 自变量取值范围（各维独立均匀采样）
X_MIN = -5.0
X_MAX = 5.0

# 噪声（高斯）
NOISE_STD_TRAIN = 2.0    # 训练集噪声标准差
NOISE_STD_TEST = 2.0     # 测试集噪声标准差（设为 0.0 可生成无噪声测试标签）

# 其它
SEED = 42
OUT_DIR = "./input"
FMT = "%.6f"             # 保存到 txt 时的小数格式
# ============================================================


def make_linear_data(n: int, d: int, w: np.ndarray, b: float,
                     x_min: float, x_max: float, noise_std: float,
                     rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
    """生成 (X, y)，其中 X∈R^{n×d}, y∈R^{n}，y = Xw + b + ε。"""
    X = rng.uniform(low=x_min, high=x_max, size=(n, d))
    if noise_std > 0.0:
        eps = rng.normal(loc=0.0, scale=noise_std, size=n)
    else:
        eps = 0.0
    y = X @ w + b + eps
    return X, y


def main() -> None:
    # 基本检查
    if D < 1:
        raise ValueError("D must be >= 1.")
    rng = np.random.RandomState(SEED)

    # 权重
    if W is None:
        w = rng.uniform(0.5, 1.5, size=D)  # 适中的随机权重
    else:
        w = np.asarray(W, dtype=np.float64).reshape(-1)
        if w.shape[0] != D:
            raise ValueError(f"Length of W must be {D}, got {w.shape[0]}.")

    b = float(B)

    # 生成数据
    X_tr, y_tr = make_linear_data(N_TRAIN, D, w, b, X_MIN, X_MAX, NOISE_STD_TRAIN, rng)
    X_te, y_te = make_linear_data(N_TEST,  D, w, b, X_MIN, X_MAX, NOISE_STD_TEST,  rng)

    # 保存
    os.makedirs(OUT_DIR, exist_ok=True)
    train_path = os.path.join(OUT_DIR, "train.txt")
    testX_path = os.path.join(OUT_DIR, "test_X.txt")
    testy_path = os.path.join(OUT_DIR, "test_y.txt")

    # 训练数据：拼接 [X, y]
    train_mat = np.concatenate([X_tr, y_tr.reshape(-1, 1)], axis=1)
    np.savetxt(train_path, train_mat, fmt=FMT, delimiter=" ")

    # 测试数据：X 与 y 分开保存
    np.savetxt(testX_path, X_te, fmt=FMT, delimiter=" ")
    np.savetxt(testy_path, y_te.reshape(-1, 1), fmt=FMT, delimiter=" ")

    # 摘要
    print("[OK] Saved datasets to:", OUT_DIR)
    print(f"  train.txt   -> shape (N={N_TRAIN}, D+1={D+1})  [last column is y]")
    print(f"  test_X.txt  -> shape (N={N_TEST}, D={D})")
    print(f"  test_y.txt  -> shape (N={N_TEST}, 1)")
    print(f"True params: w={w}, b={b}")
    print(f"Noise (σ): train={NOISE_STD_TRAIN}, test={NOISE_STD_TEST}")
    # 简单统计
    print(f"X range per dim ~ [{X_MIN}, {X_MAX}]")
    print("y_train mean/std ~", float(y_tr.mean()), float(y_tr.std(ddof=1)))


if __name__ == "__main__":
    main()
