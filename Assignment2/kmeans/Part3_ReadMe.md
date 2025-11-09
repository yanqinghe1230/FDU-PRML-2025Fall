# Assignment 2 - Part 3: K-Means 聚类及其分析

本实验围绕K-Means聚类展开，要求实现支持多种距离度量的 K-Means 聚类，并在鸢尾花数据集上进行聚类效果评估及可视化。

分数占比 - 35%

仓库主要文件: `kmeans/`
- KMeans 主实现兼学生入口：`kmeans_student.py`
- 聚类可视化：`viz_kmeans.py`
- 聚类评估指标：`test_kmeans.py`
- 鸢尾花数据：`data/iris_dataset.csv`
- 图片保存：`output/`


## 实验目标

- 理解 K-Means 算法实现的核心流程
- 掌握不同的距离度量的实现（包括 Euclidean, Chebyshev, Minkowski）
- 掌握轮廓系数（silhouette）与 Davies-Bouldin 指数等聚类内部评价指标
- 在二维平面上可视化聚类结果
- 分析不同距离度量/聚类中心数（k）的影响



## 实验主体内容

你需要在`kmeans_student.py`中完成 `TODO` 部分

- 补全不同距离度量的计算
- 补全算法的主要流程


## 运行与评测

- 运行主文件 `kmeans_student.py` 完成所有配置实验，自动输出每种聚类方式的聚类可视化图片（如 `kmeans_k3_euclidean.png`），终端将输出指标

- 图片和终端结果便于比较不同距离独立和K值的影响

## 分析内容

- 在k=3的条件下，对比不同距离度量的结果。（在本次实验的数据集上可能影响不大，但可通过相互对比，验证实现是否正确

- 在使用欧几里得距离的条件下，分析不同k值的影响。

- 选做：尝试自定义 epsilon 等 KMeans 超参数，观察结果


## 提交内容

- 以实验报告为准

