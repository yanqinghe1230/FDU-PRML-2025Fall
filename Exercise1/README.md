# Lab 1: 线性回归

本实验围绕单变量与多变量线性回归展开，要求完成两种训练方法（闭式解与梯度下降），生成预测结果并用给定脚本评测。

仓库主要文件：
- 数据生成：`data_generate.py`
- 讲义/示例：`Exercise1_Linear_Regression.ipynb`
- 单变量实现与入口：`linear_regression.py`
- 多变量实现与入口：`multi_linear_regression.py`
- 评测脚本：`evaluation.py`
- 数据与输出：`input/`、`input_multi/`、`output/`、`output_multi/`

## 实验目标
- 理解线性回归的定义与解法。
- 掌握两种求解方法：
	- 闭式解（Normal Equation / Least Squares, LSE）。
	-（小批量）随机梯度下降（SGD/mini-batch SGD）。
- 使用提供脚本计算 RMSE 指标，进行结果对比与分析。

## 实验内容
1) 数据与文件说明
- 一元线性回归训练/测试数据位于 `input/` 目录：
	- `input/train.txt`：训练数据（格式以代码要求为准）。
	- `input/test_X.txt`：测试特征。
	- `input/test_y.txt`：测试真值标签（用于评测）。
- 多元线性回归训练/测试数据位于 `input_multi/` 目录
  
2) 你需要在代码中完成`TODO`部分。

3) 评测与运行
- 评测脚本：`evaluation.py`（默认从 `input/test_y.txt` 与 `output/predict.npy` 读取，计算 RMSE）。

4) 选做内容
- 使用 `data_generate.py` 自行合成数据（可调噪声、真值参数、维度），对比不同设置对训练与泛化的影响。

## 提交内容
运行Exercise1_Linear_Regression.ipynb，并转为PDF或html（需包含代码运行结果和三道Questions的回答），并提交到elearning.


