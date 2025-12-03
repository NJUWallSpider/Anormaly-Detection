# 晶圆制造过程异常检测

该项目包含一系列用于半导体制造过程中晶圆数据异常检测的 Python 脚本。项目利用多种机器学习和深度学习模型，旨在识别生产过程中的异常样本。

## 文件结构和说明

-   `data.xlsx`: 包含52个sheet的原始数据文件，每个sheet代表一个晶圆的生产数据。

-   `feature_engineering.py`: 核心数据处理模块。负责从 `data.xlsx` 加载数据，并进行特征工程，包括数据标准化、特征选择（基于方差和相关性）以及为不同模型准备数据。

-   `encoder.py`: 使用 PyTorch 实现了一个基本的前馈神经网络自编码器（Autoencoder）用于异常检测。

-   `LSTM.py`: 使用 PyTorch 实现了一个基于 LSTM 的自编码器模型，专门用于处理时序数据的异常检测。

-   `Transformer新.py`: 使用 PyTorch 实现了一个基于 Transformer 的自编码器模型，同样用于时序数据的异常检测，并包含特征重要性分析。

-   `forward_detection.py`: 实现了多种经典的无监督异常检测算法，如孤立森林（Isolation Forest）、局部异常因子（Local Outlier Factor）等，并使用 PCA 进行降维和结果可视化。

-   `online_MSPC.py`: 实现了一个在线版本的多元统计过程控制（MSPC）方法，使用霍特林 T2 和 Q 统计量（SPE）进行在线异常检测。

-   `online_detection.py`: 实现了一个基于 Transformer 自编码器的在线异常检测系统。该系统逐个处理样本，如果样本被判定为正常，则将其加入训练集以更新模型。

-   `random_test.py`: 用于测试 `online_detection.py` 脚本的鲁棒性。它通过多次运行并打乱数据顺序，来验证在线检测结果的一致性。

## 环境依赖

项目所需的 Python 库已在 `requirements.txt` 文件中列出。您可以通过以下命令安装所有依赖：

```bash
pip install -r requirements.txt
```

## 如何运行

您可以根据需要选择运行不同的检测脚本。例如，要运行基于 Transformer 的在线检测，可以执行：

```bash
python online_detection.py
```

要运行鲁棒性测试，可以执行：

```bash
python random_test.py
```
