# 基于机器学习的电池容量预测

## 1. 项目概述

本项目旨在使用机器学习技术，特别是 XGBoost 模型，根据电池的运行数据（如电压曲线、充放电倍率、温度等）来预测其剩余容量（Capacity）。准确预测电池容量对于电池管理系统（BMS）、电动汽车续航估计以及电池健康状态（SOH）评估至关重要。

项目主要包含以下几个部分：
*   **数据加载与预处理**: 使用 `BatteryDataloader` 类处理原始数据，特别是从复杂的电压序列数据中提取有效特征。
*   **模型训练与调优**: 使用 `XGBoostModel` 类实现模型的训练、基于网格搜索的超参数优化以及模型评估。
*   **结果分析与可视化**: 对模型性能进行评估，并可视化特征重要性、预测结果、残差分布等，以深入理解模型行为和数据特性。

## 2. 数据集介绍

*   **数据来源**: 本项目使用的数据集在 `data/`。这是一个包含镍钴铝（NCA）和NCM锂离子电池循环数据的数据集。
*   **数据格式**: CSV 文件格式。
*   **主要字段**:
    *   `cycle`: 电池循环次数。
    *   `Voltages`: 一个字符串，表示在一个循环或特定阶段内的电压采样点序列。通常需要解析和特征提取。
    *   `rate`: 充放电倍率（C-rate）。
    *   `Tem`: 电池工作温度（摄氏度）。
    *   `Capacity`: 电池在该循环下的容量（Ah 或 mAh），是本项目的预测目标。
*   **数据特点**:
    *   `Voltages` 列包含嵌套的序列数据，需要进行特征工程才能被标准机器学习模型使用。
    *   数据可能包含不同工况（倍率、温度）下的循环。
    *   电池容量会随着循环次数增加而衰减。

## 3. 技术方案与实验设计

### 3.1 数据预处理 (`src/dataset.py`)

1.  **加载数据**: 使用 `pandas` 读取原始 CSV 文件。
2.  **电压特征提取**:
    *   定义 `extract_voltage_features` 静态方法，用于处理 `Voltages` 列的字符串。
    *   该方法首先清理字符串（去除括号、换行符），然后将字符串转换为 NumPy 数组。
    *   计算电压序列的统计特征，包括：
        *   `voltage_mean`: 平均电压
        *   `voltage_std`: 电压标准差
        *   `voltage_max`: 最大电压
        *   `voltage_min`: 最小电压
    *   将这些新提取的特征添加到 DataFrame 中。
3.  **数据整合**: 将原始特征（cycle, rate, Tem）与新提取的电压特征合并，移除原始的 `Voltages` 列，形成处理后的数据集 `processed_df`。
4.  **数据集划分**: 提供 `train_test_split` 方法，使用 `sklearn.model_selection.train_test_split` 将处理后的数据划分为训练集和测试集，默认测试集比例为 20%，并使用固定的 `random_state` 以保证结果可复现。

### 3.2 模型构建与训练 (`src/model.py`)

1.  **模型选择**: 选择 XGBoost (Extreme Gradient Boosting) 作为核心预测模型。XGBoost 是一种高效、灵活且准确的梯度提升树算法，特别适合处理表格数据，并能有效处理特征间的复杂关系。
2.  **模型封装**: 创建 `XGBoostModel` 类来管理模型的整个生命周期。
    *   **初始化 (`__init__`)**: 设置 XGBoost 的基本参数（如 `objective`, `n_estimators`, `learning_rate`, `max_depth` 等），初始化模型对象，并设置 Matplotlib 的中文字体参数（例如，设置为 'WenQuanYi Micro Hei' 或 'Noto Sans CJK SC'）。
    *   **数据加载 (`load_data`)**: (已移除，训练数据直接传入 `train` 方法)
    *   **训练 (`train`)**: 接收训练和测试数据，使用 `xgb.XGBRegressor.fit()` 方法训练基础模型，并在训练完成后自动评估模型性能。
    *   **评估 (`evaluate_model`)**: 计算并打印关键回归指标：均方误差 (MSE)、均方根误差 (RMSE)、平均绝对误差 (MAE) 和 R 方 (R²)。结果存储在 `self.metrics` 字典中。
    *   **网格搜索 (`grid_search`)**: 使用 `sklearn.model_selection.GridSearchCV` 进行超参数调优。定义参数网格 `param_grid`（或使用默认网格），通过交叉验证寻找最佳参数组合。训练完成后，将最佳模型存储在 `self.best_model` 中，并评估其性能。
    *   **可视化**: 提供多个绘图方法：
        *   `plot_feature_importance`: 可视化模型认为最重要的特征。
        *   `plot_predictions`: 绘制真实值 vs. 预测值的散点图，直观展示模型预测效果。
        *   `plot_capacity_vs_cycle`: 绘制容量随循环次数变化的散点图，并叠加模型预测结果。
        *   `plot_residuals`: 绘制残差分布图和残差散点图，用于检查模型假设和误差模式。
        *   `plot_learning_curve`: 绘制学习曲线，帮助判断模型是否过拟合或欠拟合。
    *   **模型比较 (`compare_models`)**: 以表格形式展示基础模型和优化后模型的性能指标对比。
    *   **模型持久化**: 提供 `save_model` 和 `load_model` 方法，使用 `joblib` 保存和加载训练好的模型。

### 3.3 实验流程

1.  **环境准备**: 安装所需的 Python 库（见 `requirements.txt`）。确保系统已安装中文字体（如 Noto Sans CJK）以正确显示图表。
2.  **数据加载与预处理**: 实例化 `BatteryDataloader`，加载 `Dataset_1_NCA_battery.csv` 并进行预处理。
3.  **数据划分**: 调用 `loader.train_test_split()` 获取训练集和测试集 (`X_train`, `X_test`, `y_train`, `y_test`)。
4.  **模型初始化**: 实例化 `XGBoostModel`，可以指定初始超参数。
5.  **基础模型训练与评估**: 调用 `xgb_model.train(X_train, y_train, X_test, y_test)` 训练基础模型并查看其性能。
6.  **超参数调优**: 调用 `xgb_model.grid_search()`（可传入自定义参数网格）寻找最佳模型。
7.  **结果分析**:
    *   调用 `xgb_model.compare_models()` 对比基础模型和最佳模型的性能。
    *   调用各种 `plot_*` 方法，对最佳模型进行深入分析（特征重要性、预测效果、残差等）。
8.  **模型保存**: (可选) 调用 `xgb_model.save_model()` 保存训练好的最佳模型以备后用。

## 4. 项目结构

```
battery/
├── data/
│   ├── Dataset_1_NCA_battery.csv  # 原始数据
│   └── processed_NCA_battery.csv  # (可选) 保存处理后的数据
├── src/
│   ├── dataset.py             # BatteryDataloader 类
│   ├── model.py               # XGBoostModel 类
│   ├── NCA_EDA.ipynb          # (可选) 探索性数据分析 Notebook
│   └── train.py               # (可选) 主训练脚本
├── figures/                     # 保存生成的图表
│   ├── feature_importance.png
│   └── ...
├── models/                      # 保存训练好的模型
│   └── best_xgboost_model.pkl
├── .venv/                       # Python 虚拟环境
├── README.md                    # 本文档
└── requirements.txt             # 项目依赖
```

## 5. 环境设置与安装

1.  **克隆仓库**:
    ```bash
    git clone https://github.com/Rvosuke/battery.git
    cd battery
    ```
2.  **创建虚拟环境** (推荐):
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    # .venv\Scripts\activate  # Windows
    ```
3.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```
    *主要依赖包括: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `joblib`, `rich`, `tabulate`*
4.  **安装中文字体** (如果绘图需要显示中文):
    *   **Debian/Ubuntu**:
        ```bash
        sudo apt-get update
        sudo apt-get install fonts-noto-cjk fonts-wqy-microhei
        sudo fc-cache -fv
        ```
    *   **其他系统**: 请参考相应系统的字体安装方法。
    *   **清除 Matplotlib 缓存**: (如果安装字体后仍有问题)
        ```bash
        rm -rf ~/.cache/matplotlib
        ```

## 6. 使用说明

1.  **准备数据**: 将 `Dataset_1_NCA_battery.csv` 或其他文件放置在 `data/` 目录下。
2.  **运行训练与评估**:
    `python main.py`  注意设置datapath
    > 当进行迁移学习时，设置pretrain=True，我们将使用在NCAdataset中训练完毕的模型来初始化NCM_NCA_dataset模型。
3.  **查看结果**:
    *   训练过程中的评估指标会打印到终端。
    *   生成的图表会显示出来，或者根据代码设置保存到 `results/` 目录。
    *   训练好的模型会保存到 `models/` 目录（如果调用了 `save_model`）。

## 7. 结果与讨论

*(在此处总结模型的最终性能指标，例如最佳模型的 RMSE 和 R²。讨论特征重要性分析的结果，哪些特征对容量预测最重要。讨论预测图和残差图反映的模型优缺点。)*

**示例:**

*   基础 XGBoost 模型在测试集上达到了 RMSE: X.XXX, R²: Y.YYY。
*   经过网格搜索优化后，最佳模型的性能提升至 RMSE: A.AAA, R²: B.BBB。
*   特征重要性分析显示，`cycle`（循环次数）、`voltage_mean`（平均电压）和 `Tem`（温度）是对电池容量影响最大的三个因素。
*   预测结果图表明模型能较好地拟合大部分数据点，但在容量较低的区域预测误差略有增大。
*   残差分析显示残差大致呈正态分布，且没有明显模式，表明模型假设基本满足。

## 8. 未来工作

*   尝试更复杂的特征工程方法，例如从电压曲线中提取更多形状相关的特征。
*   探索其他机器学习或深度学习模型（如 LSTM、GRU）处理电压序列数据。
*   将模型应用于更多不同类型（如 LFP、NMC）的电池数据集，评估其泛化能力。
*   集成模型预测结果到电池管理系统（BMS）仿真或实际应用中。
