import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from typing import Dict, Tuple, List, Optional, Union, Any
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class XGBoostModel:
    """
    XGBoost模型类，用于电池数据的容量预测

    Parameters:
        model: XGBoost模型实例
        best_model: 网格搜索后的最佳模型
        params: 模型参数
        X_train: 训练集特征
        X_test: 测试集特征
        y_train: 训练集目标
        y_test: 测试集目标
        metrics: 模型评估指标
    """

    def __init__(
        self,
        task_name: str = "battery_capacity_prediction",
        objective: str = "reg:squarederror",
        n_estimators: int = 200,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_child_weight: int = 1,
        subsample: float = 0.9,
        random_state: int = 42,
    ) -> None:
        """
        初始化XGBoost模型

        Parameters:
            task_name: 任务名称
            objective: 目标函数
            n_estimators: 树的数量
            learning_rate: 学习率
            max_depth: 树的最大深度
            min_child_weight: 决定叶节点所需的最小样本权重和
            subsample: 每棵树的样本采样比例
            random_state: 随机种子
        """
        self.task_name = task_name
        self.params = {
            "objective": objective,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "min_child_weight": min_child_weight,
            "subsample": subsample,
            "random_state": random_state,
        }

        self.model = xgb.XGBRegressor(**self.params)
        self.best_model = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # 存储评估指标
        self.metrics = {"base_model": {}, "best_model": {}}

        # 中文字体设置
        plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
        plt.rcParams["axes.unicode_minus"] = False

    def train(self, x_train, y_train, x_test, y_test) -> None:
        """
        训练基础XGBoost模型
        """
        if x_train is None or y_train is None:
            raise ValueError("请先加载数据")
        self.X_train = x_train
        self.X_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        print("开始训练基础XGBoost模型...")
        self.model.fit(x_train, y_train)
        print("基础模型训练完成")

        self.evaluate_model(self.model, "base_model")

    def evaluate_model(
        self, model: xgb.XGBRegressor, model_type: str
    ) -> Dict[str, float]:
        """
        评估模型性能

        Args:
            model: 需要评估的模型
            model_type: 模型类型，'base_model' 或 'best_model'

        Returns:
            包含评估指标的字典
        """
        y_pred = model.predict(self.X_test)

        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        metrics = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R²": r2}

        self.metrics[model_type] = metrics

        print(f"{model_type.replace('_', ' ').title()} 评估结果:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")

        return metrics

    def grid_search(
        self,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        cv: int = 5,
        verbose: int = 1,
        n_jobs: int = -1,
    ) -> None:
        """
        执行网格搜索以找到最佳Args

        Args:
            param_grid: Args网格，如果为None则使用默认网格
            cv: 交叉验证折数
            verbose: 详细程度
            n_jobs: 并行作业数，-1表示使用所有处理器
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("请先加载数据")

        if param_grid is None:
            param_grid = {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7],
                "min_child_weight": [1, 3, 5],
                "subsample": [0.7, 0.8, 0.9],
            }

        print("开始网格搜索最佳Args...")
        grid_search = GridSearchCV(
            estimator=xgb.XGBRegressor(
                objective=self.params["objective"],
                random_state=self.params["random_state"],
            ),
            param_grid=param_grid,
            scoring="neg_mean_squared_error",
            cv=cv,
            verbose=verbose,
            n_jobs=n_jobs,
        )

        grid_search.fit(self.X_train, self.y_train)

        self.best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        print(f"网格搜索完成，最佳Args: {best_params}")

        # 评估最佳模型
        self.evaluate_model(self.best_model, "best_model")

        # 更新Args
        for key, value in best_params.items():
            self.params[key] = value

    def plot_feature_importance(
        self, model_type: str = "best_model", figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """
        可视化特征重要性

        Args:
            model_type: 'base_model' 或 'best_model'
            figsize: 图形大小
        """
        model = (
            self.best_model
            if model_type == "best_model" and self.best_model is not None
            else self.model
        )
        model_name = (
            "最佳"
            if model_type == "best_model" and self.best_model is not None
            else "基础"
        )

        plt.figure(figsize=figsize)
        sorted_idx = np.argsort(model.feature_importances_)
        plt.barh(range(len(sorted_idx)), model.feature_importances_[sorted_idx])
        plt.yticks(range(len(sorted_idx)), self.X_train.columns[sorted_idx])
        plt.title(f"{model_name}XGBoost模型特征重要性")
        plt.tight_layout()
        plt.savefig(
            f"./results/{self.task_name}/{model_name}_xgboost_feature_importance.png"
        )
        plt.close()

    def plot_predictions(
        self, model_type: str = "best_model", figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """
        可视化预测结果

        Args:
            model_type: 'base_model' 或 'best_model'
            figsize: 图形大小
        """
        model = (
            self.best_model
            if model_type == "best_model" and self.best_model is not None
            else self.model
        )
        model_name = (
            "最佳"
            if model_type == "best_model" and self.best_model is not None
            else "基础"
        )

        y_pred = model.predict(self.X_test)

        plt.figure(figsize=figsize)
        plt.scatter(self.y_test, y_pred, alpha=0.5)
        plt.plot(
            [self.y_test.min(), self.y_test.max()],
            [self.y_test.min(), self.y_test.max()],
            "r--",
        )
        plt.xlabel("真实容量")
        plt.ylabel("预测容量")
        plt.title(f"{model_name}XGBoost模型预测结果对比")
        plt.tight_layout()
        plt.savefig(
            f"./results/{self.task_name}/{model_name}_xgboost_TEST_predictions.png"
        )
        plt.close()

    def plot_capacity_vs_cycle(
        self,
        full_data: pd.DataFrame,
        model_type: str = "best_model",
        figsize: Tuple[int, int] = (12, 6),
    ) -> None:
        """
        可视化循环次数与容量的关系

        Args:
            full_data: 完整数据集
            model_type: 'base_model' 或 'best_model'
            figsize: 图形大小
        """
        model = (
            self.best_model
            if model_type == "best_model" and self.best_model is not None
            else self.model
        )
        model_name = (
            "最佳"
            if model_type == "best_model" and self.best_model is not None
            else "基础"
        )

        plt.figure(figsize=figsize)
        plt.scatter(
            full_data["cycle"], full_data["Capacity"], alpha=0.5, label="真实数据"
        )

        # 按循环次数排序
        y_pred = model.predict(self.X_test)
        sorted_indices = np.argsort(self.X_test["cycle"].values)
        sorted_cycles = self.X_test["cycle"].values[sorted_indices]
        sorted_predictions = y_pred[sorted_indices]

        plt.scatter(
            sorted_cycles,
            sorted_predictions,
            alpha=0.5,
            c="r",
            label=f"{model_name}模型预测",
        )
        plt.xlabel("循环次数")
        plt.ylabel("电池容量")
        plt.title(f"循环次数与电池容量关系({model_name}模型预测)")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"./results/{self.task_name}/{model_name}_capacity_vs_cycle.png")
        plt.close()

    def plot_residuals(
        self, model_type: str = "best_model", figsize: Tuple[int, int] = (12, 4)
    ) -> None:
        """
        可视化残差分析

        Args:
            model_type: 'base_model' 或 'best_model'
            figsize: 图形大小
        """
        model = (
            self.best_model
            if model_type == "best_model" and self.best_model is not None
            else self.model
        )
        model_name = (
            "最佳"
            if model_type == "best_model" and self.best_model is not None
            else "基础"
        )

        y_pred = model.predict(self.X_test)
        residuals = self.y_test - y_pred

        plt.figure(figsize=figsize)

        # 残差分布图
        plt.subplot(121)
        sns.histplot(residuals, kde=True)
        plt.title("残差分布")
        plt.xlabel("残差")

        # 残差散点图
        plt.subplot(122)
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color="r", linestyle="--")
        plt.xlabel("预测值")
        plt.ylabel("残差")
        plt.title("残差散点图")

        plt.tight_layout()
        plt.savefig(
            f"./results/{self.task_name}/{model_name}_xgboost_TEST_residual_analysis.png"
        )
        plt.close()

    def plot_learning_curve(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        train_sizes_ratio: np.ndarray = np.linspace(0.1, 1.0, 10),
        figsize: Tuple[int, int] = (10, 6),
    ) -> None:
        """
        绘制学习曲线

        Args:
            X: 完整特征集
            y: 完整目标变量集
            cv: 交叉验证折数
            train_sizes_ratio: 训练集大小比例
            figsize: 图形大小
        """
        train_sizes, train_scores, test_scores = learning_curve(
            self.model if self.best_model is None else self.best_model,
            X,
            y,
            cv=cv,
            n_jobs=-1,
            train_sizes=train_sizes_ratio,
            scoring="neg_mean_squared_error",
        )

        train_scores_mean = -np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = -np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure(figsize=figsize)
        plt.fill_between(
            train_sizes,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color="r",
        )
        plt.fill_between(
            train_sizes,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color="g",
        )
        plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="训练集得分")
        plt.plot(train_sizes, test_scores_mean, "o-", color="g", label="验证集得分")
        plt.xlabel("训练样本数")
        plt.ylabel("均方误差")
        plt.title("XGBoost学习曲线")
        plt.legend(loc="best")
        plt.grid()
        plt.savefig(
            f"./results/{self.task_name}/xgboost_k折交叉验证_{self.params['n_estimators']}.png"
        )
        plt.close()

    def compare_models(self) -> pd.DataFrame:
        """
        比较基础模型和最佳模型的性能

        Returns:
            包含性能对比的DataFrame
        """
        if not self.metrics["base_model"] or (
            self.best_model is not None and not self.metrics["best_model"]
        ):
            raise ValueError("请先评估模型")

        base_metrics = self.metrics["base_model"]
        best_metrics = (
            self.metrics["best_model"] if self.best_model is not None else None
        )

        if best_metrics:
            comparison_df = pd.DataFrame(
                {
                    "评估指标": list(base_metrics.keys()),
                    "基础模型": list(base_metrics.values()),
                    "最佳模型": list(best_metrics.values()),
                }
            )
            print("\n模型性能对比:")
            print(comparison_df)
        else:
            comparison_df = pd.DataFrame(
                {
                    "评估指标": list(base_metrics.keys()),
                    "基础模型": list(base_metrics.values()),
                }
            )
            print("\n基础模型性能:")
            print(comparison_df)

        # 将对比结果保存到CSV文件
        comparison_df.to_csv(
            f"./results/{self.task_name}/模型性能对比.csv", index=False
        )
        return comparison_df

    def save_model(self, model_path: str, model_type: str = "best_model") -> None:
        """
        保存模型到文件

        Args:
            model_path: 保存路径
            model_type: 'base_model' 或 'best_model'
        """
        model = (
            self.best_model
            if model_type == "best_model" and self.best_model is not None
            else self.model
        )
        model_name = (
            "最佳"
            if model_type == "best_model" and self.best_model is not None
            else "基础"
        )

        joblib.dump(model, model_path)
        print(f"{model_name}模型已保存至 '{model_path}'")

    def load_model(self, model_path: str) -> "XGBoostModel":
        """
        从文件加载模型

        Args:
            model_path: 模型文件路径

        Returns:
            加载模型的XGBoostModel实例
        """
        model = joblib.load(model_path)

        # 是否为XGBoost模型
        if isinstance(model, xgb.XGBRegressor):
            self.model = model
            print(f"模型已从 '{model_path}' 加载")

        else:
            raise TypeError(f"加载的模型类型 {type(model)} 不是 XGBRegressor")


if __name__ == "__main__":
    # 示例用法
    processed_df = pd.read_csv("../data/processed_NCA_battery.csv")

    # 准备特征和目标变量
    X = processed_df.drop("Capacity", axis=1)
    y = processed_df["Capacity"]

    # 初始化模型
    xgb_model = XGBoostModel(n_estimators=100, learning_rate=0.1, max_depth=5)

    # 加载数据
    xgb_model.load_data(X, y, test_size=0.2)

    # 训练基础模型
    xgb_model.train()

    # 可视化特征重要性
    xgb_model.plot_feature_importance(model_type="base_model")

    # 可视化预测结果
    xgb_model.plot_predictions(model_type="base_model")

    # 网格搜索最佳Args（注意：此步骤可能会很耗时）
    # 如果需要快速测试，可以使用较小的Args网格
    small_param_grid = {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
    }
    xgb_model.grid_search(param_grid=small_param_grid)

    # 比较模型性能
    xgb_model.compare_models()

    # 可视化最佳模型结果
    xgb_model.plot_predictions(model_type="best_model")
    xgb_model.plot_feature_importance(model_type="best_model")

    # 分析循环次数与容量的关系
    xgb_model.plot_capacity_vs_cycle(processed_df, model_type="best_model")

    # 残差分析
    xgb_model.plot_residuals(model_type="best_model")

    # 学习曲线
    xgb_model.plot_learning_curve(X, y)

    # 保存最佳模型
    # xgb_model.save_model('../models/best_xgboost_model.pkl', 'best_model')
