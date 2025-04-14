import pandas as pd
import numpy as np
from typing import Tuple, Union, Optional, Dict, Any
from sklearn.model_selection import train_test_split


class BatteryDataloader:
    """
    电池数据加载器类，用于加载、预处理和划分电池数据集

    Args:
        data_path (str): 数据文件路径
        processed_df (pd.DataFrame): 处理后的数据
        X_train (pd.DataFrame, optional): 训练集特征
        X_test (pd.DataFrame, optional): 测试集特征
        y_train (pd.Series, optional): 训练集目标值
        y_test (pd.Series, optional): 测试集目标值
        target_column (str): 目标列名，默认为'Capacity'
        random_state (int): 随机种子
    """

    def __init__(
        self,
        data_path: str,
        target_column: str = "Capacity",
        random_state: int = 42,
        preprocess: bool = True,
        pretrain: bool = False,
    ) -> None:
        """
        初始化电池数据加载器

        Args:
            data_path (str): 数据文件路径
            target_column (str, optional): 目标列名，默认为'Capacity'
            random_state (int, optional): 随机种子，默认为42
            preprocess (bool, optional): 是否在初始化时预处理数据，默认为True
        """
        self.data_path = data_path
        self.target_column = target_column
        self.random_state = random_state
        self.processed_df = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # 读取原始数据
        self.raw_df = pd.read_csv(data_path)

        if preprocess:
            self.preprocess_data(pretrain)

    def preprocess_data(self, pretrain=False) -> pd.DataFrame:
        """
        预处理数据，将电压特征提取并生成新的特征列

        Returns:
            pd.DataFrame: 处理后的数据框
        """
        # 提取电压特征
        voltage_features = self.raw_df["Voltages"].apply(self.extract_voltage_features)

        # 合并特征
        self.processed_df = pd.concat([self.raw_df, voltage_features], axis=1)

        # 删除原始电压列
        self.processed_df.drop(columns=["Voltages"], inplace=True)

        if pretrain:
            # 删除目标列
            self.processed_df.drop(columns=["D_rate"], inplace=True)
            # 重命名目标列
            self.processed_df.rename(columns={"C_rate": "rate"}, inplace=True)

        return self.processed_df

    @staticmethod
    def extract_voltage_features(voltage_str: str) -> pd.Series:
        """
        处理电压数据字符串并提取统计特征

        Args:
            voltage_str (str): 电压数据字符串

        Returns:
            pd.Series: 包含电压统计特征的Series
        """
        # 清理字符串，将字符串转换为数字列表
        cleaned_str = voltage_str.strip("[]").replace("\n", "").split()
        voltage_array = np.array([float(x) for x in cleaned_str])

        return pd.Series(
            {
                "voltage_mean": np.mean(voltage_array),
                "voltage_std": np.std(voltage_array),
                "voltage_max": np.max(voltage_array),
                "voltage_min": np.min(voltage_array),
            }
        )

    def train_test_split(
        self, test_size: float = 0.2, stratify: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        划分训练集和测试集

        Args:
            test_size (float, optional): 测试集比例，默认为0.2
            stratify (pd.Series, optional): 分层抽样依据，默认为None

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            (X_train, X_test, y_train, y_test)
        """
        if self.processed_df is None:
            raise ValueError("请先调用preprocess_data方法处理数据")

        # 准备特征和目标
        X = self.processed_df.drop(self.target_column, axis=1)
        y = self.processed_df[self.target_column]

        # 划分数据
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=stratify
        )

        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_data(self) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
        """
        获取处理后的数据集

        Returns:
            Dict[str, Union[pd.DataFrame, pd.Series]]: 包含训练集和测试集的字典
        """
        if any(
            data is None
            for data in [self.X_train, self.X_test, self.y_train, self.y_test]
        ):
            raise ValueError("请先调用train_test_split方法划分数据")

        return {
            "X_train": self.X_train,
            "X_test": self.X_test,
            "y_train": self.y_train,
            "y_test": self.y_test,
        }

    def save_processed_data(self, output_path: str) -> None:
        """
        保存处理后的数据

        Args:
            output_path (str): 输出文件路径
        """
        if self.processed_df is None:
            raise ValueError("请先调用preprocess_data方法处理数据")

        self.processed_df.to_csv(output_path, index=False)
        print(f"处理后的数据已保存至: {output_path}")

    def get_feature_names(self) -> list:
        """
        获取特征名称列表

        Returns:
            list: 特征名称列表
        """
        if self.processed_df is None:
            raise ValueError("请先调用preprocess_data方法处理数据")

        return [col for col in self.processed_df.columns if col != self.target_column]

    def get_data_info(self) -> Dict[str, Any]:
        """
        获取数据集基本信息

        Returns:
            Dict[str, Any]: 包含数据集基本信息的字典
        """
        if self.processed_df is None:
            raise ValueError("请先调用preprocess_data方法处理数据")

        info = {
            "总样本量": len(self.processed_df),
            "特征数量": len(self.get_feature_names()),
            "目标列名": self.target_column,
            "特征类型": {
                col: str(self.processed_df[col].dtype)
                for col in self.get_feature_names()
            },
            "缺失值统计": self.processed_df.isnull().sum().to_dict(),
        }

        # 如果已经划分了训练集和测试集，添加相关信息
        if self.X_train is not None:
            info.update(
                {"训练集样本量": len(self.X_train), "测试集样本量": len(self.X_test)}
            )

        return info


if __name__ == "__main__":
    # 示例用法
    loader = BatteryDataloader("./data/Dataset_1_NCA_battery.csv")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = loader.train_test_split(test_size=0.2)

    # 查看数据集信息
    data_info = loader.get_data_info()
    print(f"总样本量: {data_info['总样本量']}")
    print(f"训练集样本量: {data_info['训练集样本量']}")
    print(f"测试集样本量: {data_info['测试集样本量']}")

    # 保存处理后的数据（可选）
    # loader.save_processed_data("../data/processed_NCA_battery.csv")
