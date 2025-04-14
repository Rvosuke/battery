import os
from pprint import pprint
from src import BatteryDataloader, XGBoostModel


def main():
    # 加载数据
    data_path = "./data/Dataset_3_NCM_NCA_battery.csv"
    pretrain = True

    task_name = data_path.split("/")[-1].split(".")[0].split("_")[2:]
    task_name = "_".join(task_name) if not pretrain else "pretrain"
    print(f"任务名称: {task_name}")
    os.makedirs(f"./checkpoints/{task_name}", exist_ok=True)
    os.makedirs(f"./results/{task_name}", exist_ok=True)
    print(f"数据路径: {data_path}")
    print("加载数据中...")
    loader = BatteryDataloader(data_path, pretrain=pretrain)
    loader.train_test_split(test_size=0.2)
    data_info = loader.get_data_info()
    print("数据集基本信息:")
    pprint(data_info, indent=2, width=80)

    # 初始化模型
    model = XGBoostModel(task_name=task_name)

    if pretrain:
        model.load_model(f"./checkpoints/NCA_battery/best_xgboost_model.ckpt")
    model.train(loader.X_train, loader.y_train, loader.X_test, loader.y_test)

    # 训练模型
    if not pretrain:
        model.grid_search()
    model.save_model(f"./checkpoints/{task_name}/best_xgboost_model.ckpt", "best_model")
    model.plot_feature_importance()
    model.plot_capacity_vs_cycle(loader.processed_df)
    model.plot_predictions()
    model.plot_residuals()
    model.plot_learning_curve(loader.X_train, loader.y_train)
    model.compare_models()


if __name__ == "__main__":
    main()
