# test_rolling_prediction.py
import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets.data_processor import TimeSeriesDataProcessor
from model.SingleStepTransformer import SingleStepTransformer
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# 配置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def test_rolling_prediction():
    """
    测试滚动预测功能：
    1. 载入训练好的模型
    2. 使用测试集前90天作为初始窗口
    3. 滚动预测未来90天
    """

    # 配置参数（与训练代码保持一致）
    MODEL_PATH = "90_best_single_step.pt"  # 模型文件路径
    TRAIN_PATH = "datasets/daily_power_train2.csv"  # 训练数据路径
    TEST_PATH = "datasets/daily_power_test2.csv"  # 测试数据路径
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PRED_DAYS = 90
    BATCH_SIZE = 16

    # 检查文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"错误：模型文件 {MODEL_PATH} 不存在！")
        return

    print("=== 开始滚动预测测试 ===")
    print(f"使用设备: {DEVICE}")

    # 1. 初始化数据处理器（与训练代码保持一致）
    processor = TimeSeriesDataProcessor(
        train_path=TRAIN_PATH,
        test_path=TEST_PATH,
        sequence_length=PRED_DAYS,
        prediction_length=1  # 单步预测
    )

    # 2. 处理数据
    print("正在处理数据...")
    data = processor.process_data(add_time_features=True, time_encoding="sincos")
    n_features = data['n_features']
    print(f"特征数量: {n_features}")
    print(f"训练序列数量: {data['X_train'].shape[0]}")
    print(f"测试序列数量: {data['X_test'].shape[0]}")

    # 3. 载入模型（与训练代码保持一致）
    print("正在载入模型...")
    model = SingleStepTransformer(
        n_features=n_features,
        input_len=PRED_DAYS  # 使用与训练相同的参数
    )

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print("模型载入成功！")
    except Exception as e:
        print(f"模型载入失败: {e}")
        return

    # 4. 准备测试数据
    # 获取测试集的原始数据（未标准化）
    test_df = pd.read_csv(TEST_PATH)
    test_df = processor.handle_missing(test_df)
    test_df = processor.add_time_features(test_df, encoding="sincos")

    # 获取可用特征列
    available_features = [col for col in processor.feature_columns if col in test_df.columns]
    test_raw = test_df[available_features].values

    print(f"测试数据形状: {test_raw.shape}")
    print(f"使用特征: {available_features}")

    # 5. 使用前90天作为初始窗口
    initial_window = test_raw[:90]  # shape: [90, F]
    print(f"初始窗口形状: {initial_window.shape}")

    # 6. 执行滚动预测
    print("开始滚动预测...")
    predictions = rolling_predict_90_days(model, initial_window, processor, DEVICE)
    print(f"预测完成！预测结果形状: {predictions.shape}")

    # 7. 获取真实值用于比较（如果测试集足够长）
    if len(test_raw) >= 180:  # 至少需要90+90天数据
        true_values = test_raw[90:180, 0]  # 取第91-180天的Global_active_power
        # 对真实值进行反标准化处理
        true_values_normalized = processor.feature_scaler.transform(test_raw[90:180])[:, 0]
        true_values = processor.inverse_transform_target(true_values_normalized)

        # 计算评估指标
        mse = mean_squared_error(true_values, predictions)
        mae = mean_absolute_error(true_values, predictions)
        rmse = np.sqrt(mse)

        print(f"\n=== 评估结果 ===")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"RMSE: {rmse:.6f}")

        # 绘制对比图
        plot_results(true_values, predictions, save_path="prediction_comparison.png")

    else:
        print("测试集数据不足，无法进行真实值对比")
        true_values = None

    # 8. 保存预测结果
    save_predictions(predictions, true_values, save_path="prediction_results.csv")

    # 9. 显示预测统计信息
    print(f"\n=== 预测统计信息 ===")
    print(f"预测值范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"预测值均值: {predictions.mean():.4f}")
    print(f"预测值标准差: {predictions.std():.4f}")

    return predictions, true_values


def rolling_predict_90_days(model, initial_window, processor, device):
    """
    滚动预测未来90天

    Args:
        model: 训练好的模型
        initial_window: 初始90天窗口，shape: [90, F]
        processor: 数据处理器
        device: 计算设备

    Returns:
        predictions: 预测结果，shape: [90]
    """
    model.eval()
    history = initial_window.copy()
    predictions = []

    print("滚动预测进度:")
    for step in range(90):
        # 取最近90天作为输入
        current_window = history[-90:]  # shape: [90, F]

        # 标准化输入
        x_scaled = processor.feature_scaler.transform(current_window)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0).to(device)

        # 预测下一天
        with torch.no_grad():
            y_pred_scaled = model(x_tensor)  # shape: [1]
            y_pred = processor.inverse_transform_target(y_pred_scaled.cpu().numpy())[0]

        predictions.append(y_pred)

        # 构造下一天的特征（复制最后一天的特征，更新目标值）
        next_day_features = current_window[-1].copy()
        next_day_features[0] = y_pred  # 第0维是Global_active_power

        # 滚动更新历史窗口
        history = np.vstack([history, next_day_features])

        # 显示进度
        if (step + 1) % 10 == 0:
            print(f"  完成 {step + 1}/90 天预测")

    return np.array(predictions)


def plot_results(true_values, predictions, save_path="prediction_comparison.png"):
    """绘制预测结果对比图"""
    plt.figure(figsize=(15, 6))

    days = np.arange(1, 91)
    plt.plot(days, true_values, label='Ground Truth (真实值)', color='blue', alpha=0.7, linewidth=2)
    plt.plot(days, predictions, label='Prediction (预测值)', color='red', linestyle='--', alpha=0.7, linewidth=2)

    plt.xlabel('时间序列数据点 (Data Points)', fontsize=12)
    plt.ylabel('每日总有功功率 (kW)', fontsize=12)
    plt.title('电量 (Power) 滚动预测结果对比曲线图（90天）', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"对比图已保存到: {save_path}")


def save_predictions(predictions, true_values=None, save_path="prediction_results.csv"):
    """保存预测结果到CSV文件"""
    result_df = pd.DataFrame({
        'Day': np.arange(1, 91),
        'Predicted_Power_kW': predictions
    })

    if true_values is not None:
        result_df['True_Power_kW'] = true_values
        result_df['Error'] = predictions - true_values
        result_df['Absolute_Error'] = np.abs(result_df['Error'])
        result_df['Relative_Error_%'] = (result_df['Absolute_Error'] / np.abs(true_values)) * 100

    result_df.to_csv(save_path, index=False)
    print(f"预测结果已保存到: {save_path}")

    # 显示前几行结果
    print("\n预测结果预览:")
    print(result_df.head(10))


if __name__ == "__main__":
    # 运行测试
    try:
        predictions, true_values = test_rolling_prediction()
        print("\n=== 测试完成 ===")

    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback

        traceback.print_exc()