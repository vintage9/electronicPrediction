import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import math
import warnings

warnings.filterwarnings('ignore')


# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


set_seed(42)


# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


# Transformer模型
class PowerTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=8, num_layers=6, output_dim=1, dropout=0.1, max_len=1000):
        super(PowerTransformer, self).__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_projection = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, seq_len):
        # src: (batch_size, seq_len, input_dim)
        # 转换为 (seq_len, batch_size, input_dim)
        src = src.transpose(0, 1)

        # 输入投影
        src = self.input_projection(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        src = self.dropout(src)

        # Transformer编码
        memory = self.transformer_encoder(src)

        # 只取最后seq_len个时间步进行预测
        output = memory[-seq_len:, :, :]  # (seq_len, batch_size, d_model)
        output = self.output_projection(output)  # (seq_len, batch_size, output_dim)

        # 转换回 (batch_size, seq_len, output_dim)
        output = output.transpose(0, 1)

        return output


# 数据集类
class PowerDataset(Dataset):
    def __init__(self, data, input_len=90, output_len=90):
        self.data = data
        self.input_len = input_len
        self.output_len = output_len

        # 创建样本
        self.samples = []
        for i in range(len(data) - input_len - output_len + 1):
            input_seq = data[i:i + input_len]
            output_seq = data[i + input_len:i + input_len + output_len]
            self.samples.append((input_seq, output_seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_seq, output_seq = self.samples[idx]
        return torch.FloatTensor(input_seq), torch.FloatTensor(output_seq)


# 数据预处理函数
def preprocess_data(train_path, test_path):
    """
    数据预处理函数
    """
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # 合并数据进行统一预处理
    all_data = pd.concat([train_data, test_data], ignore_index=True)

    # 处理缺失值
    all_data = all_data.fillna(method='ffill').fillna(method='bfill')

    # 选择特征列
    feature_columns = ['global_active_power', 'global_reactive_power', 'voltage',
                       'global_intensity', 'sub_metering_1', 'sub_metering_2',
                       'sub_metering_3', 'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']

    # 确保所有特征列都存在
    available_columns = [col for col in feature_columns if col in all_data.columns]
    if len(available_columns) < len(feature_columns):
        print(f"警告: 缺少特征列: {set(feature_columns) - set(available_columns)}")

    # 使用可用的特征列
    data = all_data[available_columns].values

    # 标准化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # 分割训练和测试集
    train_size = len(train_data)
    train_scaled = data_scaled[:train_size]
    test_scaled = data_scaled[train_size:]

    return train_scaled, test_scaled, scaler, available_columns


# 训练函数
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=50):
    model.train()
    train_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, targets.size(1))

            # 只使用第一个特征（global_active_power）作为预测目标
            loss = criterion(outputs[:, :, 0], targets[:, :, 0])

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.6f}')

    return train_losses


# 测试函数
def test_model(model, test_loader, criterion, device, scaler, target_col_idx=0):
    model.eval()
    test_losses = []
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, targets.size(1))

            # 只使用第一个特征作为预测目标
            loss = criterion(outputs[:, :, 0], targets[:, :, 0])
            test_losses.append(loss.item())

            # 收集预测结果
            all_predictions.append(outputs[:, :, 0].cpu().numpy())
            all_targets.append(targets[:, :, 0].cpu().numpy())

    # 合并所有预测结果
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # 反标准化（假设global_active_power是第一个特征）
    predictions_rescaled = predictions * scaler.scale_[target_col_idx] + scaler.mean_[target_col_idx]
    targets_rescaled = targets * scaler.scale_[target_col_idx] + scaler.mean_[target_col_idx]

    # 计算MSE和MAE
    mse = np.mean((predictions_rescaled - targets_rescaled) ** 2)
    mae = np.mean(np.abs(predictions_rescaled - targets_rescaled))

    return mse, mae, predictions_rescaled, targets_rescaled, np.mean(test_losses)


# 绘制预测结果
def plot_predictions(predictions, targets, title="Power Consumption Prediction"):
    plt.figure(figsize=(15, 8))

    # 只绘制前200个预测点以便可视化
    plot_len = min(200, len(predictions))
    time_steps = range(plot_len)

    plt.plot(time_steps, targets[:plot_len].flatten(), 'b-', label='Ground Truth', linewidth=2)
    plt.plot(time_steps, predictions[:plot_len].flatten(), 'r--', label='Prediction', linewidth=2)

    plt.xlabel('Time Steps')
    plt.ylabel('Global Active Power (kW)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# 主函数
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 数据路径
    train_path = 'train.csv'
    test_path = 'test.csv'

    # 数据预处理
    print("正在加载和预处理数据...")
    train_data, test_data, scaler, feature_columns = preprocess_data(train_path, test_path)

    print(f"训练集大小: {train_data.shape}")
    print(f"测试集大小: {test_data.shape}")
    print(f"特征列数: {len(feature_columns)}")

    # 模型参数
    input_dim = len(feature_columns)
    d_model = 64
    nhead = 8
    num_layers = 6
    dropout = 0.1

    # 用于存储多次实验结果
    short_term_results = {'mse': [], 'mae': []}
    long_term_results = {'mse': [], 'mae': []}

    # 进行5次实验
    for run in range(5):
        print(f"\n=== 实验 {run + 1}/5 ===")

        # 短期预测 (90天)
        print("训练短期预测模型 (90天)...")
        short_dataset = PowerDataset(train_data, input_len=90, output_len=90)
        short_loader = DataLoader(short_dataset, batch_size=32, shuffle=True)

        short_test_dataset = PowerDataset(test_data, input_len=90, output_len=90)
        short_test_loader = DataLoader(short_test_dataset, batch_size=32, shuffle=False)

        short_model = PowerTransformer(input_dim, d_model, nhead, num_layers, 1, dropout).to(device)
        short_criterion = nn.MSELoss()
        short_optimizer = optim.Adam(short_model.parameters(), lr=0.001)

        # 训练短期模型
        train_model(short_model, short_loader, short_criterion, short_optimizer, device, num_epochs=50)

        # 测试短期模型
        short_mse, short_mae, short_pred, short_target, _ = test_model(
            short_model, short_test_loader, short_criterion, device, scaler)

        short_term_results['mse'].append(short_mse)
        short_term_results['mae'].append(short_mae)

        print(f"短期预测 - MSE: {short_mse:.6f}, MAE: {short_mae:.6f}")

        # 长期预测 (365天)
        print("训练长期预测模型 (365天)...")
        long_dataset = PowerDataset(train_data, input_len=90, output_len=365)
        long_loader = DataLoader(long_dataset, batch_size=16, shuffle=True)

        long_test_dataset = PowerDataset(test_data, input_len=90, output_len=365)
        long_test_loader = DataLoader(long_test_dataset, batch_size=16, shuffle=False)

        long_model = PowerTransformer(input_dim, d_model, nhead, num_layers, 1, dropout).to(device)
        long_criterion = nn.MSELoss()
        long_optimizer = optim.Adam(long_model.parameters(), lr=0.001)

        # 训练长期模型
        train_model(long_model, long_loader, long_criterion, long_optimizer, device, num_epochs=50)

        # 测试长期模型
        long_mse, long_mae, long_pred, long_target, _ = test_model(
            long_model, long_test_loader, long_criterion, device, scaler)

        long_term_results['mse'].append(long_mse)
        long_term_results['mae'].append(long_mae)

        print(f"长期预测 - MSE: {long_mse:.6f}, MAE: {long_mae:.6f}")

        # 绘制最后一次实验的结果
        if run == 4:
            plot_predictions(short_pred, short_target, "短期预测结果 (90天)")
            plot_predictions(long_pred, long_target, "长期预测结果 (365天)")

    # 计算平均结果和标准差
    print("\n=== 最终结果 ===")
    print("短期预测 (90天):")
    print(f"MSE: {np.mean(short_term_results['mse']):.6f} ± {np.std(short_term_results['mse']):.6f}")
    print(f"MAE: {np.mean(short_term_results['mae']):.6f} ± {np.std(short_term_results['mae']):.6f}")

    print("\n长期预测 (365天):")
    print(f"MSE: {np.mean(long_term_results['mse']):.6f} ± {np.std(long_term_results['mse']):.6f}")
    print(f"MAE: {np.mean(long_term_results['mae']):.6f} ± {np.std(long_term_results['mae']):.6f}")


if __name__ == "__main__":
    main()