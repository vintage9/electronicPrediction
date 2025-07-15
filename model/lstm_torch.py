# c:\Users\jincan\Documents\研一机器学习\机器学习期末考核\power\lstm_torch.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import datetime
import os

# 设置随机种子以确保结果可复现
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================================================
# PYTORCH LSTM PREDICTION RESULTS SUMMARY (Average of 5 runs)
# ==================================================

# Short-term prediction (90 days):
#   MSE: 228519.4326 ± 76825.7857
#   MAE: 367.3210 ± 49.3370

# Long-term prediction (365 days):
#   MSE: 196169.7817 ± 34921.2745
#   MAE: 343.3337 ± 37.1153

# Results have been saved to 'results' folder.


# 创建保存结果的文件夹
if not os.path.exists('results'):
    os.makedirs('results')

class LSTMModel(nn.Module):
    """PyTorch LSTM模型"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)
        
        # 全连接输出层
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 只使用最后一个时间步的输出
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        
        return out

def load_data(filepath):
    """加载数据并进行基本处理"""
    # 预定义列名
    expected_cols = ['DateTime', 'Global_active_power', 'Global_reactive_power', 
                    'Voltage', 'Global_intensity', 'Sub_metering_1', 
                    'Sub_metering_2', 'Sub_metering_3', 'RR', 'NBJRR1', 
                    'NBJRR5', 'NBJRR10', 'NBJBROU']
    
    # 读取数据，现在训练和测试文件都有列名头部
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Loaded data with header. Shape: {df.shape}")
    print(f"Original columns: {df.columns.tolist()}")
    
    # 如果列名数量或格式不匹配，重新设置列名
    if len(df.columns) != len(expected_cols) or df.columns.tolist() != expected_cols:
        print("Column names don't match expected format, setting standard column names...")
        if len(df.columns) <= len(expected_cols):
            df.columns = expected_cols[:len(df.columns)]
        else:
            # 如果实际列数更多，只使用前13列
            df = df.iloc[:, :len(expected_cols)]
            df.columns = expected_cols
    
    print(f"Final columns: {df.columns.tolist()}")
    
    # 处理日期时间列
    datetime_col = df.columns[0]
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
    
    # 删除无法解析的日期时间行
    df = df.dropna(subset=[datetime_col])
    df.set_index(datetime_col, inplace=True)
    
    # 将非日期时间列转换为数值类型
    for col in df.columns:
        if col != datetime_col:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"Final shape after preprocessing: {df.shape}")
    return df
        

def load_test_data(filepath):
    """加载测试数据"""
    # 现在测试数据也有列名头部，使用相同的加载方式
    return load_data(filepath)

def preprocess_data(df):
    """预处理数据"""
    # 检查和处理缺失值
    print(f"Missing values before imputation: {df.isna().sum().sum()}")
    
    # 只对数值列进行插值
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='time')
    df = df.bfill().ffill()  # 使用新的方法替代已弃用的fillna方法
    
    print(f"Missing values after imputation: {df.isna().sum().sum()}")
    
    # 计算新的属性：sub_metering_remainder
    if all(col in df.columns for col in ['Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']):
        df['Sub_metering_remainder'] = (df['Global_active_power'] * 1000 / 60) - (
            df['Sub_metering_1'] + df['Sub_metering_2'] + df['Sub_metering_3']
        )
        print("Added calculated feature: Sub_metering_remainder")
    else:
        print("Warning: Cannot calculate Sub_metering_remainder due to missing columns")
    
    # 提取时间特征
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['weekday'] = df.index.weekday
    
    return df

def create_sequences(data, target_col, seq_length, pred_length):
    """创建序列数据用于LSTM训练"""
    X, y = [], []
    for i in range(len(data) - seq_length - pred_length + 1):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length:i + seq_length + pred_length, target_col])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape, output_length, units=50):
    """构建PyTorch LSTM模型"""
    input_size = input_shape[1]  # 特征数量
    model = LSTMModel(
        input_size=input_size,
        hidden_size=units,
        num_layers=1,
        output_size=output_length,
        dropout_rate=0.1
    )
    return model.to(device)

def evaluate_predictions(true, pred, scaler=None, target_col=0):
    """评估预测结果"""
    # 如果提供了scaler，则将预测和真实值反归一化
    if scaler is not None:
        # 创建适当形状的数组进行反变换
        true_reshaped = np.zeros((len(true), scaler.scale_.shape[0]))
        true_reshaped[:, target_col] = true
        true = scaler.inverse_transform(true_reshaped)[:, target_col]
        
        pred_reshaped = np.zeros((len(pred), scaler.scale_.shape[0]))
        pred_reshaped[:, target_col] = pred
        pred = scaler.inverse_transform(pred_reshaped)[:, target_col]
    
    # 计算评估指标
    mse = mean_squared_error(true, pred)
    mae = mean_absolute_error(true, pred)
    
    return {
        'MSE': mse,
        'MAE': mae,
    }

def train_and_predict(train_df, test_df, feature_cols, target_col, seq_length, pred_length, epochs=50, batch_size=32, num_runs=5):
    """训练LSTM模型并在独立测试集上进行预测，进行多次运行取平均"""
    # 提取训练数据特征和目标列
    train_data = train_df[feature_cols + [target_col]].values
    test_data = test_df[feature_cols + [target_col]].values
    
    # 检查数据长度是否足够
    min_required_train_length = seq_length + 100  # 训练数据最少需要的长度
    min_required_test_length = seq_length + pred_length  # 测试数据最少需要的长度
    
    if len(train_data) < min_required_train_length:
        print(f"Error: Training data length ({len(train_data)}) is too short")
        print(f"Minimum required length: {min_required_train_length}")
        return None, None, None
        
    if len(test_data) < min_required_test_length:
        print(f"Error: Test data length ({len(test_data)}) is too short")
        print(f"Minimum required length: {min_required_test_length}")
        return None, None, None
    
    # 归一化数据 - 使用训练数据的统计信息
    scaler = MinMaxScaler()

    # 添加：输出原始数据范围
    target_idx = feature_cols.index(target_col) if target_col in feature_cols else len(feature_cols)
    print(f"\nOriginal {target_col} data range:")
    print(f"Training data: min={train_data[:, target_idx].min():.4f}, max={train_data[:, target_idx].max():.4f}")
    print(f"Test data: min={test_data[:, target_idx].min():.4f}, max={train_data[:, target_idx].max():.4f}")
    

    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)
    
    # 确定目标列在缩放后数据中的索引
    target_idx = feature_cols.index(target_col) if target_col in feature_cols else len(feature_cols)
    
    # 创建训练序列 - 使用全部训练数据
    X_train, y_train = create_sequences(train_data_scaled, target_idx, seq_length, pred_length)
    print(f"Training sequences shape: {X_train.shape}, Target shape: {y_train.shape}")
    # 创建测试序列 - 使用测试数据
    X_test, y_test = create_sequences(test_data_scaled, target_idx, seq_length, pred_length)
    print(f"Test sequences shape: {X_test.shape}, Target shape: {y_test.shape}")

    # 检查是否有足够的训练和测试数据
    if len(X_train) == 0:
        print(f"Error: No training sequences generated. Data too short for seq_length={seq_length} and pred_length={pred_length}")
        return None, None, None
    
    if len(X_test) == 0:
        print(f"Error: No test sequences generated.")
        return None, None, None
        
    print(f"Training sequences: {len(X_train)}, Test sequences: {len(X_test)}")
    
    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)
    
    # 进行多次训练和预测
    all_metrics = []
    final_model = None
    final_y_pred = None
    final_train_losses = []
    final_val_losses = []
    
    print(f"\nStarting {num_runs} training runs...")
    
    for run in range(num_runs):
        print(f"\n--- Run {run + 1}/{num_runs} ---")
        
        # 设置不同的随机种子确保每次训练的差异性
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42 + run)
        
        # 构建模型
        model = build_lstm_model((X_train.shape[1], X_train.shape[2]), pred_length, units=300)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 创建数据加载器
        train_size = int(0.8 * len(X_train))
        val_size = len(X_train) - train_size
        
        # 分割训练和验证数据
        train_X, val_X = X_train[:train_size], X_train[train_size:]
        train_y, val_y = y_train[:train_size], y_train[train_size:]
        
        train_dataset = TensorDataset(train_X, train_y)
        val_dataset = TensorDataset(val_X, val_y)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 训练模型
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        # 恢复最佳模型权重
        model.load_state_dict(best_model_state)
        
        # 在测试集上预测
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test)
        
        # 转换回CPU和numpy
        y_pred_np = y_pred.cpu().numpy()
        y_test_np = y_test.cpu().numpy()
        
        print(f"Predictions shape: {y_pred_np.shape}")
        
        # 评估
        metrics = evaluate_predictions(y_test_np.flatten(), y_pred_np.flatten(), scaler, target_idx)
        all_metrics.append(metrics)
        
        print(f"Run {run + 1} Results:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        
        # 保存最后一次运行的结果用于制图
        if run == num_runs - 1:
            final_model = model
            final_y_pred = y_pred_np
            final_train_losses = train_losses
            final_val_losses = val_losses
    
    # 计算平均指标
    avg_metrics = {}
    for metric_name in all_metrics[0].keys():
        values = [m[metric_name] for m in all_metrics]
        avg_metrics[f'avg_{metric_name}'] = np.mean(values)
        avg_metrics[f'std_{metric_name}'] = np.std(values)
    
    print(f"\nAverage Results over {num_runs} runs:")
    for metric_name in all_metrics[0].keys():
        avg_val = avg_metrics[f'avg_{metric_name}']
        std_val = avg_metrics[f'std_{metric_name}']
        print(f"  {metric_name}: {avg_val:.4f} ± {std_val:.4f}")

    # 使用最后一次预测结果绘制图表
    plt.figure(figsize=(12, 6))
    
    num_sequences = y_test_np.shape[0]
    
    # 只绘制第一个预测序列作为示例
    if num_sequences > 0:
        y_test_sample = y_test_np[0]
        y_pred_sample = final_y_pred[0]
        
        # 计算对应的测试日期范围
        start_idx = seq_length
        end_idx = start_idx + pred_length
        test_dates = test_df.index[start_idx:end_idx]
        
        # 确保日期数量与预测数量匹配
        if len(test_dates) != len(y_test_sample):
            min_len = min(len(test_dates), len(y_test_sample))
            test_dates = test_dates[:min_len]
            y_test_sample = y_test_sample[:min_len]
            y_pred_sample = y_pred_sample[:min_len]
        
        # 反归一化
        y_test_reshaped = np.zeros((len(y_test_sample), scaler.scale_.shape[0]))
        y_test_reshaped[:, target_idx] = y_test_sample
        y_test_inv = scaler.inverse_transform(y_test_reshaped)[:, target_idx]
        
        y_pred_reshaped = np.zeros((len(y_pred_sample), scaler.scale_.shape[0]))
        y_pred_reshaped[:, target_idx] = y_pred_sample
        y_pred_inv = scaler.inverse_transform(y_pred_reshaped)[:, target_idx]
        
        plt.plot(test_dates, y_test_inv, label='Actual')
        plt.plot(test_dates, y_pred_inv, label='Predicted', alpha=0.7)
        plt.title(f'PyTorch LSTM Prediction for {pred_length} time steps (Average of {num_runs} runs)')
        plt.xlabel('Date')
        plt.ylabel(target_col)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'results/pytorch_lstm_prediction_{pred_length}_steps_avg{num_runs}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 绘制最后一次训练的损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(final_train_losses, label='Train Loss')
    plt.plot(final_val_losses, label='Validation Loss')
    plt.title(f'PyTorch Model Loss (Final Run of {num_runs} runs)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'results/pytorch_lstm_loss_{pred_length}_steps_avg{num_runs}.png')
    plt.close()
    
    return final_model, scaler, avg_metrics

def main():
    """主函数"""
    # 加载训练数据
    print("Loading training data...")
    df = load_data('train.csv')

    # 预处理数据
    print("Preprocessing training data...")
    df = preprocess_data(df)

    # 加载测试数据
    print("Loading test data...")
    test_df = load_test_data('test.csv')

    print("Preprocessing test data...")
    test_df = preprocess_data(test_df)

    # 由于数据是1分钟粒度，90天和365天的预测步长很大
    # 考虑重采样到天级别以减少计算复杂度，使用不同的聚合方式
    
    # 定义不同类型特征的聚合方式
    agg_dict = {
        'Global_active_power': 'sum',      # 按天取总和
        'Global_reactive_power': 'sum',    # 按天取总和
        'Sub_metering_1': 'sum',          # 按天取总和
        'Sub_metering_2': 'sum',          # 按天取总和
        'Sub_metering_3': 'sum',          # 按天取总和
        'Sub_metering_remainder': 'sum',   # 新增：按天取总和
        'Voltage': 'mean',                # 按天取平均
        'Global_intensity': 'mean',       # 按天取平均
        'RR': 'first',                    # 取当天的任意一个数据
        'NBJRR1': 'first',               # 取当天的任意一个数据
        'NBJRR5': 'first',               # 取当天的任意一个数据
        'NBJRR10': 'first',              # 取当天的任意一个数据
        'NBJBROU': 'first',              # 取当天的任意一个数据
        'hour': 'mean',                   # 时间特征取平均
        'day': 'first',                   # 天特征取第一个
        'month': 'first',                 # 月特征取第一个
        'weekday': 'first'                # 星期特征取第一个
    }
    
    # 只对存在的列进行聚合
    train_agg_dict = {col: method for col, method in agg_dict.items() if col in df.columns}
    test_agg_dict = {col: method for col, method in agg_dict.items() if col in test_df.columns}
    
    df_daily = df.resample('1D').agg(train_agg_dict)
    test_df_daily = test_df.resample('1D').agg(test_agg_dict)

    # 确保所有列都是数值类型
    for col in df_daily.columns:
        if col not in ['hour', 'day', 'month', 'weekday']:
            df_daily[col] = pd.to_numeric(df_daily[col], errors='coerce')
    
    # 对测试数据也进行相同处理
    for col in test_df_daily.columns:
        if col not in ['hour', 'day', 'month', 'weekday']:
            test_df_daily[col] = pd.to_numeric(test_df_daily[col], errors='coerce')
    
    # 特征列和目标列
    feature_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
                   'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'Sub_metering_remainder',
                   'hour', 'day', 'month', 'weekday']

    # 检查哪些特征列在训练和测试数据中都存在
    available_features = [col for col in feature_cols if col in df_daily.columns and col in test_df_daily.columns]
    print(f"Available features: {available_features}")
    
    target_col = 'Global_active_power'  # 预测总有功功率
    
    if target_col not in df_daily.columns or target_col not in test_df_daily.columns:
        print(f"Target column '{target_col}' not found in training or test data.")
        return
    
    # 短期预测：使用过去90天预测未来90天
    print("\nTraining short-term prediction model (past 90 days -> future 90 days)...")
    seq_length = 90  # 使用过去90天的数据
    pred_length = 90  # 预测未来90天
    
    
    short_term_model, short_term_scaler, short_term_metrics = train_and_predict(
        df_daily, test_df_daily, available_features, target_col, seq_length, pred_length, epochs=100, num_runs=5
    )
    
    # 长期预测：使用过去90天预测未来365天
    print("\nTraining long-term prediction model (past 90 days -> future 365 days)...")
    seq_length = 90  # 使用过去90天的数据
    pred_length = 365  # 预测未来365天
    
    result = train_and_predict(
        df_daily, test_df_daily, available_features, target_col, seq_length, pred_length, epochs=100, num_runs=5
    )
    
    if result[0] is not None:
        long_term_model, long_term_scaler, long_term_metrics = result
        # 保存模型
        torch.save(long_term_model.state_dict(), 'results/long_term_pytorch_lstm_model.pth')
    else:
        long_term_model, long_term_scaler, long_term_metrics = None, None, {}
        print("Long-term prediction failed due to insufficient data")
    
    # 保存短期模型
    if short_term_model is not None:
        torch.save(short_term_model.state_dict(), 'results/short_term_pytorch_lstm_model.pth')
    
    # 保存评估结果
    results_summary = {
        'short_term_metrics': short_term_metrics,
        'long_term_metrics': long_term_metrics
    }
    
    # 打印结果摘要
    print("\n" + "="*50)
    print("PYTORCH LSTM PREDICTION RESULTS SUMMARY (Average of 5 runs)")
    print("="*50)
    
    if short_term_metrics:
        print("\nShort-term prediction (90 days):")
        for metric in ['MSE', 'MAE']:
            if f'avg_{metric}' in short_term_metrics:
                avg_val = short_term_metrics[f'avg_{metric}']
                std_val = short_term_metrics[f'std_{metric}']
                print(f"  {metric}: {avg_val:.4f} ± {std_val:.4f}")
    
    if long_term_metrics:
        print("\nLong-term prediction (365 days):")
        for metric in ['MSE', 'MAE']:
            if f'avg_{metric}' in long_term_metrics:
                avg_val = long_term_metrics[f'avg_{metric}']
                std_val = long_term_metrics[f'std_{metric}']
                print(f"  {metric}: {avg_val:.4f} ± {std_val:.4f}")
    
    print("\nResults have been saved to 'results' folder.")

if __name__ == "__main__":
    main()