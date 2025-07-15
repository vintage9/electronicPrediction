import torch
import torch.nn as nn
import math
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Assuming data_processor.py is available and correctly imported
# from data_processor import TimeSeriesDataProcessor, TimeSeriesDataset, create_data_loaders

# --- Re-defining the data_processor classes here for completeness and direct execution ---
# (In a real project, you would just import them)

class TimeSeriesDataProcessor:
    def __init__(self, train_path, test_path, sequence_length=90, prediction_length=90):
        self.train_path = train_path
        self.test_path = test_path
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length

        self.feature_columns = [
            'Global_active_power', 'global_reactive_power', 'voltage',
            'global_intensity', 'sub_metering_1', 'sub_metering_2', 'sub_metering_3'
        ]
        self.target_column = 'Global_active_power'
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

    def load_data(self):
        train_data = pd.read_csv(self.train_path)
        test_data = pd.read_csv(self.test_path)
        return train_data, test_data

    def handle_missing(self, df):
        df = df.fillna(method='ffill').fillna(method='bfill')
        return df

    def add_time_features(self, df, encoding='sincos'):
        # Ensure 'Date' column is present before converting to datetime
        if 'Date' not in df.columns:
            # Placeholder for 'Date' if not present, assuming index could be date-like
            # Or raise an error if 'Date' is strictly required
            df['Date'] = pd.to_datetime(df.index) # This might need adjustment based on actual data
        else:
            df['Date'] = pd.to_datetime(df['Date'])

        df['dayofweek'] = df['Date'].dt.dayofweek
        df['month'] = df['Date'].dt.month
        df['season'] = df['month'] % 12 // 3

        if encoding == 'sincos':
            df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
            df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            # Only add to feature_columns if not already present
            new_time_features = ['dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos']
            for feature in new_time_features:
                if feature not in self.feature_columns:
                    self.feature_columns.append(feature)
        else:
            new_time_features = ['dayofweek', 'month', 'season']
            for feature in new_time_features:
                if feature not in self.feature_columns:
                    self.feature_columns.append(feature)

        return df

    def normalize_features(self, train_df, test_df):
        available = [col for col in self.feature_columns if col in train_df.columns]
        X_train = train_df[available].values
        y_train = train_df[self.target_column].values
        X_test = test_df[available].values
        y_test = test_df[self.target_column].values

        X_train = self.feature_scaler.fit_transform(X_train)
        X_test = self.feature_scaler.transform(X_test)
        y_train = self.target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test = self.target_scaler.transform(y_test.reshape(-1, 1)).flatten()

        return X_train, y_train, X_test, y_test, available

    def create_sequences(self, X, y):
        Xs, Ys = [], []
        total_len = self.sequence_length + self.prediction_length
        for i in range(len(X) - total_len + 1):
            Xs.append(X[i:i + self.sequence_length])
            Ys.append(y[i + self.sequence_length : i + self.sequence_length + self.prediction_length])
        return np.array(Xs), np.array(Ys)

    def process_data(self, add_time_features=True, time_encoding="sincos"):
        train_df, test_df = self.load_data()
        train_df = self.handle_missing(train_df)
        test_df = self.handle_missing(test_df)

        # Apply daily aggregation as per problem statement
        # For 'Global_active_power', 'global_reactive_power', 'sub_metering_1', 'sub_metering_2' sum daily
        # For 'voltage', 'global_intensity' average daily
        # For 'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU' take any daily (e.g., first or last)

        # Ensure 'Date' column is set as index for resampling, assuming it exists or is created
        train_df['Date'] = pd.to_datetime(train_df['Date'])
        test_df['Date'] = pd.to_datetime(test_df['Date'])
        train_df = train_df.set_index('Date')
        test_df = test_df.set_index('Date')

        # Define aggregation rules
        agg_funcs = {
            'Global_active_power': 'sum',
            'global_reactive_power': 'sum',
            'sub_metering_1': 'sum',
            'sub_metering_2': 'sum',
            'sub_metering_3': 'sum',
            'voltage': 'mean',
            'global_intensity': 'mean',
            # Assuming weather data columns are also present and need aggregation
            # 'RR': 'first',
            # 'NBJRR1': 'first',
            # 'NBJRR5': 'first',
            # 'NBJRR10': 'first',
            # 'NBJBROU': 'first'
        }
        # Filter agg_funcs to include only columns present in the dataframe
        train_agg_funcs = {k: v for k, v in agg_funcs.items() if k in train_df.columns}
        test_agg_funcs = {k: v for k, v in agg_funcs.items() if k in test_df.columns}


        train_df = train_df.resample('D').agg(train_agg_funcs) # Resample to daily
        test_df = test_df.resample('D').agg(test_agg_funcs)   # Resample to daily

        # Re-handle missing after resampling if any new NaNs introduced
        train_df = self.handle_missing(train_df)
        test_df = self.handle_missing(test_df)

        # Reset index to make 'Date' a column again for add_time_features
        train_df = train_df.reset_index()
        test_df = test_df.reset_index()


        if add_time_features:
            train_df = self.add_time_features(train_df, encoding=time_encoding)
            test_df = self.add_time_features(test_df, encoding=time_encoding)

        X_train, y_train, X_test, y_test, feat_cols = self.normalize_features(train_df, test_df)
        X_train, y_train = self.create_sequences(X_train, y_train)
        X_test, y_test = self.create_sequences(X_test, y_test)

        self.processed_data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'feature_names': feat_cols,
            'n_features': len(feat_cols)
        }
        return self.processed_data

    def inverse_transform_target(self, y):
        # Ensure y is a numpy array before reshaping and inverse transforming
        if not isinstance(y, np.ndarray):
            y = y.numpy()
        return self.target_scaler.inverse_transform(y.reshape(-1, 1)).flatten()


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        # Ensure X and y are already numpy arrays before converting to torch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_data_loaders(data, batch_size=32, shuffle=True):
    train_set = TimeSeriesDataset(data['X_train'], data['y_train'])
    test_set = TimeSeriesDataset(data['X_test'], data['y_test'])
    return DataLoader(train_set, batch_size=batch_size, shuffle=shuffle), DataLoader(test_set, batch_size=batch_size)

# --- End of data_processor classes ---


# 1. Positional Encoding
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
        # x shape: (sequence_length, batch_size, d_model)
        return x + self.pe[:x.size(0), :]

# 2. Transformer Model
class TransformerPredictor(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, n_encoder_layers, n_decoder_layers, dim_feedforward, dropout=0.1, prediction_length=90):
        super(TransformerPredictor, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.prediction_length = prediction_length

        self.input_linear = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward, dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, n_decoder_layers)

        self.output_linear = nn.Linear(d_model, 1) # Output a single value for target prediction

    def forward(self, src):
        # src shape: (batch_size, sequence_length, input_dim)
        # For Transformer, input_dim is the number of features.
        # Target input for decoder (tgt) will be shifted ground truth or an initial token.
        # For time series forecasting, common to use a 'start of sequence' token or just predict sequentially.
        # Here, we'll predict directly from the encoder output after some manipulation or
        # use a simple autoregressive approach in the decoder.

        # Input Embedding and Positional Encoding
        src = self.input_linear(src) * math.sqrt(self.d_model) # (batch_size, seq_len, d_model)
        src = self.positional_encoding(src.permute(1, 0, 2)).permute(1, 0, 2) # (batch_size, seq_len, d_model)

        # Encoder
        memory = self.transformer_encoder(src) # (batch_size, seq_len, d_model)

        # Decoder Input (for a prediction_length output, we need prediction_length tokens)
        # A common approach for time series prediction with Transformer decoder is to use a "dummy" or
        # learnable token for each prediction step, or an autoregressive setup.
        # For simplicity here, let's assume we want to predict a sequence of `prediction_length` values
        # based on the last encoded state or by feeding a sequence of zeros/learnable embeddings.

        # Simplistic approach: use the last state of the encoder output to initiate decoder or
        # create a target sequence of zeros for the decoder to learn to fill.
        # Let's create a "target" sequence of zeros with positional encoding for the decoder.
        # This acts as a query for the decoder to generate future steps.
        
        # tgt shape: (batch_size, prediction_length, d_model)
        # In this approach, the decoder is effectively learning to generate a sequence of predictions
        # given the context from the encoder (memory).
        tgt = torch.zeros((src.size(0), self.prediction_length, self.d_model), device=src.device)
        tgt = self.positional_encoding(tgt.permute(1, 0, 2)).permute(1, 0, 2) # Add positional encoding to decoder input

        # Decoder
        # The decoder will attend to the encoder's output (memory) and its own (masked) input (tgt).
        # We need a tgt_mask to prevent attention to future elements in the decoder input sequence.
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.prediction_length).to(src.device)

        # Output from decoder
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask) # (batch_size, prediction_length, d_model)

        # Final linear layer to get the prediction for each step
        output = self.output_linear(output) # (batch_size, prediction_length, 1)

        return output.squeeze(-1) # (batch_size, prediction_length)

# Training and Evaluation Functions
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model(model, test_loader, criterion, device, data_processor):
    model.eval()
    total_mse = 0
    total_mae = 0
    predictions = []
    actuals = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Inverse transform for evaluation metrics
            output_inv = data_processor.inverse_transform_target(output.cpu().numpy())
            target_inv = data_processor.inverse_transform_target(target.cpu().numpy())

            # Reshape to ensure correct calculation for batches
            output_inv_flat = output_inv.reshape(-1, model.prediction_length)
            target_inv_flat = target_inv.reshape(-1, model.prediction_length)

            total_mse += np.mean((output_inv_flat - target_inv_flat)**2) * data.size(0)
            total_mae += np.mean(np.abs(output_inv_flat - target_inv_flat)) * data.size(0)

            predictions.extend(output_inv_flat.tolist())
            actuals.extend(target_inv_flat.tolist())

    avg_mse = total_mse / len(test_loader.dataset)
    avg_mae = total_mae / len(test_loader.dataset)
    return avg_mse, avg_mae, predictions, actuals

# Main execution block
if __name__ == "__main__":
    # Define paths to your data files
    train_file_path = 'datasets/daily_power_train2.csv' # Replace with your actual train.csv path
    test_file_path = 'datasets/daily_power_test2.csv'   # Replace with your actual test.csv path

    # --- Short-term Prediction (90 days) ---
    print("--- Running Short-term Prediction (90 days) ---")
    sequence_length_short = 90
    prediction_length_short = 90

    data_processor_short = TimeSeriesDataProcessor(
        train_path=train_file_path,
        test_path=test_file_path,
        sequence_length=sequence_length_short,
        prediction_length=prediction_length_short
    )
    processed_data_short = data_processor_short.process_data(add_time_features=True, time_encoding="sincos")

    train_loader_short, test_loader_short = create_data_loaders(processed_data_short, batch_size=32)

    input_dim_short = processed_data_short['n_features']
    d_model = 64
    n_heads = 4
    n_encoder_layers = 2    
    n_decoder_layers = 2
    dim_feedforward = 128
    dropout = 0.1
    learning_rate = 0.001
    num_epochs = 50 # You might need to adjust this

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run multiple experiments for short-term prediction
    mse_scores_short = []
    mae_scores_short = []

    for i in range(5): # At least five experiments
        print(f"\nShort-term Experiment {i+1}/5:")
        model_short = TransformerPredictor(
            input_dim=input_dim_short,
            d_model=d_model,
            n_heads=n_heads,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            prediction_length=prediction_length_short
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model_short.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            train_loss = train_model(model_short, train_loader_short, criterion, optimizer, device)
            # print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

        mse_short, mae_short, preds_short, actuals_short = evaluate_model(model_short, test_loader_short, criterion, device, data_processor_short)
        mse_scores_short.append(mse_short)
        mae_scores_short.append(mae_short)
        print(f"Short-term Prediction - MSE: {mse_short:.4f}, MAE: {mae_short:.4f}")

    print("\n--- Short-term Prediction Results (90 days) ---")
    print(f"Average MSE: {np.mean(mse_scores_short):.4f} +/- {np.std(mse_scores_short):.4f}")
    print(f"Average MAE: {np.mean(mae_scores_short):.4f} +/- {np.std(mae_scores_short):.4f}")
    
    # 保存短期预测结果到文件
    short_results = {
        'MSE_scores': mse_scores_short,
        'MAE_scores': mae_scores_short,
        'MSE_mean': np.mean(mse_scores_short),
        'MSE_std': np.std(mse_scores_short),
        'MAE_mean': np.mean(mae_scores_short),
        'MAE_std': np.std(mae_scores_short)
    }
    
    # 保存最后一次实验的预测结果用于绘图
    final_preds_short = preds_short
    final_actuals_short = actuals_short

    # --- Long-term Prediction (365 days) ---
    print("\n--- Running Long-term Prediction (365 days) ---")
    sequence_length_long = 90 # Still use 90 days of input
    prediction_length_long = 365 # Predict 365 days into the future

    data_processor_long = TimeSeriesDataProcessor(
        train_path=train_file_path,
        test_path=test_file_path,
        sequence_length=sequence_length_long,
        prediction_length=prediction_length_long
    )
    processed_data_long = data_processor_long.process_data(add_time_features=True, time_encoding="sincos")

    train_loader_long, test_loader_long = create_data_loaders(processed_data_long, batch_size=32)

    input_dim_long = processed_data_long['n_features'] # Should be same as short-term input_dim
    # Model parameters can be the same, but note:
    # "长期预测的模型参数不能用于短期预测" -> "long-term prediction model parameters cannot be used for short-term prediction"
    # So we train a separate model instance.

    # Run multiple experiments for long-term prediction
    mse_scores_long = []
    mae_scores_long = []

    for i in range(5): # At least five experiments
        print(f"\nLong-term Experiment {i+1}/5:")
        model_long = TransformerPredictor(
            input_dim=input_dim_long,
            d_model=d_model,
            n_heads=n_heads,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            prediction_length=prediction_length_long # Crucially, change prediction_length
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model_long.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            train_loss = train_model(model_long, train_loader_long, criterion, optimizer, device)
            # print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

        mse_long, mae_long, preds_long, actuals_long = evaluate_model(model_long, test_loader_long, criterion, device, data_processor_long)
        mse_scores_long.append(mse_long)
        mae_scores_long.append(mae_long)
        print(f"Long-term Prediction - MSE: {mse_long:.4f}, MAE: {mae_long:.4f}")

    print("\n--- Long-term Prediction Results (365 days) ---")
    print(f"Average MSE: {np.mean(mse_scores_long):.4f} +/- {np.std(mse_scores_long):.4f}")
    print(f"Average MAE: {np.mean(mae_scores_long):.4f} +/- {np.std(mae_scores_long):.4f}")

    # 保存长期预测结果到文件
    long_results = {
        'MSE_scores': mse_scores_long,
        'MAE_scores': mae_scores_long,
        'MSE_mean': np.mean(mse_scores_long),
        'MSE_std': np.std(mse_scores_long),
        'MAE_mean': np.mean(mae_scores_long),
        'MAE_std': np.std(mae_scores_long)
    }
    
    # 保存实验结果到txt文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"model/results/transformer_experiment_results_{timestamp}.txt"
    
    with open(results_filename, 'w', encoding='utf-8') as f:
        f.write("Transformer 电力预测实验结果\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 短期预测结果
        f.write("短期预测结果 (90天):\n")
        f.write("-" * 30 + "\n")
        f.write(f"实验轮数: 5轮\n")
        f.write(f"序列长度: {sequence_length_short}天\n")
        f.write(f"预测长度: {prediction_length_short}天\n\n")
        
        f.write("各轮MSE结果:\n")
        for i, mse in enumerate(mse_scores_short, 1):
            f.write(f"  第{i}轮: {mse:.6f}\n")
        f.write(f"MSE平均值: {short_results['MSE_mean']:.6f}\n")
        f.write(f"MSE标准差: {short_results['MSE_std']:.6f}\n\n")
        
        f.write("各轮MAE结果:\n")
        for i, mae in enumerate(mae_scores_short, 1):
            f.write(f"  第{i}轮: {mae:.6f}\n")
        f.write(f"MAE平均值: {short_results['MAE_mean']:.6f}\n")
        f.write(f"MAE标准差: {short_results['MAE_std']:.6f}\n\n")
        
        # 长期预测结果
        f.write("长期预测结果 (365天):\n")
        f.write("-" * 30 + "\n")
        f.write(f"实验轮数: 5轮\n")
        f.write(f"序列长度: {sequence_length_long}天\n")
        f.write(f"预测长度: {prediction_length_long}天\n\n")
        
        f.write("各轮MSE结果:\n")
        for i, mse in enumerate(mse_scores_long, 1):
            f.write(f"  第{i}轮: {mse:.6f}\n")
        f.write(f"MSE平均值: {long_results['MSE_mean']:.6f}\n")
        f.write(f"MSE标准差: {long_results['MSE_std']:.6f}\n\n")
        
        f.write("各轮MAE结果:\n")
        for i, mae in enumerate(mae_scores_long, 1):
            f.write(f"  第{i}轮: {mae:.6f}\n")
        f.write(f"MAE平均值: {long_results['MAE_mean']:.6f}\n")
        f.write(f"MAE标准差: {long_results['MAE_std']:.6f}\n\n")
        
        # 模型参数
        f.write("模型参数:\n")
        f.write("-" * 30 + "\n")
        f.write(f"d_model: {d_model}\n")
        f.write(f"n_heads: {n_heads}\n")
        f.write(f"n_encoder_layers: {n_encoder_layers}\n")
        f.write(f"n_decoder_layers: {n_decoder_layers}\n")
        f.write(f"dim_feedforward: {dim_feedforward}\n")
        f.write(f"dropout: {dropout}\n")
        f.write(f"learning_rate: {learning_rate}\n")
        f.write(f"num_epochs: {num_epochs}\n")
        f.write(f"batch_size: 32\n")
        
    print(f"\n实验结果已保存到: {results_filename}")

    # 绘制预测结果对比图
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime, timedelta
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建时间轴（假设从某个起始日期开始）
    start_date = datetime(2023, 1, 1)  # 可以根据实际数据调整起始日期
    
    # 90天预测结果图
    plt.figure(figsize=(15, 10))
    
    # 选择第一个样本进行可视化
    sample_idx = 0
    if len(final_preds_short) > sample_idx and len(final_actuals_short) > sample_idx:
        plt.subplot(2, 1, 1)
        
        # 创建90天的时间序列
        dates_90 = [start_date + timedelta(days=i) for i in range(90)]
        
        plt.plot(dates_90, final_actuals_short[sample_idx], 'b-', label='实际值', linewidth=2, alpha=0.8)
        plt.plot(dates_90, final_preds_short[sample_idx], 'r--', label='预测值', linewidth=2, alpha=0.8)
        
        plt.title('90天电力消耗预测结果对比', fontsize=16, fontweight='bold')
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('全球有功功率 (kW)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 格式化x轴日期显示
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.xticks(rotation=45)
        
        # 添加统计信息
        mse_90 = np.mean((np.array(final_actuals_short[sample_idx]) - np.array(final_preds_short[sample_idx]))**2)
        mae_90 = np.mean(np.abs(np.array(final_actuals_short[sample_idx]) - np.array(final_preds_short[sample_idx])))
        plt.text(0.02, 0.98, f'MSE: {mse_90:.4f}\nMAE: {mae_90:.4f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 365天预测结果图
    if len(preds_long) > sample_idx and len(actuals_long) > sample_idx:
        plt.subplot(2, 1, 2)
        
        # 创建365天的时间序列
        dates_365 = [start_date + timedelta(days=i) for i in range(365)]
        
        plt.plot(dates_365, actuals_long[sample_idx], 'b-', label='实际值', linewidth=2, alpha=0.8)
        plt.plot(dates_365, preds_long[sample_idx], 'r--', label='预测值', linewidth=2, alpha=0.8)
        
        plt.title('365天电力消耗预测结果对比', fontsize=16, fontweight='bold')
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('全球有功功率 (kW)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 格式化x轴日期显示
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        
        # 添加统计信息
        mse_365 = np.mean((np.array(actuals_long[sample_idx]) - np.array(preds_long[sample_idx]))**2)
        mae_365 = np.mean(np.abs(np.array(actuals_long[sample_idx]) - np.array(preds_long[sample_idx])))
        plt.text(0.02, 0.98, f'MSE: {mse_365:.4f}\nMAE: {mae_365:.4f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('model/results/transformer_prediction_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 绘制误差分布图
    plt.figure(figsize=(15, 6))
    
    # 90天误差分布
    if len(final_preds_short) > sample_idx and len(final_actuals_short) > sample_idx:
        plt.subplot(1, 2, 1)
        error_90 = np.array(final_actuals_short[sample_idx]) - np.array(final_preds_short[sample_idx])
        plt.hist(error_90, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('90天预测误差分布', fontsize=14, fontweight='bold')
        plt.xlabel('预测误差 (实际值 - 预测值)', fontsize=12)
        plt.ylabel('频次', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息
        plt.axvline(np.mean(error_90), color='red', linestyle='--', label=f'均值: {np.mean(error_90):.4f}')
        plt.axvline(np.median(error_90), color='green', linestyle='--', label=f'中位数: {np.median(error_90):.4f}')
        plt.legend()
    
    # 365天误差分布
    if len(preds_long) > sample_idx and len(actuals_long) > sample_idx:
        plt.subplot(1, 2, 2)
        error_365 = np.array(actuals_long[sample_idx]) - np.array(preds_long[sample_idx])
        plt.hist(error_365, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.title('365天预测误差分布', fontsize=14, fontweight='bold')
        plt.xlabel('预测误差 (实际值 - 预测值)', fontsize=12)
        plt.ylabel('频次', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息
        plt.axvline(np.mean(error_365), color='red', linestyle='--', label=f'均值: {np.mean(error_365):.4f}')
        plt.axvline(np.median(error_365), color='green', linestyle='--', label=f'中位数: {np.median(error_365):.4f}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('model/results/transformer_error_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 绘制散点图显示预测精度
    plt.figure(figsize=(15, 6))
    
    # 90天散点图
    if len(final_preds_short) > sample_idx and len(final_actuals_short) > sample_idx:
        plt.subplot(1, 2, 1)
        plt.scatter(final_actuals_short[sample_idx], final_preds_short[sample_idx], alpha=0.6, color='blue')
        
        # 添加完美预测线
        min_val = min(min(final_actuals_short[sample_idx]), min(final_preds_short[sample_idx]))
        max_val = max(max(final_actuals_short[sample_idx]), max(final_preds_short[sample_idx]))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='完美预测线')
        
        plt.title('90天预测精度散点图', fontsize=14, fontweight='bold')
        plt.xlabel('实际值', fontsize=12)
        plt.ylabel('预测值', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 计算相关系数
        correlation_90 = np.corrcoef(final_actuals_short[sample_idx], final_preds_short[sample_idx])[0, 1]
        plt.text(0.05, 0.95, f'相关系数: {correlation_90:.4f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 365天散点图
    if len(preds_long) > sample_idx and len(actuals_long) > sample_idx:
        plt.subplot(1, 2, 2)
        plt.scatter(actuals_long[sample_idx], preds_long[sample_idx], alpha=0.6, color='red')
        
        # 添加完美预测线
        min_val = min(min(actuals_long[sample_idx]), min(preds_long[sample_idx]))
        max_val = max(max(actuals_long[sample_idx]), max(preds_long[sample_idx]))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='完美预测线')
        
        plt.title('365天预测精度散点图', fontsize=14, fontweight='bold')
        plt.xlabel('实际值', fontsize=12)
        plt.ylabel('预测值', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 计算相关系数
        correlation_365 = np.corrcoef(actuals_long[sample_idx], preds_long[sample_idx])[0, 1]
        plt.text(0.05, 0.95, f'相关系数: {correlation_365:.4f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('model/results/transformer_accuracy_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n图表已保存到 model/results/ 目录下")
    print("- transformer_prediction_comparison.png: 预测值与实际值对比")
    print("- transformer_error_distribution.png: 误差分布图")
    print("- transformer_accuracy_scatter.png: 预测精度散点图")
    print(f"- {results_filename}: 详细实验结果数据")