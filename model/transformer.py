import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Embedding
from tensorflow.keras.callbacks import EarlyStopping
import datetime
import os

# 设置随机种子以确保结果可复现
np.random.seed(42)
tf.random.set_seed(42)

# 创建保存结果的文件夹
if not os.path.exists('results'):
    os.makedirs('results')

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
    """创建序列数据用于Transformer训练"""
    X, y = [], []
    for i in range(len(data) - seq_length - pred_length + 1):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length:i + seq_length + pred_length, target_col])
    return np.array(X), np.array(y)

def positional_encoding(seq_len, d_model):
    """创建正弦余弦位置编码"""
    angle_rads = np.arange(seq_len)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
    
    # 对偶数索引应用sin
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # 对奇数索引应用cos
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)

def scaled_dot_product_attention(q, k, v, mask=None):
    """计算缩放点积注意力"""
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    
    # 缩放 matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    # 添加掩码
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    # softmax
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    
    return output, attention_weights

class MultiHeadAttentionLayer(tf.keras.layers.Layer):
    """多头注意力层"""
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
        })
        return config
        
    def build(self, input_shape):
        self.wq = tf.keras.layers.Dense(self.d_model)
        self.wk = tf.keras.layers.Dense(self.d_model)
        self.wv = tf.keras.layers.Dense(self.d_model)
        self.dense = tf.keras.layers.Dense(self.d_model)
        super(MultiHeadAttentionLayer, self).build(input_shape)
        
    def split_heads(self, x, batch_size):
        """分拆最后一个维度到 (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        
        concat_attention = tf.reshape(scaled_attention, 
                                    (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
        return output

def point_wise_feed_forward_network(d_model, dff):
    """点式前馈网络"""
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

class EncoderLayer(tf.keras.layers.Layer):
    """编码器层"""
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "rate": self.rate,
        })
        return config
        
    def build(self, input_shape):
        self.mha = MultiHeadAttentionLayer(self.d_model, self.num_heads)
        self.ffn = point_wise_feed_forward_network(self.d_model, self.dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(self.rate)
        self.dropout2 = tf.keras.layers.Dropout(self.rate)
        super(EncoderLayer, self).build(input_shape)
    
    def call(self, x, training=None, mask=None):
        attn_output = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        
        return out2

class Encoder(tf.keras.layers.Layer):
    """编码器"""
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.maximum_position_encoding = maximum_position_encoding
        self.rate = rate
        self.num_heads = num_heads
        self.dff = dff
        self.input_vocab_size = input_vocab_size
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "input_vocab_size": self.input_vocab_size,
            "maximum_position_encoding": self.maximum_position_encoding,
            "rate": self.rate,
        })
        return config
        
    def build(self, input_shape):
        self.input_embedding = tf.keras.layers.Dense(self.d_model)
        self.pos_encoding = positional_encoding(self.maximum_position_encoding, self.d_model)
        
        self.enc_layers = [EncoderLayer(self.d_model, self.num_heads, self.dff, self.rate) 
                          for _ in range(self.num_layers)]
        
        self.dropout = tf.keras.layers.Dropout(self.rate)
        super(Encoder, self).build(input_shape)
        
    def call(self, x, training=None, mask=None):
        seq_len = tf.shape(x)[1]
        
        # 输入嵌入和位置编码
        x = self.input_embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        
        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training, mask=mask)
        
        return x  # (batch_size, input_seq_len, d_model)

def create_transformer_model(input_shape, d_model=128, num_heads=8, num_layers=4, dff=512, dropout_rate=0.1):
    """创建标准Transformer模型"""
    
    inputs = tf.keras.Input(shape=input_shape)
    
    # 编码器
    encoder = Encoder(num_layers=num_layers, 
                     d_model=d_model, 
                     num_heads=num_heads, 
                     dff=dff,
                     input_vocab_size=input_shape[1],
                     maximum_position_encoding=input_shape[0], 
                     rate=dropout_rate)
    
    enc_output = encoder(inputs, training=True)  # (batch_size, inp_seq_len, d_model)
    
    # 全局平均池化
    pooled = tf.keras.layers.GlobalAveragePooling1D()(enc_output)
    
    # 输出层
    outputs = tf.keras.layers.Dense(1, activation='linear')(pooled)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model

def evaluate_predictions(true, pred, scaler=None, target_col=0):
    """评估预测结果"""
    # 确保pred是一维数组
    if pred.ndim > 1:
        pred = pred.flatten()
    
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

def create_single_step_sequences(data, target_col, seq_length):
    """为单步预测创建序列数据"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length, target_col])  # 下一个时间步的单个值
    return np.array(X), np.array(y)

def train_and_predict(train_df, test_df, feature_cols, target_col, seq_length, pred_length, epochs=50, batch_size=32):
    """训练标准Transformer模型并进行多步预测"""
    # 提取训练数据特征和目标列
    train_data = train_df[feature_cols + [target_col]].values
    test_data = test_df[feature_cols + [target_col]].values
    
    # 检查数据长度是否足够
    min_required_train_length = seq_length + 100
    min_required_test_length = seq_length + pred_length
    
    if len(train_data) < min_required_train_length:
        print(f"Error: Training data length ({len(train_data)}) is too short")
        print(f"Minimum required length: {min_required_train_length}")
        return None, None, None
        
    if len(test_data) < min_required_test_length:
        print(f"Error: Test data length ({len(test_data)}) is too short")
        print(f"Minimum required length: {min_required_test_length}")
        return None, None, None
    
    # 归一化数据
    scaler = MinMaxScaler()
    target_idx = feature_cols.index(target_col) if target_col in feature_cols else len(feature_cols)
    
    print(f"\nOriginal {target_col} data range:")
    print(f"Training data: min={train_data[:, target_idx].min():.4f}, max={train_data[:, target_idx].max():.4f}")
    print(f"Test data: min={test_data[:, target_idx].min():.4f}, max={test_data[:, target_idx].max():.4f}")
    
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)
    
    # 创建单步预测训练序列
    X_train, y_train = create_single_step_sequences(train_data_scaled, target_idx, seq_length)
    
    print(f"Training sequences: {len(X_train)}")
    print(f"Input shape: {X_train.shape}, Output shape: {y_train.shape}")
    
    # 构建标准Transformer模型
    model = create_transformer_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        d_model=128,
        num_heads=8,
        num_layers=4,
        dff=512,
        dropout_rate=0.1
    )
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    print("Using standard Transformer model")
    print(f"Model parameters: {model.count_params()}")
    
    # 设置回调函数
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=20, 
        restore_best_weights=True,
        verbose=1
    )
    
    # 训练模型
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # 多步预测：使用递归方式预测多个时间步
    print(f"Making {pred_length}-step predictions on test set...")
    
    # 从测试集开始位置开始预测
    current_sequence = test_data_scaled[0:seq_length].copy()
    predictions = []
    
    for step in range(pred_length):
        # 准备输入
        X_input = current_sequence.reshape(1, seq_length, -1)
        
        # 预测下一步
        next_pred = model.predict(X_input, verbose=0)
        
        # 确保预测结果是标量
        if isinstance(next_pred, np.ndarray):
            if next_pred.ndim > 0:
                next_pred = next_pred.flatten()[0]  # 取第一个元素作为标量
        
        # 更新序列：移除第一个时间步，添加预测值
        new_row = current_sequence[-1].copy()  # 复制最后一行的特征
        new_row[target_idx] = next_pred  # 更新目标列为预测值
        
        current_sequence = np.vstack([current_sequence[1:], new_row])
        predictions.append(next_pred)
    
    predictions = np.array(predictions)
    
    # 获取真实值用于评估
    if len(test_data_scaled) >= seq_length + pred_length:
        y_test = test_data_scaled[seq_length:seq_length + pred_length, target_idx]
    else:
        available_steps = len(test_data_scaled) - seq_length
        y_test = test_data_scaled[seq_length:seq_length + available_steps, target_idx]
        predictions = predictions[:available_steps]
        pred_length = available_steps
    
    # 评估
    metrics = evaluate_predictions(y_test, predictions, scaler, target_idx)
    print(f"\nEvaluation Metrics for {pred_length} time steps prediction:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # 绘制结果
    plt.figure(figsize=(15, 8))
    
    # 计算对应的测试日期范围
    start_idx = seq_length
    end_idx = start_idx + len(predictions)
    test_dates = test_df.index[start_idx:end_idx]
    
    # 反归一化
    y_test_reshaped = np.zeros((len(y_test), scaler.scale_.shape[0]))
    y_test_reshaped[:, target_idx] = y_test
    y_test_inv = scaler.inverse_transform(y_test_reshaped)[:, target_idx]
    
    # 确保predictions是一维数组
    predictions_flat = predictions.flatten() if predictions.ndim > 1 else predictions
    pred_reshaped = np.zeros((len(predictions_flat), scaler.scale_.shape[0]))
    pred_reshaped[:, target_idx] = predictions_flat
    pred_inv = scaler.inverse_transform(pred_reshaped)[:, target_idx]
    
    plt.plot(test_dates, y_test_inv, label='Actual', linewidth=2)
    plt.plot(test_dates, pred_inv, label='Predicted', alpha=0.8, linewidth=2)
    plt.title(f'Standard Transformer Multi-Step Prediction ({pred_length} steps)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(target_col, fontsize=12)
    plt.legend(fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/standard_transformer_prediction_{pred_length}_steps.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制训练历史
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE', linewidth=2)
    plt.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    plt.title('Model MAE', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/standard_transformer_training_{pred_length}_steps.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return model, scaler, metrics

def run_multiple_experiments(train_df, test_df, available_features, target_col, seq_length, pred_length, n_runs=5, epochs=150):
    """运行多次实验并计算统计结果"""
    all_metrics = []
    
    for run in range(1, n_runs + 1):
        print(f"\n{'='*50}")
        print(f"RUN {run}/{n_runs} - {pred_length} days prediction")
        print(f"{'='*50}")
        
        # 设置不同的随机种子以确保每次运行的差异
        np.random.seed(42 + run)
        tf.random.set_seed(42 + run)
        
        model, scaler, metrics = train_and_predict(
            train_df, test_df, available_features, target_col, seq_length, pred_length, epochs=epochs
        )
        
        if metrics is not None:
            all_metrics.append(metrics)
            print(f"Run {run} Results - MSE: {metrics['MSE']:.4f}, MAE: {metrics['MAE']:.4f}")
        else:
            print(f"Run {run} failed!")
    
    if all_metrics:
        # 计算统计结果
        mse_values = [m['MSE'] for m in all_metrics]
        mae_values = [m['MAE'] for m in all_metrics]
        
        stats = {
            'MSE_mean': np.mean(mse_values),
            'MSE_std': np.std(mse_values),
            'MAE_mean': np.mean(mae_values),
            'MAE_std': np.std(mae_values),
            'MSE_values': mse_values,
            'MAE_values': mae_values
        }
        
        print(f"\n{'='*50}")
        print(f"SUMMARY FOR {pred_length} DAYS PREDICTION ({n_runs} runs)")
        print(f"{'='*50}")
        print(f"MSE: {stats['MSE_mean']:.4f} ± {stats['MSE_std']:.4f}")
        print(f"MAE: {stats['MAE_mean']:.4f} ± {stats['MAE_std']:.4f}")
        
        return stats
    else:
        return None

def main():
    """主函数"""
    # 加载训练数据
    print("Loading training data...")
    df = load_data('../datasets/train.csv')

    # 预处理数据
    print("Preprocessing training data...")
    df = preprocess_data(df)

    # 加载测试数据
    print("Loading test data...")
    test_df = load_test_data('../datasets/test.csv')

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
    
    # 短期预测：使用过去90天预测未来90天，进行5轮实验
    print("\n" + "="*70)
    print("TRAINING STANDARD TRANSFORMER MODEL FOR SHORT-TERM PREDICTION")
    print("="*70)
    print("Configuration: past 90 days -> future 90 days (recursive prediction)")
    print("Running 5 experiments...")
    
    seq_length = 90
    pred_length = 90
    n_runs = 5
    
    short_term_stats = run_multiple_experiments(
        df_daily, test_df_daily, available_features, target_col, 
        seq_length, pred_length, n_runs=n_runs, epochs=150
    )
    
    # 长期预测：使用过去90天预测未来365天，进行5轮实验
    print("\n" + "="*70)
    print("TRAINING STANDARD TRANSFORMER MODEL FOR LONG-TERM PREDICTION")
    print("="*70)
    print("Configuration: past 90 days -> future 365 days (recursive prediction)")
    print("Running 5 experiments...")
    
    seq_length = 90
    pred_length = 365
    
    # 检查测试数据是否足够长期预测
    if len(test_df_daily) < seq_length + pred_length:
        print(f"Test data too short for 365-day prediction. Adjusting prediction length.")
        max_pred_length = max(1, len(test_df_daily) - seq_length - 10)
        pred_length = min(pred_length, max_pred_length)
        print(f"Adjusted prediction length to {pred_length} days")
    
    long_term_stats = run_multiple_experiments(
        df_daily, test_df_daily, available_features, target_col, 
        seq_length, pred_length, n_runs=n_runs, epochs=150
    )
    
    # 保存实验结果到txt文件
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f'results/transformer_experiment_results_{timestamp}.txt'
    
    with open(results_filename, 'w', encoding='utf-8') as f:
        f.write("Standard Transformer Model - Multiple Runs Experiment Results\n")
        f.write("=" * 70 + "\n")
        f.write(f"Experiment Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of runs per prediction: {n_runs}\n")
        f.write(f"Training epochs per run: 150\n")
        f.write(f"Model architecture: Transformer (d_model=128, num_heads=8, num_layers=4)\n\n")
        
        # 短期预测结果
        if short_term_stats:
            f.write("SHORT-TERM PREDICTION (90 days)\n")
            f.write("-" * 40 + "\n")
            f.write(f"Configuration: Past 90 days -> Future 90 days (recursive prediction)\n\n")
            f.write(f"MSE Results ({n_runs} runs):\n")
            for i, mse in enumerate(short_term_stats['MSE_values'], 1):
                f.write(f"  Run {i}: {mse:.4f}\n")
            f.write(f"  Mean: {short_term_stats['MSE_mean']:.4f}\n")
            f.write(f"  Std:  {short_term_stats['MSE_std']:.4f}\n\n")
            
            f.write(f"MAE Results ({n_runs} runs):\n")
            for i, mae in enumerate(short_term_stats['MAE_values'], 1):
                f.write(f"  Run {i}: {mae:.4f}\n")
            f.write(f"  Mean: {short_term_stats['MAE_mean']:.4f}\n")
            f.write(f"  Std:  {short_term_stats['MAE_std']:.4f}\n\n")
        
        # 长期预测结果
        if long_term_stats:
            f.write("LONG-TERM PREDICTION (365 days)\n")
            f.write("-" * 40 + "\n")
            f.write(f"Configuration: Past 90 days -> Future {pred_length} days (recursive prediction)\n\n")
            f.write(f"MSE Results ({n_runs} runs):\n")
            for i, mse in enumerate(long_term_stats['MSE_values'], 1):
                f.write(f"  Run {i}: {mse:.4f}\n")
            f.write(f"  Mean: {long_term_stats['MSE_mean']:.4f}\n")
            f.write(f"  Std:  {long_term_stats['MSE_std']:.4f}\n\n")
            
            f.write(f"MAE Results ({n_runs} runs):\n")
            for i, mae in enumerate(long_term_stats['MAE_values'], 1):
                f.write(f"  Run {i}: {mae:.4f}\n")
            f.write(f"  Mean: {long_term_stats['MAE_mean']:.4f}\n")
            f.write(f"  Std:  {long_term_stats['MAE_std']:.4f}\n\n")
        
        # 比较结果
        f.write("COMPARISON SUMMARY\n")
        f.write("-" * 40 + "\n")
        if short_term_stats and long_term_stats:
            f.write(f"Short-term (90 days):  MSE = {short_term_stats['MSE_mean']:.4f} ± {short_term_stats['MSE_std']:.4f}, MAE = {short_term_stats['MAE_mean']:.4f} ± {short_term_stats['MAE_std']:.4f}\n")
            f.write(f"Long-term ({pred_length} days): MSE = {long_term_stats['MSE_mean']:.4f} ± {long_term_stats['MSE_std']:.4f}, MAE = {long_term_stats['MAE_mean']:.4f} ± {long_term_stats['MAE_std']:.4f}\n")
    
    # 打印结果摘要
    print("\n" + "="*70)
    print("STANDARD TRANSFORMER MODEL EXPERIMENT RESULTS SUMMARY")
    print("="*70)
    
    if short_term_stats:
        print(f"\nShort-term prediction (90 days) - {n_runs} runs:")
        print(f"  MSE: {short_term_stats['MSE_mean']:.4f} ± {short_term_stats['MSE_std']:.4f}")
        print(f"  MAE: {short_term_stats['MAE_mean']:.4f} ± {short_term_stats['MAE_std']:.4f}")
    
    if long_term_stats:
        print(f"\nLong-term prediction ({pred_length} days) - {n_runs} runs:")
        print(f"  MSE: {long_term_stats['MSE_mean']:.4f} ± {long_term_stats['MSE_std']:.4f}")
        print(f"  MAE: {long_term_stats['MAE_mean']:.4f} ± {long_term_stats['MAE_std']:.4f}")
    
    print(f"\nDetailed results have been saved to: {results_filename}")
    print("Model visualizations have been saved to 'results' folder.")

if __name__ == "__main__":
    main()