def process_to_daily(file_path, save_path=None):
    """
    修改后的函数，先填充缺失值再进行每日聚合。
    Args:
        file_path: 原始分钟级数据文件路径
        save_path: 可选，保存处理后数据的路径
    """
    df = pd.read_csv(file_path, sep=',', parse_dates=['DateTime'], infer_datetime_format=True, na_values='?', low_memory=False)

    print("开始处理缺失值，不简单舍弃数据...")

    # 在聚合前，对分钟级数据进行填充，以避免因缺失值而丢失整行数据
    # 首先使用前向填充 (ffill)
    df.fillna(method='ffill', inplace=True)
    # 然后使用后向填充 (bfill) 来处理序列开头的缺失值
    df.fillna(method='bfill', inplace=True)
    # 最后，如果仍有缺失值，用列均值填充作为备用方案
    df.fillna(df.mean(numeric_only=True), inplace=True)

    print("✅ 缺失值已处理完毕。")

    df['Date'] = df['DateTime'].dt.date  # 只保留年月日作为汇总依据

    # 按照项目要求，计算 sub_metering_remainder
    df['sub_metering_remainder'] = (df['Global_active_power'] * 1000 / 60) - (
            df['Sub_metering_1'] + df['Sub_metering_2'] + df['Sub_metering_3']
    )

    daily_df = df.groupby('Date').agg({
        'Global_active_power': 'sum',
        'Global_reactive_power': 'sum',
        'Voltage': 'mean',
        'Global_intensity': 'mean',
        'Sub_metering_1': 'sum',
        'Sub_metering_2': 'sum',
        'Sub_metering_3': 'sum',
        'sub_metering_remainder': 'sum',
        'RR': 'first',  # 天气数据取第一条即可
        'NBJRR1': 'first',
        'NBJRR5': 'first',
        'NBJRR10': 'first',
        'NBJBROU': 'first'
    }).reset_index()

    print("✅ 数据处理完成！共 {} 天记录".format(len(daily_df)))

    if save_path:
        daily_df.to_csv(save_path, index=False)
        print(f"💾 已保存为 {save_path}")

    return daily_df


import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')


class TimeSeriesDataProcessor:
    """时间序列数据处理类"""

    def __init__(self, train_path, test_path, sequence_length=90, prediction_length=90):
        """
        初始化数据处理器

        Args:
            train_path: 训练数据路径
            test_path: 测试数据路径
            sequence_length: 输入序列长度（90天）
            prediction_length: 预测序列长度（90天或365天）
        """
        self.train_path = train_path
        self.test_path = test_path
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length

        # 特征列名
        self.feature_columns = [
            'Global_active_power', 'global_reactive_power', 'voltage',
            'global_intensity', 'sub_metering_1', 'sub_metering_2',
            'sub_metering_3', 'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU'
        ]

        # 目标列
        self.target_column = 'Global_active_power'

        # 数据缩放器
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        # 存储处理后的数据
        self.train_data = None
        self.test_data = None
        self.processed_features = None
        self.processed_targets = None

    def load_data(self):
        """加载训练和测试数据"""
        print("正在加载数据...")

        # 加载训练数据
        self.train_data = pd.read_csv(self.train_path)
        print(f"训练数据形状: {self.train_data.shape}")

        # 加载测试数据
        self.test_data = pd.read_csv(self.test_path)
        print(f"测试数据形状: {self.test_data.shape}")

        # 显示数据基本信息
        print("\n训练数据基本信息:")
        print(self.train_data.head())
        print(f"\n缺失值统计:")
        print(self.train_data.isnull().sum())

        return self.train_data, self.test_data

    def handle_missing_values(self, data):
        """处理缺失值"""
        print("正在处理缺失值...")

        # 记录缺失值情况
        missing_before = data.isnull().sum().sum()
        print(f"处理前缺失值总数: {missing_before}")

        # 对于数值列，使用前向填充 + 后向填充
        data_filled = data.copy()

        # 首先尝试前向填充
        data_filled = data_filled.fillna(method='ffill')

        # 然后后向填充（处理开头的缺失值）
        data_filled = data_filled.fillna(method='bfill')

        # 如果仍有缺失值，使用列的均值填充
        for col in self.feature_columns:
            if col in data_filled.columns:
                data_filled[col] = data_filled[col].fillna(data_filled[col].mean())

        missing_after = data_filled.isnull().sum().sum()
        print(f"处理后缺失值总数: {missing_after}")

        return data_filled

    def create_derived_features(self, data):
        """创建衍生特征"""
        print("正在创建衍生特征...")

        data_enhanced = data.copy()

        # 计算 sub_metering_remainder
        if all(col in data.columns for col in
               ['Global_active_power', 'sub_metering_1', 'sub_metering_2', 'sub_metering_3']):
            data_enhanced['sub_metering_remainder'] = (
                    data['Global_active_power'] * 1000 / 60 -
                    (data['sub_metering_1'] + data['sub_metering_2'] + data['sub_metering_3'])
            )
            self.feature_columns.append('sub_metering_remainder')

        # 添加时间特征（如果有日期列）
        if 'Date' in data.columns:
            data_enhanced['Date'] = pd.to_datetime(data_enhanced['Date'])

            # 提取时间特征
            data_enhanced['month'] = data_enhanced['Date'].dt.month
            data_enhanced['day_of_year'] = data_enhanced['Date'].dt.dayofyear
            data_enhanced['week_of_year'] = data_enhanced['Date'].dt.isocalendar().week
            data_enhanced['season'] = data_enhanced['month'].apply(self._get_season)

            # 添加到特征列表
            time_features = ['month', 'day_of_year', 'week_of_year', 'season']
            self.feature_columns.extend(time_features)

        # 添加滞后特征（使用目标变量的滞后值）
        for lag in [1, 7, 30]:  # 1天、7天、30天前的值
            lag_col = f'Global_active_power_lag_{lag}'
            data_enhanced[lag_col] = data_enhanced['Global_active_power'].shift(lag)
            self.feature_columns.append(lag_col)

        # 添加滚动统计特征
        window_sizes = [7, 30]  # 7天和30天滚动窗口
        for window in window_sizes:
            # 滚动均值
            col_mean = f'Global_active_power_roll_mean_{window}'
            data_enhanced[col_mean] = data_enhanced['Global_active_power'].rolling(window=window).mean()
            self.feature_columns.append(col_mean)

            # 滚动标准差
            col_std = f'Global_active_power_roll_std_{window}'
            data_enhanced[col_std] = data_enhanced['Global_active_power'].rolling(window=window).std()
            self.feature_columns.append(col_std)

        print(f"特征数量: {len(self.feature_columns)}")
        return data_enhanced

    def _get_season(self, month):
        """根据月份获取季节"""
        if month in [12, 1, 2]:
            return 0  # 冬季
        elif month in [3, 4, 5]:
            return 1  # 春季
        elif month in [6, 7, 8]:
            return 2  # 夏季
        else:
            return 3  # 秋季

    def normalize_features(self, train_data, test_data):
        """特征标准化"""
        print("正在进行特征标准化...")

        # 确保所有特征列都存在
        available_features = [col for col in self.feature_columns if col in train_data.columns]
        print(f"可用特征数量: {len(available_features)}")

        # 分离特征和目标
        X_train = train_data[available_features].values
        y_train = train_data[self.target_column].values

        X_test = test_data[available_features].values
        y_test = test_data[self.target_column].values

        # 标准化特征
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)

        # 标准化目标变量
        y_train_scaled = self.target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = self.target_scaler.transform(y_test.reshape(-1, 1)).flatten()

        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, available_features

    def create_sequences(self, X, y):
        """创建时间序列窗口"""
        print(f"正在创建时间序列窗口 (输入长度: {self.sequence_length}, 预测长度: {self.prediction_length})...")

        sequences = []
        targets = []

        # 确保有足够的数据创建序列
        total_length = self.sequence_length + self.prediction_length
        if len(X) < total_length:
            print(f"警告: 数据长度 {len(X)} 小于所需长度 {total_length}")
            return np.array([]), np.array([])

        # 创建滑动窗口
        for i in range(len(X) - total_length + 1):
            # 输入序列
            seq = X[i:i + self.sequence_length]
            # 目标序列
            target = y[i + self.sequence_length:i + self.sequence_length + self.prediction_length]

            sequences.append(seq)
            targets.append(target)

        print(f"创建了 {len(sequences)} 个序列")
        return np.array(sequences), np.array(targets)

    def process_data(self):
        """完整的数据处理流程"""
        print("开始数据处理流程...")

        # 1. 加载数据
        train_data, test_data = self.load_data()

        # 2. 处理缺失值
        train_data = self.handle_missing_values(train_data)
        test_data = self.handle_missing_values(test_data)

        # 3. 创建衍生特征
        train_data = self.create_derived_features(train_data)
        test_data = self.create_derived_features(test_data)

        # 4. 再次处理缺失值（衍生特征可能产生新的缺失值）
        train_data = self.handle_missing_values(train_data)
        test_data = self.handle_missing_values(test_data)

        # 5. 特征标准化
        X_train, X_test, y_train, y_test, feature_names = self.normalize_features(train_data, test_data)

        # 6. 创建时间序列窗口
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train)
        X_test_seq, y_test_seq = self.create_sequences(X_test, y_test)

        # 保存处理结果
        self.processed_data = {
            'X_train': X_train_seq,
            'y_train': y_train_seq,
            'X_test': X_test_seq,
            'y_test': y_test_seq,
            'feature_names': feature_names,
            'n_features': len(feature_names)
        }

        print("数据处理完成!")
        print(f"训练集形状: X={X_train_seq.shape}, y={y_train_seq.shape}")
        print(f"测试集形状: X={X_test_seq.shape}, y={y_test_seq.shape}")

        return self.processed_data

    def inverse_transform_target(self, scaled_predictions):
        """反标准化目标变量"""
        return self.target_scaler.inverse_transform(scaled_predictions.reshape(-1, 1)).flatten()

    def get_data_info(self):
        """获取数据信息"""
        if self.processed_data is None:
            print("请先运行 process_data() 方法")
            return None

        info = {
            'sequence_length': self.sequence_length,
            'prediction_length': self.prediction_length,
            'n_features': self.processed_data['n_features'],
            'feature_names': self.processed_data['feature_names'],
            'train_samples': len(self.processed_data['X_train']),
            'test_samples': len(self.processed_data['X_test'])
        }

        return info


class TimeSeriesDataset(Dataset):
    """PyTorch数据集类"""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_data_loaders(processed_data, batch_size=32, shuffle=True):
    """创建PyTorch数据加载器"""

    # 创建数据集
    train_dataset = TimeSeriesDataset(processed_data['X_train'], processed_data['y_train'])
    test_dataset = TimeSeriesDataset(processed_data['X_test'], processed_data['y_test'])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# # 使用示例
# if __name__ == "__main__":
#     # 短期预测（90天）
#     processor_short = TimeSeriesDataProcessor(
#         train_path='train.csv',
#         test_path='test.csv',
#         sequence_length=90,
#         prediction_length=90
#     )
#
#     # 处理数据
#     processed_data_short = processor_short.process_data()
#
#     # 创建数据加载器
#     train_loader_short, test_loader_short = create_data_loaders(processed_data_short, batch_size=32)
#
#     # 打印数据信息
#     data_info = processor_short.get_data_info()
#     print("\n数据信息:")
#     for key, value in data_info.items():
#         print(f"{key}: {value}")
#
#     # 长期预测（365天）
#     processor_long = TimeSeriesDataProcessor(
#         train_path='train.csv',
#         test_path='test.csv',
#         sequence_length=90,
#         prediction_length=365
#     )
#
#     # 处理数据
#     processed_data_long = processor_long.process_data()
#
#     # 创建数据加载器
#     train_loader_long, test_loader_long = create_data_loaders(processed_data_long, batch_size=16)  # 长期预测使用较小batch_size


if __name__ == "__main__":
    daily_data = process_to_daily("train.csv", save_path="daily_power_train2.csv")
    daily_data2 = process_to_daily("test_with_header.csv", save_path="daily_power_test2.csv")


