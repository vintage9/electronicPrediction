# data_processor.py（已集成周期特征 sin/cos 编码 + 支持残差目标）
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

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
        df['Date'] = pd.to_datetime(df['Date'])
        df['dayofweek'] = df['Date'].dt.dayofweek
        df['month'] = df['Date'].dt.month
        df['season'] = df['month'] % 12 // 3

        if encoding == 'sincos':
            df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
            df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            self.feature_columns += ['dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos']
        else:
            self.feature_columns += ['dayofweek', 'month', 'season']

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
            Ys.append(y[i + self.sequence_length - 1 : i + self.sequence_length + self.prediction_length])
        return np.array(Xs), np.array(Ys)

    def process_data(self, add_time_features=True, time_encoding="sincos"):
        train_df, test_df = self.load_data()
        train_df = self.handle_missing(train_df)
        test_df = self.handle_missing(test_df)

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
        return self.target_scaler.inverse_transform(y.reshape(-1, 1)).flatten()


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
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
