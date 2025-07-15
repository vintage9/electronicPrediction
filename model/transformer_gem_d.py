import torch
import torch.nn as nn
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# --- TimeSeriesDataProcessor Class (Optimized for Residual Target Prediction and Fourier Features) ---
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
        self.target_scaler = StandardScaler() # Scaler for raw target, used for inverse_transform in non-residual mode
        self.residual_target_scaler = StandardScaler() # Scaler for residuals

    def load_data(self):
        train_data = pd.read_csv(self.train_path)
        test_data = pd.read_csv(self.test_path)
        return train_data, test_data

    def handle_missing(self, df):
        df = df.fillna(method='ffill').fillna(method='bfill')
        return df

    def add_time_features(self, df, encoding='sincos'):
        if 'Date' not in df.columns:
            df['Date'] = pd.to_datetime(df.index)
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
    
    def add_fourier_features(self, df, frequencies=[7, 30.437, 365.25]): # Days in week, avg days in month, days in year
        df['day_of_year'] = df['Date'].dt.dayofyear
        df['day_of_month'] = df['Date'].dt.day

        new_fourier_features = []
        # Use day of year for yearly/half-yearly frequencies
        # Use day of week for weekly frequencies
        # Use day of month for monthly frequencies
        
        # Example: Yearly periodicity
        df['year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        new_fourier_features.extend(['year_sin', 'year_cos'])

        # Example: Weekly periodicity (already covered by dayofweek_sin/cos, but can be added explicitly)
        # Assuming df['dayofweek'] is 0-6
        # df['week_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        # df['week_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        # new_fourier_features.extend(['week_sin', 'week_cos'])

        # Example: Monthly periodicity (using day_of_month)
        # df['month_day_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 30.437) # Average days in month
        # df['month_day_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 30.437)
        # new_fourier_features.extend(['month_day_sin', 'month_day_cos'])

        for feature in new_fourier_features:
            if feature not in self.feature_columns:
                self.feature_columns.append(feature)
        
        return df


    def normalize_features(self, train_df, test_df, use_residual=False):
        available = [col for col in self.feature_columns if col in train_df.columns]
        X_train = train_df[available].values
        y_train_raw = train_df[self.target_column].values # Raw target for residual calculation
        X_test = test_df[available].values
        y_test_raw = test_df[self.target_column].values # Raw target for residual calculation

        X_train = self.feature_scaler.fit_transform(X_train)
        X_test = self.feature_scaler.transform(X_test)

        # Fit target_scaler on raw target for inverse transformation later (for non-residual mode or visualization)
        self.target_scaler.fit(y_train_raw.reshape(-1, 1))

        # y_train and y_test will be the normalized raw values if not using residual for now
        # Residual calculation and scaling will happen when creating sequences
        y_train_norm = self.target_scaler.transform(y_train_raw.reshape(-1, 1)).flatten()
        y_test_norm = self.target_scaler.transform(y_test_raw.reshape(-1, 1)).flatten()


        return X_train, y_train_norm, X_test, y_test_norm, available, y_train_raw, y_test_raw


    def create_sequences(self, X, y_normalized_full, y_raw_full, use_residual=False):
        Xs, Ys = [], []
        Ys_baseline = [] # Store baselines for residual calculation later
        Ys_raw_target_segment = [] # Store raw targets for residual calculation

        total_len = self.sequence_length + self.prediction_length
        for i in range(len(X) - total_len + 1):
            Xs.append(X[i:i + self.sequence_length])

            # The target segment for prediction
            current_target_segment_raw = y_raw_full[i + self.sequence_length : i + self.sequence_length + self.prediction_length]
            Ys_raw_target_segment.append(current_target_segment_raw)

            # Baseline for the prediction window: last known actual value from input sequence
            baseline_for_pred_window = np.full(self.prediction_length, y_raw_full[i + self.sequence_length - 1])
            Ys_baseline.append(baseline_for_pred_window)

        Xs_np = np.array(Xs)
        Ys_raw_target_segment_np = np.array(Ys_raw_target_segment)
        Ys_baseline_np = np.array(Ys_baseline)

        if use_residual:
            # Calculate residuals: Actual raw target - Baseline raw target
            Ys_residual = Ys_raw_target_segment_np - Ys_baseline_np
            # Transform residuals using the already fitted residual_target_scaler
            # Reshape for scaler, then reshape back
            Ys_scaled_residual = self.residual_target_scaler.transform(Ys_residual.reshape(-1,1)).reshape(Ys_residual.shape)
            return Xs_np, Ys_scaled_residual, Ys_baseline_np
        else:
            # If not using residual, the target is the normalized actual value.
            # Extract the correct `y_normalized_full` slice for each sequence
            Ys_normalized = []
            for i in range(len(X) - total_len + 1):
                Ys_normalized.append(y_normalized_full[i + self.sequence_length : i + self.sequence_length + self.prediction_length])
            return Xs_np, np.array(Ys_normalized), None

    def process_data(self, add_time_features=True, time_encoding="sincos", use_residual=False, add_fourier_features=True):
        train_df, test_df = self.load_data()
        train_df = self.handle_missing(train_df)
        test_df = self.handle_missing(test_df)

        # Apply daily aggregation
        train_df['Date'] = pd.to_datetime(train_df['Date'])
        test_df['Date'] = pd.to_datetime(test_df['Date'])
        train_df = train_df.set_index('Date')
        test_df = test_df.set_index('Date')

        agg_funcs = {
            'Global_active_power': 'sum',
            'global_reactive_power': 'sum',
            'sub_metering_1': 'sum',
            'sub_metering_2': 'sum',
            'sub_metering_3': 'sum',
            'voltage': 'mean',
            'global_intensity': 'mean',
            # Add weather columns if available in your actual dataset
            'RR': 'first',
            'NBJRR1': 'first',
            'NBJRR5': 'first',
            'NBJRR10': 'first',
            'NBJBROU': 'first'
        }
        train_agg_funcs = {k: v for k, v in agg_funcs.items() if k in train_df.columns}
        test_agg_funcs = {k: v for k, v in agg_funcs.items() if k in test_df.columns}

        train_df = train_df.resample('D').agg(train_agg_funcs)
        test_df = test_df.resample('D').agg(test_agg_funcs)
        train_df = self.handle_missing(train_df)
        test_df = self.handle_missing(test_df)
        train_df = train_df.reset_index()
        test_df = test_df.reset_index()

        if add_time_features:
            train_df = self.add_time_features(train_df, encoding=time_encoding)
            test_df = self.add_time_features(test_df, encoding=time_encoding)
            
        # Add Fourier Features after other time features, as it might use them
        if add_fourier_features:
            train_df = self.add_fourier_features(train_df)
            test_df = self.add_fourier_features(test_df)

        X_train, y_train_norm, X_test, y_test_norm, feat_cols, y_train_raw_full, y_test_raw_full = \
            self.normalize_features(train_df, test_df, use_residual=use_residual)

        # If using residual, fit the residual_target_scaler on the training data's residuals before creating sequences
        if use_residual:
            # Temporarily calculate all training residuals to fit the scaler
            temp_y_train_residuals = []
            for i in range(len(X_train) - (self.sequence_length + self.prediction_length) + 1):
                baseline = y_train_raw_full[i + self.sequence_length - 1]
                target_seq = y_train_raw_full[i + self.sequence_length : i + self.sequence_length + self.prediction_length]
                temp_y_train_residuals.append(target_seq - baseline)
            if temp_y_train_residuals: # Ensure there are residuals to fit
                self.residual_target_scaler.fit(np.array(temp_y_train_residuals).reshape(-1, 1))

            X_train_seq, y_train_seq_residual, y_train_baseline_seq = self.create_sequences(X_train, y_train_norm, y_train_raw_full, use_residual=True)
            X_test_seq, y_test_seq_residual, y_test_baseline_seq = self.create_sequences(X_test, y_test_norm, y_test_raw_full, use_residual=True)

            self.processed_data = {
                'X_train': X_train_seq,
                'y_train': y_train_seq_residual, # Target is now residual
                'X_test': X_test_seq,
                'y_test': y_test_seq_residual,   # Target is now residual
                'y_train_raw_full': y_train_raw_full, # Keep full raw array for evaluation context
                'y_test_raw_full': y_test_raw_full, # Keep full raw array for evaluation context
                'train_baselines': y_train_baseline_seq,
                'test_baselines': y_test_baseline_seq,
                'feature_names': feat_cols,
                'n_features': len(feat_cols)
            }
        else: # Original non-residual processing
            X_train_seq, y_train_seq_norm, _ = self.create_sequences(X_train, y_train_norm, y_train_raw_full, use_residual=False)
            X_test_seq, y_test_seq_norm, _ = self.create_sequences(X_test, y_test_norm, y_test_raw_full, use_residual=False)

            self.processed_data = {
                'X_train': X_train_seq,
                'y_train': y_train_seq_norm,
                'X_test': X_test_seq,
                'y_test': y_test_seq_norm,
                'y_train_raw_full': y_train_raw_full,
                'y_test_raw_full': y_test_raw_full,
                'feature_names': feat_cols,
                'n_features': len(feat_cols)
            }
        return self.processed_data

    def inverse_transform_target(self, y_predicted, y_baseline=None, use_residual=False):
        if use_residual:
            # y_predicted here are predicted residuals (normalized)
            # First, inverse transform the normalized residuals
            # Ensure y_predicted is a numpy array before reshaping
            if not isinstance(y_predicted, np.ndarray):
                y_predicted = y_predicted.cpu().numpy()

            predicted_residuals_orig_scale = self.residual_target_scaler.inverse_transform(y_predicted.reshape(-1, 1)).flatten()
            predicted_residuals_orig_scale = predicted_residuals_orig_scale.reshape(y_predicted.shape) # Reshape back to original prediction_length

            # Then, add the baseline back to get the final prediction in original scale
            if y_baseline is None:
                raise ValueError("Baseline is required for inverse transforming residual predictions.")
            final_predictions = predicted_residuals_orig_scale + y_baseline
            return final_predictions
        else:
            # y_predicted here are normalized actual values
            # Ensure y_predicted is a numpy array before reshaping
            if not isinstance(y_predicted, np.ndarray):
                y_predicted = y_predicted.cpu().numpy()
            return self.target_scaler.inverse_transform(y_predicted.reshape(-1, 1)).flatten()

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

# --- Modified Positional Encoding (Learnable) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create a learnable embedding for positions
        self.position_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, d_model)
        # Create position indices for the current sequence length
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0) # (1, sequence_length)
        
        # Look up learnable positional embeddings and add to input
        return x + self.position_embedding(positions) # Broadcasting will handle (batch_size, seq_len, d_model)

# --- Optimized Transformer Model with CNN and Residual Prediction ---
class OptimizedTransformerPredictor(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, n_encoder_layers, n_decoder_layers, dim_feedforward, dropout=0.1, prediction_length=90):
        super(OptimizedTransformerPredictor, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.prediction_length = prediction_length
        self.sequence_length = 90 # Hardcode as per problem, input sequence length

        # 1. Input Linear Layer
        self.input_linear = nn.Linear(input_dim, d_model)

        # 2. Convolutional Layer for Local Feature Extraction
        # Input to Conv1d: (Batch, Channels, Sequence Length)
        # We want to operate on the 'feature' dimension (d_model) across the 'sequence_length'.
        # Kernel size can be tuned (e.g., 3, 5, 7 for different local spans).
        # Padding 'same' ensures output sequence length is same as input.
        self.conv_layer = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding='same')
        self.conv_activation = nn.ReLU() # Add an activation after conv

        # 3. Positional Encoding (Now Learnable)
        # max_len should be at least sequence_length + prediction_length
        self.positional_encoding = PositionalEncoding(d_model, max_len=max(self.sequence_length, self.prediction_length))

        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_encoder_layers)

        # 5. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward, dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, n_decoder_layers)

        # 6. Output Linear Layer
        self.output_linear = nn.Linear(d_model, 1)

    def forward(self, src):
        # src shape: (batch_size, sequence_length, input_dim)

        # Input Embedding
        src = self.input_linear(src) * math.sqrt(self.d_model) # (batch_size, seq_len, d_model)

        # Permute for Conv1d: (Batch, Seq_len, Features) -> (Batch, Features, Seq_len)
        src_conv_input = src.permute(0, 2, 1) # (batch_size, d_model, seq_len)

        # Apply Conv1d and activation
        src_conv_output = self.conv_activation(self.conv_layer(src_conv_input)) # (batch_size, d_model, seq_len)

        # Permute back for Positional Encoding/Transformer: (Batch, Features, Seq_len) -> (Batch, Seq_len, Features)
        src_conv_output = src_conv_output.permute(0, 2, 1) # (batch_size, seq_len, d_model)

        # Positional Encoding for Encoder Input
        src_pe = self.positional_encoding(src_conv_output)

        # Encoder
        memory = self.transformer_encoder(src_pe) # (batch_size, seq_len, d_model)

        # Decoder Input: A sequence of learnable queries or zeros with positional encoding
        # This will be `prediction_length` tokens, each with `d_model` dimensions
        tgt = torch.zeros((src.size(0), self.prediction_length, self.d_model), device=src.device)
        # Positional Encoding for Decoder Input
        tgt_pe = self.positional_encoding(tgt)

        # Decoder Mask (Subsequent Mask for autoregressive decoding)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.prediction_length).to(src.device)

        # Decoder
        output = self.transformer_decoder(tgt_pe, memory, tgt_mask=tgt_mask) # (batch_size, prediction_length, d_model)

        # Final linear layer to get the prediction for each step
        output = self.output_linear(output) # (batch_size, prediction_length, 1)

        return output.squeeze(-1) # (batch_size, prediction_length)

# --- Training and Evaluation Functions ---
def evaluate_model(model, test_loader, criterion, device, data_processor, use_residual=False):
    model.eval()
    total_mse = 0
    total_mae = 0
    predictions_list = []
    actuals_list = [] # This will store actuals in original scale

    with torch.no_grad():
        # Retrieve full raw targets and baselines from the data_processor's stored data
        # These are needed to correctly inverse transform residual predictions
        full_raw_targets = data_processor.processed_data['y_test_raw_full']
        full_baselines = data_processor.processed_data['test_baselines'] if use_residual else None
        
        current_sample_idx = 0
        for data, target_norm_or_residual in test_loader:
            data = data.to(device)
            output_norm_or_residual = model(data) # Predicted normalized residuals or normalized values

            batch_size = data.size(0)

            # Get the raw target segment for the current batch
            # `data_processor.processed_data['y_test_raw_full']` is the full raw time series.
            # We need to extract the corresponding target segments that were fed into `create_sequences`
            # For `create_sequences(X, y_normalized_full, y_raw_full)`, the `y_raw_full` slice is:
            # y_raw_full[i + self.sequence_length : i + self.sequence_length + self.prediction_length]

            # Adjust indices to get the raw target for the current batch
            # `test_loader.dataset.X` directly maps to the processed `X_test_seq` from data_processor.
            # We need the `raw_full` array and the sequence indices.
            
            # The `evaluate_model` now directly uses the `y_test_raw_seq` and `test_baselines` from processed_data.

            start_idx_in_seq_data = current_sample_idx
            end_idx_in_seq_data = min(current_sample_idx + batch_size, len(test_loader.dataset))

            # This part needs to correctly map back to the original raw target array.
            # The processed_data['X_test'] (which is test_loader.dataset.X) corresponds to the start of each input sequence.
            # We need the *target* part of the raw data.
            
            # Calculate the true start index in the raw full data for the current batch's *targets*
            # This logic needs to be robust for all batches
            actual_target_segments = []
            for i in range(start_idx_in_seq_data, end_idx_in_seq_data):
                # The start index for the target window in the original full raw data
                # is the start of the input sequence (i) plus the sequence_length
                raw_data_start_idx = i + data_processor.sequence_length
                actual_target_segments.append(
                    data_processor.processed_data['y_test_raw_full'][raw_data_start_idx : raw_data_start_idx + data_processor.prediction_length]
                )
            current_raw_target_seqs = np.array(actual_target_segments)

            if use_residual:
                current_baselines = data_processor.processed_data['test_baselines'][start_idx_in_seq_data:end_idx_in_seq_data]
                output_inv = data_processor.inverse_transform_target(output_norm_or_residual, current_baselines, use_residual=True)
            else:
                output_inv = data_processor.inverse_transform_target(output_norm_or_residual, use_residual=False)
            
            target_inv_flat = current_raw_target_seqs.flatten()
            output_inv_flat = output_inv.flatten() # Flatten for MSE/MAE calculation

            total_mse += np.sum((output_inv_flat - target_inv_flat)**2)
            total_mae += np.sum(np.abs(output_inv_flat - target_inv_flat))

            predictions_list.extend(output_inv.tolist())
            actuals_list.extend(current_raw_target_seqs.tolist())
            
            current_sample_idx += batch_size

    avg_mse = total_mse / len(test_loader.dataset) / model.prediction_length
    avg_mae = total_mae / len(test_loader.dataset) / model.prediction_length

    return avg_mse, avg_mae, predictions_list, actuals_list

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

# --- Main execution block ---
if __name__ == "__main__":
    # Define paths to your data files
    # IMPORTANT: Replace these with the actual paths to your train.csv and test.csv
    train_file_path = 'datasets/daily_power_train2.csv' # Replace with your actual train.csv path
    test_file_path = 'datasets/daily_power_test2.csv'   # Replace with your actual test.csv path

    use_residual_prediction = True # Set to True to enable residual prediction
    add_fourier_features_to_input = True # Set to True to add Fourier features

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
    processed_data_short = data_processor_short.process_data(
        add_time_features=True,
        time_encoding="sincos",
        use_residual=use_residual_prediction,
        add_fourier_features=add_fourier_features_to_input # Enable Fourier features
    )

    train_loader_short, test_loader_short = create_data_loaders(processed_data_short, batch_size=32)

    input_dim_short = processed_data_short['n_features']
    d_model = 64
    n_heads = 8
    n_encoder_layers = 4
    n_decoder_layers = 4
    dim_feedforward = 128
    dropout = 0.1
    learning_rate = 0.001
    num_epochs = 50 # You might need to adjust this for convergence

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run multiple experiments for short-term prediction
    mse_scores_short = []
    mae_scores_short = []
    
    final_preds_short = []
    final_actuals_short = []

    for i in range(5): # At least five experiments
        print(f"\nShort-term Experiment {i+1}/5:")
        model_short = OptimizedTransformerPredictor( # Use the optimized model
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
            # You can uncomment to print training loss per epoch
            # if (epoch + 1) % 10 == 0:
            #     print(f"  Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

        mse_short, mae_short, preds_short, actuals_short = evaluate_model(
            model_short, test_loader_short, criterion, device, data_processor_short,
            use_residual=use_residual_prediction
        )
        mse_scores_short.append(mse_short)
        mae_scores_short.append(mae_short)
        print(f"Short-term Prediction - MSE: {mse_short:.4f}, MAE: {mae_short:.4f}")
        
        if i == 4: # Save results from the last run for plotting
            final_preds_short = preds_short
            final_actuals_short = actuals_short

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

    # --- Long-term Prediction (365 days) ---
    print("\n--- Running Long-term Prediction (365 days) ---")
    sequence_length_long = 90
    prediction_length_long = 365

    data_processor_long = TimeSeriesDataProcessor(
        train_path=train_file_path,
        test_path=test_file_path,
        sequence_length=sequence_length_long,
        prediction_length=prediction_length_long
    )
    processed_data_long = data_processor_long.process_data(
        add_time_features=True,
        time_encoding="sincos",
        use_residual=use_residual_prediction,
        add_fourier_features=add_fourier_features_to_input # Enable Fourier features
    )

    train_loader_long, test_loader_long = create_data_loaders(processed_data_long, batch_size=32)

    input_dim_long = processed_data_long['n_features']

    mse_scores_long = []
    mae_scores_long = []
    
    final_preds_long = []
    final_actuals_long = []

    for i in range(5): # At least five experiments
        print(f"\nLong-term Experiment {i+1}/5:")
        model_long = OptimizedTransformerPredictor( # Use the optimized model
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
            # if (epoch + 1) % 10 == 0:
            #     print(f"  Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

        mse_long, mae_long, preds_long, actuals_long = evaluate_model(
            model_long, test_loader_long, criterion, device, data_processor_long,
            use_residual=use_residual_prediction
        )
        mse_scores_long.append(mse_long)
        mae_scores_long.append(mae_long)
        print(f"Long-term Prediction - MSE: {mse_long:.4f}, MAE: {mae_long:.4f}")

        if i == 4: # Save results from the last run for plotting
            final_preds_long = preds_long
            final_actuals_long = actuals_long

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
    results_filename = f"result_d/optimized_transformer_experiment_results_{timestamp}.txt"
    
    with open(results_filename, 'w', encoding='utf-8') as f:
        f.write("优化Transformer 电力预测实验结果\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"残差预测模式: {'是' if use_residual_prediction else '否'}\n")
        f.write(f"傅里叶特征: {'是' if add_fourier_features_to_input else '否'}\n\n")
        
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
        f.write("优化模型参数:\n")
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
        f.write(f"使用残差预测: {use_residual_prediction}\n")
        f.write(f"使用傅里叶特征: {add_fourier_features_to_input}\n")
        f.write(f"卷积核大小: 3\n")
        f.write(f"位置编码: 可学习\n")
        
    print(f"\n实验结果已保存到: {results_filename}")

    # --- Plotting Results ---
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei'] # For Chinese characters
    plt.rcParams['axes.unicode_minus'] = False # To display minus signs correctly

    # Assume a starting date for plotting to make the X-axis meaningful
    plot_start_date = datetime(2009, 1, 1) # Example start date, adjust as per your test data range

    # Plotting 90-day prediction results
    plt.figure(figsize=(15, 6))
    if final_preds_short and final_actuals_short:
        sample_idx = 0 # Plot the first sample
        dates_90 = [plot_start_date + timedelta(days=i) for i in range(prediction_length_short)]
        
        plt.plot(dates_90, final_actuals_short[sample_idx], 'b-', label='实际值', linewidth=2, alpha=0.8)
        plt.plot(dates_90, final_preds_short[sample_idx], 'r--', label='预测值', linewidth=2, alpha=0.8)
        
        plt.title('90天电力消耗预测结果对比 (优化模型)', fontsize=16, fontweight='bold')
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('全球有功功率 (kW)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.xticks(rotation=45)
        
        mse_90 = np.mean((np.array(final_actuals_short[sample_idx]) - np.array(final_preds_short[sample_idx]))**2)
        mae_90 = np.mean(np.abs(np.array(final_actuals_short[sample_idx]) - np.array(final_preds_short[sample_idx])))
        plt.text(0.02, 0.98, f'MSE: {mse_90:.4f}\nMAE: {mae_90:.4f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.tight_layout()
    plt.savefig('result_d/optimized_short_term_prediction.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plotting 365-day prediction results
    plt.figure(figsize=(15, 6))
    if final_preds_long and final_actuals_long:
        sample_idx = 0 # Plot the first sample
        dates_365 = [plot_start_date + timedelta(days=i) for i in range(prediction_length_long)]
        
        plt.plot(dates_365, final_actuals_long[sample_idx], 'b-', label='实际值', linewidth=2, alpha=0.8)
        plt.plot(dates_365, final_preds_long[sample_idx], 'r--', label='预测值', linewidth=2, alpha=0.8)
        
        plt.title('365天电力消耗预测结果对比 (优化模型)', fontsize=16, fontweight='bold')
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('全球有功功率 (kW)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        
        mse_365 = np.mean((np.array(final_actuals_long[sample_idx]) - np.array(final_preds_long[sample_idx]))**2)
        mae_365 = np.mean(np.abs(np.array(final_actuals_long[sample_idx]) - np.array(final_preds_long[sample_idx])))
        plt.text(0.02, 0.98, f'MSE: {mse_365:.4f}\nMAE: {mae_365:.4f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.tight_layout()
    plt.savefig('result_d/optimized_long_term_prediction.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n图表已保存：")
    print("- optimized_short_term_prediction.png: 90天预测与实际值对比")
    print("- optimized_long_term_prediction.png: 365天预测与实际值对比")
    print(f"- {results_filename}: 详细实验结果数据")