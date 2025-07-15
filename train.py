import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from model.SingleStepTransformer import SingleStepTransformer
from model.RecursiveTransformer import RecursiveTransformer, MultiStepTransformer, HybridTransformer
from datasets.data_processor import TimeSeriesDataProcessor, create_data_loaders

# 配置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# ===== 实验配置 =====
PRED_DAYS = 90
BATCH_SIZE = 32  # 降低批次大小以适应更复杂的模型
LEARNING_RATE = 3e-4  # 进一步降低学习率
EPOCHS = 200  # 增加训练轮数以充分利用残差连接
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EXPERIMENTS = 5  # 进行5次实验
TRAINING_STRATEGY = "rolling"  # 可选: "single", "rolling", "teacher_forcing", "recursive"
MODEL_TYPE = "multi_step"         # 可选: "single_step", "recursive", "multi_step", "hybrid"
MODEL_PATH = f"{PRED_DAYS}_{MODEL_TYPE}_{TRAINING_STRATEGY}_residual_best_model.pt"

# ================== 训练函数 ==================
def train_single_step(model, train_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.SmoothL1Loss()
    best_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb[:, 0].to(DEVICE)
            optimizer.zero_grad()
            output = model(xb)
            # 处理不同模型的输出格式
            if isinstance(output, tuple):
                preds = output[0]  # 取预测值
            else:
                preds = output
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb[:, 0].to(DEVICE)
                output = model(xb)
                # 处理不同模型的输出格式
                if isinstance(output, tuple):
                    preds = output[0]  # 取预测值
                else:
                    preds = output
                loss = loss_fn(preds, yb)
                val_losses.append(loss.item())
        if np.mean(val_losses) < best_loss:
            best_loss = np.mean(val_losses)
            torch.save(model.state_dict(), MODEL_PATH)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{EPOCHS}, Train Loss: {np.mean(train_losses):.4f}, Val Loss: {np.mean(val_losses):.4f}")
    return best_loss

def train_multi_step(model, train_loader, val_loader, prediction_steps=90):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.SmoothL1Loss()
    best_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            output = model(xb)
            # 处理不同模型的输出格式
            if isinstance(output, tuple):
                preds = output[0]  # 取预测值
            else:
                preds = output
            if preds.dim() == 1:
                preds = preds.unsqueeze(1)
            loss = loss_fn(preds, yb[:, :preds.size(1)])
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                output = model(xb)
                # 处理不同模型的输出格式
                if isinstance(output, tuple):
                    preds = output[0]  # 取预测值
                else:
                    preds = output
                if preds.dim() == 1:
                    preds = preds.unsqueeze(1)
                loss = loss_fn(preds, yb[:, :preds.size(1)])
                val_losses.append(loss.item())
        if np.mean(val_losses) < best_loss:
            best_loss = np.mean(val_losses)
            torch.save(model.state_dict(), MODEL_PATH)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{EPOCHS}, Train Loss: {np.mean(train_losses):.4f}, Val Loss: {np.mean(val_losses):.4f}")
    return best_loss

def train_recursive_model(model, train_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.SmoothL1Loss()
    best_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb[:, 0].to(DEVICE)  # 取第一天的预测值
            optimizer.zero_grad()
            output = model(xb)
            # 处理不同模型的输出格式
            if isinstance(output, tuple):
                preds, uncertainty = output
                # 可以选择加入不确定性作为正则化项
                loss = loss_fn(preds, yb) + 0.01 * torch.mean(uncertainty)
            else:
                preds = output
                loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb[:, 0].to(DEVICE)  # 取第一天的预测值
                output = model(xb)
                # 处理不同模型的输出格式
                if isinstance(output, tuple):
                    preds, uncertainty = output
                    loss = loss_fn(preds, yb) + 0.01 * torch.mean(uncertainty)
                else:
                    preds = output
                    loss = loss_fn(preds, yb)
                val_losses.append(loss.item())
        if np.mean(val_losses) < best_loss:
            best_loss = np.mean(val_losses)
            torch.save(model.state_dict(), MODEL_PATH)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{EPOCHS}, Train Loss: {np.mean(train_losses):.4f}, Val Loss: {np.mean(val_losses):.4f}")
    return best_loss

def train_rolling_prediction(model, train_loader, val_loader, max_steps=10):
    """使用改进的滚动预测训练策略 - 针对残差网络优化"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # 学习率调度器 - 余弦退火
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE*0.1)
    loss_fn = torch.nn.SmoothL1Loss()
    best_loss = float('inf')
    patience = 30  # 早停耐心值
    no_improve = 0
    
    # 动态调整滚动步数和损失权重
    step_weights = torch.tensor([1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55]).to(DEVICE)
    
    for epoch in range(EPOCHS):
        model.train()
        train_losses = []
        
        # 动态调整滚动步数：更渐进的增长
        current_max_steps = min(2 + epoch // 15, max_steps)
        
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            
            # 滚动预测：逐步预测多个时间步
            rolling_loss = 0
            current_input = xb
            total_weight = 0
            
            for step in range(min(current_max_steps, yb.size(1))):
                output = model(current_input)
                if isinstance(output, tuple):
                    pred = output[0]
                    uncertainty = output[1] if len(output) > 1 else None
                else:
                    pred = output
                    uncertainty = None
                
                target = yb[:, step]
                
                # 改进的加权损失
                weight = step_weights[step] if step < len(step_weights) else 0.1
                step_loss = loss_fn(pred, target) * weight
                
                # 不确定性正则化 - 鼓励模型在预测不准确时输出高不确定性
                if uncertainty is not None:
                    prediction_error = torch.abs(pred - target)
                    uncertainty_loss = torch.mean(torch.abs(uncertainty - prediction_error.detach()))
                    step_loss += 0.005 * uncertainty_loss * weight
                
                rolling_loss += step_loss
                total_weight += weight
                
                # 更新输入：使用更智能的特征构造
                if step < current_max_steps - 1:
                    pred_expanded = pred.unsqueeze(1)
                    last_features = current_input[:, -1:, :]
                    new_features = last_features.clone()
                    new_features[:, :, 0] = pred_expanded
                    
                    # 改进的特征传播
                    if current_input.size(2) > 1:
                        if current_input.size(1) >= 3:  # 使用更长的历史
                            # 使用指数加权移动平均计算趋势
                            recent_trend = 0.5 * (current_input[:, -1:, 1:] - current_input[:, -2:-1, 1:]) + \
                                         0.3 * (current_input[:, -2:-1, 1:] - current_input[:, -3:-2, 1:])
                            new_features[:, :, 1:] = current_input[:, -1:, 1:] + 0.05 * recent_trend
                        elif current_input.size(1) >= 2:
                            trend = current_input[:, -1:, 1:] - current_input[:, -2:-1, 1:]
                            new_features[:, :, 1:] = current_input[:, -1:, 1:] + 0.05 * trend
                    
                    current_input = torch.cat([current_input[:, 1:, :], new_features], dim=1)
            
            loss = rolling_loss / total_weight if total_weight > 0 else rolling_loss
            loss.backward()
            
            # 自适应梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            train_losses.append(loss.item())
        
        # 验证阶段
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                
                rolling_loss = 0
                current_input = xb
                total_weight = 0
                
                for step in range(min(current_max_steps, yb.size(1))):
                    output = model(current_input)
                    if isinstance(output, tuple):
                        pred = output[0]
                        uncertainty = output[1] if len(output) > 1 else None
                    else:
                        pred = output
                        uncertainty = None
                    
                    target = yb[:, step]
                    weight = step_weights[step] if step < len(step_weights) else 0.1
                    step_loss = loss_fn(pred, target) * weight
                    
                    if uncertainty is not None:
                        prediction_error = torch.abs(pred - target)
                        uncertainty_loss = torch.mean(torch.abs(uncertainty - prediction_error))
                        step_loss += 0.005 * uncertainty_loss * weight
                    
                    rolling_loss += step_loss
                    total_weight += weight
                    
                    if step < current_max_steps - 1:
                        pred_expanded = pred.unsqueeze(1)
                        last_features = current_input[:, -1:, :]
                        new_features = last_features.clone()
                        new_features[:, :, 0] = pred_expanded
                        
                        if current_input.size(2) > 1:
                            if current_input.size(1) >= 3:
                                recent_trend = 0.5 * (current_input[:, -1:, 1:] - current_input[:, -2:-1, 1:]) + \
                                             0.3 * (current_input[:, -2:-1, 1:] - current_input[:, -3:-2, 1:])
                                new_features[:, :, 1:] = current_input[:, -1:, 1:] + 0.05 * recent_trend
                            elif current_input.size(1) >= 2:
                                trend = current_input[:, -1:, 1:] - current_input[:, -2:-1, 1:]
                                new_features[:, :, 1:] = current_input[:, -1:, 1:] + 0.05 * trend
                        
                        current_input = torch.cat([current_input[:, 1:, :], new_features], dim=1)
                
                loss = rolling_loss / total_weight if total_weight > 0 else rolling_loss
                val_losses.append(loss.item())
        
        # 学习率调度
        scheduler.step()
        
        # 模型保存和早停
        current_val_loss = np.mean(val_losses)
        if current_val_loss < best_loss:
            best_loss = current_val_loss
            no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            no_improve += 1
        
        if epoch % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch}/{EPOCHS}, Train Loss: {np.mean(train_losses):.4f}, Val Loss: {current_val_loss:.4f}, Steps: {current_max_steps}, LR: {current_lr:.6f}")
        
        # 早停
        if no_improve >= patience:
            print(f"早停触发于第 {epoch} 轮，验证损失连续 {patience} 轮未改善")
            break
    
    return best_loss

def train_teacher_forcing(model, train_loader, val_loader, teacher_forcing_ratio=0.5):
    """使用教师强制训练策略"""
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.SmoothL1Loss()
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            
            # 随机决定是否使用教师强制
            use_teacher_forcing = np.random.random() < teacher_forcing_ratio
            
            if use_teacher_forcing:
                # 使用真实值作为下一步的输入
                output = model(xb)
                if isinstance(output, tuple):
                    preds = output[0]
                else:
                    preds = output
                loss = loss_fn(preds, yb[:, 0])
            else:
                # 使用预测值作为下一步的输入（类似滚动预测）
                output = model(xb)
                if isinstance(output, tuple):
                    preds = output[0]
                else:
                    preds = output
                loss = loss_fn(preds, yb[:, 0])
            
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb[:, 0].to(DEVICE)
                output = model(xb)
                if isinstance(output, tuple):
                    preds = output[0]
                else:
                    preds = output
                loss = loss_fn(preds, yb)
                val_losses.append(loss.item())
        
        if np.mean(val_losses) < best_loss:
            best_loss = np.mean(val_losses)
            torch.save(model.state_dict(), MODEL_PATH)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{EPOCHS}, Train Loss: {np.mean(train_losses):.4f}, Val Loss: {np.mean(val_losses):.4f}")
    
    return best_loss

# ================== 评估函数 ==================
def evaluate_rolling_prediction(model, processed, processor, window_size=90, pred_days=90, verbose=False):
    """改进的90天滚动预测评估"""
    model.eval()
    
    # 获取原始测试数据
    try:
        _, test_df = processor.load_data()
        test_df = processor.handle_missing(test_df)
        test_data = test_df[processor.target_column].values
        test_data_scaled = processor.target_scaler.transform(test_data.reshape(-1, 1)).flatten()
        print(f"测试数据长度: {len(test_data)}")
    except Exception as e:
        print(f"获取测试数据失败: {e}")
        return np.array([]), np.array([])
    
    # 获取测试集的原始特征数据
    try:
        # 重新处理测试数据以获取完整的特征
        test_features = processor.extract_features(test_df)
        test_features_scaled = processor.feature_scaler.transform(test_features)
        n_features = test_features_scaled.shape[1]
        print(f"使用完整特征数据，特征数量: {n_features}")
    except Exception as e:
        print(f"获取完整特征失败，使用简化方法: {e}")
        n_features = processed.get('n_features', 9)
        test_features_scaled = np.tile(test_data_scaled.reshape(-1, 1), (1, n_features))
    
    target_scaler = processed['target_scaler'] if 'target_scaler' in processed else processor.target_scaler
    
    # 选择测试数据的起始点
    start_idx = window_size
    end_idx = len(test_data) - pred_days
    
    if end_idx <= start_idx:
        print(f"测试数据不足以进行{pred_days}天预测")
        return np.array([]), np.array([])
    
    # 获取初始输入窗口（使用真实的多维特征）
    initial_features = test_features_scaled[start_idx-window_size:start_idx]
    print(f"初始特征窗口形状: {initial_features.shape}")
    
    # 进行改进的滚动预测
    predictions = []
    current_window = initial_features.copy()
    
    # 用于存储预测历史，帮助改进后续预测
    prediction_history = []
    
    with torch.no_grad():
        for day in range(pred_days):
            # 确保窗口大小正确
            if current_window.shape[0] != window_size:
                if current_window.shape[0] > window_size:
                    current_window = current_window[-window_size:]
                else:
                    padding_needed = window_size - current_window.shape[0]
                    last_row = current_window[-1:] if len(current_window) > 0 else np.zeros((1, current_window.shape[1]))
                    padding = np.repeat(last_row, padding_needed, axis=0)
                    current_window = np.vstack([current_window, padding])
            
            # 准备输入
            input_tensor = torch.FloatTensor(current_window).unsqueeze(0).to(DEVICE)
            
            # 预测下一天
            output = model(input_tensor)
            if isinstance(output, tuple):
                pred = output[0]
                uncertainty = output[1] if len(output) > 1 else None
            else:
                pred = output
                uncertainty = None
            
            pred_value = pred.cpu().numpy()[0]
            predictions.append(pred_value)
            prediction_history.append(pred_value)
            
            # 构造下一个时间步的特征向量
            if day < pred_days - 1:
                # 取当前窗口最后一个时间步作为模板
                last_features = current_window[-1:].copy()
                
                # 更新目标值特征（假设是第0个特征）
                last_features[0, 0] = pred_value
                
                # 对其他特征进行更智能的更新
                if current_window.shape[1] > 1:
                    # 计算最近几个时间步的特征变化趋势
                    if len(prediction_history) >= 2:
                        # 使用预测历史来估计趋势
                        recent_trend = prediction_history[-1] - prediction_history[-2]
                        # 将趋势应用到相关特征上（比例缩放）
                        for i in range(1, current_window.shape[1]):
                            if current_window.shape[0] >= 2:
                                feature_trend = current_window[-1, i] - current_window[-2, i]
                                # 结合历史趋势和特征趋势
                                combined_trend = 0.7 * feature_trend + 0.3 * recent_trend * 0.1
                                last_features[0, i] = current_window[-1, i] + combined_trend
                    else:
                        # 如果没有足够的预测历史，使用简单的趋势延续
                        if current_window.shape[0] >= 2:
                            for i in range(1, current_window.shape[1]):
                                trend = current_window[-1, i] - current_window[-2, i]
                                last_features[0, i] = current_window[-1, i] + 0.5 * trend
                
                # 更新窗口
                current_window = np.vstack([current_window[1:], last_features])
            
            if verbose and (day + 1) % 10 == 0:
                print(f"已完成 {day + 1}/{pred_days} 天预测")
                if uncertainty is not None:
                    print(f"  预测值: {pred_value:.2f}, 不确定性: {uncertainty.cpu().numpy()[0]:.4f}")
    
    predictions = np.array(predictions)
    
    # 获取真实值
    true_values = test_data_scaled[start_idx:start_idx + pred_days]
    
    # 反标准化
    predictions_denorm = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    true_values_denorm = target_scaler.inverse_transform(true_values.reshape(-1, 1)).flatten()
    
    return predictions_denorm, true_values_denorm

def plot_comparison(y_true, y_pred, title="预测结果对比", save_path=None):
    """绘制预测结果对比图"""
    plt.figure(figsize=(15, 6))
    plt.plot(y_true, label='真实值', color='blue', alpha=0.7)
    plt.plot(y_pred, label='预测值', color='red', linestyle='--', alpha=0.7)
    plt.title(title, fontsize=16)
    plt.xlabel('时间序列数据点', fontsize=12)
    plt.ylabel('每日总有功功率 (kW)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}/{title}.png", dpi=300, bbox_inches='tight')
    
    plt.show()
    return mean_squared_error(y_true, y_pred), mean_absolute_error(y_true, y_pred)

# ================== 主流程 ==================
def main():
    print(f"开始训练实验 - 模型类型: {MODEL_TYPE}, 训练策略: {TRAINING_STRATEGY}")
    print(f"设备: {DEVICE}")
    print(f"将进行 {NUM_EXPERIMENTS} 轮实验，每轮训练 {EPOCHS} 个epoch")
    print("=" * 80)
    
    # 存储所有实验结果
    all_results = {
        'rolling_mse': [],
        'rolling_mae': [],
        'experiment_details': []
    }
    
    # 获取数据处理器（在循环外创建以保持一致性）
    processor = TimeSeriesDataProcessor("datasets/daily_power_train2.csv", "datasets/daily_power_test2.csv", 90, 1)
    processed = processor.process_data()
    
    print(f"数据特征数: {processed['n_features']}")
    # 检查processed字典中的实际键
    print(f"可用的数据键: {list(processed.keys())}")
    # 获取数据加载器来检查数据大小
    train_loader_temp, val_loader_temp = create_data_loaders(processed, batch_size=BATCH_SIZE)
    print(f"训练批次数: {len(train_loader_temp)}")
    print(f"验证批次数: {len(val_loader_temp)}")
    
    # 检查数据维度
    sample_x, sample_y = next(iter(train_loader_temp))
    print(f"输入数据维度: {sample_x.shape} (batch_size, sequence_length, n_features)")
    print(f"目标数据维度: {sample_y.shape} (batch_size, prediction_length)")
    print(f"模型实际预测天数: {sample_y.shape[1]} 天")
    print(f"但训练时只使用: 第1天 (索引0)")
    print("-" * 80)
    
    # 用于存储最后一轮的滚动预测结果用于绘图
    last_rolling_true = None
    last_rolling_pred = None
    
    # --- 循环运行5轮实验 ---
    for experiment_num in range(NUM_EXPERIMENTS):
        print(f"\n{'='*20} 第 {experiment_num + 1} 轮实验开始 {'='*20}")
        experiment_start_time = np.datetime64('now')
        
        # 重新创建数据加载器以增加随机性
        train_loader, val_loader = create_data_loaders(processed, batch_size=BATCH_SIZE)
        
        # 选择模型
        if MODEL_TYPE == "single_step":
            model = SingleStepTransformer(n_features=processed['n_features'], input_len=PRED_DAYS).to(DEVICE)
            print("✓ 使用单步Transformer模型")
        elif MODEL_TYPE == "recursive":
            model = RecursiveTransformer(n_features=processed['n_features'], input_len=PRED_DAYS).to(DEVICE)
            print("✓ 使用递归Transformer模型")
        elif MODEL_TYPE == "multi_step":
            model = MultiStepTransformer(n_features=processed['n_features'], input_len=PRED_DAYS, output_len=90).to(DEVICE)
            print("✓ 使用多步Transformer模型")
        elif MODEL_TYPE == "hybrid":
            model = HybridTransformer(n_features=processed['n_features'], input_len=PRED_DAYS).to(DEVICE)
            print("✓ 使用混合Transformer模型")
        else:
            model = SingleStepTransformer(n_features=processed['n_features'], input_len=PRED_DAYS).to(DEVICE)
            print("✓ 使用默认单步Transformer模型")
        
        # 训练模型
        print(f"\n🚀 开始第 {experiment_num + 1} 轮训练 ({EPOCHS} epochs)...")
        training_start_time = np.datetime64('now')
        
        if TRAINING_STRATEGY == "single":
            print("📈 使用单步训练策略")
            best_loss = train_single_step(model, train_loader, val_loader)
        elif TRAINING_STRATEGY == "multi_step":
            print("📈 使用多步训练策略")
            best_loss = train_multi_step(model, train_loader, val_loader)
        elif TRAINING_STRATEGY == "recursive":
            print("📈 使用递归训练策略")
            best_loss = train_recursive_model(model, train_loader, val_loader)
        elif TRAINING_STRATEGY == "rolling":
            print("📈 使用滚动预测训练策略")
            best_loss = train_rolling_prediction(model, train_loader, val_loader)
        elif TRAINING_STRATEGY == "teacher_forcing":
            print("📈 使用教师强制训练策略")
            best_loss = train_teacher_forcing(model, train_loader, val_loader)
        else:
            print("📈 使用默认单步训练策略")
            best_loss = train_single_step(model, train_loader, val_loader)
        
        training_end_time = np.datetime64('now')
        print(f"✅ 第 {experiment_num + 1} 轮训练完成，最佳验证损失: {best_loss:.6f}")
        
        # 评估90天滚动预测性能
        print(f"\n� 第 {experiment_num + 1} 轮90天滚动预测评估中...")
        eval_start_time = np.datetime64('now')
        rolling_pred, rolling_true = evaluate_rolling_prediction(
            model, processed, processor, window_size=90, pred_days=90, verbose=False
        )
        
        if len(rolling_pred) > 0 and len(rolling_true) > 0:
            rolling_mse = mean_squared_error(rolling_true, rolling_pred)
            rolling_mae = mean_absolute_error(rolling_true, rolling_pred)
            
            all_results['rolling_mse'].append(rolling_mse)
            all_results['rolling_mae'].append(rolling_mae)
            
            # 存储最后一轮的滚动预测结果用于绘图
            if experiment_num == NUM_EXPERIMENTS - 1:
                last_rolling_true = rolling_true
                last_rolling_pred = rolling_pred
            
            print(f"📈 第 {experiment_num + 1} 轮90天滚动预测结果:")
            print(f"   MSE: {rolling_mse:.6f}")
            print(f"   MAE: {rolling_mae:.6f}")
            
            rolling_success = True
        else:
            print(f"❌ 第 {experiment_num + 1} 轮90天滚动预测失败")
            rolling_success = False
            rolling_mse = rolling_mae = None
        
        eval_end_time = np.datetime64('now')
        experiment_end_time = np.datetime64('now')
        
        # 记录本轮实验详细信息
        experiment_detail = {
            'experiment_num': experiment_num + 1,
            'best_validation_loss': best_loss,
            'rolling_mse': rolling_mse,
            'rolling_mae': rolling_mae,
            'rolling_success': rolling_success,
            'training_time': str(training_end_time - training_start_time),
            'evaluation_time': str(eval_end_time - eval_start_time),
            'total_time': str(experiment_end_time - experiment_start_time)
        }
        all_results['experiment_details'].append(experiment_detail)
        
        print(f"\n⏱️  第 {experiment_num + 1} 轮实验用时: {experiment_detail['total_time']}")
        print(f"{'='*20} 第 {experiment_num + 1} 轮实验结束 {'='*20}")
    
    # --- 计算并输出最终统计结果 ---
    print(f"\n{'='*80}")
    print(f"🎯 {NUM_EXPERIMENTS} 轮实验完成 - 最终结果统计")
    print(f"{'='*80}")
    
    # 计算90天滚动预测统计
    if all_results['rolling_mse']:
        rolling_mse_mean = np.mean(all_results['rolling_mse'])
        rolling_mse_std = np.std(all_results['rolling_mse'])
        rolling_mae_mean = np.mean(all_results['rolling_mae'])
        rolling_mae_std = np.std(all_results['rolling_mae'])
        
        print(f"\n🔄 90天滚动预测统计结果:")
        print(f"   MSE - 平均值: {rolling_mse_mean:.6f} ± {rolling_mse_std:.6f}")
        print(f"   MAE - 平均值: {rolling_mae_mean:.6f} ± {rolling_mae_std:.6f}")
        print(f"   成功率: {len(all_results['rolling_mse'])}/{NUM_EXPERIMENTS} ({len(all_results['rolling_mse'])/NUM_EXPERIMENTS*100:.1f}%)")
    else:
        print(f"\n❌ 90天滚动预测全部失败")
        rolling_mse_mean = rolling_mse_std = rolling_mae_mean = rolling_mae_std = None
    
    # 保存详细结果到文件
    results_filename = f"{PRED_DAYS}_{MODEL_TYPE}_{TRAINING_STRATEGY}_detailed_results.txt"
    
    with open(results_filename, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"实验详细结果报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("实验配置:\n")
        f.write("-" * 40 + "\n")
        f.write(f"模型类型: {MODEL_TYPE}\n")
        f.write(f"训练策略: {TRAINING_STRATEGY}\n")
        f.write(f"实验轮数: {NUM_EXPERIMENTS}\n")
        f.write(f"每轮训练轮数: {EPOCHS}\n")
        f.write(f"批次大小: {BATCH_SIZE}\n")
        f.write(f"学习率: {LEARNING_RATE}\n")
        f.write(f"预测天数: {PRED_DAYS}\n")
        f.write(f"设备: {DEVICE}\n\n")
        
        f.write("总体统计结果:\n")
        f.write("-" * 40 + "\n")
        
        if rolling_mse_mean is not None:
            f.write("90天滚动预测结果:\n")
            f.write(f"  MSE - 平均值: {rolling_mse_mean:.6f}\n")
            f.write(f"  MSE - 标准差: {rolling_mse_std:.6f}\n")
            f.write(f"  MAE - 平均值: {rolling_mae_mean:.6f}\n")
            f.write(f"  MAE - 标准差: {rolling_mae_std:.6f}\n")
            f.write(f"  成功率: {len(all_results['rolling_mse'])}/{NUM_EXPERIMENTS} ({len(all_results['rolling_mse'])/NUM_EXPERIMENTS*100:.1f}%)\n\n")
        else:
            f.write("90天滚动预测结果: 全部实验失败\n\n")
        
        f.write("各轮实验详细结果:\n")
        f.write("-" * 40 + "\n")
        for detail in all_results['experiment_details']:
            f.write(f"第 {detail['experiment_num']} 轮实验:\n")
            f.write(f"  最佳验证损失: {detail['best_validation_loss']:.6f}\n")
            if detail['rolling_success']:
                f.write(f"  90天预测 MSE: {detail['rolling_mse']:.6f}\n")
                f.write(f"  90天预测 MAE: {detail['rolling_mae']:.6f}\n")
            else:
                f.write(f"  90天预测: 失败\n")
            f.write(f"  训练用时: {detail['training_time']}\n")
            f.write(f"  评估用时: {detail['evaluation_time']}\n")
            f.write(f"  总用时: {detail['total_time']}\n")
            f.write("\n")
        
        f.write("原始数据:\n")
        f.write("-" * 40 + "\n")
        if all_results['rolling_mse']:
            f.write("90天预测 MSE: " + ", ".join([f"{x:.6f}" for x in all_results['rolling_mse']]) + "\n")
            f.write("90天预测 MAE: " + ", ".join([f"{x:.6f}" for x in all_results['rolling_mae']]) + "\n")
    
    print(f"\n💾 详细结果已保存到: {results_filename}")
    
    # 创建结果保存目录
    import os
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"创建结果目录: {results_dir}")
    
    # 绘制最后一轮的滚动预测结果对比图
    if last_rolling_true is not None and last_rolling_pred is not None:
        print(f"📈 绘制90天滚动预测结果对比图...")
        rolling_title = f"90天滚动预测结果对比-{MODEL_TYPE}_{TRAINING_STRATEGY}"
        plot_comparison(last_rolling_true, last_rolling_pred, 
                       title=rolling_title, save_path=results_dir)
        print(f"💾 90天滚动预测图片已保存到: {results_dir}/{rolling_title}.png")
    
    print(f"\n🎉 训练完成！最终模型已保存到: {MODEL_PATH}")
    print("=" * 80)

if __name__ == "__main__":
    main()
