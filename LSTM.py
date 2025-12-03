import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import math

import feature_engineering_01


class LSTMAutoencoder(nn.Module):
    """优化后的LSTM自编码器模型"""

    def __init__(self, input_dim, seq_len, hidden_dim=64, num_layers=2, latent_dim=16):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim

        # 编码器 (加深网络并添加非线性)
        self.encoder_lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.encoder_lstm2 = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True
        )
        # 更强的瓶颈层约束
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim // 2, latent_dim),
            nn.Tanh()  # 添加非线性约束
        )

        # 解码器 (减少容量)
        self.decoder_lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True
        )
        self.decoder_linear = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        print(f"[DEBUG] 优化后的LSTM模型初始化完成 - 输入维度: {input_dim}, 序列长度: {seq_len}")
        print(f"        隐藏层维度: {hidden_dim}→{hidden_dim // 2}, 瓶颈层: {latent_dim} (带Tanh约束)")

    def forward(self, x):
        # 编码
        x, _ = self.encoder_lstm1(x)
        x, (hidden, _) = self.encoder_lstm2(x)
        encoded = self.bottleneck(hidden[-1])  # 取最后一层的输出

        # 解码 (重复latent vector作为每个时间步的输入)
        decoded = encoded.unsqueeze(1).repeat(1, self.seq_len, 1)
        decoded, _ = self.decoder_lstm(decoded)
        decoded = self.decoder_linear(decoded)
        return decoded


def load_wafer_data(filepath: str) -> tuple[list[np.ndarray], np.ndarray]:
    """
    严格实现功能：
    1. 读取52个Sheet的Excel文件
    2. 每个Sheet转为 (time_steps, 129) 的NumPy数组
    3. 生成标签（前38=0正常，后14=1异常）
    """
    multi_profile_data = feature_engineering_01.load_wafer_data(filepath)
    labels = np.concatenate([np.zeros(38), np.ones(14)])
    return multi_profile_data, labels


def preprocess_data(data, labels):
    """数据预处理（为LSTM返回三维序列: (batch, seq_len, input_dim)）"""
    print("\n=== 数据预处理 ===")
    # data_array: (n_samples, seq_len, n_sensors)
    data_array, min_length = feature_engineering_01.preprocess_data(data)

    # 定义测试/训练索引（与打印用样本一致）
    test_indices = [0, 16, 18, 25, 27, 32, 40, 43, 48]
    n_samples = data_array.shape[0]
    train_indices = [i for i in range(n_samples) if i not in test_indices]

    # 仅基于训练集选择方差最高的传感器，避免信息泄漏
    n_sensors_to_select = 38
    n_sensors = data_array.shape[2]
    train_data = data_array[train_indices]  # (n_train, seq_len, n_sensors)
    sensor_variances = np.var(train_data.reshape(-1, n_sensors), axis=0)
    top_sensor_indices = np.argsort(sensor_variances)[-n_sensors_to_select:]
    top_sensor_indices = np.sort(top_sensor_indices)
    print(f"[DEBUG] 选择的Top传感器索引: {top_sensor_indices.tolist()}")

    # 选择这些传感器，保持序列维度不变
    data_selected = data_array[:, :, top_sensor_indices]  # (n_samples, seq_len, n_selected)

    # 用训练集拟合标准化参数，并应用到全部（避免泄漏）
    train_selected = data_selected[train_indices]  # (n_train, seq_len, n_selected)
    # 按传感器维度计算均值/方差（跨样本与时间步聚合）
    mean_per_sensor = train_selected.reshape(-1, train_selected.shape[2]).mean(axis=0)
    std_per_sensor = train_selected.reshape(-1, train_selected.shape[2]).std(axis=0)
    std_per_sensor[std_per_sensor < 1e-8] = 1.0
    data_selected = (data_selected - mean_per_sensor) / std_per_sensor

    # 划分数据集（保持三维）
    X_train = data_selected[train_indices]
    X_test = data_selected[test_indices]
    print(f"[DEBUG] 训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")

    return X_train, X_test


def train_autoencoder(X_train, device):
    """训练过程（优化损失函数）"""
    print("\n=== 模型训练 ===")
    seq_len, input_dim = X_train.shape[1], X_train.shape[2]
    model = LSTMAutoencoder(input_dim, seq_len).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_data = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_train, dtype=torch.float32),
        torch.arange(len(X_train), dtype=torch.long)  # 确保索引是long类型
    )
    loader = DataLoader(train_data, batch_size=8, shuffle=True)

    # 训练循环
    model.train()
    for epoch in range(150):  # 增加训练轮次
        total_loss = 0
        for batch_x, batch_y, batch_indices in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_indices = batch_indices.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = torch.mean((outputs - batch_y) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step(total_loss / len(loader))
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/150 - Loss: {total_loss / len(loader):.4f} - LR: {optimizer.param_groups[0]['lr']:.2e}")

    # 计算训练集误差
    model.eval()
    with torch.no_grad():
        train_recon = model(torch.tensor(X_train, dtype=torch.float32).to(device)).cpu().numpy()
    train_loss = np.mean((X_train - train_recon) ** 2, axis=(1, 2))

    return model, train_loss


def detect_anomalies(model, X_test, train_loss, device):
    """异常检测（保持不变）"""
    print("\n=== 异常检测 ===")
    model.eval()
    with torch.no_grad():
        test_recon = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
    test_scores = np.mean((X_test - test_recon) ** 2, axis=(1, 2))

    normal_mean = np.mean(train_loss)
    normal_std = np.std(train_loss)
    anomaly_scores = (test_scores - normal_mean) / normal_std

    print("[DEBUG] 测试集异常分数统计:")
    print(f"  最小分数: {np.min(anomaly_scores):.2f}")
    print(f"  最大分数: {np.max(anomaly_scores):.2f}")
    print(f"  阈值建议: > {np.mean(anomaly_scores) + 3 * np.std(anomaly_scores):.2f}")
    return anomaly_scores


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n=== 使用设备: {device} ===")

    data, labels = load_wafer_data("data.xlsx")
    X_train, X_test = preprocess_data(data, labels)
    model, train_loss = train_autoencoder(X_train, device)
    anomaly_scores = detect_anomalies(model, X_test, train_loss, device)

    yangben = [0, 16, 18, 25, 27, 32, 40, 43, 48]
    threshold = 1
    print("\nLSTM === 最终结果 ===")
    for i, score in enumerate(anomaly_scores, start=39):
        status = "⚠️ 异常" if score > threshold else "✅ 正常"
        print(f"样本 {yangben[i - 39]:2d}: 异常分数 = {score:7.2f} | {status}")


if __name__ == "__main__":
    main()