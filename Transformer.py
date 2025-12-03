import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import math
import random
import feature_engineering_01
import matplotlib.pyplot as plt

def set_random_seeds(seed: int = 42):
    """设置所有随机种子，确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TransformerAutoencoder(nn.Module):
    """基于Transformer的自编码器（修正版）"""

    def __init__(self, input_dim, seq_len, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.d_model = d_model  # 修正：保存为成员变量

        # 1. 输入嵌入层
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # 2. Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Linear(d_model * seq_len, 32),
            nn.Tanh()
        )

        # 4. 解码器输入投影
        self.latent_projection = nn.Linear(32, d_model * seq_len)

        # 5. Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 6. 输出层
        self.output_layer = nn.Linear(d_model, input_dim)

        print(f"[DEBUG] Transformer模型初始化完成 - 输入维度: {input_dim}, 序列长度: {seq_len}")
        print(f"        d_model: {d_model}, 头数: {nhead}, 层数: {num_layers}")

    def forward(self, x):
        # 编码阶段
        x = self.input_embedding(x) * math.sqrt(self.d_model)  # 修正：使用self.d_model
        x = self.pos_encoder(x)
        memory = self.encoder(x)

        # 瓶颈压缩
        batch_size = memory.size(0)
        latent = self.bottleneck(memory.reshape(batch_size, -1))

        # 解码阶段
        decoder_input = self.latent_projection(latent).view(batch_size, self.seq_len, self.d_model)
        output = self.decoder(decoder_input, memory)
        output = self.output_layer(output)
        return output


class PositionalEncoding(nn.Module):
    """位置编码层（保持不变）"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


def load_wafer_data(filepath: str) -> list[np.ndarray]:
    """
    严格实现功能：读取52个Sheet的Excel文件
    """
    multi_profile_data = feature_engineering_01.load_wafer_data(filepath)
    return multi_profile_data


def preprocess_data(data):
    """数据预处理（无标签版本，返回三维序列: (batch, seq_len, input_dim)）"""
    print("\n=== 数据预处理 ===")
    data_array, min_length = feature_engineering_01.preprocess_data(data)

    test_indices = [0, 16, 18, 25, 27, 32, 40, 43, 48]
    n_samples = data_array.shape[0]
    train_indices = [i for i in range(n_samples) if i not in test_indices]

    n_sensors_to_select = 36
    n_sensors = data_array.shape[2]
    train_data = data_array[train_indices]
    sensor_variances = np.var(train_data.reshape(-1, n_sensors), axis=0)
    top_sensor_indices = np.argsort(sensor_variances)[-n_sensors_to_select:]
    top_sensor_indices = np.sort(top_sensor_indices)
    print(f"[DEBUG] 选择的Top传感器索引: {top_sensor_indices.tolist()}")

    data_selected = data_array[:, :, top_sensor_indices]

    train_selected = data_selected[train_indices]
    mean_per_sensor = train_selected.reshape(-1, train_selected.shape[2]).mean(axis=0)
    std_per_sensor = train_selected.reshape(-1, train_selected.shape[2]).std(axis=0)
    std_per_sensor[std_per_sensor < 1e-8] = 1.0
    data_selected = (data_selected - mean_per_sensor) / std_per_sensor

    X_train = data_selected[train_indices]
    X_test = data_selected[test_indices]
    print(f"[DEBUG] 训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")

    return X_train, X_test


def train_autoencoder(X_train, device):
    """训练过程（无标签版本，纯MSE自编码器训练）"""
    set_random_seeds(42)

    seq_len, input_dim = X_train.shape[1], X_train.shape[2]
    model = TransformerAutoencoder(input_dim, seq_len).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_train, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=min(8, len(X_train)), shuffle=True)

    model.train()
    for epoch in range(100):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = torch.mean((outputs - batch_y) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step(total_loss / max(1, len(loader)))

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/100 | Loss: {total_loss / len(loader):.4f}")

    model.eval()
    with torch.no_grad():
        recon = model(torch.tensor(X_train, dtype=torch.float32).to(device)).cpu().numpy()
    train_loss = np.mean((X_train - recon) ** 2, axis=(1, 2))

    print("\n[DEBUG] 训练误差统计:")
    print(f"训练样本MSE: μ={np.mean(train_loss):.4f} ± {np.std(train_loss):.4f}")
    return model, train_loss


def detect_anomalies(model, X_test, train_loss, device):
    """异常检测（无标签版本）"""
    model.eval()
    with torch.no_grad():
        test_recon = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()

    test_scores = np.mean((X_test - test_recon) ** 2, axis=(1, 2))

    normal_mean = np.mean(train_loss)
    normal_std = np.std(train_loss)
    anomaly_scores = (test_scores - normal_mean) / normal_std

    print("\n[DEBUG] 异常分数统计:")
    print(f"范围: [{np.min(anomaly_scores):.2f}, {np.max(anomaly_scores):.2f}]")
    print(f"动态阈值: {np.mean(anomaly_scores) + 3 * np.std(anomaly_scores):.2f}")
    return anomaly_scores


def analyze_with_reconstruction_error(X_test, test_recon, sample_indices, top_n=10):
    """
    直接使用重建误差分析异常样本的特征贡献
    :param X_test: 原始测试数据 (n_samples, seq_len, n_features)
    :param test_recon: 重建数据 (n_samples, seq_len, n_features)
    :param sample_indices: 要分析的样本索引列表
    :param top_n: 显示最重要的前n个特征
    """
    total_errors = np.zeros(X_test.shape[2])

    biaohao = [1, 2, 3, 4, 8, 9, 14, 23, 24, 26, 28, 29, 30, 32, 33, 35, 37, 38, 39, 40, 44, 45, 46, 51, 52, 63, 65, 66, 69, 70, 72, 106, 108, 110, 120, 122]
    shun = [0, 16, 18, 25, 27, 32, 40, 43, 48]

    print("\n=== 各样本重构误差Top传感器分析 ===")
    for idx in range(9):
        errors = (X_test[idx] - test_recon[idx]) ** 2
        mean_errors = np.mean(errors, axis=0)
        total_errors += mean_errors
        top_indices = np.argsort(mean_errors)[-top_n:][::-1]
        top_errors = mean_errors[top_indices]

        print(f"\n样本 {shun[idx]+1} 重构误差Top 4传感器:", end=' ')
        tt = 0
        for i, (sensor_idx, error) in enumerate(zip(top_indices, top_errors), 1):
            tt += 1
            print(f"{i}. 传感器{biaohao[sensor_idx]+1}: {error:.2f}", end='  ')
            if tt == 4:
                tt = 0
                break

    print("\n=== 所有异常样本传感器误差汇总 ===")

    top_total_indices = np.argsort(total_errors)[-top_n:][::-1]
    top_total_errors = total_errors[top_total_indices]

    plt.figure(figsize=(12, 6))
    plt.bar(range(top_n), top_total_errors, tick_label=top_total_indices)
    plt.title(f"Top {top_n} Most Anomalous Sensors Across All Samples (Total Reconstruction Error)")
    plt.xlabel("Sensor Index")
    plt.ylabel("Total Reconstruction Error")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    print("\n传感器总误差排名:")
    for i, (sensor_idx, error) in enumerate(zip(top_total_indices, top_total_errors), 1):
        print(f"{i}. 传感器{biaohao[sensor_idx]+1}: 总误差={error:.4f}")

    return total_errors


def main():
    set_random_seeds(42)

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    print(f"使用设备: {device}")

    data = load_wafer_data("data.xlsx")
    X_train, X_test = preprocess_data(data)

    model, train_loss = train_autoencoder(X_train, device)
    scores = detect_anomalies(model, X_test, train_loss, device)

    sample_indices = [0, 16, 18, 25, 27, 32, 40, 43, 48]  # 要分析的异常样本索引
    analyze_with_reconstruction_error(X_test, model(torch.tensor(X_test, dtype=torch.float32).to(device)).detach().cpu().numpy(), sample_indices, top_n=10)


    threshold = 8
    print("\nTansformer === 检测结果 ===")
    for i, (sid, score) in enumerate(zip(sample_indices, scores), 39):
        status = "⚠️异常" if score > threshold else "✅正常"
        print(f"样本{sid+1:2d}: 分数={score:7.2f} | {status}")


if __name__ == "__main__":
    main()
