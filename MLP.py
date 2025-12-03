import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import feature_engineering_01

class Autoencoder(nn.Module):
    """PyTorch 自编码器模型"""

    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),  # 瓶颈层
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        print(f"[DEBUG] 模型初始化完成 - 输入维度: {input_dim}, 瓶颈层: 32")

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
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
    """数据预处理（带调试输出）"""
    print("\n=== 数据预处理 ===")
    data_array, min_length = feature_engineering_01.preprocess_data(data)
    features_array, feature_names = feature_engineering_01.feature_extraction(data_array, n_sensors_to_select=38)
    features_array = feature_engineering_01.feature_flatten(features_array)
    test_indices = [0, 16, 18, 25, 27, 32, 40, 43, 48]
    train_indices = [i for i in range(52) if i not in test_indices]
    scaler = StandardScaler()
    # 划分数据集
    X_train = np.array(features_array[train_indices])
    X_train = scaler.fit_transform(X_train)
    X_test = np.array(features_array[test_indices])
    X_test = scaler.transform(X_test)
    print(f"[DEBUG] 训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")

    return X_train, X_test


def train_autoencoder(X_train, device):
    """训练自编码器（PyTorch版）"""
    print("\n=== 模型训练 ===")
    input_dim = X_train.shape[1]
    model = Autoencoder(input_dim).to(device)

    optimizer = optim.Adam(model.parameters())
    train_data = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_train, dtype=torch.float32),
        torch.arange(len(X_train))  # 添加样本索引
    )
    loader = DataLoader(train_data, batch_size=8, shuffle=True)

    # 3. 训练循环
    model.train()
    for epoch in range(50):
        total_loss = 0
        for batch_x, batch_y, batch_indices in loader:  # 现在获取三个值
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_indices = batch_indices.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = torch.mean((outputs - batch_y) ** 2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/50 - Loss: {total_loss / len(loader):.4f}")

    # 4. 计算训练集误差
    model.eval()
    with torch.no_grad():
        train_recon = model(torch.tensor(X_train, dtype=torch.float32).to(device)).cpu().numpy()
    train_loss = np.mean((X_train - train_recon) ** 2, axis=1)
    print("\n[DEBUG] 训练集误差统计:")
    print(f"  样本误差均值: {np.mean(train_loss):.4f}")
    return model, train_loss


def detect_anomalies(model, X_test, train_loss, device):
    """异常检测（PyTorch版）"""
    print("\n=== 异常检测 ===")
    model.eval()
    with torch.no_grad():
        test_recon = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
    test_scores = np.mean((X_test - test_recon) ** 2, axis=1)

    # 基于训练集正常样本计算Z-score
    normal_mean = np.mean(train_loss)
    normal_std = np.std(train_loss)
    anomaly_scores = (test_scores - normal_mean) / normal_std
    # anomaly_scores = test_scores

    print("[DEBUG] 测试集异常分数统计:")
    print(f"  最小分数: {np.min(anomaly_scores):.2f}")
    print(f"  最大分数: {np.max(anomaly_scores):.2f}")
    print(f"  阈值建议: > 3.0 (即均值+3σ)")
    return anomaly_scores


def main():
    """主函数"""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n=== 使用设备: {device} ===")

    # 1. 加载数据
    data, labels = load_wafer_data("data.xlsx")

    # 2. 预处理
    X_train, X_test = preprocess_data(data, labels)

    # 3. 训练
    model, train_loss = train_autoencoder(X_train, device)

    # 4. 检测
    anomaly_scores = detect_anomalies(model, X_test, train_loss, device)

    # 5. 输出结果
    samples = [0, 16, 18, 25, 27, 32, 40, 43, 48]
    print("\nMLP === 最终结果 ===")
    for i, score in enumerate(anomaly_scores):
        status = "⚠️ 异常" if score > 500 else "✅ 正常"
        print(f"样本 {samples[i]:2d}: 异常分数 = {score:7.2f} | {status}")


if __name__ == "__main__":
    main()