import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random

# 导入同目录下的数据处理模块
import feature_engineering_01


def set_random_seeds(seed: int = 42):
    """统一设置随机种子，保证实验结果可重复"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda_manual_seed_all = getattr(torch, "cuda_manual_seed_all", lambda *_: None)
    torch.cuda_manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TransformerAutoencoder(nn.Module):
    """基于 Transformer 的时序自编码器"""

    def __init__(self, input_dim, seq_len, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.d_model = d_model

        # 输入映射到 d_model 维度
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 压缩到瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Linear(d_model * seq_len, 32),
            nn.Tanh()
        )

        # 瓶颈反投影
        self.latent_projection = nn.Linear(32, d_model * seq_len)

        # 解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 输出映射回原始维度
        self.output_layer = nn.Linear(d_model, input_dim)

        print(f"[DEBUG] Transformer 初始化完成 - 输入维度: {input_dim}, 序列长度: {seq_len}")
        print(f"        d_model: {d_model}, 头数: {nhead}, 层数: {num_layers}")

    def forward(self, x):
        # 编码
        x = self.input_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        memory = self.encoder(x)

        # 压缩
        batch_size = memory.size(0)
        latent = self.bottleneck(memory.reshape(batch_size, -1))

        # 解码
        decoder_input = self.latent_projection(latent).view(batch_size, self.seq_len, self.d_model)
        output = self.decoder(decoder_input, memory)
        output = self.output_layer(output)
        return output


class PositionalEncoding(nn.Module):
    """位置编码"""

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
    读取晶圆生产数据（Excel 文件，52 个 Sheet）
    每个 Sheet 转为 (time_steps, 129) 的数组
    """
    multi_profile_data = feature_engineering_01.load_wafer_data(filepath)
    return multi_profile_data


TEST_INDICES = [0, 16, 18, 25, 27, 32, 40, 43, 48]


def train_autoencoder_online(X_train: np.ndarray, device: torch.device,
                             epochs: int = 100, seed: int = 42) -> tuple[TransformerAutoencoder, np.ndarray]:
    """训练自编码器，返回模型与训练集逐样本 MSE"""
    set_random_seeds(seed)

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
    for epoch in range(epochs):
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

    # 计算训练集 MSE
    model.eval()
    with torch.no_grad():
        recon = model(torch.tensor(X_train, dtype=torch.float32).to(device)).cpu().numpy()
    train_loss = np.mean((X_train - recon) ** 2, axis=(1, 2))

    print(f"\n[DEBUG] 训练误差: μ={np.mean(train_loss):.4f} ± {np.std(train_loss):.4f}")
    return model, train_loss


def select_top_sensors_by_variance(data_array: np.ndarray, train_indices: list[int], num_sensors: int = 36) -> np.ndarray:
    """按训练集方差选择变化最大的传感器"""
    train_data = data_array[train_indices]
    num_all_sensors = train_data.shape[2]
    num_sensors = min(num_sensors, num_all_sensors)
    sensor_variances = np.var(train_data.reshape(-1, num_all_sensors), axis=0)
    top_sensor_indices = np.argsort(sensor_variances)[-num_sensors:]
    return np.sort(top_sensor_indices)


def standardize_by_training(data_selected: np.ndarray, train_indices: list[int]) -> np.ndarray:
    """用当前训练集的均值和方差进行标准化"""
    train_selected = data_selected[train_indices]
    mean_per_sensor = train_selected.reshape(-1, train_selected.shape[2]).mean(axis=0)
    std_per_sensor = train_selected.reshape(-1, train_selected.shape[2]).std(axis=0)
    std_per_sensor[std_per_sensor < 1e-8] = 1.0
    return (data_selected - mean_per_sensor) / std_per_sensor


def evaluate_sample_mse(model: nn.Module, sample: np.ndarray, device: torch.device) -> float:
    """单样本重构误差"""
    x = torch.tensor(sample[None, ...], dtype=torch.float32).to(device)
    with torch.no_grad():
        recon = model(x).cpu().numpy()[0]
    return float(np.mean((sample - recon) ** 2))


def compute_anomaly_score(test_mse: float, normal_train_losses: np.ndarray) -> float:
    """基于训练集 MSE 分布计算 z-score 异常分数"""
    normal_mean = np.mean(normal_train_losses)
    normal_std = np.std(normal_train_losses)
    if normal_std < 1e-8:
        return 0.0
    return (test_mse - normal_mean) / normal_std


def build_detection_order(n_samples: int) -> list[int]:
    """构造检测顺序：先训练集后测试集"""
    test_set = set(TEST_INDICES)
    train_indices = [i for i in range(n_samples) if i not in test_set]
    random.seed(42)
    random.shuffle(train_indices)
    return train_indices + TEST_INDICES


def main():
    set_random_seeds(42)

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    print(f"使用设备: {device}")

    data = load_wafer_data("data.xlsx")
    data_array, _ = feature_engineering_01.preprocess_data(data)
    n_samples, seq_len, n_sensors = data_array.shape
    print(f"[DEBUG] 原始数据维度: {data_array.shape}")

    detection_order = build_detection_order(n_samples)
    print(f"[DEBUG] 检测顺序示例: {detection_order[:10]} ...")

    offline_train_indices = [i for i in range(n_samples) if i not in set(TEST_INDICES)]
    top_sensor_indices = select_top_sensors_by_variance(data_array, offline_train_indices, num_sensors=36)
    print(f"[DEBUG] 选取的传感器索引: {top_sensor_indices.tolist()}")
    data_selected = data_array[:, :, top_sensor_indices]

    INIT_TRAIN_SIZE = 10
    current_train_indices = detection_order[:INIT_TRAIN_SIZE]
    step = INIT_TRAIN_SIZE - 1

    decisions = []

    while step < len(detection_order) - 1:
        step += 1
        candidate_index = detection_order[step]

        standardized_data = standardize_by_training(data_selected, current_train_indices)
        X_train = standardized_data[current_train_indices]

        model, train_losses = train_autoencoder_online(X_train, device=device, epochs=100, seed=42)

        sample_std = standardized_data[candidate_index]
        sample_mse = evaluate_sample_mse(model, sample_std, device)
        anomaly_score = compute_anomaly_score(sample_mse, train_losses)

        threshold = 10
        is_normal = anomaly_score <= threshold

        decisions.append({
            "sample_id": int(candidate_index),
            "mse": float(sample_mse),
            "anomaly_score": float(anomaly_score),
            "threshold": float(threshold),
            "decision": "normal" if is_normal else "anomaly",
            "train_size_before": int(len(current_train_indices)),
        })

        if is_normal:
            current_train_indices.append(candidate_index)

        tag = "✅正常" if is_normal else "⚠️异常"
        print(f"样本 {candidate_index+1:2d} | 异常分数={anomaly_score:7.2f} | 阈值={threshold:4.1f} | {tag} | 训练集规模={len(current_train_indices)}")

    total_normals = sum(1 for d in decisions if d["decision"] == "normal")
    total_anomalies = len(decisions) - total_normals
    print("\n=== 检测完成 ===")
    print(f"累计纳入训练: {len(current_train_indices)} / {n_samples}")
    print(f"正常: {total_normals}，异常: {total_anomalies}")

    print("\n=== 测试集结果 ===")
    test_sample_scores = []
    for d in decisions:
        if d["sample_id"] in TEST_INDICES:
            test_sample_scores.append((d["sample_id"], d["anomaly_score"]))

    test_sample_scores.sort()
    for sample_id, score in test_sample_scores:
        status = "⚠️异常" if score > threshold else "✅正常"
        print(f"样本{sample_id+1:2d}: 分数={score:7.2f} | {status}")


if __name__ == "__main__":
    main()
