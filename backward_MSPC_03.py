import random
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
from typing import List, Tuple

import feature_engineering_01
     
def hotelling_t2_spe_analysis(file_path, normal_indices, test_indices, pca_components=0.95):
    """
    使用霍特林T2统计量和SPE(Q统计量)进行异常检测
    
    参数:
        file_path: Excel数据文件路径
        normal_indices: 正常样本的索引列表(0-based)
        test_indices: 待检测样本的索引列表(0-based)
        pca_components: PCA保留的组件数或方差比例
    """
    features_scaled = feature_engineering_01.run_flatten()
    
    # 5. 划分正常样本和测试样本
    X_train = features_scaled[normal_indices, :]
    X_test = features_scaled[test_indices, :]
    
    # 6. 在正常样本上训练PCA模型
    print("\n训练PCA模型...")
    pca = PCA(n_components=pca_components)
    pca.fit(X_train)
    
    # 7. 计算所有样本的PCA得分
    scores_all = pca.transform(features_scaled)
    
    # 8. 计算T2统计量
    print("\n计算T2统计量...")
    # 计算协方差矩阵的伪逆
    cov_matrix = np.cov(scores_all[normal_indices, :], rowvar=False)
    try:
        inv_cov_matrix = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        inv_cov_matrix = np.linalg.pinv(cov_matrix)
    
    # 计算每个样本的T2统计量
    mean_scores = np.mean(scores_all[normal_indices, :], axis=0)
    t2_scores = []
    for i in range(features_scaled.shape[0]):
        diff = scores_all[i, :] - mean_scores
        t2 = diff @ inv_cov_matrix @ diff.T
        t2_scores.append(t2)
    t2_scores = np.array(t2_scores)
    
    # 计算T2的控制限(95%分位数)
    t2_limit = np.percentile(t2_scores[normal_indices], 95)
    
    # 9. 计算SPE(Q统计量)
    print("\n计算SPE统计量...")
    spe_scores = []
    for i in range(features_scaled.shape[0]):
        # 重构误差
        reconstructed = pca.inverse_transform(scores_all[i, :].reshape(1, -1))
        error = features_scaled[i, :] - reconstructed
        spe = np.sum(error**2)
        spe_scores.append(spe)
    spe_scores = np.array(spe_scores)
    
    # 计算SPE的控制限(95%分位数)
    spe_limit = np.percentile(spe_scores[normal_indices], 95)
    print('SPE的控制限（95%分位数）', spe_limit)
    spe_limit = 1900
    print('SPE的控制限设置为', spe_limit)
    
    # 10. 检测结果
    print("\n检测结果:")
    print("样本索引 | T2统计量 | T2异常 | SPE统计量 | SPE异常 ")
    print("---------------------------------------------------")
    
    for idx in test_indices:
        t2_anomaly = t2_scores[idx] > t2_limit
        spe_anomaly = spe_scores[idx] > spe_limit
        
        print(f"{idx:4d}    | {t2_scores[idx]:7.2f} | {'异常' if t2_anomaly else '正常':^6} | "
              f"{spe_scores[idx]:7.2f} | {'异常' if spe_anomaly else '正常':^6} | ")
    
    return t2_scores, spe_scores, t2_limit, spe_limit

# 在线版：按顺序逐个样本检测，通过则加入训练集（先 normal_indices，后 suspect_indices）
def hotelling_t2_spe_online(file_path, normal_indices, suspect_indices, pca_components=0.95, min_train_size=2, cov_epsilon=1e-6):
    """
    在线检测版本：每次仅对一个候选样本进行T2和SPE检测，若通过则加入训练集并重训PCA。

    参数:
        file_path: Excel数据文件路径
        normal_indices: 认为较为正常的样本索引序列（优先被检测）
        suspect_indices: 可疑样本索引序列（在 normal_indices 之后被检测）
        pca_components: PCA保留的组件数或方差比例

    返回:
        decisions: 每个被检测样本的判定记录列表
        final_train_indices: 最终训练集索引
    """
    features_all = feature_engineering_01.run_flatten()

    # 在线检测顺序：先 normal_indices 再 suspect_indices
    detection_order = list(normal_indices) + list(suspect_indices)
    # 去重且保持顺序（如有交集，以第一次出现为准）
    seen = set()
    detection_order = [i for i in detection_order if not (i in seen or seen.add(i))]

    if len(detection_order) == 0:
        print("没有候选样本可检测")
        return [], []

    # 4) 初始化训练集：至少包含 min_train_size 个样本（直接纳入，不检测）
    init_size = min(max(min_train_size, 1), len(detection_order))
    current_train_indices = detection_order[:init_size]
    decisions = []

    # 辅助函数：基于当前训练集的标准化
    def standardize_by_training_stats(features_matrix, train_idx_list):
        X_train_raw = features_matrix[train_idx_list, :]
        scaler = StandardScaler()
        scaler.fit(X_train_raw)
        return lambda x: scaler.transform(x)

    # 在线循环：从第 init_size 个候选开始逐个检测
    for step in range(init_size, len(detection_order)):
        candidate_idx = detection_order[step]

        # 4.1) 基于当前训练集的标准化
        transform_fn = standardize_by_training_stats(features_all, current_train_indices)
        X_train = transform_fn(features_all[current_train_indices, :])
        x_candidate = transform_fn(features_all[candidate_idx, :].reshape(1, -1))

        # 4.2) 训练/重训 PCA（主成分数不超过样本上限）
        max_components_by_samples = max(1, min(X_train.shape[0] - 1, X_train.shape[1]))
        if isinstance(pca_components, float):
            n_components = max_components_by_samples
        else:
            n_components = min(int(pca_components), max_components_by_samples)
        pca = PCA(n_components=n_components)
        pca.fit(X_train)

        # 4.3) 计算训练集的 T2/SPE 分布与控制限（95%分位数）
        scores_train = pca.transform(X_train)
        cov_matrix = np.atleast_2d(np.cov(scores_train, rowvar=False))
        # 协方差正则化，避免奇异
        if cov_matrix.ndim == 2 and cov_matrix.shape[0] == cov_matrix.shape[1]:
            cov_matrix = cov_matrix + cov_epsilon * np.eye(cov_matrix.shape[0])
        try:
            inv_cov = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov_matrix)
        mean_scores = np.mean(scores_train, axis=0)

        # 训练集 T2/SPE
        t2_train = []
        spe_train = []
        for i in range(scores_train.shape[0]):
            diff_i = scores_train[i, :] - mean_scores
            t2_i = diff_i @ inv_cov @ diff_i.T
            # SPE：重构误差
            recon_i = pca.inverse_transform(scores_train[i, :].reshape(1, -1))
            err_i = X_train[i, :] - recon_i
            spe_i = float(np.sum(err_i**2))
            t2_train.append(float(t2_i))
            spe_train.append(float(spe_i))
        t2_limit = float(np.mean(t2_train) + 3 * np.std(t2_train))
        spe_limit = float(np.mean(spe_train) + 180 * np.std(spe_train))

        # 4.4) 评估候选样本
        score_c = pca.transform(x_candidate)
        diff_c = score_c[0, :] - mean_scores
        t2_c = float(diff_c @ inv_cov @ diff_c.T)
        recon_c = pca.inverse_transform(score_c)
        err_c = x_candidate - recon_c
        spe_c = float(np.sum(err_c**2))

        is_normal = (t2_c <= t2_limit) and (spe_c <= spe_limit)
        if is_normal:
            current_train_indices.append(candidate_idx)

        # 记录与打印
        decisions.append({
            "sample_id": int(candidate_idx),
            "t2": t2_c,
            "spe": spe_c,
            "t2_limit": t2_limit,
            "spe_limit": spe_limit,
            "decision": "normal" if is_normal else "anomaly",
            "train_size_before": int(len(current_train_indices) - (1 if is_normal else 0)),
            "train_size_after": int(len(current_train_indices)),
        })

        tag = "✅正常" if is_normal else "⚠️异常"
        print(f"样本 {candidate_idx:2d} | T2={t2_c:8.3f} | SPE={spe_c:8.3f} | "
              f"限值(T2={t2_limit:8.3f}, SPE={spe_limit:8.3f}) | {tag} | 训练集规模={len(current_train_indices)}")

    print("\n=== MSPC 在线检测完成 ===")
    print(f"最终训练集规模: {len(current_train_indices)}")
    return decisions, current_train_indices

def main(normal_indices_shuffled, test_indices_shuffled):
    """
    Main function for online MSPC analysis.
    Takes shuffled normal and test indices and returns detected anomalies.
    """
    # 在线版MSPC，其中suspect_indices被视为测试集
    decisions, final_train = hotelling_t2_spe_online(
        "data.xlsx",
        normal_indices=normal_indices_shuffled,
        suspect_indices=test_indices_shuffled,
        pca_components=5,
        min_train_size=7,
        cov_epsilon=1e-6
    )
    
    anomalies = [d['sample_id'] for d in decisions if d['decision'] == 'anomaly']
    return sorted(anomalies)

# 示例使用
if __name__ == "__main__":
    # 正常样本索引(0-based): 所有样本中排除可疑样本和已知异常样本
    all_indices = set(range(52))
    
    # 正常样本 = 所有样本 - 已知异常 - 可疑样本
    normal_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 20, 21, 22, 23, 24, 26, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 41, 42, 44, 45, 46, 47, 49, 50, 51]
    test_indices = sorted(list(all_indices - set(normal_indices)))

    # 随机打乱顺序
    random.shuffle(normal_indices)
    random.shuffle(test_indices)
    
    print(f"正常样本数量: {len(normal_indices)}")
    print(f"测试样本数量: {len(test_indices)}")
    
    # 运行分析
    detected_anomalies = main(normal_indices, test_indices)
    print(f"检测到的异常样本: {detected_anomalies}")
