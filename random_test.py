import random
import backward_transformer_04  # 修改导入
from tqdm import tqdm
from collections import Counter

def run_robustness_test(num_runs=100):
    """
    Performs a robustness test on the online_detection main function.
    """
    # 基于 online_detection.py 的 TEST_INDICES
    all_indices = set(range(52))
    test_indices_orig = sorted([0, 16, 18, 25, 27, 32, 40, 43, 48])
    normal_indices_orig = sorted(list(all_indices - set(test_indices_orig)))

    all_results = []
    anomaly_counts = Counter()
    print(f"开始进行 {num_runs} 次随机顺序测试...")

    # 使用 tqdm 显示进度条
    for _ in tqdm(range(num_runs), desc="Robustness Test"):
        normal_indices_shuffled = list(normal_indices_orig)
        test_indices_shuffled = list(test_indices_orig)
        
        random.shuffle(normal_indices_shuffled)
        random.shuffle(test_indices_shuffled)
        
        # 调用 online_detection 的 main 函数，禁止其内部打印
        detected_anomalies = backward_transformer_04.main()
        all_results.append(frozenset(detected_anomalies))
        anomaly_counts.update(detected_anomalies)

    print("\n测试完成。")

    # 检查所有结果是否一致
    unique_results = set(all_results)
    
    if len(unique_results) == 1:
        print("✅ 鲁棒性测试通过！")
        print(f"在 {num_runs} 次测试中，检测到的异常样本始终一致。")
        # 从 frozenset 转换回 list 并排序
        result_list = sorted(list(unique_results.pop()))
        print(f"检测到的异常样本: {result_list}")
    else:
        print("❌ 鲁棒性测试失败！")
        print(f"在 {num_runs} 次测试中，检测到的异常样本发生了变化。")
        print("以下是所有不同的检测结果:")
        for i, result_set in enumerate(unique_results):
            # 从 frozenset 转换回 list 并排序
            result_list = sorted(list(result_set))
            print(f"  结果 {i+1}: {result_list}")

    # 统计并输出每个样本被检测为异常的频数
    print("\n--- 异常样本检测频数统计 ---")
    if not anomaly_counts:
        print("在所有测试中均未检测到异常样本。")
    else:
        print(f"在 {num_runs} 次测试中，各样本被检测为异常的次数如下：")
        # 按样本ID排序后输出
        sorted_counts = sorted(anomaly_counts.items())
        for sample_id, count in sorted_counts:
            print(f"  样本 {sample_id:2d}: {count:3d} 次")

if __name__ == "__main__":
    run_robustness_test()
