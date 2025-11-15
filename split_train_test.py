import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# 设置随机种子，确保结果可复现
np.random.seed(42)

# 数据路径
data_dir = r"D:\对比模型\fine_tuning_coxpasnet\process_data"
gene_file = os.path.join(data_dir, "1024genes.csv")
image_file = os.path.join(data_dir, "768img_features.csv")

# 输出文件路径
train_gene_file = os.path.join(data_dir, "train_1024genes.csv")
test_gene_file = os.path.join(data_dir, "test_1024genes.csv")
train_image_file = os.path.join(data_dir, "train_768img_features.csv")
test_image_file = os.path.join(data_dir, "test_768img_features.csv")


def main():
    # 1. 加载数据
    print("加载数据...")
    # 加载基因特征数据
    gene_df = pd.read_csv(gene_file)
    # 加载图像特征数据
    image_df = pd.read_csv(image_file)

    # 2. 检查数据是否对齐
    print("检查数据对齐...")
    # 假设数据包含'sample_id'列用于匹配
    if 'sample_id' not in gene_df.columns or 'sample_id' not in image_df.columns:
        # 如果没有sample_id列，假设行顺序一致，创建临时sample_id
        gene_df['sample_id'] = range(len(gene_df))
        image_df['sample_id'] = range(len(image_df))
        print("数据中未发现'sample_id'列，假设行顺序一致并创建临时ID")

    # 确保两个数据集的样本ID完全一致
    common_samples = set(gene_df['sample_id']).intersection(set(image_df['sample_id']))
    if len(common_samples) != len(gene_df) or len(common_samples) != len(image_df):
        print(f"警告：基因数据和图像数据的样本不完全匹配，仅使用共有的{len(common_samples)}个样本")
        # 只保留共同样本
        gene_df = gene_df[gene_df['sample_id'].isin(common_samples)]
        image_df = image_df[image_df['sample_id'].isin(common_samples)]

    # 按sample_id排序，确保顺序一致
    gene_df = gene_df.sort_values('sample_id').reset_index(drop=True)
    image_df = image_df.sort_values('sample_id').reset_index(drop=True)

    # 再次确认样本ID完全一致
    assert (gene_df['sample_id'].values == image_df['sample_id'].values).all(), \
        "基因数据和图像数据的样本ID不匹配，无法继续"

    # 3. 划分训练集和测试集（8:2比例）
    print("划分训练集和测试集...")
    sample_ids = gene_df['sample_id'].values
    train_ids, test_ids = train_test_split(sample_ids, test_size=0.2, random_state=42)

    # 4. 生成训练集和测试集
    print("生成训练集和测试集...")
    # 基因特征
    train_gene = gene_df[gene_df['sample_id'].isin(train_ids)]
    test_gene = gene_df[gene_df['sample_id'].isin(test_ids)]

    # 图像特征
    train_image = image_df[image_df['sample_id'].isin(train_ids)]
    test_image = image_df[image_df['sample_id'].isin(test_ids)]

    # 5. 保存结果
    print("保存结果...")
    train_gene.to_csv(train_gene_file, index=False)
    test_gene.to_csv(test_gene_file, index=False)
    train_image.to_csv(train_image_file, index=False)
    test_image.to_csv(test_image_file, index=False)

    # 6. 打印划分结果
    print(f"总样本数: {len(gene_df)}")
    print(f"训练集样本数: {len(train_gene)} ({len(train_gene) / len(gene_df):.2%})")
    print(f"测试集样本数: {len(test_gene)} ({len(test_gene) / len(gene_df):.2%})")
    print(f"训练基因特征保存至: {train_gene_file}")
    print(f"测试基因特征保存至: {test_gene_file}")
    print(f"训练图像特征保存至: {train_image_file}")
    print(f"测试图像特征保存至: {test_image_file}")
    print("数据划分完成！")


if __name__ == "__main__":
    main()
