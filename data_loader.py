import pandas as pd
import torch


def load_data_with_id_and_labels(path, feature_dim, dtype=torch.float32):
    """保持不变的特征和标签加载函数"""
    df = pd.read_csv(path)

    if 'sample_id' in df.columns:
        ids = df['sample_id'].values
        features_labels = df.drop('sample_id', axis=1).values
    else:
        ids = df.index.values
        features_labels = df.values

    features = features_labels[:, :feature_dim]
    labels = {
        'cancer_or_not': features_labels[:, -3],
        'survival_time': features_labels[:, -2],
        'survival_end': features_labels[:, -1]
    }

    if features.shape[1] != feature_dim:
        raise ValueError(f"特征维度应为{feature_dim}，实际为{features.shape[1]}")

    return (torch.tensor(features, dtype=dtype),
            labels,
            ids)


def load_gene_data(path, dtype=torch.float32):
    return load_data_with_id_and_labels(path, 1024, dtype)


def load_image_data(path, dtype=torch.float32):
    return load_data_with_id_and_labels(path, 768, dtype)


def load_pathway(path, dtype=torch.float32):
    """加载通路掩码（忽略第一列通路名称，形状为[37, 1024]）"""
    df = pd.read_csv(path)

    # 跳过第一列（通路名称），取后面的1024列基因
    pathway_mask = df.iloc[:, 1:].values  # 从第二列开始取数据

    # 验证形状：37个通路 × 1024个基因
    if pathway_mask.shape != (37, 1024):
        raise ValueError(f"通路掩码形状应为(37, 1024)，实际为{pathway_mask.shape}")

    return torch.tensor(pathway_mask, dtype=dtype)


def check_data_alignment(train_ids, train_image_ids, test_ids, test_image_ids):
    if not (train_ids == train_image_ids).all():
        raise ValueError("训练集基因与图像样本ID不匹配")
    if not (test_ids == test_image_ids).all():
        raise ValueError("测试集基因与图像样本ID不匹配")
    print("所有样本ID对齐检查通过！")
