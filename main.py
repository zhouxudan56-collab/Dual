import torch
import os
from data_loader import load_gene_data, load_image_data, load_pathway, check_data_alignment
from model import PASNetFeatureExtractor, MultimodalPASNet
from train_eval import train_multimodal_pasnet, plot_results, auc, f1
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 设备设置
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 配置参数
params = {
    'in_nodes': 1024,
    'pathway_nodes': 37,
    'hidden_nodes': 100,
    'fusion_dim': 256,
    'lr': 0.001,
    'l2': 0.001,
    'epochs': 50,
    'dropout': [0.5, 0.4],
    'save_dir': 'multimodal_results'
}

# 数据路径（请确认路径正确）
data_paths = {
    'train_gene': r"D:\对比模型\fine_tuning_coxpasnet\process_data\train_1024genes.csv",
    'test_gene': r"D:\对比模型\fine_tuning_coxpasnet\process_data\test_1024genes.csv",
    'train_image': r"D:\对比模型\fine_tuning_coxpasnet\process_data\train_768img_features.csv",
    'test_image': r"D:\对比模型\fine_tuning_coxpasnet\process_data\test_768img_features.csv",
    'pathway': r"D:\对比模型\fine_tuning_coxpasnet\process_data\pathway_mask.csv"  # 请确认通路文件路径正确
}


def main():
    # 步骤1：加载数据（包含特征和标签分离）
    print("加载数据...")
    # 加载基因数据（1024特征 + 3标签）
    x_train_gene, train_gene_labels, train_ids = load_gene_data(data_paths['train_gene'])
    x_test_gene, test_gene_labels, test_ids = load_gene_data(data_paths['test_gene'])

    # 加载图像数据（768特征 + 3标签）
    x_train_image, train_image_labels, train_image_ids = load_image_data(data_paths['train_image'])
    x_test_image, test_image_labels, test_image_ids = load_image_data(data_paths['test_image'])

    # 检查数据对齐
    check_data_alignment(train_ids, train_image_ids, test_ids, test_image_ids)

    # 提取标签（使用基因数据的标签，与图像标签应一致）
    y_train = {
        'cancer_or_not': torch.tensor(train_gene_labels['cancer_or_not'], dtype=torch.float32).unsqueeze(1),
        'survival_time': torch.tensor(train_gene_labels['survival_time'], dtype=torch.float32).unsqueeze(1),
        'survival_end': torch.tensor(train_gene_labels['survival_end'], dtype=torch.float32).unsqueeze(1)
    }

    y_test = {
        'cancer_or_not': torch.tensor(test_gene_labels['cancer_or_not'], dtype=torch.float32).unsqueeze(1),
        'survival_time': torch.tensor(test_gene_labels['survival_time'], dtype=torch.float32).unsqueeze(1),
        'survival_end': torch.tensor(test_gene_labels['survival_end'], dtype=torch.float32).unsqueeze(1)
    }

    # 步骤2：数据转移到设备
    x_train_gene = x_train_gene.to(device)
    x_train_image = x_train_image.to(device)
    x_test_gene = x_test_gene.to(device)
    x_test_image = x_test_image.to(device)

    y_train = {k: v.to(device) for k, v in y_train.items()}
    y_test = {k: v.to(device) for k, v in y_test.items()}

    # 步骤3：加载通路掩码
    pathway_mask = load_pathway(data_paths['pathway']).to(device)

    # 步骤4：训练模型（使用cancer_or_not作为分类目标）
    print("开始训练...")
    model, train_losses, val_losses, train_aucs, val_aucs, pred_test = train_multimodal_pasnet(
        x_train_gene, x_train_image, y_train['cancer_or_not'],  # 使用肿瘤标签作为主要训练目标
        x_test_gene, x_test_image, y_test['cancer_or_not'],
        pathway_mask, params, device
    )

    # 步骤5：评估模型
    print("评估模型...")
    test_auc = auc(y_test['cancer_or_not'].cpu(), pred_test.cpu())
    test_f1 = f1(y_test['cancer_or_not'].cpu(), pred_test.cpu())
    print(f"测试集性能:")
    print(f"AUC: {test_auc:.4f}")
    print(f"F1分数: {test_f1:.4f}")

    # 提取注意力权重
    model.eval()
    with torch.no_grad():
        _, attn_weights = model(x_test_gene, x_test_image)

    # 步骤6：可视化结果
    print("生成可视化结果...")
    plot_results(
        train_losses, val_losses, train_aucs, val_aucs,
        attn_weights, save_dir=params['save_dir']
    )

    # 步骤7：保存模型
    print("保存模型...")
    os.makedirs(os.path.join(params['save_dir'], 'models'), exist_ok=True)
    torch.save(model.state_dict(),
               os.path.join(params['save_dir'], 'models', 'multimodal_pasnet_weights.pth'))
    print(f"模型已保存至 {os.path.join(params['save_dir'], 'models')}")


if __name__ == "__main__":
    main()
