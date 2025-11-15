import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # 指定图形后端
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import torch
import torch.nn.functional as F

# ---------------------- 1. 初始化参数 ----------------------
num_pathways = 3  # 通路数
num_genes = 5  # 每个通路的基因数

# 生成先验掩码（转为PyTorch张量，避免后续类型问题）
init_mask_np = np.array([
    [1, 1, 0, 0, 0],  # 通路1初始关联基因1、2
    [0, 0, 1, 1, 0],  # 通路2初始关联基因3、4
    [0, 0, 0, 1, 1]  # 通路3初始关联基因4、5
], dtype=np.float32)
init_mask = torch.tensor(init_mask_np, dtype=torch.float32)  # 转为张量

# 初始化可学习的logits参数
mask_logits = torch.tensor(
    np.log(init_mask_np + 1e-8),
    dtype=torch.float32,
    requires_grad=True
)
temperature = torch.tensor(0.5, dtype=torch.float32)
optimizer = torch.optim.Adam([mask_logits], lr=0.1)

# ---------------------- 2. 定义动画更新函数 ----------------------
fig, ax = plt.subplots(figsize=(8, 6))


def update(frame):
    global mask_logits

    # 1. 生成当前掩码（带梯度）
    current_mask = F.softmax(mask_logits / temperature, dim=-1)

    # 2. 计算损失（确保两个操作数都是张量，避免类型混合）
    loss = torch.norm(current_mask - init_mask, p=2)  # 现在两边都是张量

    # 3. 反向传播更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 4. 生成用于绘图的掩码（分离梯度，转为NumPy）
    with torch.no_grad():  # 临时禁用梯度计算
        gumbel_noise = torch.rand_like(mask_logits) * 1e-10
        dynamic_mask = F.softmax((mask_logits + gumbel_noise) / temperature, dim=-1)
        dynamic_mask_np = dynamic_mask.cpu().numpy()  # 安全转换为NumPy

    # 5. 绘制热力图
    ax.clear()
    sns.heatmap(
        dynamic_mask_np,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        ax=ax,
        cbar=True,
        vmin=0, vmax=1
    )
    ax.set_title(f'动态通路掩码训练过程 - 步骤 {frame + 1}')
    ax.set_xlabel('基因索引')
    ax.set_ylabel('通路索引')

    return ax,


# ---------------------- 3. 生成动画并保存 ----------------------
ani = animation.FuncAnimation(
    fig,
    update,
    frames=50,
    interval=300,
    blit=False
)

ani.save(
    'dynamic_pathway_mask.gif',
    writer='pillow',
    fps=3,
    dpi=100
)

plt.close()
