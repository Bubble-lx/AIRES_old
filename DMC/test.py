import torch

# 假设attn的形状为[1024, 4, 11, 11]，这里用随机数据模拟
attn = torch.rand(1024, 4, 11, 11)

# 提取特定元素
selected_attn = attn[:, :, [1, 3, 5, 7, 9], [2, 4, 6, 8, 10]]

# 在每个注意力头取平均值
mean_attn = selected_attn.mean(dim=1)

mean_attn.shape, mean_attn
