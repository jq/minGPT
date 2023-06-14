'''
对每个数据样本的每一层进行规范化，使得该层的输出具有零均值和单位方差variance。
LayerNorm的一个主要特性是它独立地规范化每个样本，而不考虑批次中的其他样本。
它可以使网络在训练过程中更稳定，因为它确保了不论前面的层如何改变其输出，
每一层的输入分布都将保持不变。这可以帮助模型更快地收敛，也能提高模型的总体性能。
'''

import torch
import torch.nn as nn

# 模拟输入
x = torch.randn(2, 3)
print(x.size()[1:])
print(x)

# 创建LayerNorm对象
ln = nn.LayerNorm(x.size()[1:])

# 对输入进行LayerNorm
output = ln(x)

print(output)
