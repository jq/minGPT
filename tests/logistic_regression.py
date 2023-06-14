import torch
import torch.nn as nn

# 定义逻辑回归模型
class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear(x)
        out = self.sigmoid(out)
        return out

# 测试代码
# 创建输入数据和标签
inputs = torch.tensor([[0.2, 0.4, 0.6], [0.8, 0.6, 0.4], [0.1, 0.3, 0.5]], dtype=torch.float32)
labels = torch.tensor([[1], [0], [1]], dtype=torch.float32)

# 初始化逻辑回归模型
input_size = inputs.shape[1]
model = LogisticRegression(input_size)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 进行训练
num_epochs = 100
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # 反向传播和优化
    optimizer.zero_grad()
    # https://medium.com/@monadsblog/pytorch-backward-function-e5e2b7e60140
    loss.backward()
    optimizer.step()

    # 打印训练信息
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试模型
test_inputs = torch.tensor([[0.3, 0.5, 0.7], [0.9, 0.7, 0.5]], dtype=torch.float32)
with torch.no_grad():
    predictions = model(test_inputs)
    predicted_labels = (predictions >= 0.5).float()
    print(f'Predictions: {predicted_labels}')
