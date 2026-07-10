import torch
import torch.nn as nn

# 1. 生成数据：一组分布在两个区域的点
torch.manual_seed(42)
X = torch.randn(200, 2)
y = (X[:, 0] ** 2 + X[:, 1] ** 2 < 2).float().unsqueeze(1)

# 2. 定义神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 10)
        self.layer2 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        return x

# 3. 初始化
model = SimpleNN()
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 4. 训练
epochs = 100
for epoch in range(epochs):
    pred = model(X)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        acc = ((pred > 0.5) == y).float().mean().item()
        print(f"Epoch {epoch+1:3d}  Loss: {loss.item():.4f}  Acc: {acc:.2f}")

# 5. 可视化
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
with torch.no_grad():
    pred_labels = (model(X) > 0.5).float()

plt.figure(figsize=(6, 5))
colors = ["red" if p == 1 else "blue" for p in pred_labels.squeeze()]
plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.6)
plt.title("训练结果：红色=圆内  蓝色=圆外")
plt.savefig("nn_result.png")
print("\n分类结果图已保存到 nn_result.png")
