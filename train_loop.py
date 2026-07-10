import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 1. 生成数据：2000个点
torch.manual_seed(42)
X = torch.randn(2000, 2)
y = (X[:, 0] ** 2 + X[:, 1] ** 2 < 2).float().unsqueeze(1)

# 2. DataLoader：分批喂数据
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 3. 网络（和会话2一样）
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

model = SimpleNN()
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 4. 训练并记录loss
epochs = 50
loss_history = []

for epoch in range(epochs):
    epoch_loss = 0.0

    for batch_x, batch_y in train_loader:
        pred = model(batch_x)
        loss = loss_fn(pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    loss_history.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}  Loss: {avg_loss:.4f}")

# 5. 画loss曲线
plt.figure(figsize=(8, 4))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss下降曲线")
plt.grid(True)
plt.savefig("loss_curve.png")
print("\nLoss曲线已保存到 loss_curve.png")

# 6. 保存模型
torch.save(model.state_dict(), "model.pth")
print("模型已保存到 model.pth")

# 7. 加载模型验证
model2 = SimpleNN()
model2.load_state_dict(torch.load("model.pth", weights_only=True))
model2.eval()

with torch.no_grad():
    pred_all = model2(X)
    acc = ((pred_all > 0.5) == y).float().mean().item()
print(f"\n加载后的模型准确率：{acc:.2%}")
