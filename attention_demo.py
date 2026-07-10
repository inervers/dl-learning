import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 假设一句话有3个词，每个词用4维向量表示
words = ["我", "爱", "你"]
X = torch.tensor([
    [1.0, 1.0, 0.0, 0.0],   # "我" 的向量
    [0.5, 0.8, 1.0, 0.2],   # "爱" 的向量
    [0.0, 1.0, 1.0, 1.0],   # "你" 的向量
])

print("=== 注意力机制 ===")
# 2. 自己跟自己算注意力分数（自注意力）
#    每个词和其他所有词算相似度（点积）
scores = torch.mm(X, X.T)  # 3x3 矩阵
print(f"\n注意力分数（原始）：")
print(scores.numpy())

# 3. 缩放并转成概率分布
d_k = X.shape[1]  # 向量维度=4
attention_weights = F.softmax(scores / d_k ** 0.5, dim=-1)
print(f"\n注意力权重（每行和为1）：")
print(attention_weights.numpy())

# 4. 用注意力权重加权求和，得到输出
output = torch.mm(attention_weights, X)
print(f"\n输出的新向量（每个词融合了上下文信息）：")
print(output.numpy())

print("\n仔细观察：")
print(f"  '我' 的新向量中融入了 {attention_weights[0,1]:.2%} 的'爱' 和 {attention_weights[0,2]:.2%} 的'你'")
print(f"  '爱' 的新向量中融入了 {attention_weights[1,0]:.2%} 的'我' 和 {attention_weights[1,2]:.2%} 的'你'")

# 5. 用一个具体问题演示注意力的作用
print(f"\n\n=== 可视化：查询'爱'对各个词的关注度 ===")
query = X[1]  # "爱"
for i, word in enumerate(words):
    score = torch.dot(query, X[i])
    weight = F.softmax(scores[1] / d_k ** 0.5, dim=-1)[i]
    print(f"  '爱' → '{word}'：分数={score:.2f}，权重={weight:.2%}")

# 6. 完整包装成一个简单注意力层
class SimpleAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        scores = torch.mm(x, x.T)
        weights = F.softmax(scores / self.d_model ** 0.5, dim=-1)
        return torch.mm(weights, x)

attn = SimpleAttention(4)
output = attn(X)
print(f"\n用包装好的注意力层计算：")
print(f"输入形状：{X.shape}")
print(f"输出形状：{output.shape}")
