import torch
import torch.nn as nn
import torch.optim as optim
import re
import os
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# =============================================
# 第一部分：你自己的模型（从零构建）
# =============================================

# 1. 数据
corpus = [
    ("this movie is fantastic", 1),
    ("i love every minute", 1),
    ("absolutely brilliant film", 1),
    ("best movie i have seen", 1),
    ("amazing story and acting", 1),
    ("really enjoyed this film", 1),
    ("fantastic performance", 1),
    ("great directing", 1),
    ("this movie is terrible", 0),
    ("i hate this boring film", 0),
    ("worst movie ever made", 0),
    ("complete waste of time", 0),
    ("awful acting and plot", 0),
    ("boring and predictable", 0),
    ("disappointing experience", 0),
    ("terrible film", 0),
]

# 2. 词表
vocab = {"<PAD>": 0, "<UNK>": 1}
for text, _ in corpus:
    for word in text.split():
        word = word.lower().strip(".,!?")
        if word not in vocab:
            vocab[word] = len(vocab)

VOCAB_SIZE = len(vocab)
EMBED_DIM = 16
MAX_LEN = 6

def tokenize(text):
    words = [w.lower().strip(".,!?") for w in text.split()]
    ids = [vocab.get(w, vocab["<UNK>"]) for w in words[:MAX_LEN]]
    ids += [0] * (MAX_LEN - len(ids))
    return torch.tensor(ids)

X = torch.stack([tokenize(t) for t, _ in corpus])
y = torch.tensor([[l] for _, l in corpus], dtype=torch.float32)

# 3. 你的模型：Embedding + 可选的注意力层
class YourClassifier(nn.Module):
    def __init__(self, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        if use_attention:
            self.query = nn.Linear(EMBED_DIM, EMBED_DIM)
            self.key = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.fc = nn.Linear(EMBED_DIM, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        emb = self.embedding(x)  # (batch, max_len, embed_dim)

        if self.use_attention:
            Q = self.query(emb)
            K = self.key(emb)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (EMBED_DIM ** 0.5)
            weights = torch.softmax(scores, dim=-1)
            context = torch.matmul(weights, emb)
            pooled = context.mean(dim=1)
        else:
            pooled = emb.mean(dim=1)

        return self.sigmoid(self.fc(pooled))

# 4. 训练函数
def train_model(model, name, epochs=200):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()
    losses = []

    for epoch in range(epochs):
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"{name}:")
    print(f"  训练准确率: {((pred > 0.5) == y).float().mean().item():.0%}")
    print(f"  最终 Loss: {losses[-1]:.4f}")
    return losses

# 5. 分别训练两个版本
print("=" * 50)
print("从零构建的模型训练")
print("=" * 50)

baseline_model = YourClassifier(use_attention=False)
attn_model = YourClassifier(use_attention=True)

loss_baseline = train_model(baseline_model, "  平均池化（无注意力）")
loss_attn = train_model(attn_model, "  加注意力层")

# =============================================
# 第二部分：对比测试
# =============================================

test_sentences = [
    "great movie",
    "i love this",
    "boring film",
    "terrible acting",
    "i hate this",
    "amazing movie",
    "worst film ever",
    "good directing",
]

print("\n" + "=" * 50)
print("测试结果对比")
print("=" * 50)
print(f"{'句子':25s} {'真实':7s} {'平均池化':12s} {'加注意力':12s}")
print("-" * 55)

truth_map = {
    "great movie": 1, "i love this": 1, "amazing movie": 1,
    "good directing": 1, "boring film": 0, "terrible acting": 0,
    "i hate this": 0, "worst film ever": 0,
}

with torch.no_grad():
    for s in test_sentences:
        ids = tokenize(s).unsqueeze(0)
        p1 = baseline_model(ids).item()
        p2 = attn_model(ids).item()
        t = truth_map.get(s, "?")
        l1 = "正" if p1 > 0.5 else "负"
        l2 = "正" if p2 > 0.5 else "负"
        print(f"{s:25s} {str(t):7s} {l1}({p1:.0%}){'':5s} {l2}({p2:.0%})")

# =============================================
# 第三部分：用 BERT 做同样测试
# =============================================

import sys
sys.path.insert(0, r"C:\Users\inervers\Desktop\OH-WorkSpace\dl-learning\pip-target")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), "hf_cache")
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from transformers import pipeline

print("\n" + "=" * 50)
print("BERT 对比")
print("=" * 50)

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
)

for s in test_sentences:
    r = classifier(s)[0]
    label = "正" if r["label"] == "POSITIVE" else "负"
    print(f"  {s:25s} → {label} (信心:{r['score']:.0%})")

# =============================================
# 第四部分：Loss 对比图
# =============================================

plt.figure(figsize=(8, 4))
plt.plot(loss_baseline, label="平均池化（无注意力）", alpha=0.8)
plt.plot(loss_attn, label="加注意力层", alpha=0.8)
plt.xlabel("训练轮次")
plt.ylabel("Loss")
plt.title("训练曲线对比")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("final_comparison.png", dpi=100)
print(f"\n训练曲线图已保存：final_comparison.png")
