import torch
import torch.nn as nn
import torch.optim as optim

# 1. 更多训练数据
sentences = [
    # 正面
    ("this movie is great", 1),
    ("i love this film", 1),
    ("amazing story and acting", 1),
    ("very good movie", 1),
    ("fantastic experience", 1),
    ("excellent film", 1),
    ("i really enjoyed it", 1),
    ("wonderful performance", 1),
    ("one of the best movies", 1),
    ("highly recommended", 1),
    ("i love this story", 1),
    ("great acting and plot", 1),
    # 负面
    ("terrible waste of time", 0),
    ("bad acting and boring plot", 0),
    ("worst movie ever", 0),
    ("i hate this film", 0),
    ("awful and disappointing", 0),
    ("very bad movie", 0),
    ("boring and useless", 0),
    ("terrible experience", 0),
    ("not worth watching", 0),
    ("disappointing film", 0),
    ("poor quality movie", 0),
    ("hate this terrible story", 0),
]

# 2. 构建词表
vocab = {"<PAD>": 0, "<UNK>": 1}
for text, _ in sentences:
    for word in text.split():
        if word not in vocab:
            vocab[word] = len(vocab)

vocab_size = len(vocab)
embed_dim = 8
max_len = 6

def encode(text):
    ids = [vocab.get(word, vocab["<UNK>"]) for word in text.split()]
    ids = ids[:max_len]
    ids += [vocab["<PAD>"]] * (max_len - len(ids))
    return ids

X = torch.tensor([encode(text) for text, _ in sentences])
y = torch.tensor([[label] for _, label in sentences], dtype=torch.float32)

class SentimentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        pooled = embedded.mean(dim=1)
        out = self.sigmoid(self.fc(pooled))
        return out

model = SentimentClassifier()
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 3. 训练
print("训练开始：")
for epoch in range(500):
    pred = model(X)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        acc = ((pred > 0.5) == y).float().mean().item()
        print(f"  Epoch {epoch+1:4d}  Loss: {loss.item():.4f}  Acc: {acc:.2%}")

# 4. 测试
test_sentences = [
    "good movie",
    "i love this story",
    "amazing film",
    "boring film",
    "terrible acting",
    "bad movie",
    "hate it",
]

print(f"\n测试新句子：")
model.eval()
with torch.no_grad():
    for s in test_sentences:
        ids = torch.tensor([encode(s)])
        prob = model(ids).item()
        label = "正面" if prob > 0.5 else "负面"
        print(f"  {s:20s} → {label}  (信心：{prob:.0%})")
