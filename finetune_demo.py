import sys
sys.path.insert(0, r"C:\Users\inervers\Desktop\OH-WorkSpace\dl-learning\pip-target")

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), "hf_cache")
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn

# 1. 加载预训练模型
model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 2. 测试：含无情感词的编程语句
test_texts = [
    # 常规电影情感（BERT 已经会的）
    "i really like this movie",
    "this is a bad film",
    # ---------- 分界线 ----------
    # 技术语句：没有明显的情感词，但语境决定情感
    # "low memory footprint" 在编程里是褒义，对 BERT 来说是中性
    "low memory footprint",
    # "clean code" 也是褒义
    "this code is clean",
    # "high cpu usage" 是贬义
    "high cpu usage",
    # "runtime error" 是贬义
    "runtime error detected",
]

def predict(texts):
    enc = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        outputs = model(**enc)
        probs = torch.softmax(outputs.logits, dim=-1)
    return probs[:, 1]

print("=== 微调前：BERT 对技术语境的情感判断 ===")
for text in test_texts:
    p = predict([text]).item()
    print(f"  {text:35s} → {'正面' if p > 0.5 else '负面'}  (正面信心：{p:.1%})")

# 3. 准备训练数据
train_texts = [
    # 技术正面
    "low memory footprint",
    "this code is clean",
    "fast execution speed",
    "elegant design pattern",
    "good test coverage",
    # 技术负面
    "high cpu usage",
    "runtime error detected",
    "memory leak critical bug",
    "spaghetti code no structure",
    "ugly hacky workaround",
]
train_labels = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

# 4. 分词
encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt")
input_ids = encodings["input_ids"]
attention_mask = encodings["attention_mask"]

# 5. 手动微调
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

model.train()
print("\n微调中...")
for epoch in range(3):
    total_loss = 0
    for i in range(0, len(train_texts), 5):
        batch_ids = input_ids[i:i+5]
        batch_mask = attention_mask[i:i+5]
        batch_labels = train_labels[i:i+5]

        optimizer.zero_grad()
        outputs = model(batch_ids, attention_mask=batch_mask, labels=batch_labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_acc = ((predict(train_texts) > 0.5) == train_labels).float().mean().item()
    print(f"  Epoch {epoch+1}/3  Loss: {total_loss/2:.4f}  Train Acc: {train_acc:.0%}")

# 6. 微调后再测试
print("\n=== 微调后 ===")
for text in test_texts:
    p = predict([text]).item()
    print(f"  {text:35s} → {'正面' if p > 0.5 else '负面'}  (正面信心：{p:.1%})")
