import re
from collections import Counter

# 1. 按空格分词（最基础的方式）
text = "I love deep learning. Deep learning is amazing!"
tokens = text.lower().split()
print("1. 按空格分词：", tokens)

# 2. 用正则分词（去掉标点）
tokens2 = re.findall(r"\b\w+\b", text.lower())
print("2. 去掉标点：", tokens2)

# 3. 构建词表
vocab = {word: i for i, word in enumerate(set(tokens2))}
print("\n3. 词表：")
for word, idx in vocab.items():
    print(f"   {idx}: {word}")

# 4. 文本转ID序列
text_ids = [vocab[word] for word in tokens2]
print(f"\n4. 文本转ID序列：{text_ids}")

# 5. 用Counter统计词频
freq = Counter(tokens2)
print(f"\n5. 词频统计：")
for word, count in freq.most_common():
    print(f"   {word}: {count}")

# 6. 用词频构建更实用的词表（按频率排序）
sorted_words = sorted(freq, key=freq.get, reverse=True)
vocab2 = {word: i + 1 for i, word in enumerate(sorted_words)}
vocab2["<UNK>"] = 0  # 0号留给未知词
print(f"\n6. 按频率排序的词表：")
for word, idx in vocab2.items():
    print(f"   {idx}: {word}")

# 7. 模拟词嵌入的查找
import torch

embedding = torch.nn.Embedding(len(vocab2), 4)
sample_ids = torch.tensor([vocab2.get(w, 0) for w in "deep learning".split()])
vectors = embedding(sample_ids)
print(f"\n7. 'deep learning' 的嵌入向量：")
print(f"   形状：{vectors.shape}")
print(f"   向量：\n{vectors}")
