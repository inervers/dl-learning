import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), "hf_cache")
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from transformers import pipeline

# 1. 情感分析
classifier = pipeline("sentiment-analysis")

tests = [
    "I love this movie!",
    "This is the worst film ever",
    "The acting was amazing",
    "Terrible waste of time",
    "Not bad, actually quite good",
]

print("=== BERT 情感分类 ===")
for text in tests:
    result = classifier(text)[0]
    label = "正面" if result["label"] == "POSITIVE" else "负面"
    print(f"  {text:40s} → {label}  (信心：{result['score']:.1%})")
