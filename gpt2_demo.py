import sys
sys.path.insert(0, r"C:\Users\inervers\Desktop\OH-WorkSpace\dl-learning\pip-target")

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), "hf_cache")
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 1. 加载 GPT-2
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 2. 提示词
prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt")

# 3. 生成文本
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.8,
    pad_token_id=tokenizer.eos_token_id,
)

generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("=== GPT-2 文本生成 ===")
print(f"\n输入：{prompt}")
print(f"输出：{generated}")
