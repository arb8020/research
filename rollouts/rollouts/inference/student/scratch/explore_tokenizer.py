# /// script
# dependencies = ["torch", "transformers", "accelerate"]
# ///
"""Explore Qwen3-0.6B tokenizer and model loading. Questions to answer:
- What does apply_chat_template produce?
- What are the special token ids?
- Where do weights land with device_map="auto" vs explicit .to("cuda:0")?
- What does model.eval() change?
- How do we verify the model is actually on GPU?
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"

print("=== tokenizer ===")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
print(f"eos_token_id:        {tokenizer.eos_token_id}")
print(f"pad_token_id:        {tokenizer.pad_token_id}")
print(f"special_tokens_map:  {tokenizer.special_tokens_map}")

print("\n=== chat template output ===")
messages = [{"role": "user", "content": "hello"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(repr(text))

token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
print(f"token_ids type: {type(token_ids)}")
print(f"token_ids attrs: {[a for a in dir(token_ids) if not a.startswith('_')]}")

if hasattr(token_ids, "ids"):
    token_ids = token_ids.ids
print(f"token_ids: {token_ids}")
tensor = torch.tensor(token_ids)
print(f"shape: {tensor.shape}")

print("\n=== model load ===")
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()

print("\n=== weight placement (first 5 params) ===")
for name, param in list(model.named_parameters())[:5]:
    print(f"  {name}: device={param.device} dtype={param.dtype} shape={param.shape}")

print("\n=== memory ===")
print(f"  allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"  reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
