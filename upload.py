import json
from datasets import Dataset

data = []

with open("/workspace/logits_len/multilingual_dataset3.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            print(f"[Warning] Skipping empty line at line {i}")
            continue
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"[Error] Failed to parse line {i}: {e}")
            print(line)
            continue  

dataset = Dataset.from_list(data)
dataset.push_to_hub("rica40325/multestdata")
