import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
def load_banned_token_ids_from_file(file_path):
    banned_token_ids = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                try:
                    token_id = int(parts[0])
                    banned_token_ids.append(token_id)
                except ValueError:
                    continue
    return banned_token_ids

def filter_english_in_hidden(model, hidden_states, layer_idx, tokenizer, topk=5):
    token_file_path = "/workspace/logits-hidden/english_tokens_with_text.txt"
    banned_token_ids = load_banned_token_ids_from_file(token_file_path)
    banned_token_ids_set = set(banned_token_ids)
    modified_hidden = hidden_states.clone()
    seq_len = hidden_states.shape[1]
    λ = 2  # 基础抑制系数

    # Precompute normed hidden states and logits
    if hasattr(model.model, 'norm'):
        normed_hidden = model.model.norm(hidden_states)
    else:
        normed_hidden = hidden_states
    logits = model.lm_head(normed_hidden)  # [batch, seq, vocab]

    for pos in range(seq_len):
        current_logits = logits[0, pos, :]
        topk_values, topk_indices = torch.topk(current_logits, 1000)
        
        # 收集被禁止token及其对应的logits
        current_banned_ids = []
        current_banned_logits = []
        for tid, val in zip(topk_indices, topk_values):
            tid_item = tid.item()
            if tid_item in banned_token_ids_set:
                current_banned_ids.append(tid_item)
                current_banned_logits.append(val)

        if not current_banned_ids:
            continue


        logits_tensor = torch.stack(current_banned_logits)
        weights = torch.softmax(logits_tensor, dim=0)
        λ_weights = λ * weights  

        V = model.lm_head.weight[current_banned_ids, :]
        V = torch.nn.functional.normalize(V, dim=-1).to(hidden_states.dtype)

        h = hidden_states[0, pos, :]

        dots = torch.matmul(V, h)  # [num_banned]
        projection = torch.sum((dots * λ_weights).unsqueeze(1) * V, dim=0)
        modified_h = h - projection  

        modified_hidden[0, pos, :] = modified_h

        with torch.no_grad():
            mod_h = modified_h.unsqueeze(0)
            if hasattr(model.model, 'norm'):
                mod_h = model.model.norm(mod_h)
            mod_logits = model.lm_head(mod_h)
            
            orig_top = torch.topk(current_logits.unsqueeze(0), topk)
            mod_top = torch.topk(mod_logits, topk)
            
            if not torch.equal(orig_top.indices, mod_top.indices):
                print(f"\nLayer {layer_idx} Position {pos}:")
                orig_tokens = [tokenizer.decode([t]) for t in orig_top.indices[0].tolist()]
                mod_tokens = [tokenizer.decode([t]) for t in mod_top.indices[0].tolist()]
                print(f"Original: {orig_tokens}\nModified: {mod_tokens}")

    return modified_hidden

def forward_with_layerwise_intervention(model, inputs, tokenizer, start_layer, end_layer):
    hidden = model.model.embed_tokens(inputs["input_ids"])
    rotary_emb = model.model.rotary_emb
    layers = model.model.layers
    seq_len = hidden.shape[1]
    position_ids = torch.arange(seq_len, device=hidden.device).unsqueeze(0)
    cos, sin = rotary_emb(hidden, position_ids)
    attention_mask = inputs["attention_mask"][:, :seq_len].to(hidden.dtype)

    for layer_idx, layer in enumerate(layers):
        hidden = layer(
            hidden,
            position_embeddings=(cos, sin),
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False
        )[0]

        if start_layer <= layer_idx <= end_layer:
            hidden = filter_english_in_hidden(model, hidden, layer_idx, tokenizer)

    hidden = model.model.norm(hidden)
    return model.lm_head(hidden)

def main():
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # text = "请你跟着我数，一，二，三，四，五"
    # text = "Please count with me: one, two, three, four, five."
    text = "世界上面积最大的国家是俄罗斯"
    # text = "The largest country in the world by area is Russia."
    # text = "世界で最も面積が大きい国はロシアです"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    inputs["position_ids"] = torch.arange(inputs["input_ids"].shape[1], device=model.device).unsqueeze(0)

    # Original output
    with torch.no_grad():
        outputs = model(**inputs)
        print("\nOriginal Output:")
        print(tokenizer.decode(torch.argmax(outputs.logits, dim=-1)[0]))

    # Intervened output
    with torch.no_grad():
        logits = forward_with_layerwise_intervention(model, inputs, tokenizer, 1, 32)
        print("\nModified Output:")
        print(tokenizer.decode(torch.argmax(logits, dim=-1)[0]))

if __name__ == "__main__":
    main()