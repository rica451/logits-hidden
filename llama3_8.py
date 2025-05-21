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
    λ = 0  # 基础抑制系数

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
                # print(f"\nLayer {layer_idx} Position {pos}:")
                orig_tokens = [tokenizer.decode([t]) for t in orig_top.indices[0].tolist()]
                mod_tokens = [tokenizer.decode([t]) for t in mod_top.indices[0].tolist()]
                # print(f"Original: {orig_tokens}\nModified: {mod_tokens}")

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

def generate_with_intervention(
    model, tokenizer, prompt, start_layer=1, end_layer=32, max_new_tokens=50, topk=5
):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    attention_mask = torch.ones_like(input_ids)

    generated = input_ids.clone()
    for step in range(max_new_tokens):
        inputs = {
            "input_ids": generated,
            "attention_mask": attention_mask,
            "position_ids": torch.arange(generated.shape[1], device=generated.device).unsqueeze(0),
        }

        hidden = model.model.embed_tokens(inputs["input_ids"])
        rotary_emb = model.model.rotary_emb
        seq_len = hidden.shape[1]
        position_ids = torch.arange(seq_len, device=hidden.device).unsqueeze(0)
        cos, sin = rotary_emb(hidden, position_ids)
        attention_mask_step = inputs["attention_mask"][:, :seq_len].to(hidden.dtype)

        # forward through transformer layers
        for layer_idx, layer in enumerate(model.model.layers):
            hidden = layer(
                hidden,
                position_embeddings=(cos, sin),
                attention_mask=attention_mask_step,
                position_ids=position_ids,
                use_cache=False,
            )[0]

            if start_layer <= layer_idx <= end_layer:
                hidden = filter_english_in_hidden(model, hidden, layer_idx, tokenizer, topk=topk)

        hidden = model.model.norm(hidden)
        logits = model.lm_head(hidden)
        next_token_logits = logits[:, -1, :]  # 取最后一个位置的logits
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

        generated = torch.cat([generated, next_token], dim=-1)
        attention_mask = torch.ones_like(generated)

        decoded_token = tokenizer.decode(next_token[0])
        print(decoded_token, end="", flush=True)

        if next_token.item() == tokenizer.eos_token_id:
            break

    print("\n\nFinal Decoding:\n", tokenizer.decode(generated[0]))


def main():
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    prompt = "写一个关于小猪的小说："
    print("\nGenerating with intervention:\n")
    generate_with_intervention(model, tokenizer, prompt, start_layer=1, end_layer=32, max_new_tokens=50)

if __name__ == "__main__":
    main()