import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
def load_banned_token_ids_from_file(file_path):
    banned_token_ids = []
    with open(file_path, 'r', encoding='utf-8') as f:
        i = 0
        # f = f.readlines()[10000:]
        for line in f:
            parts = line.strip().split()
            if parts and i<1000:
                try:
                    token_id = int(parts[0])
                    banned_token_ids.append(token_id)
                    i += 1
                except ValueError:
                    continue
    return banned_token_ids

def filter_english_in_hidden(model, hidden_states, layer_idx, tokenizer, topk=5):
    import torch

    token_file_path = "/workspace/logits-hidden/english_tokens_with_text.txt"
    orig_dtype = hidden_states.dtype
    modifiled_hidden = hidden_states.clone()
    seq_len = hidden_states.shape[1]

    banned_token_ids = [
        tokenizer.encode("correct", add_special_tokens=False)[0],
    ]
    λ = 0.6


    hidden_states_fp32 = hidden_states.float()
    V = model.lm_head.weight[banned_token_ids].float()
    V = torch.nn.functional.normalize(V, dim=-1)
    VVt = V @ V.T
    VVt_inv = torch.linalg.pinv(VVt)
    P = V.T @ VVt_inv @ V

    projection = hidden_states_fp32 @ P  # [B, T, D]
    cos_sim = torch.nn.functional.cosine_similarity(hidden_states_fp32, projection, dim=-1)  # [B, T]
    print(f"cos_sim: {cos_sim}")
    mask = (cos_sim > 0).float().unsqueeze(-1)  # [B, T, 1]

    modifiled_hidden_fp32 = hidden_states_fp32 - λ * mask * projection
    modifiled_hidden = modifiled_hidden_fp32.to(orig_dtype) 

    print(f"\n===== Layer {layer_idx} Top-{topk} Token Changes with Logits =====")
    for pos in range(seq_len):
        orig_hidden = hidden_states[0, pos, :].unsqueeze(0)
        mod_hidden = modifiled_hidden[0, pos, :].unsqueeze(0)

        if hasattr(model.model, 'norm'):
            orig_hidden = model.model.norm(orig_hidden)
            mod_hidden = model.model.norm(mod_hidden)

        orig_logits = model.lm_head(orig_hidden)
        mod_logits = model.lm_head(mod_hidden)

        orig_topk = torch.topk(orig_logits, topk, dim=-1)
        mod_topk = torch.topk(mod_logits, topk, dim=-1)

        orig_ids = orig_topk.indices[0].tolist()
        mod_ids = mod_topk.indices[0].tolist()
        orig_tokens = [tokenizer.decode([tid]) for tid in orig_ids]
        mod_tokens = [tokenizer.decode([tid]) for tid in mod_ids]
        orig_logit_vals = orig_topk.values[0].tolist()
        mod_logit_vals = mod_topk.values[0].tolist()

        if orig_ids != mod_ids:
            print(f"[Position {pos}]")
            print("  Before:")
            for token, logit in zip(orig_tokens, orig_logit_vals):
                print(f"    {token:15s} | logit: {logit:.4f}")
            print("  After :")
            for token, logit in zip(mod_tokens, mod_logit_vals):
                print(f"    {token:15s} | logit: {logit:.4f}")

    return modifiled_hidden


def forward_with_layerwise_intervention(model, inputs, tokenizer, start_layer, end_layer):
    # 嵌入阶段
    hidden = model.model.embed_tokens(inputs["input_ids"])
    rotary_emb = model.model.rotary_emb

    layers = model.model.layers
    seq_len = hidden.shape[1]
    position_ids = torch.arange(seq_len, device=hidden.device).unsqueeze(0)
    cos, sin = rotary_emb(hidden, position_ids)
    attention_mask = inputs["attention_mask"][:, :seq_len].to(hidden.dtype)

    # 逐层推进并在指定层干预
    for i, layer in enumerate(layers):
        hidden = layer(
            hidden,
            position_embeddings=(cos, sin),
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False
        )[0]

        if start_layer <= i <= end_layer:
            hidden = filter_english_in_hidden(model, hidden, i, tokenizer)

    hidden = model.model.norm(hidden)
    logits = model.lm_head(hidden)
    return logits


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

    # 输入
    text = "请你跟着我数，一，二，三，四，五"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    inputs["position_ids"] = torch.arange(inputs["input_ids"].shape[1], device=inputs["input_ids"].device).unsqueeze(0)

    # 原始预测
    with torch.no_grad():
        outputs = model(**inputs)
        pred_ids = torch.argmax(outputs.logits, dim=-1)[0]

        print("\n📝 原始模型输出（逐个token）:")
        for i, tid in enumerate(pred_ids.tolist()):
            token_str = tokenizer.decode([tid])
            print(f"{i:02d}: {token_str!r}")
    # 干预范围
    start_layer = 24
    end_layer = 25

    # 干预后预测
    with torch.no_grad():
        logits = forward_with_layerwise_intervention(model, inputs, tokenizer, start_layer, end_layer)
        final_ids = torch.argmax(logits, dim=-1)[0]

        print("\n📝 干预后模型输出（逐个token）:")
        for i, tid in enumerate(final_ids.tolist()):
            token_str = tokenizer.decode([tid])
            print(f"{i:02d}: {token_str!r}")



if __name__ == "__main__":
    main()
