from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
# from gcld3 import NNetLanguageIdentifier
def is_english(token):
    # identifier = NNetLanguageIdentifier(0, 1000)
    # raw_detection = identifier.FindLanguage(token)
    # if raw_detection.language == "en":
    #     return True
    return False
def show_token(hidden_states ,layer_idx ,token_idx ,model ,top_k=10):

    hidden = hidden_states[layer_idx][0, token_idx, :].unsqueeze(0)
    normalized_hidden = model.model.norm(hidden)
            
    logits = model.lm_head(normalized_hidden)
    topk = torch.topk(logits, k=top_k)
            
    # 修正候选token的显示
    # decoded_tokens = correct_chinese_bytes(
    #     tokenizer.convert_ids_to_tokens(topk.indices[0])
    # )
    # preds = " | ".join([f"{t} ({s:.2f})" for t, s in zip(decoded_tokens, topk.values[0].tolist())])
    token_ids = topk.indices[0].tolist()
    preds = " | ".join([f"{tid} ({s:.2f})" for tid, s in zip(token_ids, topk.values[0].tolist())])
    print(f"Layer {layer_idx:2d}: {preds}")

def decode_hidden_to_tokens(model, inputs, modified_hidden, layer_idx):
    with torch.no_grad():
        hidden = model.model.embed_tokens(inputs["input_ids"])

        # # 从LlamaModel获取共享的rotary_emb
        rotary_emb = model.model.rotary_emb
        
        # # 前向传播到目标层之前的层
        for i, layer in enumerate(model.model.layers[:layer_idx]):
            seq_len = hidden.shape[1]
            position_ids = torch.arange(seq_len, device=hidden.device).unsqueeze(0)
            
            # 生成cos/sin
            cos, sin = rotary_emb(hidden, position_ids)
            
            # 调用decoder layer时需要传递position_embeddings
            hidden = layer(
                hidden,
                position_embeddings=(cos, sin),
                attention_mask=inputs["attention_mask"].to(hidden.dtype),
                position_ids=position_ids,
                use_cache=False
            )[0]

        # 应用修改后的hidden_states
        hidden = modified_hidden
        new_seq_len = hidden.shape[1]
        
        for layer in model.model.layers[layer_idx:]:
            # 生成新的位置编码
            position_ids = torch.arange(new_seq_len, device=hidden.device).unsqueeze(0)
            cos, sin = rotary_emb(hidden, position_ids)
            
            attention_mask = inputs["attention_mask"][:, :new_seq_len] if inputs["attention_mask"] is not None else None
            
            hidden = layer(
                hidden,
                position_embeddings=(cos, sin),
                attention_mask=attention_mask.to(hidden.dtype),
                position_ids=position_ids,
                use_cache=False
            )[0]

        hidden = model.model.norm(hidden)
        logits = model.lm_head(hidden)
    return logits



# def filter_english_in_hidden(model, hidden_states, layer_idx, tokenizer):
#     modifiled_hidden = hidden_states.clone()
#     seq_len = hidden_states.shape[1]
#     banned_token_ids = [9093, 1234, 5678]
#     V = model.lm_head.weight[banned_token_ids]  # [k, hidden_dim]
#     V_float32 = V.float()
#     V = torch.nn.functional.normalize(V, dim=-1)
#     VVt = V_float32 @ V_float32.T
#     VVt_inv = torch.linalg.pinv(VVt)
#     P_float32 = V_float32.T @ VVt_inv @ V_float32
#     P = P_float32.to(hidden_states.dtype)
#     # 正交投影矩阵 P = V^T (V V^T)^-1 V
#     # P = V.T @ torch.inverse(V @ V.T) @ V  # [hidden_dim, hidden_dim]

#     # 投影 hidden 到屏蔽子空间
#     projection = hidden_states @ P  # [batch, seq_len, hidden_dim]
#     modifiled_hidden = hidden_states - projection
#     # for pos in range(seq_len):
#     #     hidden_vector = hidden_states[0, pos, :].unsqueeze(0)
#     #     normalized = model.model.norm(hidden_vector)
#     #     logits = model.lm_head(normalized)
#     #     token_id = torch.argmax(logits, dim=-1).item()

#     #     token = tokenizer.decode([token_id])
#     #     if is_english(token):
#     #         print(f"Filtering English token at layer {layer_idx} position {pos}: {token}")
#     #         modifiled_hidden[0, pos, :] = 0
#     return modifiled_hidden

def filter_english_in_hidden(model, hidden_states, layer_idx, tokenizer, topk=5):
    modifiled_hidden = hidden_states.clone()
    seq_len = hidden_states.shape[1]
    aklı_id = tokenizer.encode("esimal", add_special_tokens=False)[0]
    banned_token_ids = [aklı_id, 27946, 1097, 108504, 29061, 58731, 42888,69952, 56405, 23881,42888, 117283, 69952, 56405, 23881,84555, 85869, 80979, 63524,7126, 6699, 1027, 47797, 19144]
    λ = 1
    V = model.lm_head.weight[banned_token_ids]
    V_float32 = V.float()
    V = torch.nn.functional.normalize(V, dim=-1)
    VVt = V_float32 @ V_float32.T
    VVt_inv = torch.linalg.pinv(VVt)
    P_float32 = V_float32.T @ VVt_inv @ V_float32
    P = P_float32.to(hidden_states.dtype)
    

    # 计算 projection 并得到修改后的隐藏状态
    projection = hidden_states @ P
    modifiled_hidden = hidden_states - λ * projection

    print(f"\n===== Layer {layer_idx} Top-{topk} Token Changes =====")
    for pos in range(seq_len):
        orig_hidden = hidden_states[0, pos, :].unsqueeze(0)  # [1, hidden_dim]
        mod_hidden = modifiled_hidden[0, pos, :].unsqueeze(0)

        # 正常化（如果模型有 norm 层）
        if hasattr(model.model, 'norm'):
            orig_hidden = model.model.norm(orig_hidden)
            mod_hidden = model.model.norm(mod_hidden)

        orig_logits = model.lm_head(orig_hidden)  # [1, vocab_size]
        mod_logits = model.lm_head(mod_hidden)

        orig_topk = torch.topk(orig_logits, topk, dim=-1)
        mod_topk = torch.topk(mod_logits, topk, dim=-1)

        orig_ids = orig_topk.indices[0].tolist()
        mod_ids = mod_topk.indices[0].tolist()
        orig_tokens = [tokenizer.decode([tid]) for tid in orig_ids]
        mod_tokens = [tokenizer.decode([tid]) for tid in mod_ids]

        if orig_ids != mod_ids:
            print(f"[Position {pos}]")
            print(f"  Before: {orig_tokens}")
            print(f"  After : {mod_tokens}")
    aklı_id = tokenizer.encode("aklı", add_special_tokens=False)[0]
    aklı_vector = model.lm_head.weight[aklı_id]  # [hidden_dim]

    # 投影分量大小
    proj = aklı_vector @ P  # [hidden_dim]
    proj_norm = torch.norm(proj)
    aklı_norm = torch.norm(aklı_vector)
    cosine_proj = proj_norm / aklı_norm
    print(f"aklı's vector projection magnitude: {cosine_proj:.4f}")
    return modifiled_hidden

def main():
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
        trust_remote_code=True
    )
    model.eval()
    text = "请你跟着我数，0，1，2，3，4"
    # text = "你好，我是你爸爸，超弟的名字叫生命"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    inputs["position_ids"] = torch.arange(
        inputs["input_ids"].shape[1], device=inputs["input_ids"].device
    ).unsqueeze(0)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits

        predicted_ids = torch.argmax(logits, dim=-1)[0]  
        predicted_tokens = [tokenizer.decode([token_id], skip_special_tokens=True) for token_id in predicted_ids.tolist()]

        print("📝 Model Predicted Tokens:")
        for i, token in enumerate(predicted_tokens):
            print(f"{i+1}. {token}")
        print("📝 Model Predicted Text:")
        print(predicted_tokens)

        hidden_states = outputs.hidden_states[20]

    modified_hidden = filter_english_in_hidden(model, hidden_states, 20, tokenizer)
    logits = decode_hidden_to_tokens(model, inputs, modified_hidden, 20)

    final_tokens = torch.argmax(logits, dim=-1)[0]
    final_tokens = tokenizer.decode(final_tokens.tolist())
    
    print(f"Final tokens: {final_tokens}")

if __name__ == "__main__":
    main()