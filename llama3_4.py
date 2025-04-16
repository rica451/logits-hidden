from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# import gcld3

def forward_with_modified_hidden(model, inputs, modified_hidden, layer_idx, token_position):
    """
    手动替换指定层的某个 token 的 hidden，然后从这一层继续 forward，直到输出 logits。
    """
    with torch.no_grad():
        # Step 1: Run 原始 embedding 到 Layer N 的前一层
        all_hidden_states = []
        hidden = model.model.embed_tokens(inputs["input_ids"])
        hidden = model.model.norm(hidden) if hasattr(model.model, "norm") else hidden

        for i, layer in enumerate(model.model.layers):
            if i == layer_idx:
                break
            hidden = layer(hidden)[0]  # 这里忽略 attention 输出

        # Step 2: 替换指定位置的 hidden state
        hidden[:, token_position, :] = modified_hidden  # batch size = 1 假设

        # Step 3: 从该层继续 forward 到最后一层
        for i, layer in enumerate(model.model.layers[layer_idx:], start=layer_idx):
            hidden = layer(hidden)[0]

        # Step 4: 最终归一化 + lm_head 映射 logits
        hidden = model.model.norm(hidden)
        logits = model.lm_head(hidden)

        return logits  # shape: [1, seq_len, vocab_size]

def intervene_and_generate(model, tokenizer, inputs, intervention_layer=10, token_idx=5, top_k=5):
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
    
    hidden = hidden_states[intervention_layer][0, token_idx, :].unsqueeze(0)
    normalized_hidden = model.model.norm(hidden)
    logits = model.lm_head(normalized_hidden)

    # === 干预 logits ===
    topk = torch.topk(logits, k=top_k)
    decode_tokens = tokens_to_chinese(tokenizer, topk.indices[0].cpu().numpy())
    print(f"[Before Intervention] Layer {intervention_layer} Top-{top_k}:", decode_tokens)

    # === 示例干预：屏蔽所有 ASCII 范围 token（英文） ===
    logits_clone = logits.clone()
    vocab_size = logits.shape[-1]
    min_val = torch.finfo(logits_clone.dtype).min
    for token_id in range(vocab_size):
        token = tokenizer.decode([token_id], skip_special_tokens=True)
        if all(32 <= ord(c) <= 126 for c in token):  # ASCII 范围
            logits_clone[0, token_id] = min_val  # 使用 fp16 能表示的最小值

    # 用 argmax 或采样生成一个干预 token
    new_token_id = torch.argmax(logits_clone, dim=-1).to(model.device)  # shape: [1]
    new_token_str = tokenizer.decode(new_token_id.tolist())
    print(f"[After Intervention] New token: {new_token_str}") 

    # === 构造新的输入继续 forward ===
    new_input_ids = torch.cat([inputs["input_ids"][0][:token_idx+1], new_token_id], dim=0).unsqueeze(0)
    new_attention_mask = torch.ones_like(new_input_ids)

    new_inputs = {
        "input_ids": new_input_ids,
        "attention_mask": new_attention_mask
    }

    print("\n[Generating from intervened sequence...]")
    generation = model.generate(
        **new_inputs,
        max_new_tokens=1,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    print(tokenizer.decode(generation[0], skip_special_tokens=True))


def tokens_to_chinese(tokenizer, token_ids):
    """将token序列转换为：
    1. 字节流列表（每个token的原始字节）
    2. 解码后的中文字符列表（每个token单独解码）
    """
    byte_stream = []  # 存储每个token的字节
    decoded_texts = []  # 存储每个token解码后的字符串
    
    for tid in token_ids:
        token = tokenizer.decode([tid], skip_special_tokens=True)
        try:
            # 编码为UTF-8字节
            token_bytes = token.encode('utf-8')
            byte_stream.append(token_bytes)
            decoded_texts.append(token)  
        except UnicodeEncodeError:
            token_bytes = bytes([tid % 256])
            byte_stream.append(token_bytes)
            decoded_texts.append(f"<0x{tid:02x}>")  
    
    # return byte_stream, decoded_texts 
    return decoded_texts


def main():
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
        trust_remote_code=True
    )
    model.eval()

    # text = "类似你提供的日志可帮助诊断模型决策路径，适合调试生成质量。"
    text = "How are you? I am fine. This is a test. Hello world!"

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states

    # 语言分布存储
    layer_languages = []
    token_ids = inputs["input_ids"][0].tolist()
    byte_stream = tokens_to_chinese(tokenizer, token_ids)
    print(byte_stream)
    top_k = 5
    for token_idx in range(inputs["input_ids"].shape[1]):
        current_token = inputs["input_ids"][0, token_idx]
        current_bytes = tokens_to_chinese(tokenizer, [int(current_token)])
        print(f"\n=== Token {token_idx}: '{current_bytes}' ===")

        for layer_idx in range(len(hidden_states)):
            hidden = hidden_states[layer_idx][0, token_idx, :].unsqueeze(0)
            normalized_hidden = model.model.norm(hidden)
            layer_logits = model.lm_head(normalized_hidden)
            topk = torch.topk(layer_logits, k=top_k)
            topk_values = topk.values[0].detach().cpu().numpy()  
            decode_tokens = tokens_to_chinese(tokenizer, topk.indices[0].cpu().numpy())
            print(f"Layer {layer_idx:2d}: {decode_tokens} | Logits: {topk_values}")
    intervene_and_generate(model, tokenizer, inputs, intervention_layer=10, token_idx=10, top_k=50)


if __name__ == "__main__":
    main()