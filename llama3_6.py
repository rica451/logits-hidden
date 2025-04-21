from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# from gcld3 import NNetLanguageIdentifier

# def is_english(token):
#     identifier = NNetLanguageIdentifier(0, 1000)
#     raw_detection = identifier.FindLanguage(token)
#     return raw_detection.language == "en"

def tokens_to_readable(tokenizer, token_ids):
    decoded_texts = []
    for tid in token_ids:
        token = tokenizer.decode([tid], skip_special_tokens=True)
        decoded_texts.append(token)
    return decoded_texts

def filter_english_tokens(model, hidden_states, tokenizer):
    modified_hidden = hidden_states.clone()
    seq_len = hidden_states.shape[1]
    for pos in range(seq_len):
        token_hidden = hidden_states[0, pos, :].unsqueeze(0)
        normed = model.model.norm(token_hidden)
        logits = model.lm_head(normed)
        pred_id = torch.argmax(logits, dim=-1).item()
        token = tokenizer.decode([pred_id])
        if is_english(token):
            print(f"[Filtered] Pos {pos} -> '{token}'")
            modified_hidden[0, pos, :] = 0
    return modified_hidden

def decode_from_layer(model, inputs, hidden, start_layer):
    rotary_emb = model.model.rotary_emb
    for i, layer in enumerate(model.model.layers[start_layer:]):
        position_ids = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        cos, sin = rotary_emb(hidden, position_ids)
        attn_mask = inputs["attention_mask"][:, :hidden.shape[1]]
        hidden = layer(
            hidden,
            position_embeddings=(cos, sin),
            attention_mask=attn_mask.to(hidden.dtype),
            position_ids=position_ids,
            use_cache=False
        )[0]
    normed = model.model.norm(hidden)
    return model.lm_head(normed)

def analyze_all_layers(model, tokenizer, inputs, top_k=10, intervene_layer=None):
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        for token_idx in range(inputs["input_ids"].shape[1]):
            print(f"\n=== Token {token_idx} '{tokenizer.decode([inputs['input_ids'][0, token_idx]])}' ===")
            for layer_idx, layer_hidden in enumerate(hidden_states):
                token_hidden = layer_hidden[0, token_idx, :].unsqueeze(0)
                normed = model.model.norm(token_hidden)
                logits = model.lm_head(normed)
                topk = torch.topk(logits, k=top_k)
                tokens = tokens_to_readable(tokenizer, topk.indices[0].cpu().numpy())
                print(f"Layer {layer_idx:02d}: {tokens}")

        # 可选干预
        if intervene_layer is not None:
            print(f"\n=== 🔧 Intervening at layer {intervene_layer} ===")
            modified = filter_english_tokens(model, hidden_states[intervene_layer], tokenizer)
            final_logits = decode_from_layer(model, inputs, modified, intervene_layer)
            final_ids = torch.argmax(final_logits, dim=-1)[0]
            final_text = tokenizer.decode(final_ids.tolist(), skip_special_tokens=True)
            print(f"\n🧠 Final Output After Intervention:\n{final_text}")

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

    text = "你好，我是你爸爸，超弟的名字叫生命"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    inputs["position_ids"] = torch.arange(inputs["input_ids"].shape[1], device=inputs["input_ids"].device).unsqueeze(0)

    analyze_all_layers(model, tokenizer, inputs, top_k=10, intervene_layer=20)

if __name__ == "__main__":
    main()
