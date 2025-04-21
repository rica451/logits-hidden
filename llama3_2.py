from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt

def correct_chinese_bytes(tokens):
    corrected = []
    for t in tokens:
        try:
            raw_bytes = t.encode('latin-1')
            # 优先尝试UTF-8解码
            fixed = raw_bytes.decode('utf-8')
            corrected.append(fixed)
        except UnicodeDecodeError:
            # UTF-8失败时尝试其他常见编码
            try:
                fixed = raw_bytes.decode('iso-8859-11')  
                corrected.append(fixed)
            except:
                corrected.append(t)  
        except:
            corrected.append(t)
    # 处理空格标记
    return [t.replace('Ġ', '▁') for t in corrected]

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


    text = "爸爸的爸爸叫爷爷，爸爸的妈妈叫奶奶。"
    # text = "Le bateau naviguait en douceur sur l"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    
    print("\n=== 修正后的分词结果 ===")
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    corrected_tokens = correct_chinese_bytes(tokens)
    print(corrected_tokens)

    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states

    # Logits Lens分析（带修正的显示）
    top_k = 5
    for token_idx in range(input_ids.shape[1]):
        current_token = corrected_tokens[token_idx]
        print(f"\n=== Token {token_idx}: '{current_token}' ===")
        
        for layer_idx in range(len(hidden_states)):  # 显式遍历各层
            hidden = hidden_states[layer_idx][0, token_idx, :].unsqueeze(0)
            normalized_hidden = model.model.norm(hidden)
            
            logits = model.lm_head(normalized_hidden)
            topk = torch.topk(logits, k=top_k)
            
            # 修正候选token的显示
            decoded_tokens = correct_chinese_bytes(
                tokenizer.convert_ids_to_tokens(topk.indices[0])
            )
            preds = " | ".join([f"{t} ({s:.2f})" for t, s in zip(decoded_tokens, topk.values[0].tolist())])
            # token_ids = topk.indices[0].tolist()
            # preds = " | ".join([f"{tid} ({s:.2f})" for tid, s in zip(token_ids, topk.values[0].tolist())])
            print(f"Layer {layer_idx:2d}: {preds}")

    # 强制触发绘图的增强实现
    # plt.figure(figsize=(10, 6))
    # similarities = []
    # with torch.no_grad():
    #     final_hidden = model.model.norm(hidden_states[-1][:, -1, :])
    #     for layer in hidden_states:
    #         h = model.model.norm(layer[:, -1, :])
    #         sim = F.cosine_similarity(h, final_hidden, dim=-1)
    #         similarities.append(sim.mean().item())
    
    # plt.plot(similarities, marker='o', color='#FF6B6B', linewidth=2)
    # plt.title("层表示相似度演化", fontsize=14)
    # plt.xlabel("网络层深度")
    # plt.ylabel("余弦相似度")
    # plt.grid(True, alpha=0.3)
    # plt.savefig('layer_similarity.png', bbox_inches='tight', dpi=300)
    # print("\n=== 绘图已保存到 layer_similarity.png ===")

if __name__ == "__main__":
    main()