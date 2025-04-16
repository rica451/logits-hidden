from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from gcld3 import NNetLanguageIdentifier

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
banned_token_ids = [1]
V = model.lm_head.weight[banned_token_ids]  # [k, hidden_dim]
V = torch.nn.functional.normalize(V, dim=-1)

# 正交投影矩阵 P = V^T (V V^T)^-1 V
P = V.T @ torch.inverse(V @ V.T) @ V  # [hidden_dim, hidden_dim]

# 投影 hidden 到屏蔽子空间
projection = hidden @ P  # [batch, seq_len, hidden_dim]
hidden = hidden - projection

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# import gcld3
import matplotlib.pyplot as plt

# def tokens_to_bytes(tokenizer, token_ids):
#     """将token序列转换为原始字节流"""
#     byte_stream = b''
#     for tid in token_ids:
#         token = tokenizer.decode([tid], skip_special_tokens=True)
#         try:
#             # 优先尝试UTF-8解码获取字节
#             byte_stream += token.encode('utf-8')
#         except UnicodeEncodeError:
#             # 回退到原始字节表示
#             byte_stream += bytes([tid % 256])
#     return byte_stream

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
    
    return decoded_texts


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from gcld3 import NNetLanguageIdentifier
def is_english(token):
    identifier = NNetLanguageIdentifier(0, 1000)
    raw_detection = identifier.FindLanguage(token)
    if raw_detection.language == "en":
        return True
    return False

def decode_hidden_to_tokens(model, inputs, modified_hidden, layer_idx):
    with torch.no_grad():
        # hidden = model.model.embed_tokens(inputs["input_ids"])
        # hidden = model.model.norm(hidden)

        # 从LlamaModel获取共享的rotary_emb
        rotary_emb = model.model.rotary_emb
        
        # # 前向传播到目标层之前的层
        # for i, layer in enumerate(model.model.layers[:layer_idx]):
        #     seq_len = hidden.shape[1]
        #     position_ids = torch.arange(seq_len, device=hidden.device).unsqueeze(0)
            
        #     # 生成cos/sin
        #     cos, sin = rotary_emb(hidden, position_ids)
            
        #     # 调用decoder layer时需要传递position_embeddings
        #     hidden = layer(
        #         hidden,
        #         position_embeddings=(cos, sin),
        #         attention_mask=inputs["attention_mask"].to(hidden.dtype),
        #         position_ids=position_ids,
        #         use_cache=False
        #     )[0]

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


def filter_english_in_hidden(model, hidden_states, layer_idx, tokenizer):
    modifiled_hidden = hidden_states.clone()
    seq_len = hidden_states.shape[1]

    for pos in range(seq_len):
        hidden_vector = hidden_states[0, pos, :].unsqueeze(0)
        normalized = model.model.norm(hidden_vector)
        logits = model.lm_head(normalized)
        token_id = torch.argmax(logits, dim=-1).item()

        token = tokenizer.decode([token_id])
        if is_english(token):
            print(f"Filtering English token at layer {layer_idx} position {pos}: {token}")
            modifiled_hidden[0, pos, :] = 0

    return modifiled_hidden

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
    # text = "请你跟着我数，0，1，2，3，4"
    text = "你好，我是你爸爸，超弟的名字叫生命"
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
