from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from openai import OpenAI
from tqdm import tqdm
# 初始化本地 OpenAI 客户端
client_user = OpenAI(
    base_url="http://localhost:10002/v1",
    api_key="sk-xxxxx",  # 你的 API 密钥
)

# 初始化 tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

def generate_user_response(token: str, max_retries: int = 3) -> bool:
    for attempt in range(max_retries):
        openai_messages = [
            {"role": "system", "content": (
                "You are a helpful assistant. Your task is to determine whether the given token is an English word. "
                "If it is an English word, return True. Otherwise, return False. "
                "Return only a single Boolean value without any additional explanation.\n"
            )},{"role": "user","content": f"Input token: {token}"}
        ]
        try:
            response = client_user.chat.completions.create(
                model="diangpt",
                messages=openai_messages,
                temperature=0.1,
                max_tokens=10,
                stop=["\n", "\n\n"]
            )
            reply = response.choices[0].message.content.strip()
            if "True" in reply:
                return True
            elif "False" in reply:
                return False
            else:
                print(f"[!] Unexpected response (retrying): {reply}")
        except Exception as e:
            print(f"[!] Error during generation: {e}")

    # 如果所有尝试都失败，就默认返回 False
    print(f"[!] Failed to get valid response after {max_retries} attempts for token: {token}")
    return False




output_file = "english_tokens_with_text.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for i in tqdm(range(tokenizer.vocab_size), desc="Checking tokens"):
        token_str = tokenizer.decode([i]).strip()
        try:
            if generate_user_response(token_str):
                f.write(f"{i}\t{token_str}\n")
                f.flush()
        except Exception as e:
            print(f"[!] Exception at token id {i}: {e}")
