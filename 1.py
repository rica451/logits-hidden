import sys
from openai import OpenAI

# 读取 system prompt
try:
    with open("/workspace/2.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()
except Exception as e:
    print(f"读取 system_prompt.txt 失败: {e}", file=sys.stderr)
    system_prompt = ""

# 初始化 Deepseek 客户端
client_depseek = OpenAI(
    base_url="https://api.deepseek.com",
    api_key="sk-ec2a8aeed8b54c91a8b3d4298f371f3b",
)

def main():
    print("与 Deepseek 模型聊天，输入 exit 或 Ctrl-D 结束对话。\n")
    # 历史对话：首条 system 消息来自 txt
    history = [
        {"role": "system", "content": system_prompt}
    ]

    while True:
        try:
            user_input = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break
        if not user_input or user_input.lower() in ("exit", "quit"):
            print("退出。")
            break

        history.append({"role": "user", "content": user_input})
        try:
            resp = client_depseek.chat.completions.create(
                model="deepseek-chat",
                messages=history,
                temperature=1.0,
                max_completion_tokens=100,
            )
            assistant_msg = resp.choices[0].message.content.strip()
            print(f"Assistant: {assistant_msg}\n")
            history.append({"role": "assistant", "content": assistant_msg})
        except Exception as e:
            print(f"请求出错: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()