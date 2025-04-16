from datasets import load_dataset
from googletrans import Translator
import asyncio
import json
import random
import aiofiles

# 目标语言池（按您指定的优先级）
TARGET_LANGUAGES = [
    'de', 'es', 'it', 'fr', 'pt', 'ar', 'ru', 
    'vi', 'tr', 'th', 'gl', 'ja', 'ko'
]

async def translate_text(text, src_lang='zh-cn', dest_lang='en'):
    """带重试机制的翻译函数"""
    translator = Translator()
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = await translator.translate(text, src=src_lang, dest=dest_lang)
            return result.text
        except Exception as e:
            print(f"翻译失败（{attempt+1}/{max_retries}）: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
    return text  # 失败时返回原文

async def process_conversation(conversation):
    """处理单条对话"""
    processed = []
    total_turns = len(conversation)
    if total_turns < 4:
        print(f"对话长度不足4，跳过")
        return None

    is_zh = True
    for msg in conversation:
        try:
            detected = await Translator().detect(msg["value"])
            if detected.lang != "zh-CN":
                is_zh = False
                break
        except:
            continue
    
    if not is_zh:
        print(f"跳过非中文对话")
        return None

    # 决定处理策略：25%全英译，75%混合
    if random.random() < 0.25:  # 25%概率全英译
        for turn in conversation:
            translated = await translate_text(turn['value'], dest_lang='en')
            processed.append({
                'from': turn['from'],
                'value': f"<lang_en>{translated}"
            })
    else:  # 75%概率前25%英译，后75%多语言
        split_point = max(1, int(total_turns * 0.25))
        for turn in conversation[:split_point]:
            translated = await translate_text(turn['value'], dest_lang='en')
            processed.append({
                'from': turn['from'],
                'value': f"<lang_en>{translated}"
            })

        target_lang = random.choice(TARGET_LANGUAGES)
        for turn in conversation[split_point:]:
            try:
                translated = await translate_text(turn['value'], dest_lang=target_lang)
                processed.append({
                    'from': turn['from'],
                    'value': f"<lang_{target_lang}>{translated}"
                })
            except:
                translated = await translate_text(turn['value'], dest_lang='en')
                processed.append({
                    'from': turn['from'],
                    'value': f"<lang_en>{translated}"
                })

    return {"conversations": processed}

async def create_multilingual_dataset():
    ds = load_dataset("larryvrh/ShareGPT-Zh_Only")['train']
    print(f"加载完成，共 {len(ds)} 条对话")

    output_file = "multilingual_dataset3.jsonl"

    async with aiofiles.open(output_file, "w", encoding="utf-8") as f:
        for idx, item in enumerate(ds):
            if idx % 5 == 0:
                print(f"已处理 {idx} 条数据")
            
            conversations = item["conversations"]
            if len(conversations) > 20:
                conversations = conversations[:20]
            
            processed = await process_conversation(conversations)
            if processed:
                json_line = json.dumps(processed, ensure_ascii=False)
                await f.write(json_line + "\n")

    print("数据集构建完成（已按行写入）")

# 启动程序
asyncio.run(create_multilingual_dataset())
