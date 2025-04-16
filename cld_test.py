from gcld3 import NNetLanguageIdentifier

# 验证乱码字符串的语言检测
def detect_corrupted_string():
    # 创建语言识别器实例（参数是最小和最大字节数）
    identifier = NNetLanguageIdentifier(0, 1000)
    
    corrupted_str = "welcome"
    
    # 步骤1：直接检测乱码字符串
    raw_detection = identifier.FindLanguage(corrupted_str)
    
    # 步骤2：编码回溯解析
    try:
        # 将乱码字符串转换为原始字节
        raw_bytes = corrupted_str.encode('latin-1')  # 字节：b'\xe6\x96\x87\xe5\xad\x97'
        # 尝试用正确编码解码
        corrected_str = raw_bytes.decode('utf-8')    # 正确解码结果："文字"
        correction_detection = identifier.FindLanguage(corrected_str)
    except Exception as e:
        corrected_str = None
        correction_detection = None

    # 打印检测结果
    print("="*40)
    print(f"原始乱码字符串：{corrupted_str}")
    print(f"直接检测结果：语言={raw_detection.language} 置信度={raw_detection.probability:.2f}")
    
    if corrected_str:
        print("\n" + "-"*40)
        print(f"编码修正后字符串：{corrected_str}")
        print(f"修正后检测结果：语言={correction_detection.language} 置信度={correction_detection.probability:.2f}")
    print("="*40)

if __name__ == "__main__":
    detect_corrupted_string()