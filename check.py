import google.generativeai as genai
import os
from dotenv import load_dotenv

# .envからキーを読み込む
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

print("=== あなたのAPIキーで使えるモデル一覧 ===")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"エラーが発生しました: {e}")