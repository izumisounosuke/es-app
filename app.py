import os
import json
import re
from datetime import datetime, timezone
from urllib.parse import quote_plus

import psycopg2
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv

# .envの読み込み
load_dotenv()

app = Flask(__name__)

# DB設定
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set.")

def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

# テーブル作成（user_idカラムを追加した定義）
TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    company_name TEXT,
    category TEXT NOT NULL,
    score_logic REAL NOT NULL,
    score_specificity REAL NOT NULL,
    score_passion REAL NOT NULL,
    score_insight REAL NOT NULL,
    score_potential REAL NOT NULL,
    total_score REAL NOT NULL,
    user_id TEXT  -- ここを追加
);
"""

def init_db():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(TABLE_SCHEMA)
        conn.commit()

init_db()

# Gemini API設定
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-pro')

def get_average_scores():
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('SELECT COUNT(*) FROM results')
                count_row = cur.fetchone()
                count = count_row[0] if count_row else 0
                
                if count < 5:
                    return [16, 15, 17, 15, 16]
                cur.execute('''
                    SELECT AVG(score_logic), AVG(score_specificity), AVG(score_passion),
                           AVG(score_insight), AVG(score_potential)
                    FROM results
                ''')
                row = cur.fetchone()
        if row:
            return [round(float(x), 1) if x is not None else 0.0 for x in row]
        return [16, 15, 17, 15, 16]
    except Exception as exc:
        print(f"DB Average Error: {exc}")
        return [16, 15, 17, 15, 16]

def evaluate_es(company_name, category, es_text):
    # プロンプト（前回と同じ汎用対応版）
    prompt = f"""
    あなたは「厳格な採点アルゴリズム」です。以下のESを採点してください。
    
    【入力データ】
    - 志望企業: {company_name if company_name else "指定なし（汎用的な提出として評価）"}
    - カテゴリ: {category}
    - ES本文: {es_text}
    【重要：評価スタンス】
    1. 企業名なしの場合、特定企業への言及不足を減点しない。
    2. ガクチカの場合、企業への熱意より能力・行動特性を重視する。
    【採点基準 (各20点満点)】
    12点を基準に加点・減点法で採点。
    1. 論理性 (Logic)
    2. 具体性 (Specificity)
    3. 熱意・姿勢 (Passion/Attitude)
    4. 自己分析 (Insight)
    5. 将来性 (Potential)
    【出力フォーマット (JSONのみ)】
    ```json
    {{
      "scores": [論理性, 具体性, 熱意, 自己分析, 将来性],
      "total_score": 合計点,
      "result": "合格" or "不合格",
      "good_points": ["点1", "点2"],
      "bad_points": ["点1", "点2"],
      "advice": "改善アドバイス",
      "questions": ["質問1", "質問2", "質問3"]
    }}
    ```
    """

    try:
        generation_config = genai.types.GenerationConfig(temperature=0.0, top_k=1, top_p=1.0, max_output_tokens=8192)
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        response = model.generate_content(prompt, generation_config=generation_config, safety_settings=safety_settings)
        
        if not response.parts:
            raise ValueError("AI response was empty.")
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if not json_match:
            raise ValueError("JSON not found in AI response.")
        result_data = json.loads(json_match.group(0))
        return result_data
        
    except Exception as e:
        print(f"Error in evaluate_es: {e}")
        raise

@app.route('/api/evaluate', methods=['POST'])
def api_evaluate():
    try:
        company_name = request.form.get('company_name', '').strip()
        category = request.form.get('category', '志望動機')
        es_text = request.form.get('es_text', '').strip()
        user_id = request.form.get('user_id', None) # 👈 フロントエンドからIDを受け取る

        if not es_text:
            return jsonify({"error": "ES本文を入力してください。"}), 400
        
        # AI評価
        result_data = evaluate_es(company_name, category, es_text)
        scores = result_data.get('scores', [0]*5)
        
        # DB保存 (user_id も一緒に保存！)
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    '''
                    INSERT INTO results (
                        timestamp, company_name, category,
                        score_logic, score_specificity, score_passion, score_insight, score_potential,
                        total_score, user_id
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ''',
                    (
                        datetime.now(timezone.utc),
                        company_name or None,
                        category,
                        scores[0], scores[1], scores[2], scores[3], scores[4],
                        result_data.get('total_score', 0),
                        user_id # ここで保存
                    )
                )
            conn.commit()
        
        avg_data = get_average_scores()
        return jsonify({"result": result_data, "average": avg_data}), 200
        
    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({"error": f"サーバーエラー: {str(e)}"}), 500

@app.route('/api/rewrite', methods=['POST'])
def api_rewrite():
    """ESリライト用のAPIエンドポイント"""
    try:
        # user_id の検証（ログインユーザー限定）
        user_id = request.form.get('user_id', None)
        if not user_id:
            return jsonify({"error": "リライト機能はログインユーザー限定です。"}), 403
        
        # フォームデータの取得
        original_text = request.form.get('original_text', '').strip()
        company_name = request.form.get('company_name', '').strip()
        category = request.form.get('category', '志望動機')
        bad_points = request.form.getlist('bad_points')  # リストとして取得
        advice = request.form.get('advice', '').strip()
        
        # バリデーション
        if not original_text:
            return jsonify({"error": "元のES本文を入力してください。"}), 400
        
        # リライト用プロンプト作成
        bad_points_text = "\n".join([f"- {point}" for point in bad_points]) if bad_points else "なし"
        
        prompt = f"""
        ユーザーが送信した元のES、悪い点、アドバイスを基に、全ての指摘を解消した**論理的で具体的な新しいES本文を純粋なテキスト形式で生成せよ**。

        【元のES本文】
        {original_text}

        【応募情報】
        - 志望企業: {company_name if company_name else "指定なし"}
        - カテゴリ: {category}

        【改善すべき点（悪い点）】
        {bad_points_text}

        【アドバイス】
        {advice}

        【出力形式】
        - JSONやMarkdown記法は一切使用しないこと。
        - 純粋なテキスト形式で、新しいES本文のみを出力すること。
        - 改行は適切に入れること。
        """

        # Gemini API呼び出し
        generation_config = genai.types.GenerationConfig(
            temperature=0.7,  # リライトなので少し創造性を持たせる
            top_k=40,
            top_p=0.95,
            max_output_tokens=4096,
        )

        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        if not response.parts:
            raise ValueError(f"AI Response Empty. Finish Reason: {response.candidates[0].finish_reason}")

        rewritten_text = response.text.strip()
        
        # JSONレスポンスを返す
        return jsonify({
            "rewritten_text": rewritten_text
        }), 200
        
    except Exception as e:
        print(f"Rewrite API Error: {e}")
        return jsonify({"error": f"リライト中にエラーが発生しました: {str(e)}"}), 500

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
