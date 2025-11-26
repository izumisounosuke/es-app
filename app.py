import os
import json
import re
from datetime import datetime, timezone

import psycopg2
from flask import Flask, render_template, request
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv

# .envの読み込み
load_dotenv()

app = Flask(__name__)

# DB設定
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. Please add your Supabase/PostgreSQL connection string to the environment or .env file."
    )

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
    total_score REAL NOT NULL
);
"""


def get_db_connection():
    return psycopg2.connect(DATABASE_URL)


def init_db():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(TABLE_SCHEMA)
        conn.commit()


init_db()

# Gemini API設定
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# モデル設定
model = genai.GenerativeModel('gemini-2.5-pro')

@app.route('/', methods=['GET', 'POST'])
def index():
    result_data = None
    avg_data = get_average_scores()

    if request.method == 'POST':
        company_name = request.form.get('company_name', '')
        category = request.form.get('category', '志望動機')
        target_length = request.form.get('target_length', '')
        es_text = request.form.get('es_text', '')

        if es_text:
            # プロンプト（厳格な採点アルゴリズム）
            prompt = f"""
            あなたは「厳格な採点アルゴリズム」です。感情を排除し、以下の基準に基づいて機械的にESを採点してください。
            同じ入力に対しては、常に同じ点数を出力する必要があります。

            【入力データ】
            - 志望企業: {company_name if company_name else "指定なし"}
            - カテゴリ: {category}
            - ES本文: {es_text}

            【採点アルゴリズム (各項目20点満点)】
            各項目について、**「基準点12点」**からスタートし、以下の要素に基づいて加点・減点を行ってください。
            ※合計が20点を超えた場合は20点、0点未満の場合は0点とします。

            1. **論理性 (Logic)**
               - [基準] 12点: 話の筋道が通っている。
               - [加点] +3点: 結論ファースト。 +3点: 接続詞が適切。 +2点: 構造化されている。
               - [減点] -3点: ねじれ文。 -3点: 因果関係不明。 -5点: 結論が最後。

            2. **具体性 (Specificity)**
               - [基準] 12点: 具体的なエピソードがある。
               - [加点] +4点: 固有名詞や数字（金額、人数、期間）がある。 +4点: 5W1Hが明確。
               - [減点] -4点: 抽象語（色々な、頑張った）が多い。 -4点: 状況描写不足。

            3. **熱意 (Passion)**
               - [基準] 12点: 志望動機として成立している。
               - [加点] +4点: 企業独自の強み・理念（{company_name}）への言及。 +4点: 接点が明確。
               - [減点] -5点: コピペ可能な内容。 -3点: 受け身の姿勢。

            4. **自己分析 (Insight)**
               - [基準] 12点: 強みを理解している。
               - [加点] +4点: 原体験が深い。 +4点: 思考プロセス・価値観の開示。
               - [減点] -4点: エピソードと不一致。 -4点: 浅い強み。

            5. **将来性 (Potential)**
               - [基準] 12点: 活躍イメージがある。
               - [加点] +4点: 具体的なキャリアプラン。 +4点: カルチャーフィット。
               - [減点] -4点: ビジネス視点の欠如。 -4点: 独りよがり。

            【出力手順】
            1. 各項目の加点・減点理由を詳細に分析してください（思考プロセス）。
            2. 算出された点数をJSON形式で出力してください。

            【出力フォーマット】
            思考プロセスの後に、必ず以下のJSONブロックのみを作成してください。
            ```json
            {{
              "scores": [論理性, 具体性, 熱意, 自己分析, 将来性],
              "total_score": 合計点,
              "result": "合格" or "不合格",
              "good_points": ["良い点1", "良い点2"],
              "bad_points": ["悪い点1", "悪い点2"],
              "advice": "改善アドバイス",
              "questions": ["質問1", "質問2", "質問3"]
            }}
            ```
            """

            try:
                # 【重要変更1】トークン数を4倍に増やして「喋りすぎによる強制終了」を防ぐ
                generation_config = genai.types.GenerationConfig(
                    temperature=0.0,
                    top_k=1,
                    top_p=1.0,
                    max_output_tokens=8192,  # 2048 -> 8192 に変更
                )

                # 【重要変更2】安全フィルターを無効化（ESの内容での誤ブロックを防ぐ）
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
                
                # レスポンスが空でないか確認
                if not response.parts:
                    raise ValueError(f"AIからの応答が空でした。Finish Reason: {response.candidates[0].finish_reason}")

                raw_text = response.text
                
                # --- JSON抽出ロジック ---
                json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
                
                if not json_match:
                     # デバッグ用にテキストの一部を表示
                     print("Raw response text:", raw_text[:500])
                     raise ValueError("JSONが見つかりませんでした")

                json_str = json_match.group(0)
                result_data = json.loads(json_str)
                
                # DBへの保存
                save_result(company_name, category, result_data)
                avg_data = get_average_scores()
                
            except Exception as e:
                print(f"Error: {e}")
                result_data = {"error": f"エラー: {str(e)}"}

    # チャート用データ
    labels = ["論理性", "具体性", "熱意", "自己分析", "将来性"]
    user_scores = result_data.get("scores", avg_data) if result_data and "scores" in result_data else avg_data
    
    chart_data = {
        "labels": labels,
        "user": user_scores,
        "average": avg_data,
    }

    return render_template('index.html', result=result_data, avg=avg_data, chart_data=json.dumps(chart_data, ensure_ascii=False))

def save_result(company, category, data):
    try:
        scores = data.get('scores', [0]*5)
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    '''
                    INSERT INTO results (
                        timestamp,
                        company_name,
                        category,
                        score_logic,
                        score_specificity,
                        score_passion,
                        score_insight,
                        score_potential,
                        total_score
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ''',
                    (
                        datetime.now(timezone.utc),
                        company or None,
                        category,
                        scores[0],
                        scores[1],
                        scores[2],
                        scores[3],
                        scores[4],
                        data.get('total_score', 0),
                    )
                )
            conn.commit()
    except Exception as e:
        print(f"DB Save Error: {e}")

def get_average_scores():
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    '''
                    SELECT
                        COUNT(*),
                        AVG(score_logic),
                        AVG(score_specificity),
                        AVG(score_passion),
                        AVG(score_insight),
                        AVG(score_potential)
                    FROM results
                    '''
                )
                row = cur.fetchone()

        if row and row[0] and row[0] > 0:
            # row[1:] may include Decimal values; convert safely
            averages = [
                round(float(value), 1) if value is not None else 0.0
                for value in row[1:]
            ]
            return averages
        return [16, 15, 17, 15, 16]
    except Exception as exc:
        print(f"DB Average Error: {exc}")
        return [16, 15, 17, 15, 16]

if __name__ == '__main__':
    app.run(debug=True)