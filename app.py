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


def evaluate_es(company_name, category, es_text):
    """
    ESを評価する共通関数
    戻り値: (result_data, avg_data) のタプル
    """
    # プロンプト（汎用対応版）
    prompt = f"""
    あなたは「厳格な採点アルゴリズム」です。感情を排除し、以下の基準に基づいて機械的にESを採点してください。
    同じ入力に対しては、常に同じ点数を出力する必要があります。

    【入力データ】
    - 志望企業: {company_name if company_name else "指定なし（汎用的な提出として評価）"}
    - カテゴリ: {category}
    - ES本文: {es_text}

    【重要：評価のスタンス（ここを厳守）】
    1. **企業名が「指定なし」の場合:**
       - 特定の企業への言及や、企業理念との適合度は**評価対象外**としてください。
       - 「志望企業への言及がない」という理由での減点は**禁止**です。
       - 「入社後の活躍」については、特定の企業ではなく「一般的なビジネスシーンでどう役立つか」という汎用的な再現性を評価してください。

    2. **カテゴリが「ガクチカ」の場合:**
       - これは「志望動機」ではありません。企業への熱意よりも、「課題解決能力」「行動特性」「人柄」を重視してください。
       - 企業へのラブレターになっていないことを理由に減点しないでください。

    【採点アルゴリズム (各項目20点満点)】
    各項目について、**「基準点12点」**からスタートし、以下の要素に基づいて加点・減点を行ってください。

    1. **論理性 (Logic)**
       - [基準] 12点: 話の筋道が通っている。
       - [加点] +3点: 結論ファースト。 +3点: 接続詞が適切。 +2点: 構造化されている。
       - [減点] -3点: ねじれ文。 -3点: 因果関係不明。 -5点: 結論が最後。

    2. **具体性 (Specificity)**
       - [基準] 12点: 具体的なエピソードがある。
       - [加点] +4点: 固有名詞や数字（金額、人数、期間）がある。 +4点: 5W1Hが明確。
       - [減点] -4点: 抽象語（色々な、頑張った）が多い。 -4点: 状況描写不足。

    3. **熱意・姿勢 (Passion/Attitude)**
       - [基準] 12点: 取り組む姿勢が前向きである。
       - [加点]
         - (企業名ありの場合) +4点: 企業独自の強みへの言及。 +4点: 接点が明確。
         - (企業名なし/ガクチカの場合) +4点: 自ら主体的に行動した事実がある。 +4点: 困難から逃げずに立ち向かった姿勢。
       - [減点] -5点: コピペ可能な内容。 -3点: 受け身の姿勢。

    4. **自己分析 (Insight)**
       - [基準] 12点: 強みを理解している。
       - [加点] +4点: 原体験が深い。 +4点: 思考プロセス・価値観の開示。
       - [減点] -4点: エピソードと不一致。 -4点: 浅い強み。

    5. **将来性 (Potential)**
       - [基準] 12点: 活躍イメージがある。
       - [加点]
         - (企業名ありの場合) +4点: その企業での具体的なキャリアプラン。
         - (企業名なし/ガクチカの場合) +4点: その強みが社会人として汎用的に活かせる（再現性がある）。
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
        # 設定: トークン数確保と安全フィルター解除
        generation_config = genai.types.GenerationConfig(
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            max_output_tokens=8192,
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

        raw_text = response.text
        
        # JSON抽出
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if not json_match:
            print("Raw response:", raw_text[:500])
            raise ValueError("JSON not found")

        result_data = json.loads(json_match.group(0))
        
        # DB保存
        scores = result_data.get('scores', [0]*5)
        
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
                        company_name or None,
                        category,
                        scores[0],
                        scores[1],
                        scores[2],
                        scores[3],
                        scores[4],
                        result_data.get('total_score', 0),
                    )
                )
            conn.commit()
        
        avg_data = get_average_scores()
        return result_data, avg_data
        
    except Exception as e:
        print(f"Error: {e}")
        raise


@app.route('/api/evaluate', methods=['POST'])
def api_evaluate():
    """Reactクライアント用のAPIエンドポイント"""
    try:
        # フォームデータの取得
        company_name = request.form.get('company_name', '').strip()
        category = request.form.get('category', '志望動機')
        es_text = request.form.get('es_text', '').strip()
        
        # バリデーション
        if not es_text:
            return jsonify({"error": "ES本文を入力してください。"}), 400
        
        # 評価実行
        result_data, avg_data = evaluate_es(company_name, category, es_text)
        
        # JSONレスポンスを返す
        return jsonify({
            "result": result_data,
            "average": avg_data
        }), 200
        
    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({"error": f"エラー: {str(e)}"}), 500


@app.route('/', methods=['GET', 'POST'])
def index():
    result_data = None
    avg_data = get_average_scores()
    share_url = None

    if request.method == 'POST':
        company_name = request.form.get('company_name', '')
        category = request.form.get('category', '志望動機')
        target_length = request.form.get('target_length', '')
        es_text = request.form.get('es_text', '')

        if es_text:
            try:
                result_data, avg_data = evaluate_es(company_name, category, es_text)
                share_url = build_share_url(result_data)
            except Exception as e:
                print(f"Error: {e}")
                result_data = {"error": f"エラー: {str(e)}"}
                avg_data = get_average_scores()

    # チャート用データ
    labels = ["論理性", "具体性", "熱意", "自己分析", "将来性"]
    user_scores = result_data.get("scores", avg_data) if result_data and "scores" in result_data else avg_data
    
    chart_data = {
        "labels": labels,
        "user": user_scores,
        "average": avg_data,
    }

    if not share_url:
        share_url = build_share_url(result_data)

    return render_template(
        'index.html',
        result=result_data,
        avg=avg_data,
        chart_data=json.dumps(chart_data, ensure_ascii=False),
        share_url=share_url,
    )

def get_average_scores():
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # データの件数を確認
                cur.execute('SELECT COUNT(*) FROM results')
                count_row = cur.fetchone()
                count = count_row[0] if count_row else 0
                
                # データが5件未満なら、計算せずに「仮想の合格者平均」を返す
                if count < 5:
                    return [16, 15, 17, 15, 16]
                # 5件以上溜まったら、実際の平均を計算する
                cur.execute(
                    '''
                    SELECT
                        AVG(score_logic),
                        AVG(score_specificity),
                        AVG(score_passion),
                        AVG(score_insight),
                        AVG(score_potential)
                    FROM results
                    '''
                )
                row = cur.fetchone()
        if row:
            averages = [
                round(float(value), 1) if value is not None else 0.0
                for value in row
            ]
            return averages
        return [16, 15, 17, 15, 16]
    except Exception as exc:
        print(f"DB Average Error: {exc}")
        return [16, 15, 17, 15, 16]


def build_share_url(result_data):
    if not result_data or "total_score" not in result_data or "result" not in result_data:
        return None

    site_url = os.getenv("SITE_URL") or request.host_url.rstrip('/')
    share_text = (
        f"【ES添削】私のESスコアは{result_data.get('total_score', 0)}点でした！"
        f"判定：{result_data.get('result')}（合格/不合格） #AI鬼面接官"
    )
    return "https://twitter.com/intent/tweet?text={text}&url={url}".format(
        text=quote_plus(share_text),
        url=quote_plus(site_url),
    )

if __name__ == '__main__':
    app.run(debug=True)