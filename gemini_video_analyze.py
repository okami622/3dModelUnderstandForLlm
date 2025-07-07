import os
from dotenv import load_dotenv
import google.generativeai as genai
import time

load_dotenv()
API_KEY = os.environ["GEMINI_API_KEY"]

def wait_for_file_active(file_obj, timeout=300, interval=2):
    """ファイルがACTIVEになるまで待機"""
    start = time.time()
    while True:
        file_obj = genai.get_file(file_obj.name)
        state = file_obj.to_dict()['state']
        if str(state).upper() == "ACTIVE":
            return file_obj
        if time.time() - start > timeout:
            raise TimeoutError(f"ファイルがACTIVEになりませんでした（最終state: {state}）")
        time.sleep(interval)

def generate(video_path: str):
    genai.configure(api_key=API_KEY)

    # ファイルをアップロード（保存期間は48時間）
    video_file = genai.upload_file(
        path=video_path,
        mime_type="video/mp4",
    )
    video_file = wait_for_file_active(video_file)

    model_name = "models/gemini-2.5-flash-lite-preview-06-17"
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction="あなたは 3D 可視化と幾何解析に熟達した AI アシスタントです。\nテクスチャマッピング・寸法計測・欠陥検出・物体カテゴリ分類・総合解析を高精度かつ論理的に実施してください。"
    )

    user_prompt = """
目的:
  この 3D モデルが「何を表しているか」「だいたいのサイズ」「テクスチャ特性」「気になる欠陥」を知りたい。

入力データ:
  - 3x4グリッド回転動画（3軸×4視覚化）

お願いしたいこと:
  この動画は、3Dモデルを4つの視覚化手法で表示した統合動画です：
  
  **グリッド構成（3x4レイアウト）:**
  - 列1: ワイヤーフレーム（構造表示）
  - 列2: 点群（密度可視化） 
  - 列3: 断面図（内部構造）
  - 列4: テクスチャ（表面材質）
  - 行1-3: X, Y, Z軸周りの360度回転
  
  1. モデルが何のオブジェクトかを 1 行で推定してください。  
  2. 幅×奥行×高さ（cm 単位）をおおよそで構いませんので出してください。  
  3. 主要な特徴／パーツを 3 つ挙げてください。  
  4. テクスチャ品質と表面特性を評価してください。
  5. "穴・欠け・UV座標異常・テクスチャ不整合" など気になる箇所があれば箇条書きで。  
  6. まとめを 100 字以内で書いてください。

出力フォーマット（厳守）:

### オブジェクト推定
- **種類**: …

### 寸法 (cm, おおよそ)
| 幅 | 奥行 | 高さ |
|----|------|------|
|    |      |      |

### 主要特徴
1. …
2. …
3. …

### テクスチャ評価
- **品質**: [高品質/中品質/低品質]
- **特徴**: …
- **色彩**: …

### 欠陥・注意点
- **構造**: …
- **テクスチャ**: …
- **その他**: …

### まとめ
…

制約:
- 日本語で回答。
- 表の罫線や見出しは変更しないでください。
- 寸法は半角数字。
- テクスチャ列（右端）の情報も活用して分析してください。
- UV座標マッピングの品質やテクスチャの適用状況も評価に含めてください。
  (注: 実寸のスケール情報は提供されていません。一般的なオブジェクトのサイズを参考に推定してください。)
  - 追加で質問があれば最後に「質問: …?」と書き、私の返答を待ってから続けてください。
"""

    response = model.generate_content(
        [
            video_file,
            user_prompt
        ]
    )
    print(response.text)

if __name__ == "__main__":
    generate("/path/to/your/video.mp4")  # 動画ファイルパスを指定