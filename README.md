# 3D Model Understanding Tools

3Dモデルを解析してLLMが理解しやすい形式に変換するPythonツール集です。

## 機能

### 1. 回転動画生成ツール (`create_rotation_video.py`)
- X, Y, Z軸すべての回転解析を含む統合動画を生成
- ワイヤーフレーム、点群、断面図を同時表示
- 背面カリング、グラデーション断面表示
- 高品質MP4出力
- 対応形式: PLY, OBJ, STL, OFF, GLB, GLTF, DAE, 3MF

### 2. AI動画解析ツール (`gemini_video_analyze.py`)
- 生成された3D回転動画をGoogle Gemini AIで解析
- 3Dモデルの形状、特徴、構造を自動認識
- 自然言語での詳細な解析レポート生成

## セットアップ

### 依存関係のインストール
```bash
pip install -r requirements.txt
```

### 必要なパッケージ
- numpy>=1.21.0
- matplotlib>=3.5.0
- Pillow>=9.0.0
- trimesh[easy]>=4.0.0
- scipy>=1.7.0
- opencv-python>=4.5.0
- google-generativeai

## 使用方法

### 1. 3D回転動画の生成
```bash
# 基本使用法
python create_rotation_video.py path/to/your/model.ply

# フレーム数とFPSを調整
python create_rotation_video.py model.ply --frames 60 --fps 30

# 断面図なし
python create_rotation_video.py model.ply --no-cross-sections

# 個別動画も作成
python create_rotation_video.py model.ply --individual
```

### 2. AI動画解析の実行
```bash
# 生成された動画をGemini AIで解析
python gemini_video_analyze.py

# 注意: Google AI Studio APIキーが必要です
# 環境変数GOOGLE_API_KEYを設定してください
```

## 出力ファイル

### 回転動画の出力
```
Output/video_output_YYYYMMDD_HHMMSS/
├── model_xyz_analysis.mp4                 # 統合解析動画（メイン）
├── model_wireframe_rotation_*.mp4         # 個別ワイヤーフレーム動画
├── model_pointcloud_rotation_*.mp4        # 個別点群動画
└── model_cross_section_*.mp4              # 個別断面図動画
```

### AI解析レポート
- Gemini AIによる詳細な3Dモデル解析レポート
- 形状の特徴、構造、幾何学的性質の説明
- コンソールに出力される自然言語解析結果

## 統合動画の特徴

**3×3グリッドレイアウト:**
```
┌─────────────┬─────────────┬─────────────┐
│ Wireframe   │ Point Cloud │Cross Section│
├─────────────┼─────────────┼─────────────┤
│ X-Axis      │ X-Axis      │ X-Axis      │
├─────────────┼─────────────┼─────────────┤
│ Y-Axis      │ Y-Axis      │ Y-Axis      │
├─────────────┼─────────────┼─────────────┤
│ Z-Axis      │ Z-Axis      │ Z-Axis      │
└─────────────┴─────────────┴─────────────┘
```

**解析機能:**
- ✅ **ワイヤーフレーム** - 3D形状の輪郭表示
- ✅ **点群** - 背面カリング付き点群表示
- ✅ **断面図** - グラデーション色分け、進行状況表示
- ✅ **同期再生** - 全軸・全解析が同時に動作
- ✅ **詳細情報** - 軸ラベル、フレーム番号、断面位置表示

## 対応3Dファイル形式

- PLY (.ply)
- OBJ (.obj)
- STL (.stl)
- OFF (.off)
- GLB (.glb)
- GLTF (.gltf)
- DAE (.dae)
- 3MF (.3mf)

## 技術的特徴

- **Trimesh** - 高速で安定した3D処理
- **背面カリング** - カメラから見える面のみ表示
- **特徴点抽出** - 境界検出、曲率解析、距離ベース選択
- **透視投影** - 自然な3D表示
- **断面解析** - グラデーション色分け、輪郭線表示
- **MP4出力** - OpenCVによる高品質動画生成

## トラブルシューティング

### 空の画像が生成される場合
- モデルファイルが正しく読み込まれているか確認
- 対応形式かどうか確認
- ファイルパスに日本語が含まれていないか確認

### 動画が再生されない場合
- OpenCVが正しくインストールされているか確認
- 出力ディレクトリの権限を確認

### メモリ不足エラー
- 点群数を減らす（デフォルト10000点→5000点など）
- フレーム数を減らす（--frames 30など）

## APIキーの設定

### Google AI Studio APIキー

Gemini AIを使用するには、Google AI Studio APIキーが必要です：

1. [Google AI Studio](https://makersuite.google.com/app/apikey)でAPIキーを取得
2. 環境変数を設定：

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

## 開発者向け情報

### 主要ファイル

- `create_rotation_video.py` - 3D回転動画生成
- `gemini_video_analyze.py` - AI動画解析

### カスタマイズ可能な設定

- 点群サンプリング数
- 画像解像度
- 投影パラメータ
- 断面の厚み
- 色設定
- フレーム数・FPS

## ライセンス

このプロジェクトはオープンソースです。
