# 3D Model Understanding Tools

3DモデルをGemini AI分析用の高品質動画に変換するPythonツール集です。

## 機能

### 1. 回転動画生成ツール (`create_rotation_video.py`)

- **3x4グリッドレイアウト**: ワイヤーフレーム、点群、断面図、テクスチャを統合表示
- **高品質テクスチャマッピング**: UV座標による正確なテクスチャ適用（70,000面超対応）
- **X, Y, Z軸完全回転解析**: 各軸での360度回転を同時表示
- **幾何学的断面解析**: 正確な平面交線による内部構造可視化
- **matplotlibテクスチャ最適化**: Poly3DCollectionによる高速レンダリング
- **メモリ管理最適化**: 大規模モデル対応の安定した動画生成
- **柔軟な時間設定**: フレーム数・FPS調整による任意長動画生成
- **高品質MP4出力**: 720p, 1080p, 4K対応
- **対応形式**: PLY, OBJ+MTL, STL, OFF, GLB, GLTF, DAE, 3MF, X3D, STEP, IGES

### モジュール構成

- `create_rotation_video.py` - メインの動画作成クラス。3x4グリッドレイアウトでワイヤーフレーム、点群、断面図、テクスチャを統合した動画を生成します。
- `trimesh_texture_video_creator.py` - **NEW** テクスチャ付き動画生成モジュール。UV座標とテクスチャ画像から面ごとの色を計算し、Poly3DCollectionで高品質レンダリングを実現します。
- `texture_loader.py` - テクスチャ読み込みモジュール。OBJ+MTL, GLTF, DAEなどからテクスチャ・マテリアル情報を抽出し、モデルに適用します。
- `open3d_video_creator.py` - Open3D高速レンダリングモジュール。大規模モデルでの効率的な動画生成をサポートします。
- `x3d_loader.py` - X3Dファイル読み込みモジュール。XMLベースのX3D形式をパースし、trimeshで扱えるメッシュデータに変換します。
- `color_generator.py` - 色生成・シェーディングモジュール。位置、深度、法線などの情報に基づいて、モデルに豊かな色彩を与えます。
- `cross_section_processor.py` - 断面処理モジュール。メッシュと平面の交線を計算し、正確な幾何学的断面を生成します。

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
- open3d>=0.17.0 (高速レンダリング用)

## 使用方法

### 1. 3D回転動画の生成

```bash
# 基本使用法（3x4グリッド統合動画）
python create_rotation_video.py path/to/your/model.ply

# テクスチャ付き3x4グリッド動画生成
python create_rotation_video.py model.obj --texture --frames 20 --fps 12

# 高解像度テクスチャ動画
python create_rotation_video.py model.obj --texture --resolution 1080p --frames 30

# 断面図なし2x3グリッド動画
python create_rotation_video.py model.obj --no-cross-sections --frames 50 --fps 15

# テクスチャなし3x3グリッド動画  
python create_rotation_video.py model.ply --frames 40 --fps 12

# 個別動画も作成
python create_rotation_video.py model.obj --texture --individual
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

### **3x4グリッドレイアウト（フル機能版）**
各行が回転軸（X, Y, Z）、各列が解析タイプに対応し、テクスチャ付きモデルを完全分析します。

```
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ Wireframe   │ Point Cloud │Cross Section│   Texture   │
├─────────────┼─────────────┼─────────────┼─────────────┤
│ X-Axis      │ X-Axis      │ X-Axis      │ X-Axis      │
├─────────────┼─────────────┼─────────────┼─────────────┤
│ Y-Axis      │ Y-Axis      │ Y-Axis      │ Y-Axis      │
├─────────────┼─────────────┼─────────────┼─────────────┤
│ Z-Axis      │ Z-Axis      │ Z-Axis      │ Z-Axis      │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

### **解析機能:**

- ✅ **ワイヤーフレーム** - 3D形状の構造表示
- ✅ **点群** - 背面カリング付き密度可視化
- ✅ **断面図** - 内部構造のグラデーション表示
- ✅ **テクスチャ** - **NEW** UV座標による高品質表面材質表示
- ✅ **同期再生** - 4つの視覚化が同時動作
- ✅ **大規模対応** - 70,000面超のモデルに対応
- ✅ **メモリ最適化** - 安定した長時間動画生成

## 対応3Dファイル形式

### 3Dモデル

- OBJ (.obj) + MTL テクスチャ
- PLY (.ply)
- STL (.stl)
- GLTF/GLB (.gltf/.glb)  
- DAE (.dae) - Collada
- 3MF (.3mf)
- X3D (.x3d)
- OFF (.off)
- STEP (.step, .stp)
- IGES (.iges, .igs)

### テクスチャ

- JPG, PNG, BMP
- MTL材質定義
- GLTF埋め込みテクスチャ
- DAE XMLテクスチャ参照

## 技術的特徴

### 幾何学的断面

- mesh.section()による真の平面交線計算
- 球体断面の正確な円形表示
- サイズ変化の正確な表現

### 高品質テクスチャマッピング

- **Poly3DCollection最適化**: matplotlibでの面ごとテクスチャレンダリング
- **UV座標精密処理**: テクスチャ画像からの正確な色抽出
- **MTLファイル解析**: OBJ+MTLの完全サポート
- **大規模対応**: 70,000面超モデルでの安定動作
- **メモリ管理**: 2フレームごとの最適化クリーンアップ
- **フォールバック機能**: 頂点色・デフォルト色への自動切り替え

### 色生成

- 位置ベースHSVカラーマッピング
- 深度シェーディング
- グラデーション色生成

### その他の特徴

- **Trimesh** - 高速で安定した3D処理
- **背面カリング** - カメラから見える面のみ表示
- **特徴点抽出** - 境界検出、曲率解析、距離ベース選択
- **透視投影** - 自然な3D表示
- **MP4出力** - OpenCVによる高品質動画生成

## 使用例

### パフォーマンス最適化例

```bash
# 軽量モデル向け（高フレーム数）
python create_rotation_video.py "simple.ply" --frames 60 --fps 30

# 大規模モデル向け（低フレーム数）
python create_rotation_video.py "complex.obj" --texture --frames 10 --fps 10
```

## トラブルシューティング

### 空の画像が生成される場合

- モデルファイルが正しく読み込まれているか確認
- 対応形式かどうか確認
- ファイルパスに日本語が含まれていないか確認

### CADファイルが読み込めない場合

- STEP, IGESなどのCAD形式は、追加のライブラリ（例: FreeCAD, python-occ）が必要になる場合があります。
- エラーメッセージを確認し、必要に応じてライブラリをインストールしてください。
- 互換性の高いSTLやOBJ形式への変換も有効です。

### 動画が再生されない場合

- OpenCVが正しくインストールされているか確認
- 出力ディレクトリの権限を確認

### メモリ不足・タイムアウトエラー

- **テクスチャ動画**: フレーム数を減らす（--frames 10-20推奨）
- **大規模モデル**: 解像度を下げる（--resolution 720p）
- **点群数調整**: デフォルト10000点→5000点など
- **FPS調整**: --fps 10-15で処理負荷軽減

### テクスチャが表示されない場合

- OBJ+MTLファイルの配置を確認
- テクスチャ画像ファイル（JPG/PNG）の存在確認
- UV座標の有無を確認
- --textureオプションの指定確認

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

- `create_rotation_video.py` - メインの動画作成クラス（3x4グリッド統合）
- `trimesh_texture_video_creator.py` - テクスチャ動画生成（Poly3DCollection最適化）
- `texture_loader.py` - テクスチャ読み込みモジュール（MTL解析）
- `open3d_video_creator.py` - Open3D高速レンダリングモジュール
- `x3d_loader.py` - X3Dファイル読み込みモジュール
- `color_generator.py` - 色生成・シェーディングモジュール
- `cross_section_processor.py` - 断面処理モジュール
- `gemini_video_analyze.py` - AI動画解析

### カスタマイズ可能な設定

- 点群サンプリング数
- 画像解像度
- 投影パラメータ
- 断面の厚み
- 色設定
- フレーム数・FPS
