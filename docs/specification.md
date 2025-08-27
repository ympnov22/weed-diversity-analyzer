# Natural Farming Field Vegetation Diversity Analysis Tool (iNatAg Version) - Technical Specification

## 1. Project Overview

### 1.1 Purpose
Develop a tool to identify weed species and analyze diversity from field images in natural farming practices, with time-series visualization capabilities. Utilizes the iNatAg dataset (2,959 species, 4.7M images) with Swin Transformer + LoRA fine-tuning for high-precision vegetation analysis specialized for Hokkaido field conditions.

### 1.2 Objectives
- **High-precision weed species identification**: Large-scale classification using iNatAg (2,959 species support)
- **Hokkaido adaptation**: Regional specialization through Swin Transformer + LoRA fine-tuning
- **Accurate biodiversity metrics calculation**: High-precision computation of Shannon diversity, Hill numbers, etc.
- **Robustness to varying shooting conditions**: Adaptation to weather and time-of-day variations
- **Modular design for extensibility**: Expandability to other regions and crops

### 1.3 Target Users
- **Natural farming practitioners** (primary users): Field recording tool
- **Agricultural researchers** (future expansion): Comparative research and academic use

### 1.4 Technical Advantages
- **Large-scale dataset**: 4.7M images, 2,959 species (1.86x species coverage compared to previous versions)
- **State-of-the-art architecture**: High-precision image recognition with Swin Transformer
- **Efficient adaptation**: Lightweight regional adaptation learning through LoRA fine-tuning
- **Academic foundation**: Technical reliability backed by arXiv publications

## 2. Functional Requirements

### 2.1 Input Specifications
- **Data format**: JPEG image files
- **Directory structure**: `data/YYYY-MM-DD/*.jpg` or web upload interface
- **Image size**: 224x224 minimum (Swin Transformer optimized)
- **Shooting conditions**: Outdoor natural light, various weather and time conditions
- **Target region**: Natural farming fields in Hokkaido (LoRA adaptation target)

### 2.1.1 Image Upload Interface (NEW - Phase A Implementation)
- **Web upload endpoint**: `POST /api/upload` with multipart/form-data support
- **Supported formats**: JPEG, PNG, WebP (auto-conversion to JPEG)
- **File size limits**: Maximum 10MB per image, batch upload up to 50 images
- **Validation**: Format verification, size validation, corruption detection
- **Progress tracking**: Real-time upload progress and processing status
- **Error handling**: Comprehensive error messages and retry mechanisms

### 2.2 前処理機能

#### 2.2.1 画像補正
- **明度補正**: ヒストグラム均等化、CLAHE適用
- **色温度補正**: ホワイトバランス調整（Gray World/White Patch）
- **露出補正**: ガンマ補正による明暗調整
- **Swin対応**: 224x224リサイズ、パッチ分割最適化

#### 2.2.2 冗長性削減
- **類似度計算**: 構造的類似性指数（SSIM）、特徴量ベース類似度
- **クラスタリング**: 階層クラスタリング（Ward法）
- **代表画像選択**: 品質スコア最高の画像を選択

#### 2.2.3 品質評価
- **ブラー検出**: Laplacian分散による鮮明度評価
- **露出評価**: ヒストグラム解析による適正露出判定
- **Swin適合性**: パッチ分割品質の事前評価

### 2.3 推論機能

#### 2.3.1 主要モデル: iNatAg Swin Transformer
- **データセット**: iNatAg（470万画像、2,959種）
- **アーキテクチャ**: Swin Transformer（Tiny/Base/Large）
- **アクセス方法**: Hugging Face Hub (Project-AgML/iNatAg-models)
- **出力**: Top-k推定（k=3）+ 確信度
- **入力サイズ**: 224x224 RGB

#### 2.3.2 LoRA微調整仕様
- **対象レイヤー**: Attention層、MLP層
- **LoRAランク**: r=16（デフォルト）
- **学習率**: 1e-4
- **北海道適応**: 畑画像での追加学習
- **微調整方法**: 最終層 + LoRAアダプター

#### 2.3.3 モデルバリエーション
```python
AVAILABLE_MODELS = {
    "swin_tiny_with_lora": {
        "size": "117MB",
        "speed": "高速",
        "accuracy": "標準",
        "recommended_for": "リアルタイム処理"
    },
    "swin_base_with_lora": {
        "size": "374MB", 
        "speed": "中速",
        "accuracy": "高精度",
        "recommended_for": "バランス重視"
    },
    "swin_large_with_lora": {
        "size": "946MB",
        "speed": "低速",
        "accuracy": "最高精度",
        "recommended_for": "精度最優先"
    }
}
```

#### 2.3.4 推論後処理
- **信頼度フィルタリング**: 閾値以下は上位分類群にロールアップ
- **Top-3ソフト投票**: 重み付き平均による多様性計算
- **北海道特化補正**: 地域適応モデルによる予測調整

### 2.4 多様性評価機能

#### 2.4.1 基本指標
- **種リッチネス (R)**: ユニーク種数
- **Shannon多様度 (H')**: -Σ(pi × ln(pi))
- **Pielou均等度 (J)**: H' / ln(R)
- **Simpson多様度 (D)**: 1 - Σ(pi²)
- **Hill数**: q=0,1,2での多様度

#### 2.4.2 高度な指標
- **Chao1推定種数**: 未観測種の推定
- **カバレッジ標準化**: C*=0.8基準での標準化
- **信頼区間**: ブートストラップ法による区間推定

#### 2.4.3 撮影枚数補正
- **サブサンプリング**: 固定枚数m（30枚）での反復平均
- **レアファクション**: 種蓄積曲線による補正
- **Top-3ソフト投票**: 確信度重み付きによる多様性計算

### 2.5 出力機能

#### 2.5.1 日次サマリ (JSON)
```json
{
  "date": "2025-08-24",
  "species_richness": 15,
  "shannon_diversity": 2.34,
  "pielou_evenness": 0.86,
  "hill_numbers": {
    "q0": 15,
    "q1": 10.4,
    "q2": 8.2
  },
  "chao1_estimate": 18.5,
  "confidence_intervals": {
    "richness": [12, 18],
    "shannon": [2.1, 2.6]
  },
  "top_species": [
    {"name": "Taraxacum officinale", "abundance": 0.23, "confidence": 0.89},
    {"name": "Plantago major", "abundance": 0.18, "confidence": 0.92}
  ],
  "total_images": 45,
  "processed_images": 38,
  "processing_metadata": {
    "model_used": "inatag_swin_base_lora",
    "hokkaido_adapted": true,
    "processing_time": 127.3,
    "soft_voting_applied": true
  }
}
```

#### 2.5.2 詳細データ (CSV)
```csv
date,image_path,species_1,confidence_1,species_2,confidence_2,species_3,confidence_3,model_version,hokkaido_adapted
2025-08-24,data/2025-08-24/IMG_001.jpg,Taraxacum officinale,0.89,Plantago major,0.08,Trifolium repens,0.03,inatag_swin_base_lora,true
```

#### 2.5.3 可視化データ
- **GitHub草風カレンダー用JSON**: 日次多様性スコア
- **時系列グラフ用データ**: 種リッチネス・Shannon多様度推移

## 3. 非機能要件

### 3.1 性能要件
- **処理速度**: 
  - Tiny: 1秒/画像以下
  - Base: 2秒/画像以下
  - Large: 3秒/画像以下
- **メモリ使用量**: 
  - Tiny: 2GB以下
  - Base: 4GB以下
  - Large: 8GB以下
- **バッチ処理**: 100画像/日で10分以内

### 3.2 精度要件
- **種識別精度**: Top-1で75%以上、Top-3で90%以上（iNatAg大規模データ効果）
- **多様性指標精度**: 手動カウントとの相関係数0.85以上
- **北海道適応効果**: LoRA微調整により地域精度10%向上

### 3.3 拡張性要件
- **モジュール設計**: 各機能を独立したモジュールとして実装
- **設定ファイル**: YAML形式での設定管理
- **LoRA対応**: 他地域への適応学習容易性
- **モデル切り替え**: Tiny/Base/Large間の動的切り替え

## 4. システム構成

### 4.1 アーキテクチャ
```
Input Layer (画像データ)
    ↓
Preprocessing Layer (前処理・品質評価)
    ↓
Model Layer (iNatAg Swin Transformer + LoRA)
    ↓
Analysis Layer (多様性解析・統計補正)
    ↓
Output Layer (結果出力・可視化)
```

### 4.2 主要コンポーネント
- **ImageProcessor**: 画像前処理・品質評価
- **iNatAgClassifier**: Swin Transformer種識別
- **LoRAAdapter**: 北海道適応微調整
- **DiversityAnalyzer**: 多様性解析・統計補正
- **OutputGenerator**: 結果出力・可視化
- **ConfigManager**: 設定管理・モデル選択

### 4.3 データフロー
1. **画像読み込み** → 前処理 → 品質チェック → Swin対応リサイズ
2. **モデル推論** → LoRA適応 → Top-3予測 → 信頼度評価
3. **多様性計算** → ソフト投票 → 統計処理 → 補正適用
4. **結果出力** → JSON/CSV生成 → 可視化データ作成

## 5. 技術仕様

### 5.1 開発環境
- **言語**: Python 3.12+
- **深層学習**: PyTorch 2.0+, Transformers 4.30+
- **画像処理**: OpenCV 4.8+, Pillow 10.0+
- **数値計算**: NumPy 1.24+, SciPy 1.11+
- **データ処理**: Pandas 2.0+
- **LoRA**: peft 0.4+, loralib 0.1+

### 5.2 外部依存
- **iNatAg Models**: Hugging Face Hub (Project-AgML/iNatAg-models)
- **Swin Transformer**: timm 0.9+
- **生態学ライブラリ**: scikit-bio 0.5+, EcoSpold 2.0+

### 5.3 設定管理
```yaml
# config.yaml
app:
  name: "Weed Diversity Analyzer (iNatAg)"
  version: "2.0.0"

model:
  primary: "inatag_swin_base_lora"
  alternatives: ["inatag_swin_tiny_lora", "inatag_swin_large_lora"]
  confidence_threshold: 0.7
  hokkaido_adapted: true

lora_config:
  rank: 16
  alpha: 32
  dropout: 0.1
  target_modules: ["query", "key", "value", "dense"]
  learning_rate: 1e-4

preprocessing:
  image_size: [224, 224]  # Swin Transformer最適化
  similarity_threshold: 0.8
  quality_thresholds:
    min_blur_score: 100.0
    min_brightness: 50
    max_brightness: 200

diversity:
  soft_voting: true  # Top-3ソフト投票
  bootstrap_iterations: 1000
  coverage_target: 0.8
  subsample_size: 30
```

## 6. 品質保証

### 6.1 テスト戦略
- **単体テスト**: 各モジュールの機能テスト
- **統合テスト**: エンドツーエンドの処理テスト
- **性能テスト**: 処理速度・メモリ使用量測定
- **LoRAテスト**: 北海道適応効果の検証
- **精度テスト**: iNatAg vs 専門家同定の比較

### 6.2 検証方法
- **専門家による種同定結果との比較**
- **既知多様性データセットでの検証**
- **北海道畑画像での精度評価**
- **クロスバリデーション**

## 7. 運用・保守

### 7.1 ログ管理
- **処理ログ**: 各ステップの実行状況
- **モデルログ**: 使用モデル・LoRA適応状況
- **エラーログ**: 例外・警告の記録
- **性能ログ**: 処理時間・リソース使用量

### 7.2 モニタリング
- **精度監視**: 定期的な精度評価・LoRA効果測定
- **性能監視**: 処理時間の傾向分析
- **データ品質**: 入力画像の品質チェック
- **モデル監視**: Swin Transformer各サイズの性能比較

## 8. 将来拡張

### 8.1 機能拡張
- **他地域適応**: 本州・九州向けLoRAアダプター
- **季節変動解析**: 季節別多様性パターン分析
- **病害虫検出機能**: マルチタスク学習による拡張
- **土壌健康度推定**: 植生多様性からの推定

### 8.2 技術拡張
- **リアルタイム処理**: Swin Tiny活用
- **モバイルアプリ対応**: 軽量モデル展開
- **クラウド展開**: GPU最適化・分散処理
- **API提供**: RESTful API・GraphQL対応

### 8.3 LoRA拡張
- **マルチタスク学習**: 病害検出・成長段階推定
- **季節適応**: 季節別LoRAアダプター
- **圃場特化**: 個別圃場への適応学習
- **作物特化**: 作物種別LoRAアダプター

## 9. 制約事項

### 9.1 技術的制約
- **GPU要件**: CUDA対応GPU推奨（Large使用時必須）
- **メモリ要件**: 
  - Tiny: 最低2GB RAM
  - Base: 最低4GB RAM  
  - Large: 最低8GB RAM
- **ストレージ**: モデルファイル用に2GB以上

### 9.2 データ制約
- **画像品質**: 最低解像度224x224（Swin Transformer要件）
- **撮影条件**: 極端な逆光・暗所は除外
- **対象種**: iNatAg学習データに含まれる2,959種
- **地域制約**: 北海道以外では精度低下の可能性

### 9.3 LoRA制約
- **適応データ**: 北海道畑画像が必要
- **学習時間**: LoRA微調整に数時間必要
- **メモリ**: 微調整時は追加メモリ必要

## 10. 成功基準

### 10.1 技術的成功基準
- [ ] **種識別精度**: Top-3で90%以上（iNatAg大規模データ効果）
- [ ] **処理速度**: Base使用時2秒/画像以内
- [ ] **多様性指標精度**: 専門家評価との一致度85%以上
- [ ] **北海道適応効果**: LoRA微調整により10%精度向上
- [ ] **システム安定性**: 100画像連続処理でエラー率1%以下

### 10.2 ユーザー満足度
- [ ] **使いやすいインターフェース**: 設定変更の容易性
- [ ] **安定した動作**: 様々な撮影条件での頑健性
- [ ] **有用な洞察の提供**: 多様性トレンドの可視化
- [ ] **処理速度満足度**: 日次処理の実用的な速度

### 10.3 拡張性達成度
- [ ] **他地域展開可能性**: LoRAアダプター追加の容易性
- [ ] **新機能追加**: モジュール設計による拡張性
- [ ] **研究利用価値**: 学術研究での活用可能性

---

**文書バージョン**: 2.0（iNatAg版）  
**作成日**: 2025-08-24  
**作成者**: Devin AI  
**承認者**: ヤマシタ　ヤスヒロ  
**ベースモデル**: iNatAg (Project-AgML/iNatAg-models)  
**対応種数**: 2,959種  
**データセット規模**: 470万画像
