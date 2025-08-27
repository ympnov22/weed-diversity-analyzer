# 自然農法畑植生多様性解析ツール (Natural Farming Field Vegetation Diversity Analysis Tool)

## 概要
自然農法の畑の植生を撮影した画像から、雑草の種類と多様性を判定し、日ごとに時系列で可視化するツールです。

## 主な機能
- 畑の画像から雑草種の自動識別
- 生物多様性指標の算出
- 時系列データの可視化
- 高精度なモデルによる種判定

## Model Strategy / モデル戦略

### Current Status / 現在の状況
- **WeedNet**: Currently unavailable / 現在利用不可
- **Temporary Solution**: Using iNatAg Swin Transformer model / 一時的解決策：iNatAg Swin Transformerモデルを使用
- **Target Coverage**: 2,959 plant species / 対象範囲：2,959植物種

### Verification Plan / 検証計画
- Species identification accuracy validation / 種同定精度の検証
- Biodiversity metrics consistency check / 生物多様性指標の一貫性チェック
- Performance benchmarking against known datasets / 既知データセットでの性能ベンチマーク

## Deployment Guidelines / デプロイメントガイドライン

### Fly.io Configuration / Fly.io設定
- **Instance Type**: shared-cpu-1x / インスタンスタイプ：shared-cpu-1x
- **Memory**: 256-512MB / メモリ：256-512MB  
- **Primary Region**: nrt (Tokyo) / プライマリリージョン：nrt（東京）
- **Health Endpoint**: `/health` / ヘルスエンドポイント：`/health`

### Deployment Commands / デプロイメントコマンド
```bash
# Deploy to Fly.io / Fly.ioにデプロイ
./scripts/deploy.sh
```

## Environment Variables / 環境変数

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DATABASE_URL` | Database connection string / データベース接続文字列 | `sqlite:///./weed_diversity.db` | Yes |
| `SECRET_KEY` | Application secret key / アプリケーション秘密鍵 | - | Yes |
| `LOG_LEVEL` | Logging level / ログレベル | `INFO` | No |
| `ALLOWED_ORIGINS` | CORS allowed origins / CORS許可オリジン | `http://localhost:8000` | No |
| `PRIMARY_REGION` | Deployment region / デプロイメントリージョン | `nrt` | No |
| `PRODUCTION` | Production mode flag / 本番モードフラグ | `false` | No |
| `PORT` | Application port / アプリケーションポート | `8000` | No |

Copy `.env.example` to `.env` and configure values.
`.env.example`を`.env`にコピーして値を設定してください。

## Local Development / ローカル開発

### Quick Start / クイックスタート
```bash
# Setup environment / 環境セットアップ
./scripts/local-setup.sh

# Start development server / 開発サーバー起動
source venv/bin/activate
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Docker Compose Setup / Docker Compose セットアップ
```bash
# Build and start all services / 全サービスのビルドと起動
docker-compose up --build

# Stop services / サービス停止
docker-compose down
```

### Development Commands / 開発コマンド
```bash
# Run tests / テスト実行
python -m pytest tests/

# Code formatting / コード整形
python -m black src/ tests/

# Linting / リント
python -m flake8 src/ tests/

# Type checking / 型チェック
python -m mypy src/
```

## 技術スタック
- Python 3.12+
- PyTorch / TensorFlow (深層学習)
- OpenCV (画像処理)
- Pandas / NumPy (データ処理)
- Matplotlib / Plotly (可視化)

## プロジェクト構成
```
weed-diversity-analyzer/
├── docs/                    # ドキュメント
├── src/                     # ソースコード
├── data/                    # データディレクトリ
├── models/                  # 学習済みモデル
├── tests/                   # テストコード
├── notebooks/               # Jupyter notebooks
└── requirements.txt         # 依存関係
```

## 開発者
- 開発者: Devin AI
- 依頼者: ヤマシタ　ヤスヒロ (@ympnov22)
