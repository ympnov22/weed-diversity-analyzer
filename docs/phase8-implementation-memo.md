# Phase 8 実装メモ - 運用・デプロイメント技術仕様
# Phase 8 Implementation Memo - Production Deployment Technical Specifications

**作成日**: 2025年8月25日  
**対象**: Phase 8 運用・デプロイメント実装  
**承認待ち**: 条件付き承認済み

## 1. DBスキーマと移行

### ERD (Entity Relationship Diagram)
```
users (1) ----< (N) analysis_sessions (1) ----< (N) diversity_metrics
                                        |
                                        +----< (N) image_data (1) ----< (N) prediction_results (N) >---- (1) diversity_metrics
                                        |
                                        +----< (N) processing_results

主要テーブル:
- users: id(PK), username(UQ), email(UQ), api_key(UQ), is_active, created_at
- analysis_sessions: id(PK), user_id(FK), session_name, description, start_date, end_date, created_at
- diversity_metrics: id(PK), session_id(FK), date(IDX), species_richness, shannon_diversity, etc.
- image_data: id(PK), session_id(FK), path, date(IDX), width, height, is_processed, cluster_id
- prediction_results: id(PK), diversity_metrics_id(FK), image_data_id(FK), model_name, predictions(JSON)
- processing_results: id(PK), session_id(FK), date(IDX), processing_time_total, average_confidence
```

### Alembic マイグレーション方針
```bash
# 初期スキーマ
alembic init alembic/
alembic revision --autogenerate -m "Initial schema"
alembic upgrade head

# 将来変更の手順
alembic revision --autogenerate -m "Add new column"
alembic upgrade head  # 本番適用
alembic downgrade -1  # ロールバック
```

### 主要インデックスとユニーク制約
```sql
-- 検索性能向上のためのインデックス
CREATE INDEX idx_diversity_metrics_date ON diversity_metrics(date);
CREATE INDEX idx_diversity_metrics_session_date ON diversity_metrics(session_id, date);
CREATE INDEX idx_image_data_date ON image_data(date);
CREATE INDEX idx_image_data_session_date ON image_data(session_id, date);
CREATE INDEX idx_prediction_results_timestamp ON prediction_results(timestamp);

-- ユニーク制約
ALTER TABLE users ADD CONSTRAINT uq_users_username UNIQUE (username);
ALTER TABLE users ADD CONSTRAINT uq_users_email UNIQUE (email);
ALTER TABLE users ADD CONSTRAINT uq_users_api_key UNIQUE (api_key);
```

## 2. 認証

### APIキーの生成・保管方法
```python
# 生成: secrets.token_urlsafe(32) = 256bit エントロピー
# 最小長: 43文字 (Base64URL)
# 保管: データベースにハッシュ化せずプレーンテキスト（個人利用のため）
# ローテーション: 手動でユーザーが新しいキー生成・置換

import secrets
api_key = secrets.token_urlsafe(32)  # 例: "tymdqToKEkoKtUtNRff75tdo7acw3D-eDOpJ7F0xfkY"
```

### ヘッダー仕様とレート制限
```python
# ヘッダー仕様
Authorization: Bearer <api_key>
# または
X-API-Key: <api_key>

# レート制限: 個人利用のため当初は無制限
# 将来実装時: slowapi + Redis
# 制限例: 100 requests/minute per API key
```

### 可視化ルートの公開/認証切替
```yaml
# config/config.yaml
auth:
  enabled: true
  require_auth_for_api: false      # 可視化API は公開
  require_auth_for_data_upload: true  # データ投稿は認証必須
  
# 実装: optional_auth dependency で制御
@app.get("/api/calendar/{metric}")
async def get_calendar_data(user: UserModel = Depends(optional_auth)):
    # 認証ユーザーは実データ、未認証はサンプルデータ
```

## 3. 設定・シークレット

### 必要な環境変数一覧
```bash
# 必須
DATABASE_URL=postgresql://user:pass@host:5432/dbname
SECRET_KEY=<32-byte-random-string>

# オプション（デフォルト値あり）
LOG_LEVEL=INFO
ALLOWED_ORIGINS=https://weed-diversity-analyzer.fly.dev
PRIMARY_REGION=nrt
PRODUCTION=true
PORT=8000
PYTHONUNBUFFERED=1

# 開発用
DEVELOPMENT=false
DEBUG=false
```

### シークレット投入方法
```bash
# Fly.io secrets
flyctl secrets set DATABASE_URL="postgresql://..."
flyctl secrets set SECRET_KEY="$(openssl rand -base64 32)"

# .env.example 更新
DATABASE_URL=sqlite:///./weed_diversity.db
SECRET_KEY=your-secret-key-here
LOG_LEVEL=INFO
ALLOWED_ORIGINS=http://localhost:8000
```

## 4. Fly.io / Docker

### fly.toml 差分
```toml
# 主要変更点
[machine]
  memory = "512mb"        # 個人利用に最適化
  cpu_kind = "shared"     # コスト削減
  cpus = 1

[http_service]
  internal_port = 8000
  auto_stop_machines = true    # コスト削減
  min_machines_running = 0     # アイドル時停止

[[http_service.checks]]
  path = "/health"       # ヘルスチェックエンドポイント
  interval = "30s"       # チェック間隔

[env]
  PRIMARY_REGION = "nrt"  # 東京リージョン
```

### ボリューム/マネージドDBの選択
```bash
# 選択: Fly PostgreSQL (マネージドDB)
flyctl postgres create --name weed-diversity-analyzer-db --region nrt

# バックアップ方針
# - 自動スナップショット: 毎日 (Fly.io標準)
# - 手動バックアップ: flyctl postgres backup create
# - 復旧手順: flyctl postgres backup restore <backup-id>

# ボリューム (データ永続化用)
flyctl volumes create weed_data --region nrt --size 1
flyctl volumes create weed_output --region nrt --size 1
```

### Dockerfile 最終サイズと起動時メモリ見積
```dockerfile
# 最終イメージサイズ: ~800MB (Python 3.12 + ML libraries)
# 起動時メモリ見積:
# - ベースメモリ: ~150MB (Python + FastAPI)
# - iNatAg モデル: ~200MB (Swin Transformer)
# - PostgreSQL接続: ~20MB
# - 予備: ~142MB
# 合計: ~512MB (制限内)

# RSS目標: 400MB 定常, 500MB ピーク
# 起動時ピーク: 480MB (モデルロード時)
```

## 5. ルーティング変更

### web_server.py の主要エンドポイント変更
```python
# 既存エンドポイント → DB サービス層リダイレクト
GET /api/calendar/{metric}     → DatabaseService.get_calendar_data()
GET /api/time-series          → DatabaseService.get_time_series_data()
GET /api/species-dashboard    → DatabaseService.get_species_dashboard_data()

# 新規エンドポイント
GET /health                   → ヘルスチェック
POST /api/upload              → 画像アップロード (認証必須)
GET /api/sessions             → セッション一覧 (認証必須)
POST /api/sessions            → セッション作成 (認証必須)

# 認証制御
- 可視化系: optional_auth (公開アクセス可、認証時は実データ)
- データ操作系: require_auth (認証必須)
```

### 互換性 (既存 JSON/CSV 出力)
```python
# 変更なし: 既存の OutputManager/JSONExporter/CSVExporter は保持
# 追加: DatabaseService が並行して動作
# 設定で切り替え可能:

if os.getenv("DATABASE_URL"):
    # データベースモード
    service = DatabaseService(db)
else:
    # ファイルモード (既存)
    service = OutputManager()
```

## 6. 検証計画

### ローカル: docker compose up 手順
```bash
# 1. 環境準備
cp .env.example .env
# DATABASE_URL, SECRET_KEY を設定

# 2. コンテナ起動
docker-compose up --build

# 3. Smoke テスト
curl http://localhost:8000/health
curl http://localhost:8000/api/calendar/shannon_diversity
curl -H "Authorization: Bearer <api_key>" http://localhost:8000/api/sessions
```

### CI: 250+ テスト実行結果サマリ
```bash
# テスト実行
python -m pytest tests/ -v --cov=src --cov-report=term-missing

# 期待結果:
# - 250+ tests passed
# - Coverage: >95%
# - 新規テスト: test_database_*.py, test_auth.py (30+ tests)
# - 既存テスト: 全て PASS (回帰なし)
```

### 本番: デプロイURL、Smoke チェックリスト
```bash
# デプロイURL (予定)
https://weed-diversity-analyzer.fly.dev

# Smoke チェックリスト
✓ GET /health → 200 OK
✓ GET / → 200 OK (メインページ表示)
✓ GET /api/calendar/shannon_diversity → 200 OK (サンプルデータ)
✓ POST /api/sessions (認証付き) → 201 Created
✓ 可視化ページ表示 → カレンダー、時系列チャート正常表示
✓ データベース接続 → PostgreSQL 正常接続
✓ ログ出力 → 構造化ログ正常出力
```

### ロールバック手順
```bash
# 1. 前バージョンへの戻し
flyctl releases list
flyctl releases rollback <previous-version>

# 2. データベースロールバック (必要時)
flyctl postgres backup list --app weed-diversity-analyzer-db
flyctl postgres backup restore <backup-id>

# 3. 確認
curl https://weed-diversity-analyzer.fly.dev/health

# 4. 緊急時: アプリ停止
flyctl apps suspend weed-diversity-analyzer
```

## 実装順序

1. **データベース層** (2-3h)
   - Alembic セットアップ
   - マイグレーション実行
   - DatabaseService 統合テスト

2. **認証システム** (1-2h)
   - API キー認証実装
   - エンドポイント保護
   - 認証テスト

3. **Web サーバー更新** (2-3h)
   - ルーティング変更
   - DB サービス統合
   - ヘルスチェック追加

4. **Docker/Fly.io 設定** (1-2h)
   - fly.toml 最終調整
   - 環境変数設定
   - デプロイ準備

5. **検証・デプロイ** (2-3h)
   - ローカルテスト
   - CI 実行
   - 本番デプロイ
   - Smoke テスト

**総見積時間**: 8-13時間

---

**承認確認**: 上記仕様で実装を開始してよろしいでしょうか？
