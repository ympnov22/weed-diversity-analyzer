# Natural Farming Field Vegetation Diversity Analysis Tool

A comprehensive tool for analyzing weed species diversity in natural farming fields using advanced computer vision and ecological analysis methods.

## 🌱 Project Status

**Current Version**: 1.0.0 (8 phases completed)  
**Status**: Production-ready visualization dashboard, **image upload functionality in development**  
**Next Release**: Phase A - Basic Image Upload Functionality

## ✨ Features

### 🔬 Analysis Engine (Fully Implemented)
- **Automatic Weed Identification**: Uses iNatAg Swin Transformer model for 2,959 species classification
- **Biodiversity Metrics**: Calculates Shannon diversity, Hill numbers, Chao1 estimation, and more
- **Advanced Analytics**: Comparative analysis, functional diversity, and statistical testing
- **Statistical Rigor**: Bootstrap confidence intervals, sample size correction, soft voting

### 📊 Visualization Dashboard (Fully Implemented)
- **Time Series Visualization**: GitHub-style calendar and interactive charts for temporal analysis
- **Interactive Dashboard**: Multiple analysis views with real-time data updates
- **Species Analysis**: Distribution charts, frequency analysis, and trend detection
- **Model Performance**: Processing statistics and accuracy metrics

### 🚧 Image Upload Interface (In Development - Phase A)
- **Web Upload**: Drag & drop interface for image uploads *(Coming Soon)*
- **Batch Processing**: Multiple image analysis *(Coming Soon)*
- **Real-time Results**: Immediate analysis feedback *(Coming Soon)*
- **Session Management**: Organize and track analysis sessions *(Coming Soon)*

## 🎯 Current Capabilities

### ✅ What Works Now
- **Complete analysis pipeline**: iNatAg model integration with 2,959 species support
- **Full visualization suite**: Interactive dashboards, calendars, and time-series charts
- **Database integration**: PostgreSQL with user authentication
- **Docker deployment**: Complete containerized environment
- **API endpoints**: RESTful API for data access and visualization

### 🔄 What's Coming Next (Phase A)
- **Image upload endpoint**: `POST /api/upload` for file uploads
- **Upload interface**: User-friendly web interface for image submission
- **Analysis integration**: Direct connection from upload to analysis pipeline
- **Result display**: Real-time analysis results and visualization updates

## 🤖 Model Strategy

### iNatAg Swin Transformer Integration
- **Dataset**: 4.7M images, 2,959 species from iNatAg
- **Architecture**: Swin Transformer (Tiny/Small/Base/Large variants)
- **Repository**: Project-AgML/iNatAg-models on Hugging Face Hub
- **License**: Apache-2.0 (research and commercial use permitted)
- **Status**: ✅ **Fully integrated and operational**

### Model Variants
- **Tiny (28M params)**: Fast inference, suitable for real-time processing
- **Small (50M params)**: Balanced performance and speed
- **Base (88M params)**: High accuracy, recommended for most use cases *(Currently loaded)*
- **Large (197M params)**: Maximum accuracy for research applications

### LoRA Fine-tuning Support
- **Efficient adaptation**: Parameter-efficient fine-tuning for regional specialization
- **Hokkaido optimization**: Specialized for northern Japan field conditions
- **Modular design**: Easy adaptation to other regions and crop types
- **Status**: ✅ **Framework implemented, ready for custom training**

## 🚀 Deployment

### Fly.io Configuration
- **Instance Type**: shared-cpu-1x
- **Memory**: 512MB
- **Primary Region**: nrt (Tokyo)
- **Health Endpoint**: `/health`
- **Status**: ✅ **Configured and ready**

### Quick Deploy
```bash
# Deploy to Fly.io
flyctl deploy

# Check deployment status
flyctl status
```

## ⚙️ Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DATABASE_URL` | Database connection string | `sqlite:///./weed_diversity.db` | Yes |
| `SECRET_KEY` | Application secret key | - | Yes |
| `LOG_LEVEL` | Logging level | `INFO` | No |
| `ALLOWED_ORIGINS` | CORS allowed origins | `http://localhost:8000` | No |
| `PRIMARY_REGION` | Deployment region | `nrt` | No |
| `PRODUCTION` | Production mode flag | `false` | No |
| `PORT` | Application port | `8000` | No |

Copy `.env.example` to `.env` and configure values.

## 💻 Local Development

### Quick Start
```bash
# Setup environment
./scripts/local-setup.sh

# Start development server
source venv/bin/activate
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Docker Compose Setup
```bash
# Build and start all services
docker-compose up --build

# Stop services
docker-compose down
```

### Development Commands
```bash
# Run tests
python -m pytest tests/

# Code formatting
python -m black src/ tests/

# Linting
python -m flake8 src/ tests/

# Type checking
python -m mypy src/
```

## 🛠️ Technical Stack
- **Python 3.12+**: Core language
- **PyTorch**: Deep learning framework
- **FastAPI**: Web framework
- **PostgreSQL**: Production database
- **Docker**: Containerization
- **OpenCV**: Image processing
- **Pandas/NumPy**: Data processing
- **Plotly/D3.js**: Visualization

## 📁 Project Structure
```
weed-diversity-analyzer/
├── docs/                    # Documentation
├── src/                     # Source code
│   ├── models/             # ML models and inference
│   ├── preprocessing/      # Image processing
│   ├── analysis/           # Diversity analysis
│   ├── visualization/      # Web interface
│   └── database/           # Data persistence
├── tests/                   # Test suite
├── docker-compose.yml      # Local development
└── requirements.txt        # Dependencies
```

## 👥 Contributors
- **Developer**: Devin AI
- **Project Owner**: ヤマシタ　ヤスヒロ (@ympnov22)

---

**Repository**: https://github.com/ympnov22/weed-diversity-analyzer  
**License**: Apache-2.0  
**Documentation**: [docs/](./docs/)
