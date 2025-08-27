"""FastAPI web server for visualization interface."""

from fastapi import FastAPI, HTTPException, Depends, Request, status, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import uvicorn
import os
from datetime import datetime

from ..utils.logger import LoggerMixin
from ..output.output_manager import OutputManager
from ..database.database import get_db
from ..database.services import DatabaseService
from ..auth.dependencies import require_auth, optional_auth
from ..database.models import UserModel
from ..models.model_manager import ModelManager
from ..utils.config import ConfigManager
from .calendar_visualizer import CalendarVisualizer
from .time_series_visualizer import TimeSeriesVisualizer
from .dashboard_generator import DashboardGenerator


class WebServer(LoggerMixin):
    """FastAPI web server for diversity visualization."""
    
    def __init__(self, output_manager: OutputManager = None, host: str = "127.0.0.1", port: int = 8000):
        """Initialize web server.
        
        Args:
            output_manager: Output manager instance
            host: Server host
            port: Server port
        """
        self.app = FastAPI(
            title="Weed Diversity Analyzer",
            description="iNatAg-based weed diversity visualization dashboard",
            version="1.0.0"
        )
        self.output_manager = output_manager or OutputManager()
        self.host = host
        self.port = port
        
        self._setup_cors()
        
        self.calendar_viz = CalendarVisualizer()
        self.time_series_viz = TimeSeriesVisualizer()
        self.dashboard_gen = DashboardGenerator()
        
        self.logger.info("Minimal mode - all model loading disabled for memory optimization")
        self.model_manager = None
        self.model_loaded = False
        
        self.static_dir = Path(__file__).parent / "static"
        self.static_dir.mkdir(exist_ok=True)
        
        self._setup_routes()
        self._setup_static_files()
    
    def _setup_cors(self):
        """Setup CORS middleware for production."""
        origins = ["*"]
        
        if os.getenv("PRODUCTION"):
            origins = [
                "https://weed-diversity-analyzer.fly.dev",
                "http://localhost:8000",
                "http://127.0.0.1:8000"
            ]
        
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Setup API routes for visualization data."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def root():
            """Main dashboard page."""
            return self._get_main_dashboard_html()
        
        @self.app.get("/api/calendar/{metric}")
        async def get_calendar_data(
            metric: str,
            user: Optional[UserModel] = Depends(optional_auth),
            db: Session = Depends(get_db)
        ):
            """Get calendar data for specified metric."""
            try:
                if user and hasattr(user, 'id'):
                    db_service = DatabaseService(db)
                    calendar_data = db_service.get_calendar_data(user.id, metric)
                    
                    if calendar_data["total_days"] > 0:
                        return JSONResponse(calendar_data)
                
                sample_data = self.calendar_viz.generate_sample_data()
                return JSONResponse(sample_data)
                
            except Exception as e:
                self.logger.error(f"Failed to get calendar data for {metric}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/time-series")
        async def get_time_series_data(
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            user: Optional[UserModel] = Depends(optional_auth),
            db: Session = Depends(get_db)
        ):
            """Get time series data for specified date range."""
            try:
                if user and hasattr(user, 'id'):
                    db_service = DatabaseService(db)
                    time_series_data = db_service.get_time_series_data(user.id, start_date, end_date)
                    
                    if time_series_data["total_points"] > 0:
                        return JSONResponse(time_series_data)
                
                sample_data = self.time_series_viz.generate_sample_time_series_data()
                return JSONResponse(sample_data)
                
            except Exception as e:
                self.logger.error(f"Failed to get time series data: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/dashboard/species")
        async def get_species_dashboard_data(
            user: Optional[UserModel] = Depends(optional_auth),
            db: Session = Depends(get_db)
        ):
            """Get species distribution dashboard data."""
            try:
                if user and hasattr(user, 'id'):
                    db_service = DatabaseService(db)
                    dashboard_data = db_service.get_species_dashboard_data(user.id)
                    
                    if dashboard_data["total_species"] > 0:
                        return JSONResponse(dashboard_data)
                
                sample_data = self.dashboard_gen.generate_sample_dashboard_data()
                return JSONResponse(sample_data["daily_summaries"])
                
            except Exception as e:
                self.logger.error(f"Failed to get species dashboard data: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/calendar", response_class=HTMLResponse)
        async def calendar_page(metric: str = "shannon_diversity", year: Optional[int] = None):
            """Calendar visualization page."""
            try:
                calendar_data_response = await get_calendar_data(metric)
                calendar_data = json.loads(calendar_data_response.body)
                
                html_content = self.calendar_viz.generate_calendar_html(
                    calendar_data=calendar_data,
                    year=year
                )
                
                return HTMLResponse(html_content)
                
            except Exception as e:
                self.logger.error(f"Failed to generate calendar page: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/time-series", response_class=HTMLResponse)
        async def time_series_page():
            """Time series visualization page."""
            try:
                time_series_response = await get_time_series_data()
                time_series_data = json.loads(time_series_response.body)
                
                html_content = self.time_series_viz.generate_diversity_trends_chart(
                    time_series_data=time_series_data
                )
                
                return HTMLResponse(html_content)
                
            except Exception as e:
                self.logger.error(f"Failed to generate time series page: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/dashboard/species", response_class=HTMLResponse)
        async def species_dashboard_page():
            """Species distribution dashboard page."""
            try:
                species_data_response = await get_species_dashboard_data()
                daily_summaries = json.loads(species_data_response.body)
                
                html_content = self.dashboard_gen.generate_species_distribution_chart(
                    daily_summaries=daily_summaries
                )
                
                return HTMLResponse(html_content)
                
            except Exception as e:
                self.logger.error(f"Failed to generate species dashboard: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/dashboard/performance", response_class=HTMLResponse)
        async def performance_dashboard_page():
            """Model performance dashboard page."""
            try:
                sample_data = self.dashboard_gen.generate_sample_dashboard_data()
                processing_metadata = sample_data["processing_metadata"]
                
                html_content = self.dashboard_gen.generate_model_performance_dashboard(
                    processing_metadata=processing_metadata
                )
                
                return HTMLResponse(html_content)
                
            except Exception as e:
                self.logger.error(f"Failed to generate performance dashboard: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/comparative-analysis", response_class=HTMLResponse)
        async def comparative_analysis():
            try:
                from ..analysis.comparative_analysis import ComparativeAnalyzer
                analyzer = ComparativeAnalyzer()
                
                sample_data = self._generate_sample_daily_summaries()
                results = analyzer.compare_temporal_diversity(sample_data)
                
                dashboard_html = self.dashboard_gen.generate_comparative_analysis_dashboard(results)
                return HTMLResponse(dashboard_html)
            except Exception as e:
                self.logger.error(f"Error generating comparative analysis: {e}")
                return HTMLResponse("<h1>Error generating comparative analysis</h1>")
        
        @self.app.get("/functional-diversity", response_class=HTMLResponse)
        async def functional_diversity():
            try:
                from ..analysis.functional_diversity import FunctionalDiversityAnalyzer
                analyzer = FunctionalDiversityAnalyzer()
                
                species_list = ["Taraxacum officinale", "Plantago major", "Trifolium repens", "Poa annua"]
                sample_traits = analyzer.generate_sample_trait_database(species_list)
                analyzer.load_traits_from_dict(sample_traits)
                
                sample_abundances = {species: 10 - i*2 for i, species in enumerate(species_list)}
                results = analyzer.calculate_functional_diversity(sample_abundances)
                
                dashboard_html = self.dashboard_gen.generate_functional_diversity_dashboard(results)
                return HTMLResponse(dashboard_html)
            except Exception as e:
                self.logger.error(f"Error generating functional diversity analysis: {e}")
                return HTMLResponse("<h1>Error generating functional diversity analysis</h1>")
        
        @self.app.get("/api/status")
        async def get_status(db: Session = Depends(get_db)):
            """Get server status and data summary."""
            try:
                output_summary = self.output_manager.get_output_summary()
                
                db_status = "connected"
                try:
                    db.execute("SELECT 1")
                except Exception:
                    db_status = "disconnected"
                
                status = {
                    "server": "running",
                    "timestamp": datetime.now().isoformat(),
                    "version": "1.0.0",
                    "database_status": db_status,
                    "inatag_species_count": 2959,
                    "output_summary": output_summary
                }
                
                return JSONResponse(status)
                
            except Exception as e:
                self.logger.error(f"Failed to get status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/auth/create-user")
        async def create_user(
            username: str,
            email: str,
            db: Session = Depends(get_db)
        ):
            """Create a new user with API key."""
            try:
                from ..auth.auth import create_api_key
                from ..database.models import UserModel
                
                api_key = create_api_key()
                user = UserModel(
                    username=username,
                    email=email,
                    api_key=api_key,
                    is_active=True
                )
                
                db.add(user)
                db.commit()
                db.refresh(user)
                
                return JSONResponse({
                    "user_id": user.id,
                    "username": user.username,
                    "api_key": user.api_key,
                    "message": "User created successfully"
                })
                
            except Exception as e:
                self.logger.error(f"Failed to create user: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/upload")
        async def upload_image(
            file: UploadFile = File(...),
            user: UserModel = Depends(require_auth),
            db: Session = Depends(get_db)
        ):
            """Upload and analyze image."""
            try:
                if not file.content_type or not file.content_type.startswith('image/'):
                    raise HTTPException(status_code=400, detail="File must be an image")
                
                allowed_types = ['image/jpeg', 'image/png', 'image/webp']
                if file.content_type not in allowed_types:
                    raise HTTPException(status_code=400, detail="Unsupported image format")
                
                contents = await file.read()
                if len(contents) > 10 * 1024 * 1024:
                    raise HTTPException(status_code=400, detail="File too large (max 10MB)")
                
                import tempfile
                import uuid
                from pathlib import Path
                
                temp_dir = Path(tempfile.gettempdir()) / "weed_analyzer_uploads"
                temp_dir.mkdir(exist_ok=True)
                
                file_id = str(uuid.uuid4())
                temp_file = temp_dir / f"{file_id}_{file.filename}"
                
                with open(temp_file, "wb") as f:
                    f.write(contents)
                
                if self.model_manager and self.model_loaded:
                    from ..inference_pipeline import InferencePipeline
                    pipeline = InferencePipeline(self.model_manager.config)
                    result = pipeline.process_single_image(temp_file)
                    
                    if result:
                        db_service = DatabaseService(db)
                        
                        sessions = db_service.get_user_sessions(user.id)
                        if sessions:
                            session = sessions[0]
                        else:
                            session = db_service.create_analysis_session(
                                user.id, 
                                f"Upload Session {datetime.now().strftime('%Y-%m-%d')}"
                            )
                        
                        image_id = db_service.store_upload_result(
                            session.id,
                            file.filename or "uploaded_image",
                            str(temp_file),
                            result["predictions"],
                            result["processing_time"],
                            result["model_info"]
                        )
                        
                        temp_file.unlink()
                        
                        return JSONResponse({
                            "success": True,
                            "file_id": file_id,
                            "image_id": image_id,
                            "filename": file.filename,
                            "predictions": result["predictions"],
                            "processing_time": result["processing_time"],
                            "model_info": result["model_info"],
                            "quality_assessment": result.get("quality_assessment", {}),
                            "session_id": session.id
                        })
                    else:
                        temp_file.unlink()
                        raise HTTPException(status_code=400, detail="Image processing failed")
                else:
                    temp_file.unlink()
                    raise HTTPException(status_code=503, detail="Model not available")
                    
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Upload failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def _generate_sample_daily_summaries(self) -> List[Dict[str, Any]]:
        """Generate sample daily summaries for testing."""
        import random
        from datetime import datetime, timedelta
        
        summaries = []
        base_date = datetime(2025, 1, 1)
        
        for i in range(30):
            date = base_date + timedelta(days=i)
            summary = {
                'date': date.strftime('%Y-%m-%d'),
                'diversity_metrics': {
                    'species_richness': random.randint(5, 15),
                    'shannon_diversity': random.uniform(1.0, 3.0),
                    'pielou_evenness': random.uniform(0.3, 0.9),
                    'simpson_diversity': random.uniform(0.5, 0.9)
                },
                'species_counts': {
                    f'Species_{j}': random.randint(1, 10) 
                    for j in range(random.randint(5, 15))
                },
                'processing_info': {
                    'total_images': random.randint(10, 50),
                    'processed_images': random.randint(8, 45),
                    'average_confidence': random.uniform(0.6, 0.9)
                }
            }
            summaries.append(summary)
        
        return summaries
    
    def _get_upload_interface_html(self) -> str:
        """Generate upload interface HTML."""
        return """
        <div class="upload-section">
            <h2>🌱 画像アップロード解析</h2>
            <div class="upload-area" id="uploadArea">
                <div class="upload-content">
                    <div class="upload-icon">📷</div>
                    <p>画像をここにドラッグ&ドロップするか <button type="button" id="selectFile">ファイルを選択</button></p>
                    <p class="upload-hint">対応形式: JPEG, PNG, WebP (最大10MB)</p>
                </div>
                <input type="file" id="fileInput" accept="image/*" style="display: none;">
            </div>
            <div class="upload-progress" id="uploadProgress" style="display: none;">
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
                <p id="progressText">アップロード中...</p>
            </div>
            <div class="upload-results" id="uploadResults" style="display: none;">
                <h3>解析結果</h3>
                <div id="resultsContent"></div>
            </div>
        </div>
        """

    def _setup_static_files(self):
        """Setup static file serving."""
        if self.static_dir.exists():
            self.app.mount("/static", StaticFiles(directory=str(self.static_dir)), name="static")
    
    def _get_main_dashboard_html(self) -> str:
        """Get main dashboard HTML."""
        return f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>畑植生多様性解析ダッシュボード</title>
    <style>
        {self._get_dashboard_css()}
    </style>
</head>
<body>
    <div class="dashboard-container">
        <header class="dashboard-header">
            <h1>畑植生多様性解析ダッシュボード</h1>
            <p class="subtitle">iNatAg (2,959種対応) による自然農法畑の植生多様性分析</p>
        </header>
        
        {self._get_upload_interface_html()}
        
        <nav class="dashboard-nav">
            <div class="nav-grid">
                <a href="/calendar" class="nav-card">
                    <div class="nav-icon">📅</div>
                    <h3>カレンダー表示</h3>
                    <p>GitHub草風カレンダーで日次多様性を可視化</p>
                </a>
                
                <a href="/time-series" class="nav-card">
                    <div class="nav-icon">📈</div>
                    <h3>時系列分析</h3>
                    <p>多様性指標の時間変化と季節パターン</p>
                </a>
                
                <a href="/dashboard/species" class="nav-card">
                    <div class="nav-icon">🌱</div>
                    <h3>種分析</h3>
                    <p>検出種の分布と頻度分析</p>
                </a>
                
                <a href="/dashboard/performance" class="nav-card">
                    <div class="nav-icon">⚡</div>
                    <h3>モデル性能</h3>
                    <p>Swin Transformer & LoRA性能比較</p>
                </a>
            </div>
        </nav>
        
        <section class="dashboard-info">
            <div class="info-grid">
                <div class="info-card">
                    <h4>対応種数</h4>
                    <div class="info-value">2,959種</div>
                    <p>iNatAg大規模データセット</p>
                </div>
                
                <div class="info-card">
                    <h4>学習データ</h4>
                    <div class="info-value">470万枚</div>
                    <p>高精度分類を実現</p>
                </div>
                
                <div class="info-card">
                    <h4>モデル</h4>
                    <div class="info-value">Swin Transformer</div>
                    <p>LoRA微調整対応</p>
                </div>
                
                <div class="info-card">
                    <h4>多様性指標</h4>
                    <div class="info-value">6種類</div>
                    <p>Shannon, Hill数, Chao1等</p>
                </div>
            </div>
        </section>
        
        <footer class="dashboard-footer">
            <p>© 2025 Weed Diversity Analyzer | iNatAg-based Analysis System</p>
            <div class="footer-links">
                <a href="/api/status">API Status</a>
                <a href="https://github.com/ympnov22/weed-diversity-analyzer">GitHub</a>
            </div>
        </footer>
    </div>
    
    <script>
        {self._get_dashboard_javascript()}
    </script>
</body>
</html>"""
    
    def _get_dashboard_css(self) -> str:
        """Get CSS styles for main dashboard."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .dashboard-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .dashboard-header {
            text-align: center;
            color: white;
            margin-bottom: 40px;
        }
        
        .dashboard-header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .subtitle {
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        .dashboard-nav {
            margin-bottom: 40px;
        }
        
        .nav-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
        }
        
        .nav-card {
            background: white;
            border-radius: 12px;
            padding: 30px;
            text-decoration: none;
            color: #333;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
            text-align: center;
        }
        
        .nav-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }
        
        .nav-icon {
            font-size: 3rem;
            margin-bottom: 15px;
        }
        
        .nav-card h3 {
            font-size: 1.3rem;
            margin-bottom: 10px;
            color: #2c3e50;
        }
        
        .nav-card p {
            color: #7f8c8d;
            line-height: 1.5;
        }
        
        .dashboard-info {
            background: white;
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 40px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }
        
        .info-card {
            text-align: center;
            padding: 20px;
            border-radius: 8px;
            background: #f8f9fa;
        }
        
        .info-card h4 {
            color: #495057;
            margin-bottom: 10px;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .info-value {
            font-size: 2rem;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }
        
        .info-card p {
            color: #6c757d;
            font-size: 0.85rem;
        }
        
        .dashboard-footer {
            text-align: center;
            color: white;
            opacity: 0.8;
        }
        
        .footer-links {
            margin-top: 10px;
        }
        
        .footer-links a {
            color: white;
            text-decoration: none;
            margin: 0 15px;
            opacity: 0.8;
            transition: opacity 0.2s;
        }
        
        .footer-links a:hover {
            opacity: 1;
        }
        
        @media (max-width: 768px) {
            .dashboard-container {
                padding: 15px;
            }
            
            .dashboard-header h1 {
                font-size: 2rem;
            }
            
            .nav-grid {
                grid-template-columns: 1fr;
            }
            
            .info-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }
        
        .upload-section {
            margin: 20px 0;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 12px;
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
        }
        
        .upload-section h2 {
            color: white;
            margin-bottom: 20px;
            text-align: center;
        }
        
        .upload-area {
            border: 2px dashed rgba(255,255,255,0.3);
            border-radius: 8px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: rgba(255,255,255,0.05);
        }
        
        .upload-area:hover, .upload-area.drag-over {
            border-color: #4CAF50;
            background: rgba(76,175,80,0.1);
        }
        
        .upload-icon {
            font-size: 48px;
            margin-bottom: 10px;
        }
        
        .upload-content p {
            margin: 10px 0;
            color: white;
        }
        
        .upload-hint {
            color: rgba(255,255,255,0.7);
            font-size: 0.9em;
        }
        
            background: #4CAF50;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9em;
            transition: background 0.3s ease;
        }
        
            background: #45a049;
        }
        
        .upload-progress {
            margin: 20px 0;
        }
        
        .progress-bar {
            width: 100%;
            height: 20px;
            background: rgba(255,255,255,0.2);
            border-radius: 10px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: #4CAF50;
            width: 0%;
            transition: width 0.3s ease;
        }
        
        .upload-results {
            margin: 20px 0;
            padding: 15px;
            background: rgba(255,255,255,0.9);
            border-radius: 8px;
            color: #333;
        }
        
        .result-summary {
            margin-bottom: 15px;
            padding: 10px;
            background: white;
            border-radius: 4px;
        }
        
        .predictions {
            background: white;
            border-radius: 4px;
            padding: 10px;
        }
        
        .prediction-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }
        
        .prediction-item:last-child {
            border-bottom: none;
        }
        
        .species {
            font-weight: bold;
            color: #2c5530;
        }
        
        .confidence {
            color: #4CAF50;
            font-weight: bold;
        }
        
        @media (max-width: 480px) {
            .info-grid {
                grid-template-columns: 1fr;
            }
        }
        """
    
    def _get_dashboard_javascript(self) -> str:
        """Get JavaScript for dashboard interactivity."""
        return """
        document.addEventListener('DOMContentLoaded', function() {
            initializeUpload();
            
            const navCards = document.querySelectorAll('.nav-card');
            
            navCards.forEach(card => {
                card.addEventListener('click', function(e) {
                    e.preventDefault();
                    const href = this.getAttribute('href');
                    
                    this.style.transform = 'scale(0.95)';
                    setTimeout(() => {
                        window.location.href = href;
                    }, 150);
                });
            });
            
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    console.log('Dashboard status:', data);
                })
                .catch(error => {
                    console.error('Failed to fetch status:', error);
                });
        });
        
        function initializeUpload() {
            const uploadArea = document.getElementById('uploadArea');
            const fileInput = document.getElementById('fileInput');
            const selectButton = document.getElementById('selectFile');
            const progressDiv = document.getElementById('uploadProgress');
            const resultsDiv = document.getElementById('uploadResults');
            
            if (!uploadArea || !fileInput || !selectButton) return;
            
            selectButton.addEventListener('click', () => fileInput.click());
            fileInput.addEventListener('change', handleFileSelect);
            
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('drag-over');
            });
            
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('drag-over');
            });
            
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('drag-over');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    uploadFile(files[0]);
                }
            });
            
            function handleFileSelect(e) {
                const file = e.target.files[0];
                if (file) uploadFile(file);
            }
            
            async function uploadFile(file) {
                const formData = new FormData();
                formData.append('file', file);
                
                progressDiv.style.display = 'block';
                resultsDiv.style.display = 'none';
                
                const progressFill = document.getElementById('progressFill');
                const progressText = document.getElementById('progressText');
                let progress = 0;
                const progressInterval = setInterval(() => {
                    progress += 10;
                    progressFill.style.width = progress + '%';
                    if (progress >= 90) clearInterval(progressInterval);
                }, 100);
                
                try {
                    const response = await fetch('/api/upload', {
                        method: 'POST',
                        headers: {
                            'X-API-Key': getApiKey()
                        },
                        body: formData
                    });
                    
                    clearInterval(progressInterval);
                    progressFill.style.width = '100%';
                    
                    const result = await response.json();
                    
                    if (response.ok && result.success) {
                        displayResults(result);
                    } else {
                        throw new Error(result.detail || 'Upload failed');
                    }
                } catch (error) {
                    clearInterval(progressInterval);
                    alert('アップロードに失敗しました: ' + error.message);
                } finally {
                    setTimeout(() => {
                        progressDiv.style.display = 'none';
                        progressFill.style.width = '0%';
                    }, 1000);
                }
            }
            
            function displayResults(result) {
                const content = document.getElementById('resultsContent');
                content.innerHTML = `
                    <div class="result-summary">
                        <p><strong>ファイル:</strong> ${result.filename}</p>
                        <p><strong>処理時間:</strong> ${result.processing_time.toFixed(2)}秒</p>
                        <p><strong>モデル:</strong> ${result.model_info.model_name || 'iNatAg Swin Transformer'}</p>
                        <p><strong>セッションID:</strong> ${result.session_id}</p>
                    </div>
                    <div class="predictions">
                        <h4>上位予測結果:</h4>
                        ${result.predictions.slice(0, 5).map(pred => `
                            <div class="prediction-item">
                                <span class="species">${pred.species_name}</span>
                                <span class="confidence">${(pred.confidence * 100).toFixed(1)}%</span>
                            </div>
                        `).join('')}
                    </div>
                `;
                resultsDiv.style.display = 'block';
                
                resultsDiv.scrollIntoView({ behavior: 'smooth' });
            }
            
            function getApiKey() {
                return 'demo-api-key';
            }
        }
        """
    
    def run(self, debug: bool = False):
        """Run the web server.
        
        Args:
            debug: Enable debug mode
        """
        try:
            self.logger.info(f"Starting web server at http://{self.host}:{self.port}")
            uvicorn.run(
                self.app,
                host=self.host,
                port=self.port,
                reload=debug,
                log_level="info" if debug else "warning"
            )
        except Exception as e:
            self.logger.error(f"Failed to start web server: {e}")
            raise
    
    def get_app(self):
        """Get FastAPI app instance for testing.
        
        Returns:
            FastAPI application instance
        """
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint for deployment monitoring."""
            try:
                database_status = "disconnected"
                if os.getenv("DATABASE_URL"):
                    try:
                        with next(get_db()) as db:
                            db.execute("SELECT 1")
                        database_status = "connected"
                    except Exception:
                        database_status = "error"
                else:
                    database_status = "file-based"
                
                return {
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat(),
                    "version": "1.0.0",
                    "database": database_status,
                    "model_loaded": self.model_loaded,
                    "model_info": {
                        "primary_model": self.model_manager.primary_model.config.model_name if self.model_manager and self.model_manager.primary_model else None,
                        "fallback_models": len(self.model_manager.fallback_models) if self.model_manager else 0
                    } if self.model_loaded else None
                }
            except Exception as e:
                self.logger.error(f"Health check failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Service unhealthy"
                )

        @self.app.get("/api/sessions")
        async def get_user_sessions(
            user: UserModel = Depends(require_auth),
            db: Session = Depends(get_db)
        ):
            """Get analysis sessions for authenticated user."""
            try:
                db_service = DatabaseService(db)
                sessions = db_service.get_user_sessions(user.id)
                
                return JSONResponse([{
                    "id": session.id,
                    "session_name": session.session_name,
                    "description": session.description,
                    "start_date": session.start_date.isoformat(),
                    "end_date": session.end_date.isoformat() if session.end_date else None,
                    "created_at": session.created_at.isoformat()
                } for session in sessions])
                
            except Exception as e:
                self.logger.error(f"Failed to get user sessions: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/sessions")
        async def create_analysis_session(
            request: Request,
            user: UserModel = Depends(require_auth),
            db: Session = Depends(get_db)
        ):
            """Create new analysis session for authenticated user."""
            try:
                data = await request.json()
                session_name = data.get("session_name")
                description = data.get("description")
                
                if not session_name:
                    raise HTTPException(status_code=400, detail="session_name is required")
                
                db_service = DatabaseService(db)
                session = db_service.create_analysis_session(user.id, session_name, description)
                
                return JSONResponse({
                    "id": session.id,
                    "session_name": session.session_name,
                    "description": session.description,
                    "start_date": session.start_date.isoformat(),
                    "created_at": session.created_at.isoformat()
                }, status_code=201)
                
            except Exception as e:
                self.logger.error(f"Failed to create analysis session: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        return self.app
