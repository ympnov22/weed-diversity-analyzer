"""FastAPI web server for visualization interface."""

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import uvicorn
from datetime import datetime

from ..utils.logger import LoggerMixin
from ..output.output_manager import OutputManager
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
        
        self.calendar_viz = CalendarVisualizer()
        self.time_series_viz = TimeSeriesVisualizer()
        self.dashboard_gen = DashboardGenerator()
        
        self.static_dir = Path(__file__).parent / "static"
        self.static_dir.mkdir(exist_ok=True)
        
        self._setup_routes()
        self._setup_static_files()
    
    def _setup_routes(self):
        """Setup API routes for visualization data."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def root():
            """Main dashboard page."""
            return self._get_main_dashboard_html()
        
        @self.app.get("/api/calendar/{metric}")
        async def get_calendar_data(metric: str):
            """Get calendar data for specified metric."""
            try:
                json_dir = self.output_manager.json_exporter.output_dir
                calendar_file = json_dir / f"github_calendar_{metric}.json"
                
                if not calendar_file.exists():
                    sample_data = self.calendar_viz.generate_sample_data()
                    return JSONResponse(sample_data)
                
                with open(calendar_file, 'r', encoding='utf-8') as f:
                    calendar_data = json.load(f)
                
                return JSONResponse(calendar_data)
                
            except Exception as e:
                self.logger.error(f"Failed to get calendar data for {metric}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/time-series")
        async def get_time_series_data(start_date: Optional[str] = None, end_date: Optional[str] = None):
            """Get time series data for specified date range."""
            try:
                json_dir = self.output_manager.json_exporter.output_dir
                
                if start_date and end_date:
                    time_series_file = json_dir / f"time_series_{start_date}_to_{end_date}.json"
                else:
                    time_series_files = list(json_dir.glob("time_series_*.json"))
                    if time_series_files:
                        time_series_file = time_series_files[0]
                    else:
                        time_series_file = None
                
                if not time_series_file or not time_series_file.exists():
                    sample_data = self.time_series_viz.generate_sample_time_series_data()
                    return JSONResponse(sample_data)
                
                with open(time_series_file, 'r', encoding='utf-8') as f:
                    time_series_data = json.load(f)
                
                return JSONResponse(time_series_data)
                
            except Exception as e:
                self.logger.error(f"Failed to get time series data: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/dashboard/species")
        async def get_species_dashboard_data():
            """Get species distribution dashboard data."""
            try:
                json_dir = self.output_manager.json_exporter.output_dir
                daily_files = list(json_dir.glob("daily_summary_*.json"))
                
                if not daily_files:
                    sample_data = self.dashboard_gen.generate_sample_dashboard_data()
                    return JSONResponse(sample_data["daily_summaries"])
                
                daily_summaries = []
                for file_path in sorted(daily_files):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        daily_data = json.load(f)
                        daily_summaries.append(daily_data)
                
                return JSONResponse(daily_summaries)
                
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
        async def get_status():
            """Get server status and data summary."""
            try:
                output_summary = self.output_manager.get_output_summary()
                
                status = {
                    "server": "running",
                    "timestamp": datetime.now().isoformat(),
                    "version": "1.0.0",
                    "inatag_species_count": 2959,
                    "output_summary": output_summary
                }
                
                return JSONResponse(status)
                
            except Exception as e:
                self.logger.error(f"Failed to get status: {e}")
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
        return self.app
