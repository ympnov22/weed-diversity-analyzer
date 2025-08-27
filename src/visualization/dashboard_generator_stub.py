"""Stub implementation for dashboard generator to avoid heavy dependencies."""

from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..utils.logger import LoggerMixin


class DashboardGeneratorStub(LoggerMixin):
    """Stub implementation of dashboard generator for minimal deployment."""
    
    def __init__(self) -> None:
        """Initialize dashboard generator."""
        super().__init__()
        self.logger.info("DashboardGenerator initialized in stub mode (no heavy dependencies)")
    
    def generate_species_distribution_chart(self, 
                                          species_data: Dict[str, Any],
                                          output_path: Optional[Path] = None) -> str:
        """Stub implementation for species distribution chart."""
        self.logger.warning("Species distribution chart generation not available in minimal mode")
        return self._generate_empty_dashboard("種分布チャート")
    
    def generate_model_performance_dashboard(self,
                                           performance_data: Dict[str, Any],
                                           output_path: Optional[Path] = None) -> str:
        """Stub implementation for model performance dashboard."""
        self.logger.warning("Model performance dashboard not available in minimal mode")
        return self._generate_empty_dashboard("モデル性能ダッシュボード")
    
    def generate_soft_voting_analysis(self,
                                    voting_data: Dict[str, Any],
                                    output_path: Optional[Path] = None) -> str:
        """Stub implementation for soft voting analysis."""
        self.logger.warning("Soft voting analysis not available in minimal mode")
        return self._generate_empty_dashboard("ソフト投票分析")
    
    def generate_comparative_analysis_dashboard(self,
                                              analysis_data: Dict[str, Any],
                                              output_path: Optional[Path] = None) -> str:
        """Stub implementation for comparative analysis dashboard."""
        self.logger.warning("Comparative analysis dashboard not available in minimal mode")
        return self._generate_empty_dashboard("比較分析ダッシュボード")
    
    def generate_functional_diversity_dashboard(self,
                                              diversity_data: Dict[str, Any],
                                              output_path: Optional[Path] = None) -> str:
        """Stub implementation for functional diversity dashboard."""
        self.logger.warning("Functional diversity dashboard not available in minimal mode")
        return self._generate_empty_dashboard("機能的多様性ダッシュボード")
    
    def generate_sample_dashboard_data(self) -> Dict[str, Any]:
        """Generate minimal sample dashboard data."""
        return {
            "species_distribution": {
                "species_names": ["Species A", "Species B", "Species C"],
                "counts": [0, 0, 0],
                "percentages": [0.0, 0.0, 0.0]
            },
            "model_performance": {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0
            },
            "metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "analysis_version": "1.0.0-minimal"
            }
        }
    
    def _generate_empty_dashboard(self, title: str = "ダッシュボード") -> str:
        """Generate empty dashboard placeholder."""
        return f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f6f8fa;
        }}
        .dashboard-placeholder {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 40px;
            text-align: center;
            max-width: 1200px;
            margin: 0 auto;
        }}
        .feature-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }}
        .feature-card {{
            background: #f8f9fa;
            border-radius: 6px;
            padding: 20px;
            border-left: 4px solid #0366d6;
        }}
    </style>
</head>
<body>
    <div class="dashboard-placeholder">
        <h1>{title}</h1>
        <p>ダッシュボード機能は最小モードでは利用できません。</p>
        
        <div class="feature-grid">
            <div class="feature-card">
                <h3>種分布分析</h3>
                <p>種の分布パターンと豊富さの可視化</p>
            </div>
            <div class="feature-card">
                <h3>モデル性能</h3>
                <p>機械学習モデルの精度と性能指標</p>
            </div>
            <div class="feature-card">
                <h3>時系列分析</h3>
                <p>多様性指標の時間的変化</p>
            </div>
            <div class="feature-card">
                <h3>比較分析</h3>
                <p>サイト間・期間間の多様性比較</p>
            </div>
        </div>
        
        <p style="margin-top: 30px; color: #666;">
            完全な機能を使用するには、フル版をデプロイしてください。
        </p>
    </div>
</body>
</html>"""
