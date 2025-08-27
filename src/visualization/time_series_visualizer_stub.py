"""Stub implementation for time series visualizer to avoid heavy dependencies."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..utils.logger import LoggerMixin


class TimeSeriesVisualizerStub(LoggerMixin):
    """Stub implementation of time series visualizer for minimal deployment."""
    
    def __init__(self) -> None:
        """Initialize time series visualizer."""
        super().__init__()
        self.logger.info("TimeSeriesVisualizer initialized in stub mode (no heavy dependencies)")
    
    def generate_diversity_trends_chart(
        self, 
        time_series_data: Dict[str, Any], 
        output_path: Optional[Path] = None
    ) -> str:
        """Stub implementation for diversity trends chart generation."""
        self.logger.warning("Diversity trends chart generation not available in minimal mode")
        return self._generate_empty_chart("時系列多様性トレンド")
    
    def generate_confidence_interval_chart(
        self,
        time_series_data: Dict[str, Any],
        metric: str = "shannon_diversity",
        output_path: Optional[Path] = None
    ) -> str:
        """Stub implementation for confidence interval chart generation."""
        self.logger.warning("Confidence interval chart generation not available in minimal mode")
        return self._generate_empty_chart("信頼区間チャート")
    
    def generate_seasonal_analysis_chart(
        self,
        time_series_data: Dict[str, Any],
        output_path: Optional[Path] = None
    ) -> str:
        """Stub implementation for seasonal analysis chart generation."""
        self.logger.warning("Seasonal analysis chart generation not available in minimal mode")
        return self._generate_empty_chart("季節分析チャート")
    
    def generate_sample_time_series_data(
        self, 
        start_date: str = "2025-01-01", 
        num_days: int = 365
    ) -> Dict[str, Any]:
        """Generate minimal sample time series data for testing."""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        dates = []
        
        for i in range(min(num_days, 10)):  # Limit to 10 days to avoid memory issues
            current_date = start + timedelta(days=i)
            dates.append(current_date.strftime("%Y-%m-%d"))
        
        return {
            "period": {
                "start_date": start_date,
                "end_date": (start + timedelta(days=min(num_days, 10)-1)).strftime("%Y-%m-%d"),
                "total_days": min(num_days, 10)
            },
            "diversity_trends": {
                "dates": dates,
                "shannon_diversity": [0.0] * len(dates),
                "species_richness": [0] * len(dates),
                "pielou_evenness": [0.0] * len(dates),
                "simpson_diversity": [0.0] * len(dates)
            },
            "metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "analysis_version": "1.0.0-minimal"
            }
        }
    
    def _generate_empty_chart(self, title: str = "チャート") -> str:
        """Generate empty chart placeholder."""
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
        .placeholder {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 40px;
            text-align: center;
            max-width: 800px;
            margin: 0 auto;
        }}
    </style>
</head>
<body>
    <div class="placeholder">
        <h2>{title}</h2>
        <p>チャート生成機能は最小モードでは利用できません。</p>
        <p>完全な機能を使用するには、フル版をデプロイしてください。</p>
    </div>
</body>
</html>"""
