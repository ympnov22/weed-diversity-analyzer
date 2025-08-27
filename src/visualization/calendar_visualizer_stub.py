"""Stub implementation for calendar visualizer to avoid heavy dependencies."""

from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from ..utils.logger import LoggerMixin


@dataclass
class CalendarConfig:
    """Configuration for calendar visualization."""
    cell_size: int = 12
    cell_spacing: int = 2
    week_spacing: int = 4
    month_labels: bool = True
    color_scheme: str = "green"


class CalendarVisualizerStub(LoggerMixin):
    """Stub implementation of calendar visualizer for minimal deployment."""
    
    def __init__(self, config: Optional[CalendarConfig] = None):
        super().__init__()
        self.config = config or CalendarConfig()
        self.logger.info("CalendarVisualizer initialized in stub mode (no heavy dependencies)")
    
    def generate_calendar_html(self,
                             diversity_data: Dict[str, float],
                             year: int = None,
                             output_path: Optional[Path] = None) -> str:
        """Stub implementation for calendar HTML generation."""
        self.logger.warning("Calendar visualization not available in minimal mode")
        
        if year is None:
            year = datetime.now().year
            
        html_content = self._generate_empty_calendar(year)
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            self.logger.info(f"Generated empty calendar at {output_path}")
        
        return html_content
    
    def generate_sample_data(self, year: int = None) -> Dict[str, float]:
        """Generate minimal sample data for testing."""
        if year is None:
            year = datetime.now().year
        
        sample_data = {}
        start_date = date(year, 1, 1)
        
        for i in range(0, 10):  # Only 10 sample dates
            current_date = start_date + timedelta(days=i * 36)  # Every ~month
            if current_date.year == year:
                date_str = current_date.strftime("%Y-%m-%d")
                sample_data[date_str] = 0.0
        
        return sample_data
    
    def _generate_empty_calendar(self, year: int) -> str:
        """Generate empty calendar placeholder."""
        return f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{year}年 多様性カレンダー</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f6f8fa;
        }}
        .calendar-container {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 40px;
            text-align: center;
            max-width: 1000px;
            margin: 0 auto;
        }}
        .calendar-placeholder {{
            border: 2px dashed #ddd;
            border-radius: 8px;
            padding: 60px 20px;
            margin: 20px 0;
            background: #fafbfc;
        }}
        .legend {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
            margin-top: 20px;
            font-size: 12px;
            color: #666;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .legend-color {{
            width: 12px;
            height: 12px;
            border-radius: 2px;
        }}
    </style>
</head>
<body>
    <div class="calendar-container">
        <h1>{year}年 植生多様性カレンダー</h1>
        <p>GitHub草スタイルの多様性指標可視化</p>
        
        <div class="calendar-placeholder">
            <h3>📅 カレンダー表示</h3>
            <p>カレンダー可視化機能は最小モードでは利用できません。</p>
            <p>完全な機能を使用するには、フル版をデプロイしてください。</p>
        </div>
        
        <div class="legend">
            <span>多様性レベル:</span>
            <div class="legend-item">
                <div class="legend-color" style="background: #ebedf0;"></div>
                <span>低</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #9be9a8;"></div>
                <span>中</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #40c463;"></div>
                <span>高</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #30a14e;"></div>
                <span>非常に高</span>
            </div>
        </div>
    </div>
</body>
</html>"""

CalendarVisualizer = CalendarVisualizerStub
