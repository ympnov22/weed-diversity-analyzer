"""GitHub grass-style calendar visualization for diversity metrics."""

import json
import numpy as np
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from ..utils.logger import LoggerMixin
from ..utils.config import ConfigManager


@dataclass
class CalendarConfig:
    """Configuration for calendar visualization."""
    cell_size: int = 12
    cell_spacing: int = 2
    week_spacing: int = 4
    month_labels: bool = True
    day_labels: bool = True
    color_scheme: str = "green"
    intensity_levels: int = 5
    tooltip_enabled: bool = True
    responsive: bool = True


class CalendarVisualizer(LoggerMixin):
    """Generate GitHub grass-style calendar visualization for diversity metrics."""
    
    def __init__(self, config: Optional[CalendarConfig] = None):
        """Initialize calendar visualizer.
        
        Args:
            config: Calendar visualization configuration
        """
        self.config = config or CalendarConfig()
        self.color_schemes = {
            "green": ["#ebedf0", "#9be9a8", "#40c463", "#30a14e", "#216e39"],
            "blue": ["#ebedf0", "#9ecbff", "#0969da", "#0550ae", "#033d8b"],
            "purple": ["#ebedf0", "#d0a9f5", "#8b5cf6", "#7c3aed", "#5b21b6"]
        }
    
    def generate_calendar_html(
        self,
        calendar_data: Dict[str, Any],
        output_path: Optional[Path] = None,
        year: Optional[int] = None
    ) -> str:
        """Generate HTML for GitHub-style calendar visualization.
        
        Args:
            calendar_data: Calendar data from JSON exporter
            output_path: Path to save HTML file
            year: Year to display (defaults to current year)
            
        Returns:
            HTML string for calendar visualization
        """
        try:
            year = year or datetime.now().year
            colors = self.color_schemes.get(self.config.color_scheme, self.color_schemes["green"])
            
            data_by_date = {entry["date"]: entry for entry in calendar_data.get("data", [])}
            metric_name = calendar_data.get("metric", "shannon_diversity")
            metric_description = calendar_data.get("metadata", {}).get("metric_description", "Diversity metric")
            
            calendar_svg = self._generate_calendar_svg(data_by_date, year, colors)
            
            html_content = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>畑植生多様性カレンダー - {year}</title>
    <style>
        {self._get_calendar_css()}
    </style>
</head>
<body>
    <div class="calendar-container">
        <header class="calendar-header">
            <h1>畑植生多様性カレンダー</h1>
            <p class="metric-description">{metric_description}</p>
            <div class="year-selector">
                <button onclick="changeYear({year - 1})">&lt; {year - 1}</button>
                <span class="current-year">{year}</span>
                <button onclick="changeYear({year + 1})">{year + 1} &gt;</button>
            </div>
        </header>
        
        <div class="calendar-wrapper">
            {calendar_svg}
        </div>
        
        <div class="legend">
            <span class="legend-label">少ない</span>
            <div class="legend-colors">
                {''.join(f'<div class="legend-color" style="background-color: {color}"></div>' for color in colors)}
            </div>
            <span class="legend-label">多い</span>
        </div>
        
        <div id="tooltip" class="tooltip"></div>
    </div>
    
    <script>
        {self._get_calendar_javascript(data_by_date, metric_name)}
    </script>
</body>
</html>"""
            
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                self.logger.info(f"Generated calendar HTML at {output_path}")
            
            return html_content
            
        except Exception as e:
            self.logger.error(f"Failed to generate calendar HTML: {e}")
            raise
    
    def _generate_calendar_svg(
        self, 
        data_by_date: Dict[str, Dict[str, Any]], 
        year: int, 
        colors: List[str]
    ) -> str:
        """Generate SVG for calendar grid."""
        start_date = date(year, 1, 1)
        end_date = date(year, 12, 31)
        
        days_to_sunday = start_date.weekday() + 1 if start_date.weekday() != 6 else 0
        calendar_start = start_date - timedelta(days=days_to_sunday)
        
        total_days = (end_date - calendar_start).days + 1
        total_weeks = (total_days + 6) // 7
        
        cell_size = self.config.cell_size
        cell_spacing = self.config.cell_spacing
        week_spacing = self.config.week_spacing
        
        svg_width = total_weeks * (cell_size + cell_spacing) + week_spacing * 12
        svg_height = 7 * (cell_size + cell_spacing) + 50
        
        svg_content = f'<svg width="{svg_width}" height="{svg_height}" class="calendar-svg">\n'
        
        if self.config.month_labels:
            svg_content += self._generate_month_labels(year, cell_size, cell_spacing)
        
        if self.config.day_labels:
            svg_content += self._generate_day_labels(cell_size, cell_spacing)
        
        current_date = calendar_start
        week = 0
        
        while current_date <= end_date + timedelta(days=6):
            day_of_week = current_date.weekday()
            if day_of_week == 6:
                day_of_week = 0
            else:
                day_of_week += 1
            
            x = week * (cell_size + cell_spacing) + 20
            y = day_of_week * (cell_size + cell_spacing) + 30
            
            date_str = current_date.strftime("%Y-%m-%d")
            day_data = data_by_date.get(date_str, {})
            
            level = day_data.get("level", 0)
            color = colors[min(level, len(colors) - 1)]
            
            opacity = 1.0 if current_date.year == year else 0.3
            
            svg_content += f'''
    <rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" 
          fill="{color}" opacity="{opacity}" 
          data-date="{date_str}" 
          data-value="{day_data.get('value', 0):.3f}"
          data-species="{day_data.get('species_count', 0)}"
          data-images="{day_data.get('total_images', 0)}"
          class="calendar-cell" />'''
            
            current_date += timedelta(days=1)
            if current_date.weekday() == 0:
                week += 1
        
        svg_content += '</svg>'
        return svg_content
    
    def _generate_month_labels(self, year: int, cell_size: int, cell_spacing: int) -> str:
        """Generate month labels for calendar."""
        months = ["1月", "2月", "3月", "4月", "5月", "6月", 
                 "7月", "8月", "9月", "10月", "11月", "12月"]
        
        labels = ""
        for month in range(1, 13):
            first_day = date(year, month, 1)
            days_from_start = (first_day - date(year, 1, 1)).days
            week_offset = days_from_start // 7
            
            x = week_offset * (cell_size + cell_spacing) + 20
            labels += f'<text x="{x}" y="15" class="month-label">{months[month-1]}</text>\n'
        
        return labels
    
    def _generate_day_labels(self, cell_size: int, cell_spacing: int) -> str:
        """Generate day of week labels."""
        days = ["月", "火", "水", "木", "金", "土", "日"]
        labels = ""
        
        for i, day in enumerate(days):
            y = i * (cell_size + cell_spacing) + 30 + cell_size // 2
            labels += f'<text x="10" y="{y}" class="day-label">{day}</text>\n'
        
        return labels
    
    def _get_calendar_css(self) -> str:
        """Get CSS styles for calendar."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f6f8fa;
            color: #24292e;
            line-height: 1.5;
        }
        
        .calendar-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .calendar-header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .calendar-header h1 {
            font-size: 2rem;
            margin-bottom: 10px;
            color: #24292e;
        }
        
        .metric-description {
            color: #586069;
            margin-bottom: 20px;
        }
        
        .year-selector {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
        }
        
        .year-selector button {
            background: #f6f8fa;
            border: 1px solid #d0d7de;
            border-radius: 6px;
            padding: 8px 12px;
            cursor: pointer;
            font-size: 14px;
        }
        
        .year-selector button:hover {
            background: #f3f4f6;
        }
        
        .current-year {
            font-size: 1.2rem;
            font-weight: 600;
        }
        
        .calendar-wrapper {
            display: flex;
            justify-content: center;
            margin: 30px 0;
            overflow-x: auto;
        }
        
        .calendar-svg {
            border: 1px solid #d0d7de;
            border-radius: 6px;
            background: white;
            padding: 10px;
        }
        
        .calendar-cell {
            cursor: pointer;
            stroke: rgba(27, 31, 35, 0.06);
            stroke-width: 1px;
        }
        
        .calendar-cell:hover {
            stroke: #1f2328;
            stroke-width: 2px;
        }
        
        .month-label, .day-label {
            font-size: 12px;
            fill: #656d76;
            text-anchor: middle;
            dominant-baseline: middle;
        }
        
        .legend {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            margin-top: 20px;
        }
        
        .legend-label {
            font-size: 12px;
            color: #656d76;
        }
        
        .legend-colors {
            display: flex;
            gap: 2px;
        }
        
        .legend-color {
            width: 12px;
            height: 12px;
            border-radius: 2px;
        }
        
        .tooltip {
            position: absolute;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
            z-index: 1000;
        }
        
        .tooltip.visible {
            opacity: 1;
        }
        
        @media (max-width: 768px) {
            .calendar-container {
                padding: 10px;
            }
            
            .calendar-header h1 {
                font-size: 1.5rem;
            }
            
            .calendar-wrapper {
                margin: 20px 0;
            }
        }
        """
    
    def _get_calendar_javascript(self, data_by_date: Dict[str, Any], metric_name: str) -> str:
        """Get JavaScript for calendar interactivity."""
        return f"""
        const tooltip = document.getElementById('tooltip');
        const cells = document.querySelectorAll('.calendar-cell');
        
        cells.forEach(cell => {{
            cell.addEventListener('mouseenter', (e) => {{
                const rect = e.target;
                const date = rect.getAttribute('data-date');
                const value = parseFloat(rect.getAttribute('data-value'));
                const species = rect.getAttribute('data-species');
                const images = rect.getAttribute('data-images');
                
                const dateObj = new Date(date);
                const formattedDate = dateObj.toLocaleDateString('ja-JP', {{
                    year: 'numeric',
                    month: 'long',
                    day: 'numeric'
                }});
                
                tooltip.innerHTML = `
                    <div><strong>${{formattedDate}}</strong></div>
                    <div>{metric_name}: ${{value.toFixed(3)}}</div>
                    <div>種数: ${{species}}</div>
                    <div>画像数: ${{images}}</div>
                `;
                
                tooltip.classList.add('visible');
            }});
            
            cell.addEventListener('mousemove', (e) => {{
                tooltip.style.left = e.pageX + 10 + 'px';
                tooltip.style.top = e.pageY - 10 + 'px';
            }});
            
            cell.addEventListener('mouseleave', () => {{
                tooltip.classList.remove('visible');
            }});
            
            cell.addEventListener('click', (e) => {{
                const date = e.target.getAttribute('data-date');
                showDayDetails(date);
            }});
        }});
        
        function showDayDetails(date) {{
            console.log('Show details for date:', date);
            alert(`詳細表示: ${{date}}\\n\\n実装予定: 日次詳細データの表示`);
        }}
        
        function changeYear(year) {{
            console.log('Change to year:', year);
            alert(`年変更: ${{year}}\\n\\n実装予定: 年次データの動的読み込み`);
        }}
        """
    
    def generate_sample_data(self, year: Optional[int] = None, num_days: int = 365) -> Dict[str, Any]:
        """Generate sample calendar data for testing.
        
        Args:
            year: Year to generate data for
            num_days: Number of days to generate
            
        Returns:
            Sample calendar data dictionary
        """
        year = year or datetime.now().year
        start_date = date(year, 1, 1)
        
        data = []
        for i in range(num_days):
            current_date = start_date + timedelta(days=i)
            
            day_of_year = current_date.timetuple().tm_yday
            seasonal_factor = 0.5 + 0.5 * np.sin(2 * np.pi * day_of_year / 365)
            
            shannon_diversity = np.random.normal(1.5, 0.3) * seasonal_factor
            shannon_diversity = max(0, min(3.0, shannon_diversity))
            
            if shannon_diversity < 0.5:
                level = 1
            elif shannon_diversity < 1.0:
                level = 2
            elif shannon_diversity < 2.0:
                level = 3
            else:
                level = 4
            
            data.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "value": shannon_diversity,
                "level": level,
                "species_count": int(np.random.poisson(5) * seasonal_factor) + 1,
                "total_images": np.random.randint(5, 25)
            })
        
        return {
            "metric": "shannon_diversity",
            "data": data,
            "scale": {
                "min": 0.0,
                "max": 3.0,
                "q25": 0.75,
                "q50": 1.5,
                "q75": 2.25
            },
            "metadata": {
                "total_days": len(data),
                "metric_description": "Shannon diversity index (H') - measures species diversity",
                "export_timestamp": datetime.now().isoformat()
            }
        }
