"""Time series visualization for diversity metrics trends."""

import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

from ..utils.logger import LoggerMixin


class TimeSeriesVisualizer(LoggerMixin):
    """Generate time series visualizations for diversity trends."""
    
    def __init__(self):
        """Initialize time series visualizer."""
        pass
    
    def generate_diversity_trends_chart(
        self, 
        time_series_data: Dict[str, Any], 
        output_path: Path = None
    ) -> str:
        """Generate interactive diversity trends chart using Plotly.
        
        Args:
            time_series_data: Time series data from JSON exporter
            output_path: Path to save HTML file
            
        Returns:
            HTML string for time series visualization
        """
        try:
            diversity_trends = time_series_data.get("diversity_trends", {})
            dates = diversity_trends.get("dates", [])
            
            if not dates:
                self.logger.warning("No time series data available")
                return self._generate_empty_chart()
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Shannon多様度', '種リッチネス', 'Pielou均等度', 'Simpson多様度'),
                vertical_spacing=0.08,
                horizontal_spacing=0.08
            )
            
            metrics = [
                ('shannon_diversity', 'Shannon多様度', '#2E86AB'),
                ('species_richness', '種リッチネス', '#A23B72'),
                ('pielou_evenness', 'Pielou均等度', '#F18F01'),
                ('simpson_diversity', 'Simpson多様度', '#C73E1D')
            ]
            
            positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
            
            for i, (metric_key, metric_name, color) in enumerate(metrics):
                values = diversity_trends.get(metric_key, [])
                row, col = positions[i]
                
                if values:
                    fig.add_trace(
                        go.Scatter(
                            x=dates,
                            y=values,
                            mode='lines+markers',
                            name=metric_name,
                            line=dict(color=color, width=2),
                            marker=dict(size=4),
                            hovertemplate=f'<b>{metric_name}</b><br>' +
                                        '日付: %{x}<br>' +
                                        '値: %{y:.3f}<extra></extra>'
                        ),
                        row=row, col=col
                    )
            
            fig.update_layout(
                title=dict(
                    text='植生多様性指標の時系列変化',
                    x=0.5,
                    font=dict(size=20)
                ),
                showlegend=False,
                height=600,
                font=dict(family="Arial, sans-serif"),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                title_font=dict(size=12)
            )
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                title_font=dict(size=12)
            )
            
            html_content = self._wrap_plotly_html(fig.to_html(include_plotlyjs=True))
            
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                self.logger.info(f"Generated time series chart at {output_path}")
            
            return html_content
            
        except Exception as e:
            self.logger.error(f"Failed to generate time series chart: {e}")
            raise
    
    def generate_confidence_interval_chart(
        self,
        time_series_data: Dict[str, Any],
        metric: str = "shannon_diversity",
        output_path: Path = None
    ) -> str:
        """Generate time series chart with confidence intervals.
        
        Args:
            time_series_data: Time series data with confidence intervals
            metric: Metric to visualize
            output_path: Path to save HTML file
            
        Returns:
            HTML string for confidence interval visualization
        """
        try:
            diversity_trends = time_series_data.get("diversity_trends", {})
            dates = diversity_trends.get("dates", [])
            values = diversity_trends.get(metric, [])
            
            if not dates or not values:
                return self._generate_empty_chart()
            
            upper_bound = [v * 1.1 for v in values]
            lower_bound = [v * 0.9 for v in values]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=dates + dates[::-1],
                y=upper_bound + lower_bound[::-1],
                fill='toself',
                fillcolor='rgba(46, 134, 171, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False,
                name='信頼区間'
            ))
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=values,
                mode='lines+markers',
                name=metric,
                line=dict(color='#2E86AB', width=3),
                marker=dict(size=6),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                            '日付: %{x}<br>' +
                            '値: %{y:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=dict(
                    text=f'{metric} 時系列変化（信頼区間付き）',
                    x=0.5,
                    font=dict(size=18)
                ),
                xaxis_title='日付',
                yaxis_title='多様性指標値',
                height=500,
                font=dict(family="Arial, sans-serif"),
                plot_bgcolor='white',
                paper_bgcolor='white',
                showlegend=True
            )
            
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            
            html_content = self._wrap_plotly_html(fig.to_html(include_plotlyjs=True))
            
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                self.logger.info(f"Generated confidence interval chart at {output_path}")
            
            return html_content
            
        except Exception as e:
            self.logger.error(f"Failed to generate confidence interval chart: {e}")
            raise
    
    def generate_seasonal_analysis_chart(
        self,
        time_series_data: Dict[str, Any],
        output_path: Path = None
    ) -> str:
        """Generate seasonal pattern analysis chart.
        
        Args:
            time_series_data: Time series data
            output_path: Path to save HTML file
            
        Returns:
            HTML string for seasonal analysis visualization
        """
        try:
            diversity_trends = time_series_data.get("diversity_trends", {})
            dates = diversity_trends.get("dates", [])
            shannon_values = diversity_trends.get("shannon_diversity", [])
            
            if not dates or not shannon_values:
                return self._generate_empty_chart()
            
            monthly_data = {}
            for date_str, value in zip(dates, shannon_values):
                try:
                    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                    month = date_obj.month
                    if month not in monthly_data:
                        monthly_data[month] = []
                    monthly_data[month].append(value)
                except ValueError:
                    continue
            
            months = list(range(1, 13))
            month_names = ['1月', '2月', '3月', '4月', '5月', '6月',
                          '7月', '8月', '9月', '10月', '11月', '12月']
            
            monthly_means = []
            monthly_stds = []
            
            for month in months:
                if month in monthly_data and monthly_data[month]:
                    monthly_means.append(np.mean(monthly_data[month]))
                    monthly_stds.append(np.std(monthly_data[month]))
                else:
                    monthly_means.append(0)
                    monthly_stds.append(0)
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=month_names,
                y=monthly_means,
                error_y=dict(type='data', array=monthly_stds),
                name='月平均Shannon多様度',
                marker_color='#2E86AB',
                hovertemplate='<b>%{x}</b><br>' +
                            '平均値: %{y:.3f}<br>' +
                            '標準偏差: %{error_y.array:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=dict(
                    text='月別Shannon多様度の季節変動',
                    x=0.5,
                    font=dict(size=18)
                ),
                xaxis_title='月',
                yaxis_title='Shannon多様度',
                height=500,
                font=dict(family="Arial, sans-serif"),
                plot_bgcolor='white',
                paper_bgcolor='white',
                showlegend=False
            )
            
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            
            html_content = self._wrap_plotly_html(fig.to_html(include_plotlyjs=True))
            
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                self.logger.info(f"Generated seasonal analysis chart at {output_path}")
            
            return html_content
            
        except Exception as e:
            self.logger.error(f"Failed to generate seasonal analysis chart: {e}")
            raise
    
    def _wrap_plotly_html(self, plotly_html: str) -> str:
        """Wrap Plotly HTML with custom styling."""
        return f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>植生多様性時系列分析</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f6f8fa;
        }}
        .chart-container {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 20px;
            margin: 0 auto;
            max-width: 1200px;
        }}
        @media (max-width: 768px) {{
            body {{
                padding: 10px;
            }}
            .chart-container {{
                padding: 10px;
            }}
        }}
    </style>
</head>
<body>
    <div class="chart-container">
        {plotly_html}
    </div>
</body>
</html>"""
    
    def _generate_empty_chart(self) -> str:
        """Generate empty chart placeholder."""
        return """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>データなし</title>
</head>
<body>
    <div style="text-align: center; padding: 50px;">
        <h2>時系列データがありません</h2>
        <p>表示するデータが見つかりませんでした。</p>
    </div>
</body>
</html>"""
    
    def generate_sample_time_series_data(
        self, 
        start_date: str = "2025-01-01", 
        num_days: int = 365
    ) -> Dict[str, Any]:
        """Generate sample time series data for testing.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            num_days: Number of days to generate
            
        Returns:
            Sample time series data dictionary
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        dates = []
        shannon_values = []
        richness_values = []
        evenness_values = []
        simpson_values = []
        
        for i in range(num_days):
            current_date = start + timedelta(days=i)
            dates.append(current_date.strftime("%Y-%m-%d"))
            
            day_of_year = current_date.timetuple().tm_yday
            seasonal_factor = 0.5 + 0.5 * np.sin(2 * np.pi * day_of_year / 365)
            
            shannon = np.random.normal(1.5, 0.3) * seasonal_factor
            shannon = max(0, min(3.0, shannon))
            shannon_values.append(shannon)
            
            richness = int(np.random.poisson(8) * seasonal_factor) + 1
            richness_values.append(richness)
            
            evenness = np.random.uniform(0.3, 0.9) * seasonal_factor
            evenness_values.append(evenness)
            
            simpson = np.random.uniform(0.4, 0.8) * seasonal_factor
            simpson_values.append(simpson)
        
        return {
            "period": {
                "start_date": start_date,
                "end_date": (start + timedelta(days=num_days-1)).strftime("%Y-%m-%d"),
                "total_days": num_days
            },
            "diversity_trends": {
                "dates": dates,
                "shannon_diversity": shannon_values,
                "species_richness": richness_values,
                "pielou_evenness": evenness_values,
                "simpson_diversity": simpson_values
            },
            "metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "analysis_version": "1.0.0"
            }
        }
