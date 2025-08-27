"""Stub implementation for temporal analysis to avoid heavy dependencies."""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from ..utils.logger import LoggerMixin
from ..utils.data_structures import DiversityMetrics


class TemporalAnalyzerStub(LoggerMixin):
    """Stub implementation of temporal analyzer for minimal deployment."""
    
    def __init__(self):
        super().__init__()
        self.logger.info("TemporalAnalyzer initialized in stub mode (no heavy dependencies)")
    
    def analyze_temporal_trends(self, 
                              temporal_data: List[Tuple[datetime, DiversityMetrics]],
                              **kwargs) -> Dict[str, Any]:
        """Stub implementation for temporal trend analysis."""
        self.logger.warning("Temporal trend analysis not available in minimal mode")
        return {
            'trend': 'unknown',
            'slope': 0.0,
            'r_squared': 0.0,
            'p_value': None,
            'seasonal_patterns': [],
            'change_points': [],
            'summary': 'Temporal analysis not available in minimal mode'
        }
    
    def detect_seasonal_patterns(self, 
                               temporal_data: List[Tuple[datetime, float]],
                               **kwargs) -> Dict[str, Any]:
        """Stub implementation for seasonal pattern detection."""
        self.logger.warning("Seasonal pattern detection not available in minimal mode")
        return {
            'has_seasonality': False,
            'seasonal_strength': 0.0,
            'peak_months': [],
            'low_months': [],
            'summary': 'Seasonal analysis not available in minimal mode'
        }
    
    def forecast_diversity(self, 
                         temporal_data: List[Tuple[datetime, float]],
                         forecast_periods: int = 12,
                         **kwargs) -> Dict[str, Any]:
        """Stub implementation for diversity forecasting."""
        self.logger.warning("Diversity forecasting not available in minimal mode")
        return {
            'forecast_values': [0.0] * forecast_periods,
            'confidence_intervals': [(0.0, 0.0)] * forecast_periods,
            'model_type': 'none',
            'accuracy_metrics': {},
            'summary': 'Forecasting not available in minimal mode'
        }
