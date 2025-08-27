"""Stub implementation for spatial analysis to avoid heavy dependencies."""

from typing import List, Dict, Any, Optional, Tuple
from ..utils.logger import LoggerMixin
from ..utils.data_structures import DiversityMetrics


class SpatialAnalyzerStub(LoggerMixin):
    """Stub implementation of spatial analyzer for minimal deployment."""
    
    def __init__(self):
        super().__init__()
        self.logger.info("SpatialAnalyzer initialized in stub mode (no heavy dependencies)")
    
    def analyze_spatial_patterns(self, 
                                spatial_data: List[Tuple[Tuple[float, float], DiversityMetrics]],
                                **kwargs) -> Dict[str, Any]:
        """Stub implementation for spatial pattern analysis."""
        self.logger.warning("Spatial pattern analysis not available in minimal mode")
        return {
            'spatial_autocorrelation': 0.0,
            'hotspots': [],
            'coldspots': [],
            'clusters': [],
            'spatial_trend': 'unknown',
            'summary': 'Spatial analysis not available in minimal mode'
        }
    
    def calculate_spatial_diversity(self, 
                                  site_coordinates: List[Tuple[float, float]],
                                  diversity_values: List[float],
                                  **kwargs) -> Dict[str, Any]:
        """Stub implementation for spatial diversity calculation."""
        self.logger.warning("Spatial diversity calculation not available in minimal mode")
        return {
            'gamma_diversity': 0.0,
            'alpha_diversity_mean': 0.0,
            'beta_diversity_spatial': 0.0,
            'distance_decay': 0.0,
            'summary': 'Spatial diversity calculation not available in minimal mode'
        }
    
    def interpolate_diversity_surface(self, 
                                    spatial_data: List[Tuple[Tuple[float, float], float]],
                                    **kwargs) -> Dict[str, Any]:
        """Stub implementation for diversity surface interpolation."""
        self.logger.warning("Diversity surface interpolation not available in minimal mode")
        return {
            'interpolated_grid': [],
            'interpolation_method': 'none',
            'cross_validation_score': 0.0,
            'summary': 'Surface interpolation not available in minimal mode'
        }
