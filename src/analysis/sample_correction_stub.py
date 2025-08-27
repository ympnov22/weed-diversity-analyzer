"""Stub implementation for sample correction to avoid heavy dependencies."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ..utils.logger import LoggerMixin


@dataclass
class SamplingConfig:
    """Configuration for sampling correction."""
    correction_method: str = "rarefaction"
    target_sample_size: Optional[int] = None
    bootstrap_iterations: int = 1000
    confidence_level: float = 0.95


class SampleCorrectionStub(LoggerMixin):
    """Stub implementation of sample correction for minimal deployment."""
    
    def __init__(self, config: Optional[SamplingConfig] = None):
        super().__init__()
        self.config = config or SamplingConfig()
        self.logger.info("SampleCorrection initialized in stub mode (no heavy dependencies)")
    
    def apply_rarefaction(self, 
                         abundance_data: Dict[str, int],
                         target_size: Optional[int] = None) -> Dict[str, int]:
        """Stub implementation for rarefaction."""
        self.logger.warning("Rarefaction not available in minimal mode")
        return abundance_data
    
    def bootstrap_diversity(self, 
                          abundance_data: Dict[str, int],
                          iterations: int = 1000) -> Dict[str, Any]:
        """Stub implementation for bootstrap diversity estimation."""
        self.logger.warning("Bootstrap diversity estimation not available in minimal mode")
        return {
            'mean_diversity': 0.0,
            'confidence_interval': (0.0, 0.0),
            'standard_error': 0.0
        }
    
    def correct_sampling_bias(self, 
                            diversity_metrics: Dict[str, float],
                            sample_sizes: List[int]) -> Dict[str, float]:
        """Stub implementation for sampling bias correction."""
        self.logger.warning("Sampling bias correction not available in minimal mode")
        return diversity_metrics
