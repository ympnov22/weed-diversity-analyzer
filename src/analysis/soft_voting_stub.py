"""Stub implementation for soft voting system to avoid heavy dependencies."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ..utils.logger import LoggerMixin


@dataclass
class SoftVotingConfig:
    """Configuration for soft voting system."""
    confidence_threshold: float = 0.5
    min_votes: int = 1
    use_weighted_voting: bool = True


@dataclass
class TaxonomicRollup:
    """Configuration for taxonomic rollup."""
    enable_family_rollup: bool = True
    enable_genus_rollup: bool = True
    confidence_threshold: float = 0.3


class SoftVotingSystemStub(LoggerMixin):
    """Stub implementation of soft voting system for minimal deployment."""
    
    def __init__(self, config: Optional[SoftVotingConfig] = None):
        super().__init__()
        self.config = config or SoftVotingConfig()
        self.logger.info("SoftVotingSystem initialized in stub mode (no heavy dependencies)")
    
    def aggregate_predictions(self, 
                            predictions: List[Dict[str, Any]],
                            **kwargs) -> Dict[str, Any]:
        """Stub implementation for prediction aggregation."""
        self.logger.warning("Prediction aggregation not available in minimal mode")
        
        if not predictions:
            return {'species': 'Unknown', 'confidence': 0.0, 'votes': 0}
        
        return predictions[0] if predictions else {'species': 'Unknown', 'confidence': 0.0, 'votes': 0}
    
    def apply_taxonomic_rollup(self, 
                             predictions: List[Dict[str, Any]],
                             rollup_config: Optional[TaxonomicRollup] = None) -> List[Dict[str, Any]]:
        """Stub implementation for taxonomic rollup."""
        self.logger.warning("Taxonomic rollup not available in minimal mode")
        return predictions
