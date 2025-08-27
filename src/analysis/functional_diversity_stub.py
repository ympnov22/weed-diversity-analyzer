"""Stub implementation for functional diversity to avoid heavy dependencies."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ..utils.logger import LoggerMixin


@dataclass
class FunctionalTraits:
    """Configuration for functional traits."""
    trait_categories: List[str] = None
    trait_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.trait_categories is None:
            self.trait_categories = ['growth_form', 'leaf_type', 'reproduction']
        if self.trait_weights is None:
            self.trait_weights = {trait: 1.0 for trait in self.trait_categories}


@dataclass
class FunctionalDiversityConfig:
    """Configuration for functional diversity analysis."""
    distance_metric: str = "euclidean"
    standardize_traits: bool = True
    include_phylogeny: bool = False


class FunctionalDiversityAnalyzerStub(LoggerMixin):
    """Stub implementation of functional diversity analyzer for minimal deployment."""
    
    def __init__(self, config: Optional[FunctionalDiversityConfig] = None):
        super().__init__()
        self.config = config or FunctionalDiversityConfig()
        self.logger.info("FunctionalDiversityAnalyzer initialized in stub mode (no heavy dependencies)")
    
    def calculate_functional_diversity(self, 
                                     species_traits: Dict[str, Dict[str, Any]],
                                     abundance_data: Dict[str, int],
                                     **kwargs) -> Dict[str, float]:
        """Stub implementation for functional diversity calculation."""
        self.logger.warning("Functional diversity calculation not available in minimal mode")
        return {
            'functional_richness': 0.0,
            'functional_evenness': 0.0,
            'functional_divergence': 0.0,
            'functional_dispersion': 0.0
        }
    
    def analyze_trait_space(self, 
                          species_traits: Dict[str, Dict[str, Any]],
                          **kwargs) -> Dict[str, Any]:
        """Stub implementation for trait space analysis."""
        self.logger.warning("Trait space analysis not available in minimal mode")
        return {
            'trait_space_volume': 0.0,
            'trait_correlations': {},
            'trait_loadings': {},
            'summary': 'Trait space analysis not available in minimal mode'
        }
    
    def calculate_functional_beta_diversity(self, 
                                          site_data: List[Dict[str, Any]],
                                          **kwargs) -> Dict[str, Any]:
        """Stub implementation for functional beta diversity."""
        self.logger.warning("Functional beta diversity not available in minimal mode")
        return {
            'functional_turnover': 0.0,
            'functional_nestedness': 0.0,
            'functional_beta_total': 0.0,
            'summary': 'Functional beta diversity not available in minimal mode'
        }
