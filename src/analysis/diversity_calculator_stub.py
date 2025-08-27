"""Stub implementation for diversity calculator to avoid heavy dependencies."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ..utils.logger import LoggerMixin
from ..utils.data_structures import DiversityMetrics, SpeciesPrediction


@dataclass
class DiversityConfig:
    """Configuration for diversity calculations."""
    include_rare_species: bool = True
    rare_species_threshold: int = 2
    bootstrap_iterations: int = 1000
    confidence_level: float = 0.95


class DiversityCalculatorStub(LoggerMixin):
    """Stub implementation of diversity calculator for minimal deployment."""
    
    def __init__(self):
        super().__init__()
        self.logger.info("DiversityCalculator initialized in stub mode (no heavy dependencies)")
    
    def calculate_diversity_metrics(self, 
                                  species_predictions: List[SpeciesPrediction],
                                  **kwargs) -> DiversityMetrics:
        """Stub implementation for diversity metrics calculation."""
        self.logger.warning("Diversity metrics calculation not available in minimal mode")
        
        species_count = len(set(pred.species_name for pred in species_predictions))
        total_individuals = len(species_predictions)
        
        return DiversityMetrics(
            species_richness=species_count,
            shannon_diversity=0.0,
            simpson_diversity=0.0,
            evenness=0.0,
            total_individuals=total_individuals,
            dominant_species=species_predictions[0].species_name if species_predictions else "Unknown",
            rare_species=[],
            diversity_index_interpretation="Calculation not available in minimal mode"
        )
    
    def calculate_alpha_diversity(self, abundance_data: Dict[str, int]) -> Dict[str, float]:
        """Stub implementation for alpha diversity calculation."""
        self.logger.warning("Alpha diversity calculation not available in minimal mode")
        return {
            'species_richness': len(abundance_data),
            'shannon_diversity': 0.0,
            'simpson_diversity': 0.0,
            'evenness': 0.0
        }
    
    def calculate_beta_diversity(self, 
                               site_data: List[Dict[str, int]],
                               **kwargs) -> Dict[str, Any]:
        """Stub implementation for beta diversity calculation."""
        self.logger.warning("Beta diversity calculation not available in minimal mode")
        return {
            'bray_curtis_dissimilarity': 0.0,
            'jaccard_similarity': 0.0,
            'sorensen_similarity': 0.0,
            'whittaker_beta': 0.0
        }
