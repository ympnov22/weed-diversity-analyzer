"""Diversity metrics calculation for species analysis."""

# import numpy as np  # Removed for minimal deployment
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
import math
# from scipy import stats  # Removed for minimal deployment
# from scipy.special import gammaln  # Removed for minimal deployment

from ..utils.logger import LoggerMixin
from ..utils.data_structures import SpeciesPrediction, DiversityMetrics


@dataclass
class DiversityConfig:
    """Configuration for diversity calculations."""
    
    bootstrap_iterations: int = 1000
    confidence_level: float = 0.95
    coverage_target: float = 0.8
    min_confidence_threshold: float = 0.3
    subsampling_size: int = 30
    hill_orders: Optional[List[float]] = None
    
    def __post_init__(self) -> None:
        if self.hill_orders is None:
            self.hill_orders = [0, 1, 2]


class DiversityCalculator(LoggerMixin):
    """Calculate biodiversity metrics from species predictions."""
    
    def __init__(self, config: Optional[DiversityConfig] = None):
        """Initialize diversity calculator.
        
        Args:
            config: Configuration for diversity calculations
        """
        self.config = config or DiversityConfig()
    
    def calculate_basic_metrics(self, species_counts: Dict[str, int]) -> Dict[str, float]:
        """Calculate basic diversity metrics.
        
        Args:
            species_counts: Dictionary mapping species names to counts
            
        Returns:
            Dictionary with basic diversity metrics
        """
        if not species_counts:
            return self._empty_metrics()
        
        counts = list(species_counts.values())
        total = sum(counts)
        
        if total == 0:
            return self._empty_metrics()
        
        richness = len(species_counts)
        
        proportions = [c / total for c in counts]
        shannon = -sum(p * math.log(p + 1e-10) for p in proportions)
        
        if richness > 1:
            max_shannon = math.log(richness)
            pielou = shannon / max_shannon
        else:
            pielou = 1.0
        
        hill_numbers = {}
        if self.config.hill_orders is not None:
            for q in self.config.hill_orders:
                hill_numbers[f'hill_q{q}'] = self._calculate_hill_number(proportions, q)
        
        simpson = 1.0 - sum(p ** 2 for p in proportions)
        
        metrics = {
            'species_richness': float(richness),
            'shannon_diversity': float(shannon),
            'pielou_evenness': float(pielou),
            'simpson_diversity': float(simpson),
            'total_individuals': int(total),
            **hill_numbers
        }
        
        return metrics
    
    def calculate_chao1_estimator(self, species_counts: Dict[str, int]) -> Tuple[float, float]:
        """Calculate Chao1 species richness estimator.
        
        Args:
            species_counts: Dictionary mapping species names to counts
            
        Returns:
            Tuple of (chao1_estimate, chao1_se)
        """
        if not species_counts:
            return 0.0, 0.0
        
        counts = list(species_counts.values())
        observed_richness = len(counts)
        
        f1 = sum(1 for c in counts if c == 1)  # singletons
        f2 = sum(1 for c in counts if c == 2)  # doubletons
        
        if f2 > 0:
            chao1 = observed_richness + (f1 ** 2) / (2 * f2)
        else:
            chao1 = observed_richness + f1 * (f1 - 1) / 2
        
        if f2 > 0:
            var_chao1 = f2 * ((f1 / f2) ** 2 / 2 + (f1 / f2) ** 3 / 4 + (f1 / f2) ** 4 / 4)
        else:
            var_chao1 = f1 * (f1 - 1) / 2 + f1 * (2 * f1 - 1) ** 2 / 4 - f1 ** 4 / (4 * chao1)
        
        chao1_se = math.sqrt(max(0, var_chao1))
        
        return float(chao1), float(chao1_se)
    
    def calculate_coverage_standardized_diversity(
        self, 
        species_counts: Dict[str, int]
    ) -> Dict[str, float]:
        """Calculate coverage-standardized diversity metrics.
        
        Args:
            species_counts: Dictionary mapping species names to counts
            
        Returns:
            Dictionary with coverage-standardized metrics
        """
        if not species_counts:
            return {}
        
        counts = list(species_counts.values())
        total = sum(counts)
        
        f1 = sum(1 for c in counts if c == 1)
        coverage = 1.0 - (f1 / total) if total > 0 else 0.0
        
        target_coverage = self.config.coverage_target
        
        if coverage >= target_coverage:
            standardized_counts = self._interpolate_to_coverage(counts, target_coverage)
        else:
            standardized_counts = self._extrapolate_to_coverage(counts, target_coverage)
        
        standardized_species_counts = {
            f"species_{i}": int(count) 
            for i, count in enumerate(standardized_counts) 
            if count > 0
        }
        
        metrics = self.calculate_basic_metrics(standardized_species_counts)
        metrics['sample_coverage'] = float(coverage)
        metrics['target_coverage'] = float(target_coverage)
        
        return metrics
    
    def bootstrap_confidence_intervals(
        self, 
        species_counts: Dict[str, int]
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate bootstrap confidence intervals for diversity metrics.
        
        Args:
            species_counts: Dictionary mapping species names to counts
            
        Returns:
            Dictionary mapping metric names to (lower_ci, upper_ci) tuples
        """
        if not species_counts:
            return {}
        
        species_list = []
        for species, count in species_counts.items():
            species_list.extend([species] * count)
        
        if len(species_list) == 0:
            return {}
        
        bootstrap_results: Dict[str, List[float]] = {
            'species_richness': [],
            'shannon_diversity': [],
            'pielou_evenness': [],
            'simpson_diversity': []
        }
        
        if self.config.hill_orders is not None:
            for q in self.config.hill_orders:
                bootstrap_results[f'hill_q{q}'] = []
        
        import random
        for _ in range(self.config.bootstrap_iterations):
            bootstrap_sample = [random.choice(species_list) for _ in range(len(species_list))]
            
            bootstrap_counts = Counter(bootstrap_sample)
            
            metrics = self.calculate_basic_metrics(bootstrap_counts)
            
            for metric_name, value in metrics.items():
                if metric_name in bootstrap_results:
                    bootstrap_results[metric_name].append(value)
        
        alpha = 1 - self.config.confidence_level
        confidence_intervals = {}
        
        for metric_name, values in bootstrap_results.items():
            if values:
                values.sort()
                n = len(values)
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                
                lower_idx = int(n * lower_percentile / 100)
                upper_idx = int(n * upper_percentile / 100)
                lower_ci = values[min(lower_idx, n-1)]
                upper_ci = values[min(upper_idx, n-1)]
                
                confidence_intervals[metric_name] = (float(lower_ci), float(upper_ci))
        
        return confidence_intervals
    
    def _calculate_hill_number(self, proportions: list, q: float) -> float:
        """Calculate Hill number of order q.
        
        Args:
            proportions: Array of species proportions
            q: Order of Hill number
            
        Returns:
            Hill number value
        """
        if len(proportions) == 0:
            return 0.0
        
        if q == 0:
            return float(len(proportions))
        elif q == 1:
            shannon = -sum(p * math.log(p + 1e-10) for p in proportions)
            return float(math.exp(shannon))
        else:
            if any(p <= 0 for p in proportions):
                return 0.0
            diversity = sum(p ** q for p in proportions) ** (1 / (1 - q))
            return float(diversity)
    
    def _interpolate_to_coverage(self, counts: list, target_coverage: float) -> list:
        """Interpolate species counts to target coverage."""
        total = sum(counts)
        f1 = sum(1 for c in counts if c == 1)
        current_coverage = 1.0 - (f1 / total) if total > 0 else 0.0
        
        if current_coverage <= target_coverage:
            return counts
        
        scaling_factor = target_coverage / current_coverage
        return [c * scaling_factor for c in counts]
    
    def _extrapolate_to_coverage(self, counts: list, target_coverage: float) -> list:
        """Extrapolate species counts to target coverage."""
        total = sum(counts)
        f1 = sum(1 for c in counts if c == 1)
        current_coverage = 1.0 - (f1 / total) if total > 0 else 0.0
        
        if current_coverage >= target_coverage:
            return counts
        
        scaling_factor = target_coverage / (current_coverage + 1e-10)
        return [c * scaling_factor for c in counts]
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dictionary."""
        metrics = {
            'species_richness': 0.0,
            'shannon_diversity': 0.0,
            'pielou_evenness': 0.0,
            'simpson_diversity': 0.0,
            'total_individuals': 0
        }
        
        if self.config.hill_orders is not None:
            for q in self.config.hill_orders:
                metrics[f'hill_q{q}'] = 0.0
        
        return metrics
