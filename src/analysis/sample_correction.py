"""Sample size correction and subsampling methods."""

# import numpy as np  # Removed for minimal deployment
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import random
from dataclasses import dataclass

from ..utils.logger import LoggerMixin
from .diversity_calculator import DiversityCalculator, DiversityConfig


@dataclass
class SamplingConfig:
    """Configuration for sampling correction methods."""
    
    target_sample_size: int = 30
    bootstrap_iterations: int = 1000
    confidence_level: float = 0.95
    min_sample_size: int = 5
    rarefaction_steps: int = 20


class SampleCorrection(LoggerMixin):
    """Handle sample size correction and subsampling."""
    
    def __init__(self, config: Optional[SamplingConfig] = None):
        """Initialize sample correction system.
        
        Args:
            config: Configuration for sampling methods
        """
        self.config = config or SamplingConfig()
        self.diversity_calc = DiversityCalculator()
    
    def subsample_to_fixed_size(
        self, 
        species_data: List[str], 
        target_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Subsample species data to fixed size with bootstrap confidence intervals.
        
        Args:
            species_data: List of species names (one per individual/observation)
            target_size: Target sample size (uses config default if None)
            
        Returns:
            Dictionary with subsampling results and confidence intervals
        """
        if target_size is None:
            target_size = self.config.target_sample_size
        
        if len(species_data) < self.config.min_sample_size:
            self.logger.warning(f"Sample size {len(species_data)} below minimum {self.config.min_sample_size}")
            return self._empty_subsampling_result()
        
        if len(species_data) <= target_size:
            species_counts = Counter(species_data)
            metrics = self.diversity_calc.calculate_basic_metrics(species_counts)
            
            return {
                'subsampled_metrics': metrics,
                'confidence_intervals': {},
                'original_sample_size': len(species_data),
                'target_sample_size': target_size,
                'subsampling_performed': False,
                'bootstrap_iterations': 0
            }
        
        bootstrap_results: Dict[str, List[float]] = {
            'species_richness': [],
            'shannon_diversity': [],
            'pielou_evenness': [],
            'simpson_diversity': []
        }
        
        for q in [0, 1, 2]:
            bootstrap_results[f'hill_q{q}'] = []
        
        for i in range(self.config.bootstrap_iterations):
            subsampled_data = random.sample(species_data, target_size)
            subsampled_counts = Counter(subsampled_data)
            
            metrics = self.diversity_calc.calculate_basic_metrics(subsampled_counts)
            
            for metric_name, value in metrics.items():
                if metric_name in bootstrap_results:
                    bootstrap_results[metric_name].append(value)
        
        mean_metrics = {}
        confidence_intervals = {}
        alpha = 1 - self.config.confidence_level
        
        for metric_name, values in bootstrap_results.items():
            if values:
                mean_metrics[metric_name] = sum(values) / len(values)
                
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                
                sorted_values = sorted(values)
                n = len(sorted_values)
                lower_idx = int(lower_percentile * n / 100)
                upper_idx = int(upper_percentile * n / 100)
                lower_ci = sorted_values[min(lower_idx, n-1)]
                upper_ci = sorted_values[min(upper_idx, n-1)]
                
                confidence_intervals[metric_name] = (float(lower_ci), float(upper_ci))
        
        return {
            'subsampled_metrics': mean_metrics,
            'confidence_intervals': confidence_intervals,
            'original_sample_size': len(species_data),
            'target_sample_size': target_size,
            'subsampling_performed': True,
            'bootstrap_iterations': self.config.bootstrap_iterations,
            'bootstrap_results': bootstrap_results
        }
    
    def calculate_rarefaction_curve(
        self, 
        species_data: List[str]
    ) -> Dict[str, Any]:
        """Calculate rarefaction curve for species accumulation.
        
        Args:
            species_data: List of species names
            
        Returns:
            Dictionary with rarefaction curve data
        """
        if len(species_data) < self.config.min_sample_size:
            return {'sample_sizes': [], 'mean_richness': [], 'confidence_intervals': []}
        
        max_sample_size = len(species_data)
        step_size = max(1, max_sample_size // self.config.rarefaction_steps)
        sample_sizes = list(range(1, max_sample_size + 1, step_size))
        
        if max_sample_size not in sample_sizes:
            sample_sizes.append(max_sample_size)
        
        mean_richness = []
        confidence_intervals = []
        
        for sample_size in sample_sizes:
            if sample_size > len(species_data):
                continue
            
            richness_values = []
            
            for _ in range(min(100, self.config.bootstrap_iterations)):
                subsampled_data = random.sample(species_data, sample_size)
                unique_species = len(set(subsampled_data))
                richness_values.append(unique_species)
            
            if richness_values:
                mean_rich = sum(richness_values) / len(richness_values)
                mean_richness.append(float(mean_rich))
                
                alpha = 1 - self.config.confidence_level
                sorted_richness = sorted(richness_values)
                n = len(sorted_richness)
                lower_idx = int((alpha / 2) * 100 * n / 100)
                upper_idx = int((1 - alpha / 2) * 100 * n / 100)
                lower_ci = sorted_richness[min(lower_idx, n-1)]
                upper_ci = sorted_richness[min(upper_idx, n-1)]
                confidence_intervals.append((float(lower_ci), float(upper_ci)))
            else:
                mean_richness.append(0.0)
                confidence_intervals.append((0.0, 0.0))
        
        return {
            'sample_sizes': sample_sizes,
            'mean_richness': mean_richness,
            'confidence_intervals': confidence_intervals,
            'max_sample_size': max_sample_size,
            'bootstrap_iterations': min(100, self.config.bootstrap_iterations)
        }
    
    def correct_for_unequal_sampling(
        self, 
        daily_species_data: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Correct diversity metrics for unequal sampling across days.
        
        Args:
            daily_species_data: Dictionary mapping dates to species lists
            
        Returns:
            Dictionary with corrected diversity metrics
        """
        if not daily_species_data:
            return {}
        
        sample_sizes = {date: len(species_list) for date, species_list in daily_species_data.items()}
        min_sample_size = min(sample_sizes.values())
        
        if min_sample_size < self.config.min_sample_size:
            self.logger.warning(f"Minimum sample size {min_sample_size} below threshold")
            min_sample_size = self.config.min_sample_size
        
        correction_size = min(min_sample_size, self.config.target_sample_size)
        
        corrected_metrics = {}
        
        for date, species_list in daily_species_data.items():
            if len(species_list) >= correction_size:
                result = self.subsample_to_fixed_size(species_list, correction_size)
                corrected_metrics[date] = result
            else:
                species_counts = Counter(species_list)
                metrics = self.diversity_calc.calculate_basic_metrics(species_counts)
                corrected_metrics[date] = {
                    'subsampled_metrics': metrics,
                    'confidence_intervals': {},
                    'original_sample_size': len(species_list),
                    'target_sample_size': correction_size,
                    'subsampling_performed': False,
                    'bootstrap_iterations': 0,
                    'warning': 'Sample size too small for correction'
                }
        
        summary = self._calculate_correction_summary(corrected_metrics, sample_sizes)
        
        return {
            'corrected_metrics': corrected_metrics,
            'correction_summary': summary,
            'correction_sample_size': correction_size,
            'original_sample_sizes': sample_sizes
        }
    
    def _calculate_correction_summary(
        self, 
        corrected_metrics: Dict[str, Dict[str, Any]], 
        original_sizes: Dict[str, int]
    ) -> Dict[str, Any]:
        """Calculate summary statistics for sample correction.
        
        Args:
            corrected_metrics: Corrected metrics for each day
            original_sizes: Original sample sizes for each day
            
        Returns:
            Summary statistics dictionary
        """
        if not corrected_metrics:
            return {}
        
        diversity_values: Dict[str, List[float]] = {
            'species_richness': [],
            'shannon_diversity': [],
            'pielou_evenness': [],
            'simpson_diversity': []
        }
        
        for date, result in corrected_metrics.items():
            metrics = result.get('subsampled_metrics', {})
            for metric_name in diversity_values.keys():
                if metric_name in metrics:
                    diversity_values[metric_name].append(metrics[metric_name])
        
        summary = {}
        for metric_name, values in diversity_values.items():
            if values:
                import math
                mean_val = sum(values) / len(values)
                variance = sum((x - mean_val) ** 2 for x in values) / len(values)
                std_val = math.sqrt(variance)
                sorted_vals = sorted(values)
                n = len(sorted_vals)
                median_val = sorted_vals[n // 2] if n % 2 == 1 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
                
                summary[metric_name] = {
                    'mean': float(mean_val),
                    'std': float(std_val),
                    'min': float(min(values)),
                    'max': float(max(values)),
                    'median': float(median_val)
                }
        
        original_sizes_list = list(original_sizes.values())
        import math
        mean_size = sum(original_sizes_list) / len(original_sizes_list)
        variance_size = sum((x - mean_size) ** 2 for x in original_sizes_list) / len(original_sizes_list)
        std_size = math.sqrt(variance_size)
        sorted_sizes = sorted(original_sizes_list)
        n = len(sorted_sizes)
        median_size = sorted_sizes[n // 2] if n % 2 == 1 else (sorted_sizes[n // 2 - 1] + sorted_sizes[n // 2]) / 2
        
        summary['sample_sizes'] = {
            'mean': float(mean_size),
            'std': float(std_size),
            'min': int(min(original_sizes_list)),
            'max': int(max(original_sizes_list)),
            'median': float(median_size)
        }
        
        return summary
    
    def _empty_subsampling_result(self) -> Dict[str, Any]:
        """Return empty subsampling result."""
        return {
            'subsampled_metrics': {},
            'confidence_intervals': {},
            'original_sample_size': 0,
            'target_sample_size': self.config.target_sample_size,
            'subsampling_performed': False,
            'bootstrap_iterations': 0,
            'error': 'Sample size too small'
        }
