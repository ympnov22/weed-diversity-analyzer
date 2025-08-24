"""Tests for diversity calculator implementation."""

import pytest
import numpy as np
from collections import Counter

from src.analysis.diversity_calculator import DiversityCalculator, DiversityConfig


class TestDiversityCalculator:
    """Test cases for diversity calculator."""
    
    @pytest.fixture
    def diversity_calc(self):
        """Create diversity calculator instance."""
        config = DiversityConfig(
            bootstrap_iterations=100,  # Reduced for faster testing
            confidence_level=0.95,
            coverage_target=0.8
        )
        return DiversityCalculator(config)
    
    @pytest.fixture
    def sample_species_counts(self):
        """Create sample species count data."""
        return {
            'species_a': 10,
            'species_b': 8,
            'species_c': 5,
            'species_d': 3,
            'species_e': 1
        }
    
    def test_initialization(self, diversity_calc):
        """Test diversity calculator initialization."""
        assert diversity_calc.config.bootstrap_iterations == 100
        assert diversity_calc.config.confidence_level == 0.95
        assert diversity_calc.config.coverage_target == 0.8
        assert diversity_calc.config.hill_orders == [0, 1, 2]
    
    def test_calculate_basic_metrics_success(self, diversity_calc, sample_species_counts):
        """Test successful basic metrics calculation."""
        metrics = diversity_calc.calculate_basic_metrics(sample_species_counts)
        
        assert 'species_richness' in metrics
        assert 'shannon_diversity' in metrics
        assert 'pielou_evenness' in metrics
        assert 'simpson_diversity' in metrics
        assert 'total_individuals' in metrics
        
        assert metrics['species_richness'] == 5.0
        assert metrics['total_individuals'] == 27
        assert 0 <= metrics['pielou_evenness'] <= 1
        assert 0 <= metrics['simpson_diversity'] <= 1
        assert metrics['shannon_diversity'] > 0
        
        assert 'hill_q0' in metrics
        assert 'hill_q1' in metrics
        assert 'hill_q2' in metrics
        assert metrics['hill_q0'] == 5.0  # Should equal species richness
    
    def test_calculate_basic_metrics_empty_data(self, diversity_calc):
        """Test basic metrics with empty data."""
        metrics = diversity_calc.calculate_basic_metrics({})
        
        assert metrics['species_richness'] == 0.0
        assert abs(metrics['shannon_diversity']) < 1e-9  # Near zero due to floating point precision
        assert metrics['pielou_evenness'] == 0.0
        assert metrics['simpson_diversity'] == 0.0
        assert metrics['total_individuals'] == 0
    
    def test_calculate_basic_metrics_single_species(self, diversity_calc):
        """Test basic metrics with single species."""
        single_species = {'species_a': 10}
        metrics = diversity_calc.calculate_basic_metrics(single_species)
        
        assert metrics['species_richness'] == 1.0
        assert abs(metrics['shannon_diversity']) < 1e-9  # Near zero due to floating point precision
        assert metrics['pielou_evenness'] == 1.0  # Perfect evenness for single species
        assert metrics['simpson_diversity'] == 0.0
    
    def test_chao1_estimator_success(self, diversity_calc):
        """Test Chao1 estimator calculation."""
        species_counts = {
            'species_a': 5,
            'species_b': 3,
            'species_c': 2,  # doubleton
            'species_d': 2,  # doubleton
            'species_e': 1,  # singleton
            'species_f': 1   # singleton
        }
        
        chao1, chao1_se = diversity_calc.calculate_chao1_estimator(species_counts)
        
        assert chao1 >= len(species_counts)  # Should be >= observed richness
        assert chao1_se >= 0  # Standard error should be non-negative
        assert isinstance(chao1, float)
        assert isinstance(chao1_se, float)
    
    def test_chao1_estimator_no_doubletons(self, diversity_calc):
        """Test Chao1 estimator with no doubletons."""
        species_counts = {
            'species_a': 5,
            'species_b': 3,
            'species_c': 1,  # singleton
            'species_d': 1   # singleton
        }
        
        chao1, chao1_se = diversity_calc.calculate_chao1_estimator(species_counts)
        
        assert chao1 >= len(species_counts)
        assert chao1_se >= 0
    
    def test_chao1_estimator_empty_data(self, diversity_calc):
        """Test Chao1 estimator with empty data."""
        chao1, chao1_se = diversity_calc.calculate_chao1_estimator({})
        
        assert chao1 == 0.0
        assert chao1_se == 0.0
    
    def test_coverage_standardized_diversity(self, diversity_calc, sample_species_counts):
        """Test coverage-standardized diversity calculation."""
        metrics = diversity_calc.calculate_coverage_standardized_diversity(sample_species_counts)
        
        assert 'sample_coverage' in metrics
        assert 'target_coverage' in metrics
        assert 'species_richness' in metrics
        assert 'shannon_diversity' in metrics
        
        assert 0 <= metrics['sample_coverage'] <= 1
        assert metrics['target_coverage'] == 0.8
    
    def test_bootstrap_confidence_intervals(self, diversity_calc, sample_species_counts):
        """Test bootstrap confidence intervals calculation."""
        confidence_intervals = diversity_calc.bootstrap_confidence_intervals(sample_species_counts)
        
        assert 'species_richness' in confidence_intervals
        assert 'shannon_diversity' in confidence_intervals
        assert 'pielou_evenness' in confidence_intervals
        
        for metric_name, (lower_ci, upper_ci) in confidence_intervals.items():
            assert isinstance(lower_ci, float)
            assert isinstance(upper_ci, float)
            assert lower_ci <= upper_ci
    
    def test_bootstrap_confidence_intervals_empty_data(self, diversity_calc):
        """Test bootstrap confidence intervals with empty data."""
        confidence_intervals = diversity_calc.bootstrap_confidence_intervals({})
        
        assert confidence_intervals == {}
    
    def test_hill_number_calculation(self, diversity_calc):
        """Test Hill number calculations."""
        proportions = np.array([0.4, 0.3, 0.2, 0.1])
        
        hill_0 = diversity_calc._calculate_hill_number(proportions, 0)
        assert hill_0 == 4.0
        
        hill_1 = diversity_calc._calculate_hill_number(proportions, 1)
        assert hill_1 > 0
        
        hill_2 = diversity_calc._calculate_hill_number(proportions, 2)
        assert hill_2 > 0
        
        assert hill_0 >= hill_1 >= hill_2
    
    def test_hill_number_edge_cases(self, diversity_calc):
        """Test Hill number edge cases."""
        empty_hill = diversity_calc._calculate_hill_number(np.array([]), 1)
        assert empty_hill == 0.0
        
        single_hill = diversity_calc._calculate_hill_number(np.array([1.0]), 1)
        assert abs(single_hill - 1.0) < 1e-9  # Near 1.0 due to floating point precision
        
        zero_props = np.array([0.5, 0.0, 0.5])
        hill_with_zero = diversity_calc._calculate_hill_number(zero_props, 2)
        assert hill_with_zero >= 0
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = DiversityConfig()
        
        assert config.bootstrap_iterations == 1000
        assert config.confidence_level == 0.95
        assert config.coverage_target == 0.8
        assert config.min_confidence_threshold == 0.3
        assert config.subsampling_size == 30
        assert config.hill_orders == [0, 1, 2]
    
    def test_custom_hill_orders(self):
        """Test custom Hill orders configuration."""
        custom_orders = [0, 0.5, 1, 1.5, 2]
        config = DiversityConfig(hill_orders=custom_orders)
        calc = DiversityCalculator(config)
        
        species_counts = {'species_a': 5, 'species_b': 3, 'species_c': 2}
        metrics = calc.calculate_basic_metrics(species_counts)
        
        for q in custom_orders:
            assert f'hill_q{q}' in metrics
            assert metrics[f'hill_q{q}'] >= 0
