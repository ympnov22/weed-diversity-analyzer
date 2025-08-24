"""Tests for sample correction implementation."""

import pytest
import numpy as np
from collections import Counter
from unittest.mock import patch

from src.analysis.sample_correction import SampleCorrection, SamplingConfig


class TestSampleCorrection:
    """Test cases for sample correction."""
    
    @pytest.fixture
    def sampling_config(self):
        """Create sampling configuration."""
        return SamplingConfig(
            target_sample_size=10,  # Reduced for faster testing
            bootstrap_iterations=50,  # Reduced for faster testing
            confidence_level=0.95,
            min_sample_size=3,
            rarefaction_steps=5
        )
    
    @pytest.fixture
    def sample_correction(self, sampling_config):
        """Create sample correction instance."""
        return SampleCorrection(sampling_config)
    
    @pytest.fixture
    def sample_species_data(self):
        """Create sample species data."""
        return [
            'species_a', 'species_a', 'species_a', 'species_a', 'species_a',
            'species_b', 'species_b', 'species_b', 'species_b',
            'species_c', 'species_c', 'species_c',
            'species_d', 'species_d',
            'species_e'
        ]  # 15 total individuals, 5 species
    
    def test_initialization(self, sample_correction):
        """Test sample correction initialization."""
        assert sample_correction.config.target_sample_size == 10
        assert sample_correction.config.bootstrap_iterations == 50
        assert sample_correction.config.confidence_level == 0.95
        assert sample_correction.config.min_sample_size == 3
    
    def test_subsample_to_fixed_size_larger_sample(self, sample_correction, sample_species_data):
        """Test subsampling when original sample is larger than target."""
        result = sample_correction.subsample_to_fixed_size(sample_species_data, target_size=10)
        
        assert 'subsampled_metrics' in result
        assert 'confidence_intervals' in result
        assert result['original_sample_size'] == 15
        assert result['target_sample_size'] == 10
        assert result['subsampling_performed'] is True
        assert result['bootstrap_iterations'] == 50
        
        metrics = result['subsampled_metrics']
        assert 'species_richness' in metrics
        assert 'shannon_diversity' in metrics
        assert metrics['species_richness'] > 0
        
        ci = result['confidence_intervals']
        assert 'species_richness' in ci
        lower_ci, upper_ci = ci['species_richness']
        assert lower_ci <= upper_ci
    
    def test_subsample_to_fixed_size_smaller_sample(self, sample_correction):
        """Test subsampling when original sample is smaller than target."""
        small_sample = ['species_a', 'species_b', 'species_c']
        result = sample_correction.subsample_to_fixed_size(small_sample, target_size=10)
        
        assert result['original_sample_size'] == 3
        assert result['target_sample_size'] == 10
        assert result['subsampling_performed'] is False
        assert result['bootstrap_iterations'] == 0
        
        metrics = result['subsampled_metrics']
        assert metrics['species_richness'] == 3.0
    
    def test_subsample_to_fixed_size_too_small(self, sample_correction):
        """Test subsampling with sample too small."""
        tiny_sample = ['species_a', 'species_b']  # Below min_sample_size
        result = sample_correction.subsample_to_fixed_size(tiny_sample)
        
        assert 'error' in result
        assert result['original_sample_size'] == 0
        assert result['subsampling_performed'] is False
    
    def test_subsample_to_fixed_size_default_target(self, sample_correction, sample_species_data):
        """Test subsampling with default target size."""
        result = sample_correction.subsample_to_fixed_size(sample_species_data)
        
        assert result['target_sample_size'] == 10
    
    @patch('random.sample')
    def test_subsample_reproducible_results(self, mock_sample, sample_correction):
        """Test that subsampling produces consistent results."""
        sample_data = ['species_a'] * 5 + ['species_b'] * 5
        
        mock_sample.return_value = ['species_a'] * 3 + ['species_b'] * 2
        
        result = sample_correction.subsample_to_fixed_size(sample_data, target_size=5)
        
        assert mock_sample.call_count == sample_correction.config.bootstrap_iterations
        assert result['subsampling_performed'] is True
    
    def test_calculate_rarefaction_curve(self, sample_correction, sample_species_data):
        """Test rarefaction curve calculation."""
        result = sample_correction.calculate_rarefaction_curve(sample_species_data)
        
        assert 'sample_sizes' in result
        assert 'mean_richness' in result
        assert 'confidence_intervals' in result
        assert 'max_sample_size' in result
        
        sample_sizes = result['sample_sizes']
        mean_richness = result['mean_richness']
        confidence_intervals = result['confidence_intervals']
        
        assert len(sample_sizes) == len(mean_richness)
        assert len(sample_sizes) == len(confidence_intervals)
        assert result['max_sample_size'] == 15
        
        assert mean_richness[0] <= mean_richness[-1]
        
        for lower_ci, upper_ci in confidence_intervals:
            assert lower_ci <= upper_ci
    
    def test_calculate_rarefaction_curve_small_sample(self, sample_correction):
        """Test rarefaction curve with small sample."""
        small_sample = ['species_a', 'species_b']
        result = sample_correction.calculate_rarefaction_curve(small_sample)
        
        assert result['sample_sizes'] == []
        assert result['mean_richness'] == []
        assert result['confidence_intervals'] == []
    
    def test_correct_for_unequal_sampling(self, sample_correction):
        """Test correction for unequal sampling across days."""
        daily_data = {
            'day1': ['species_a'] * 8 + ['species_b'] * 4 + ['species_c'] * 3,  # 15 total
            'day2': ['species_a'] * 5 + ['species_b'] * 3 + ['species_d'] * 2,  # 10 total
            'day3': ['species_a'] * 6 + ['species_c'] * 4,  # 10 total
        }
        
        result = sample_correction.correct_for_unequal_sampling(daily_data)
        
        assert 'corrected_metrics' in result
        assert 'correction_summary' in result
        assert 'correction_sample_size' in result
        assert 'original_sample_sizes' in result
        
        corrected = result['corrected_metrics']
        assert 'day1' in corrected
        assert 'day2' in corrected
        assert 'day3' in corrected
        
        assert result['correction_sample_size'] == 10
        
        summary = result['correction_summary']
        assert 'species_richness' in summary
        assert 'sample_sizes' in summary
        
        richness_stats = summary['species_richness']
        assert 'mean' in richness_stats
        assert 'std' in richness_stats
        assert 'min' in richness_stats
        assert 'max' in richness_stats
    
    def test_correct_for_unequal_sampling_small_samples(self, sample_correction):
        """Test correction with some samples too small."""
        daily_data = {
            'day1': ['species_a'] * 8 + ['species_b'] * 4,  # 12 total - OK
            'day2': ['species_a', 'species_b'],  # 2 total - too small
            'day3': ['species_a'] * 6 + ['species_c'] * 4,  # 10 total - OK
        }
        
        result = sample_correction.correct_for_unequal_sampling(daily_data)
        
        corrected = result['corrected_metrics']
        assert len(corrected) == 3
        
        day2_result = corrected['day2']
        assert 'warning' in day2_result
        assert day2_result['subsampling_performed'] is False
    
    def test_correct_for_unequal_sampling_empty_data(self, sample_correction):
        """Test correction with empty data."""
        result = sample_correction.correct_for_unequal_sampling({})
        assert result == {}
    
    def test_correction_summary_calculation(self, sample_correction):
        """Test correction summary statistics calculation."""
        corrected_metrics = {
            'day1': {
                'subsampled_metrics': {
                    'species_richness': 3.0,
                    'shannon_diversity': 1.2,
                    'pielou_evenness': 0.8,
                    'simpson_diversity': 0.6
                }
            },
            'day2': {
                'subsampled_metrics': {
                    'species_richness': 4.0,
                    'shannon_diversity': 1.5,
                    'pielou_evenness': 0.9,
                    'simpson_diversity': 0.7
                }
            }
        }
        
        original_sizes = {'day1': 15, 'day2': 12}
        
        summary = sample_correction._calculate_correction_summary(
            corrected_metrics, 
            original_sizes
        )
        
        assert 'species_richness' in summary
        assert 'shannon_diversity' in summary
        assert 'sample_sizes' in summary
        
        richness_stats = summary['species_richness']
        assert richness_stats['mean'] == 3.5  # (3.0 + 4.0) / 2
        assert richness_stats['min'] == 3.0
        assert richness_stats['max'] == 4.0
        
        size_stats = summary['sample_sizes']
        assert size_stats['mean'] == 13.5  # (15 + 12) / 2
        assert size_stats['min'] == 12
        assert size_stats['max'] == 15
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = SamplingConfig()
        
        assert config.target_sample_size == 30
        assert config.bootstrap_iterations == 1000
        assert config.confidence_level == 0.95
        assert config.min_sample_size == 5
        assert config.rarefaction_steps == 20
