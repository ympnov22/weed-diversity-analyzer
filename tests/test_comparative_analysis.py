"""Tests for comparative analysis implementation."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch

from src.analysis.comparative_analysis import (
    ComparativeAnalyzer, 
    ComparisonConfig
)


class TestComparativeAnalyzer:
    """Test cases for comparative analyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create comparative analyzer instance."""
        return ComparativeAnalyzer()
    
    @pytest.fixture
    def sample_daily_summaries(self):
        """Create sample daily summaries for temporal analysis."""
        base_date = datetime(2025, 8, 1)
        summaries = []
        
        for i in range(30):
            date = base_date + timedelta(days=i)
            summaries.append({
                'date': date.strftime('%Y-%m-%d'),
                'diversity_metrics': {
                    'species_richness': 5 + np.random.randint(-2, 3),
                    'shannon_diversity': 1.5 + np.random.normal(0, 0.2),
                    'pielou_evenness': 0.8 + np.random.normal(0, 0.1)
                },
                'top_species': [
                    {'species_name': 'Taraxacum officinale', 'count': 3 + np.random.randint(-1, 2), 'average_confidence': 0.85},
                    {'species_name': 'Plantago major', 'count': 2 + np.random.randint(-1, 2), 'average_confidence': 0.78},
                    {'species_name': 'Trifolium repens', 'count': 1 + np.random.randint(0, 2), 'average_confidence': 0.72}
                ]
            })
        
        return summaries
    
    @pytest.fixture
    def sample_site_data(self):
        """Create sample site data for beta diversity analysis."""
        return [
            {
                'site_name': 'Site_A',
                'species_counts': {
                    'Taraxacum officinale': 5,
                    'Plantago major': 3,
                    'Trifolium repens': 2,
                    'Bellis perennis': 1
                }
            },
            {
                'site_name': 'Site_B',
                'species_counts': {
                    'Taraxacum officinale': 2,
                    'Plantago major': 4,
                    'Capsella bursa-pastoris': 3,
                    'Stellaria media': 2
                }
            },
            {
                'site_name': 'Site_C',
                'species_counts': {
                    'Trifolium repens': 4,
                    'Bellis perennis': 3,
                    'Capsella bursa-pastoris': 1,
                    'Poa annua': 2
                }
            }
        ]
    
    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = ComparativeAnalyzer()
        assert analyzer is not None
        assert isinstance(analyzer.config, ComparisonConfig)
    
    def test_initialization_with_config(self):
        """Test analyzer initialization with custom config."""
        config = ComparisonConfig(significance_level=0.01, bootstrap_iterations=500)
        analyzer = ComparativeAnalyzer(config)
        assert analyzer.config.significance_level == 0.01
        assert analyzer.config.bootstrap_iterations == 500
    
    def test_compare_temporal_diversity_success(self, analyzer, sample_daily_summaries):
        """Test successful temporal diversity comparison."""
        result = analyzer.compare_temporal_diversity(sample_daily_summaries)
        
        assert 'period_summary' in result
        assert 'trend_analysis' in result
        assert 'statistical_tests' in result
        
        period_summary = result['period_summary']
        assert 'start_date' in period_summary
        assert 'end_date' in period_summary
        assert 'total_days' in period_summary
        assert period_summary['total_days'] == 30
        
        trend_analysis = result['trend_analysis']
        assert 'species_richness' in trend_analysis
        assert 'shannon_diversity' in trend_analysis
        
        for metric, trend_data in trend_analysis.items():
            assert 'linear_trend' in trend_data
            assert 'mann_kendall' in trend_data
            assert 'trend_direction' in trend_data
            assert 'is_significant' in trend_data
    
    def test_compare_temporal_diversity_insufficient_data(self, analyzer):
        """Test temporal diversity comparison with insufficient data."""
        insufficient_data = [
            {
                'date': '2025-08-01',
                'diversity_metrics': {'species_richness': 5}
            }
        ]
        
        result = analyzer.compare_temporal_diversity(insufficient_data)
        assert 'error' in result
        assert result['error'] == 'insufficient_data'
    
    def test_compare_beta_diversity_success(self, analyzer, sample_site_data):
        """Test successful beta diversity comparison."""
        result = analyzer.compare_beta_diversity(sample_site_data)
        
        assert 'sites' in result
        assert 'total_species' in result
        assert 'beta_diversity_indices' in result
        assert 'similarity_matrix' in result
        assert 'species_turnover' in result
        
        assert len(result['sites']) == 3
        assert result['total_species'] > 0
        
        beta_indices = result['beta_diversity_indices']
        assert 'jaccard' in beta_indices
        assert 'sorensen' in beta_indices
        
        similarity_matrix = result['similarity_matrix']
        assert 'jaccard' in similarity_matrix
        assert 'sorensen' in similarity_matrix
        
        turnover = result['species_turnover']
        assert 'pairwise_turnover' in turnover
        assert 'unique_species_per_site' in turnover
    
    def test_compare_beta_diversity_insufficient_sites(self, analyzer):
        """Test beta diversity comparison with insufficient sites."""
        insufficient_data = [
            {
                'site_name': 'Site_A',
                'species_counts': {'Species1': 5}
            }
        ]
        
        result = analyzer.compare_beta_diversity(insufficient_data)
        assert 'error' in result
        assert result['error'] == 'insufficient_data'
    
    def test_analyze_species_correlations_success(self, analyzer, sample_daily_summaries):
        """Test successful species correlation analysis."""
        result = analyzer.analyze_species_correlations(sample_daily_summaries)
        
        assert 'species_analyzed' in result
        assert 'total_species' in result
        assert 'observation_days' in result
        assert 'correlation_analysis' in result
        assert 'cooccurrence_patterns' in result
        assert 'network_analysis' in result
        assert 'significant_associations' in result
        
        assert result['total_species'] > 0
        assert result['observation_days'] == 30
        
        correlation_analysis = result['correlation_analysis']
        assert 'correlation_matrix' in correlation_analysis
        assert 'species_names' in correlation_analysis
        assert 'method' in correlation_analysis
        
        cooccurrence = result['cooccurrence_patterns']
        assert 'cooccurrence_matrix' in cooccurrence
        assert 'association_strength' in cooccurrence
    
    def test_analyze_species_correlations_insufficient_data(self, analyzer):
        """Test species correlation analysis with insufficient data."""
        insufficient_data = [
            {
                'date': '2025-08-01',
                'top_species': [{'species_name': 'Species1', 'count': 1}]
            }
        ]
        
        result = analyzer.analyze_species_correlations(insufficient_data)
        assert 'error' in result
        assert result['error'] == 'insufficient_data'
    
    def test_perform_statistical_tests_success(self, analyzer):
        """Test successful statistical tests."""
        group1_data = [
            {'diversity_metrics': {'shannon_diversity': 1.0 + np.random.normal(0, 0.1)}}
            for _ in range(20)
        ]
        
        group2_data = [
            {'diversity_metrics': {'shannon_diversity': 1.5 + np.random.normal(0, 0.1)}}
            for _ in range(20)
        ]
        
        result = analyzer.perform_statistical_tests(group1_data, group2_data)
        
        assert 'group1_summary' in result
        assert 'group2_summary' in result
        assert 'statistical_tests' in result
        assert 'effect_sizes' in result
        assert 'summary' in result
        
        assert result['group1_summary']['sample_size'] == 20
        assert result['group2_summary']['sample_size'] == 20
        
        if 'shannon_diversity' in result['statistical_tests']:
            test_result = result['statistical_tests']['shannon_diversity']
            assert 't_test' in test_result
            assert 'mann_whitney_u' in test_result
            assert 'period_comparison' in test_result
        
        if 'shannon_diversity' in result['effect_sizes']:
            effect_size = result['effect_sizes']['shannon_diversity']
            assert 'cohens_d' in effect_size
            assert 'effect_size_interpretation' in effect_size
    
    def test_perform_statistical_tests_empty_data(self, analyzer):
        """Test statistical tests with empty data."""
        result = analyzer.perform_statistical_tests([], [])
        assert 'error' in result
        assert result['error'] == 'insufficient_data'
    
    def test_bray_curtis_calculation(self, analyzer):
        """Test Bray-Curtis dissimilarity calculation."""
        abundance_matrix = np.array([
            [5, 3, 2, 0],
            [2, 4, 0, 3],
            [0, 1, 4, 2]
        ])
        
        result = analyzer._calculate_bray_curtis(abundance_matrix)
        
        assert result.shape == (3, 3)
        assert np.allclose(np.diag(result), 0)  # Diagonal should be 0
        assert np.allclose(result, result.T)  # Should be symmetric
    
    def test_jaccard_similarity_calculation(self, analyzer):
        """Test Jaccard similarity calculation."""
        abundance_matrix = np.array([
            [5, 3, 2, 0],
            [2, 4, 0, 3],
            [0, 1, 4, 2]
        ])
        
        result = analyzer._calculate_jaccard_similarity(abundance_matrix)
        
        assert result.shape == (3, 3)
        assert np.allclose(np.diag(result), 1)  # Diagonal should be 1
        assert np.allclose(result, result.T)  # Should be symmetric
        assert np.all(result >= 0) and np.all(result <= 1)  # Values between 0 and 1
    
    def test_mann_kendall_test(self, analyzer):
        """Test Mann-Kendall trend test."""
        increasing_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = analyzer._mann_kendall_test(increasing_data)
        
        assert 'trend' in result
        assert 'p_value' in result
        assert 'significant' in result
        assert result['trend'] == 'increasing'
        
        no_trend_data = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
        result = analyzer._mann_kendall_test(no_trend_data)
        assert result['trend'] == 'no trend'
    
    def test_change_point_detection(self, analyzer):
        """Test change point detection."""
        data = np.concatenate([
            np.random.normal(1, 0.1, 20),  # First period
            np.random.normal(2, 0.1, 20)   # Second period (higher mean)
        ])
        
        result = analyzer._cumsum_change_detection(data)
        
        assert isinstance(result, list)
        if result:
            change_point = result[0]
            assert 'position' in change_point
            assert 'magnitude' in change_point
            assert 'direction' in change_point
    
    def test_phi_coefficient_calculation(self, analyzer):
        """Test Phi coefficient calculation."""
        contingency_table = np.array([[10, 0], [0, 10]])
        phi = analyzer._calculate_phi_coefficient(contingency_table)
        assert abs(phi - 1.0) < 0.01
        
        contingency_table = np.array([[0, 10], [10, 0]])
        phi = analyzer._calculate_phi_coefficient(contingency_table)
        assert abs(phi + 1.0) < 0.01
        
        contingency_table = np.array([[5, 5], [5, 5]])
        phi = analyzer._calculate_phi_coefficient(contingency_table)
        assert abs(phi) < 0.01
    
    def test_effect_size_interpretation(self, analyzer):
        """Test effect size interpretation."""
        assert analyzer._interpret_effect_size(0.1) == "negligible"
        assert analyzer._interpret_effect_size(0.3) == "small"
        assert analyzer._interpret_effect_size(0.6) == "medium"
        assert analyzer._interpret_effect_size(1.0) == "large"
    
    def test_extract_metric_values(self, analyzer):
        """Test metric value extraction."""
        data_list = [
            {'diversity_metrics': {'shannon_diversity': 1.5, 'species_richness': 5}},
            {'diversity_metrics': {'shannon_diversity': 1.8, 'species_richness': 7}},
            {'diversity_metrics': {'shannon_diversity': 1.2, 'species_richness': 4}}
        ]
        
        result = analyzer._extract_metric_values(data_list)
        
        assert 'shannon_diversity' in result
        assert 'species_richness' in result
        assert len(result['shannon_diversity']) == 3
        assert len(result['species_richness']) == 3
        assert result['shannon_diversity'] == [1.5, 1.8, 1.2]
        assert result['species_richness'] == [5.0, 7.0, 4.0]
    
    def test_seasonal_pattern_analysis(self, analyzer):
        """Test seasonal pattern analysis."""
        import pandas as pd
        
        dates = pd.date_range('2025-01-01', periods=365, freq='D')
        day_of_year = dates.dayofyear
        seasonal_values = 2 + np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 0.1, 365)
        
        time_series = pd.Series(seasonal_values, index=dates)
        
        result = analyzer._analyze_seasonal_patterns(time_series)
        
        assert 'monthly_patterns' in result
        assert 'seasonal_variation_coefficient' in result
        assert 'peak_month' in result
        assert 'low_month' in result
        
        monthly_patterns = result['monthly_patterns']
        assert 'means' in monthly_patterns
        assert 'standard_deviations' in monthly_patterns
        assert len(monthly_patterns['means']) <= 12
    
    def test_network_analysis(self, analyzer):
        """Test species network analysis."""
        correlation_matrix = [
            [1.0, 0.8, 0.3, 0.1],
            [0.8, 1.0, 0.2, 0.6],
            [0.3, 0.2, 1.0, 0.7],
            [0.1, 0.6, 0.7, 1.0]
        ]
        
        species_names = ['Species_A', 'Species_B', 'Species_C', 'Species_D']
        
        result = analyzer._analyze_species_network(correlation_matrix, species_names)
        
        assert 'adjacency_matrix' in result
        assert 'network_metrics' in result
        assert 'node_metrics' in result
        
        network_metrics = result['network_metrics']
        assert 'total_nodes' in network_metrics
        assert 'total_edges' in network_metrics
        assert 'density' in network_metrics
        assert 'average_degree' in network_metrics
        
        assert network_metrics['total_nodes'] == 4
        
        node_metrics = result['node_metrics']
        for species in species_names:
            assert species in node_metrics
            assert 'degree' in node_metrics[species]
            assert 'degree_centrality' in node_metrics[species]
