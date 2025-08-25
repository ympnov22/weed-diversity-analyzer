"""Tests for functional diversity analysis implementation."""

import pytest
import numpy as np
from unittest.mock import patch

from src.analysis.functional_diversity import (
    FunctionalDiversityAnalyzer,
    FunctionalDiversityConfig,
    FunctionalTraits
)


class TestFunctionalTraits:
    """Test cases for functional traits data structure."""
    
    def test_functional_traits_creation(self):
        """Test functional traits creation."""
        traits = FunctionalTraits(
            species_name="Taraxacum officinale",
            height_cm=25.0,
            leaf_area_cm2=15.5,
            seed_mass_mg=0.8,
            growth_form="perennial",
            photosynthesis_type="C3",
            nitrogen_fixation=False
        )
        
        assert traits.species_name == "Taraxacum officinale"
        assert traits.height_cm == 25.0
        assert traits.leaf_area_cm2 == 15.5
        assert traits.seed_mass_mg == 0.8
        assert traits.growth_form == "perennial"
        assert traits.photosynthesis_type == "C3"
        assert traits.nitrogen_fixation is False
    
    def test_functional_traits_to_dict(self):
        """Test conversion of functional traits to dictionary."""
        traits = FunctionalTraits(
            species_name="Plantago major",
            height_cm=15.0,
            growth_form="perennial"
        )
        
        trait_dict = traits.to_dict()
        
        assert isinstance(trait_dict, dict)
        assert trait_dict['species_name'] == "Plantago major"
        assert trait_dict['height_cm'] == 15.0
        assert trait_dict['growth_form'] == "perennial"
        assert 'leaf_area_cm2' in trait_dict
        assert trait_dict['leaf_area_cm2'] is None


class TestFunctionalDiversityAnalyzer:
    """Test cases for functional diversity analyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create functional diversity analyzer instance."""
        return FunctionalDiversityAnalyzer()
    
    @pytest.fixture
    def sample_traits(self):
        """Create sample functional traits."""
        return {
            "Taraxacum officinale": {
                "height_cm": 25.0,
                "leaf_area_cm2": 15.5,
                "seed_mass_mg": 0.8,
                "growth_form": "perennial",
                "photosynthesis_type": "C3",
                "nitrogen_fixation": False,
                "dispersal_mode": "wind",
                "flowering_start_month": 4,
                "flowering_duration_months": 6,
                "root_depth_cm": 30.0,
                "specific_leaf_area": 25.0
            },
            "Plantago major": {
                "height_cm": 15.0,
                "leaf_area_cm2": 8.2,
                "seed_mass_mg": 0.3,
                "growth_form": "perennial",
                "photosynthesis_type": "C3",
                "nitrogen_fixation": False,
                "dispersal_mode": "animal",
                "flowering_start_month": 5,
                "flowering_duration_months": 4,
                "root_depth_cm": 20.0,
                "specific_leaf_area": 20.0
            },
            "Trifolium repens": {
                "height_cm": 10.0,
                "leaf_area_cm2": 5.8,
                "seed_mass_mg": 0.5,
                "growth_form": "perennial",
                "photosynthesis_type": "C3",
                "nitrogen_fixation": True,
                "dispersal_mode": "animal",
                "flowering_start_month": 6,
                "flowering_duration_months": 3,
                "root_depth_cm": 25.0,
                "specific_leaf_area": 30.0
            },
            "Poa annua": {
                "height_cm": 20.0,
                "leaf_area_cm2": 3.2,
                "seed_mass_mg": 0.2,
                "growth_form": "annual",
                "photosynthesis_type": "C3",
                "nitrogen_fixation": False,
                "dispersal_mode": "wind",
                "flowering_start_month": 3,
                "flowering_duration_months": 8,
                "root_depth_cm": 15.0,
                "specific_leaf_area": 35.0
            }
        }
    
    @pytest.fixture
    def sample_abundances(self):
        """Create sample species abundances."""
        return {
            "Taraxacum officinale": 10.0,
            "Plantago major": 8.0,
            "Trifolium repens": 6.0,
            "Poa annua": 4.0
        }
    
    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = FunctionalDiversityAnalyzer()
        assert analyzer is not None
        assert isinstance(analyzer.config, FunctionalDiversityConfig)
        assert analyzer.trait_database == {}
    
    def test_initialization_with_config(self):
        """Test analyzer initialization with custom config."""
        config = FunctionalDiversityConfig(
            distance_metric="manhattan",
            standardize_traits=False
        )
        analyzer = FunctionalDiversityAnalyzer(config)
        assert analyzer.config.distance_metric == "manhattan"
        assert analyzer.config.standardize_traits is False
    
    def test_add_species_traits(self, analyzer):
        """Test adding species traits."""
        traits = FunctionalTraits(
            species_name="Test species",
            height_cm=20.0,
            growth_form="annual"
        )
        
        analyzer.add_species_traits(traits)
        
        assert "Test species" in analyzer.trait_database
        assert analyzer.trait_database["Test species"] == traits
    
    def test_load_traits_from_dict(self, analyzer, sample_traits):
        """Test loading traits from dictionary."""
        analyzer.load_traits_from_dict(sample_traits)
        
        assert len(analyzer.trait_database) == 4
        assert "Taraxacum officinale" in analyzer.trait_database
        assert "Plantago major" in analyzer.trait_database
        
        taraxacum_traits = analyzer.trait_database["Taraxacum officinale"]
        assert taraxacum_traits.height_cm == 25.0
        assert taraxacum_traits.growth_form == "perennial"
    
    def test_calculate_functional_diversity_success(self, analyzer, sample_traits, sample_abundances):
        """Test successful functional diversity calculation."""
        analyzer.load_traits_from_dict(sample_traits)
        
        result = analyzer.calculate_functional_diversity(sample_abundances)
        
        assert 'species_analyzed' in result
        assert 'traits_used' in result
        assert 'functional_diversity_indices' in result
        assert 'functional_groups' in result
        assert 'trait_analysis' in result
        assert 'functional_space' in result
        
        assert len(result['species_analyzed']) == 4
        assert 'Taraxacum officinale' in result['species_analyzed']
        
        fd_indices = result['functional_diversity_indices']
        assert 'functional_richness' in fd_indices
        assert 'functional_evenness' in fd_indices
        assert 'functional_divergence' in fd_indices
        assert 'functional_dispersion' in fd_indices
        assert 'raos_quadratic_entropy' in fd_indices
        
        assert fd_indices['functional_richness'] >= 0
        assert 0 <= fd_indices['functional_evenness'] <= 1
        assert fd_indices['functional_divergence'] >= 0
        assert fd_indices['functional_dispersion'] >= 0
        assert fd_indices['raos_quadratic_entropy'] >= 0
    
    def test_calculate_functional_diversity_no_traits(self, analyzer, sample_abundances):
        """Test functional diversity calculation with no trait data."""
        result = analyzer.calculate_functional_diversity(sample_abundances)
        
        assert 'error' in result
        assert result['error'] == 'insufficient_data'
    
    def test_calculate_functional_diversity_empty_abundances(self, analyzer, sample_traits):
        """Test functional diversity calculation with empty abundances."""
        analyzer.load_traits_from_dict(sample_traits)
        
        result = analyzer.calculate_functional_diversity({})
        
        assert 'error' in result
        assert result['error'] == 'insufficient_data'
    
    def test_compare_functional_diversity_success(self, analyzer, sample_traits):
        """Test successful functional diversity comparison."""
        analyzer.load_traits_from_dict(sample_traits)
        
        community1 = {
            "Taraxacum officinale": 10.0,
            "Plantago major": 8.0
        }
        
        community2 = {
            "Trifolium repens": 6.0,
            "Poa annua": 4.0
        }
        
        result = analyzer.compare_functional_diversity(community1, community2)
        
        assert 'community1' in result
        assert 'community2' in result
        assert 'comparison' in result
        
        comparison = result['comparison']
        assert 'functional_diversity_differences' in comparison
        assert 'functional_beta_diversity' in comparison
        
        fd_differences = comparison['functional_diversity_differences']
        assert 'functional_richness' in fd_differences
        
        for metric, diff_data in fd_differences.items():
            assert 'community1_value' in diff_data
            assert 'community2_value' in diff_data
            assert 'difference' in diff_data
            assert 'relative_difference' in diff_data
    
    def test_analyze_trait_correlations_success(self, analyzer, sample_traits):
        """Test successful trait correlation analysis."""
        analyzer.load_traits_from_dict(sample_traits)
        
        species_list = list(sample_traits.keys())
        result = analyzer.analyze_trait_correlations(species_list)
        
        assert 'species_analyzed' in result
        assert 'traits_analyzed' in result
        assert 'correlation_matrix' in result
        assert 'significant_correlations' in result
        assert 'pca_analysis' in result
        assert 'trait_summary_statistics' in result
        
        correlation_matrix = result['correlation_matrix']
        assert isinstance(correlation_matrix, list)
        assert len(correlation_matrix) > 0
        
        pca_analysis = result['pca_analysis']
        if 'error' not in pca_analysis:
            assert 'explained_variance_ratio' in pca_analysis
            assert 'cumulative_variance' in pca_analysis
            assert 'loadings' in pca_analysis
    
    def test_analyze_trait_correlations_no_traits(self, analyzer):
        """Test trait correlation analysis with no trait data."""
        result = analyzer.analyze_trait_correlations(['Species1', 'Species2'])
        
        assert 'error' in result
        assert result['error'] == 'no_trait_data_available'
    
    def test_generate_sample_trait_database(self, analyzer):
        """Test sample trait database generation."""
        species_list = ['Species1', 'Species2', 'Species3']
        
        sample_traits = analyzer.generate_sample_trait_database(species_list)
        
        assert isinstance(sample_traits, dict)
        assert len(sample_traits) == 3
        
        for species in species_list:
            assert species in sample_traits
            traits = sample_traits[species]
            
            assert 'height_cm' in traits
            assert 'leaf_area_cm2' in traits
            assert 'seed_mass_mg' in traits
            assert 'growth_form' in traits
            assert 'photosynthesis_type' in traits
            assert 'nitrogen_fixation' in traits
            assert 'dispersal_mode' in traits
            assert 'flowering_start_month' in traits
            assert 'flowering_duration_months' in traits
            assert 'root_depth_cm' in traits
            assert 'specific_leaf_area' in traits
            
            assert isinstance(traits['height_cm'], float)
            assert traits['height_cm'] > 0
            assert isinstance(traits['nitrogen_fixation'], bool)
            assert 3 <= traits['flowering_start_month'] <= 7
            assert 1 <= traits['flowering_duration_months'] <= 4
    
    def test_create_trait_matrix_success(self, analyzer, sample_traits):
        """Test successful trait matrix creation."""
        analyzer.load_traits_from_dict(sample_traits)
        
        species_list = list(sample_traits.keys())
        trait_matrix, trait_names = analyzer._create_trait_matrix(species_list)
        
        assert trait_matrix.shape[0] == len(species_list)
        assert trait_matrix.shape[1] > 0
        assert len(trait_names) == trait_matrix.shape[1]
        assert isinstance(trait_names, list)
        
        assert not np.any(np.isnan(trait_matrix))
    
    def test_create_trait_matrix_with_subset(self, analyzer, sample_traits):
        """Test trait matrix creation with trait subset."""
        analyzer.load_traits_from_dict(sample_traits)
        
        species_list = list(sample_traits.keys())
        trait_subset = ['height_cm', 'leaf_area_cm2', 'seed_mass_mg']
        
        trait_matrix, trait_names = analyzer._create_trait_matrix(species_list, trait_subset)
        
        assert trait_matrix.shape[0] == len(species_list)
        assert len(trait_names) <= len(trait_subset)
        
        for trait_name in trait_names:
            assert trait_name in trait_subset
    
    def test_create_trait_matrix_empty_species(self, analyzer):
        """Test trait matrix creation with empty species list."""
        trait_matrix, trait_names = analyzer._create_trait_matrix([])
        
        assert trait_matrix.size == 0
        assert trait_names == []
    
    def test_functional_richness_calculation(self, analyzer):
        """Test functional richness calculation."""
        trait_matrix = np.array([
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0],
            [2.0, 2.0]
        ])
        
        fric = analyzer._calculate_functional_richness(trait_matrix)
        
        assert isinstance(fric, float)
        assert fric > 0
    
    def test_functional_richness_single_species(self, analyzer):
        """Test functional richness with single species."""
        trait_matrix = np.array([[1.0, 2.0]])
        
        fric = analyzer._calculate_functional_richness(trait_matrix)
        
        assert fric == 0.0
    
    def test_functional_evenness_calculation(self, analyzer):
        """Test functional evenness calculation."""
        trait_matrix = np.array([
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0]
        ])
        
        abundances = {'species1': 5.0, 'species2': 5.0, 'species3': 5.0}
        
        feve = analyzer._calculate_functional_evenness(trait_matrix, abundances)
        
        assert isinstance(feve, float)
        assert 0 <= feve <= 1
    
    def test_functional_divergence_calculation(self, analyzer):
        """Test functional divergence calculation."""
        trait_matrix = np.array([
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0]
        ])
        
        abundances = {'species1': 5.0, 'species2': 3.0, 'species3': 2.0}
        
        fdiv = analyzer._calculate_functional_divergence(trait_matrix, abundances)
        
        assert isinstance(fdiv, float)
        assert fdiv >= 0
    
    def test_functional_dispersion_calculation(self, analyzer):
        """Test functional dispersion calculation."""
        trait_matrix = np.array([
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0]
        ])
        
        abundances = {'species1': 5.0, 'species2': 3.0, 'species3': 2.0}
        
        fdis = analyzer._calculate_functional_dispersion(trait_matrix, abundances)
        
        assert isinstance(fdis, float)
        assert fdis >= 0
    
    def test_raos_quadratic_entropy_calculation(self, analyzer):
        """Test Rao's quadratic entropy calculation."""
        trait_matrix = np.array([
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0]
        ])
        
        abundances = {'species1': 5.0, 'species2': 3.0, 'species3': 2.0}
        
        raos_q = analyzer._calculate_raos_quadratic_entropy(trait_matrix, abundances)
        
        assert isinstance(raos_q, float)
        assert raos_q >= 0
    
    def test_identify_functional_groups(self, analyzer):
        """Test functional group identification."""
        trait_matrix = np.array([
            [1.0, 1.0],
            [1.1, 1.1],  # Similar to first
            [5.0, 5.0],
            [5.1, 5.1]   # Similar to third
        ])
        
        species_names = ['species1', 'species2', 'species3', 'species4']
        
        result = analyzer._identify_functional_groups(trait_matrix, species_names)
        
        assert 'functional_groups' in result
        assert 'number_of_groups' in result
        assert 'cluster_labels' in result
        
        functional_groups = result['functional_groups']
        assert len(functional_groups) > 0
        assert result['number_of_groups'] > 0
        
        all_assigned_species = []
        for group_species in functional_groups.values():
            all_assigned_species.extend(group_species)
        
        assert len(all_assigned_species) == len(species_names)
    
    def test_trait_summary_statistics(self, analyzer):
        """Test trait summary statistics calculation."""
        trait_matrix = np.array([
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [4.0, 8.0]
        ])
        
        trait_names = ['trait1', 'trait2']
        
        result = analyzer._calculate_trait_summary_stats(trait_matrix, trait_names)
        
        assert isinstance(result, dict)
        assert len(result) == 2
        
        for trait_name in trait_names:
            assert trait_name in result
            stats = result[trait_name]
            
            assert 'mean' in stats
            assert 'std' in stats
            assert 'min' in stats
            assert 'max' in stats
            assert 'median' in stats
            assert 'q25' in stats
            assert 'q75' in stats
            assert 'skewness' in stats
            assert 'kurtosis' in stats
    
    def test_skewness_calculation(self, analyzer):
        """Test skewness calculation."""
        symmetric_data = np.array([1, 2, 3, 4, 5])
        skewness = analyzer._calculate_skewness(symmetric_data)
        assert abs(skewness) < 0.1  # Should be close to 0
        
        right_skewed = np.array([1, 1, 1, 2, 5])
        skewness = analyzer._calculate_skewness(right_skewed)
        assert skewness > 0
    
    def test_kurtosis_calculation(self, analyzer):
        """Test kurtosis calculation."""
        normal_data = np.array([1, 2, 3, 4, 5])
        kurtosis = analyzer._calculate_kurtosis(normal_data)
        assert isinstance(kurtosis, float)
        
        constant_data = np.array([5, 5, 5, 5, 5])
        kurtosis = analyzer._calculate_kurtosis(constant_data)
        assert kurtosis == 0.0
    
    def test_functional_beta_diversity_calculation(self, analyzer, sample_traits):
        """Test functional beta diversity calculation."""
        analyzer.load_traits_from_dict(sample_traits)
        
        community1 = {
            "Taraxacum officinale": 10.0,
            "Plantago major": 8.0
        }
        
        community2 = {
            "Trifolium repens": 6.0,
            "Poa annua": 4.0
        }
        
        result = analyzer._calculate_functional_beta_diversity(community1, community2)
        
        assert 'functional_beta_total' in result
        assert 'functional_turnover' in result
        assert 'functional_nestedness' in result
        assert 'community1_functional_richness' in result
        assert 'community2_functional_richness' in result
        assert 'shared_functional_space' in result
        assert 'species_analyzed' in result
        
        assert 0 <= result['functional_beta_total'] <= 1
        assert 0 <= result['functional_turnover'] <= 1
        assert 0 <= result['functional_nestedness'] <= 1
        assert result['community1_functional_richness'] >= 0
        assert result['community2_functional_richness'] >= 0
        assert result['shared_functional_space'] >= 0
    
    def test_functional_beta_diversity_insufficient_data(self, analyzer):
        """Test functional beta diversity with insufficient data."""
        community1 = {"Species1": 5.0}
        community2 = {"Species2": 3.0}
        
        result = analyzer._calculate_functional_beta_diversity(community1, community2)
        
        assert 'error' in result
        assert result['error'] == 'insufficient_species_with_traits'
