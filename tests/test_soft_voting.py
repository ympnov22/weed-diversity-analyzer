"""Tests for soft voting system implementation."""

import pytest
import numpy as np
from unittest.mock import Mock

from src.analysis.soft_voting import SoftVotingSystem, SoftVotingConfig, TaxonomicRollup
from src.utils.data_structures import SpeciesPrediction
from src.models.base_model import PredictionResult


class TestTaxonomicRollup:
    """Test cases for taxonomic rollup."""
    
    @pytest.fixture
    def taxonomic_rollup(self):
        """Create taxonomic rollup instance."""
        rollup = TaxonomicRollup()
        rollup.add_taxonomic_mapping('species_a', 'genus_a', 'family_1')
        rollup.add_taxonomic_mapping('species_b', 'genus_a', 'family_1')
        rollup.add_taxonomic_mapping('species_c', 'genus_b', 'family_2')
        return rollup
    
    def test_add_taxonomic_mapping(self, taxonomic_rollup):
        """Test adding taxonomic mappings."""
        assert 'species_a' in taxonomic_rollup.species_to_genus
        assert taxonomic_rollup.species_to_genus['species_a'] == 'genus_a'
        assert taxonomic_rollup.species_to_family['species_a'] == 'family_1'
        assert taxonomic_rollup.genus_to_family['genus_a'] == 'family_1'
    
    def test_rollup_high_confidence(self, taxonomic_rollup):
        """Test rollup with high confidence prediction."""
        prediction = SpeciesPrediction(
            species_name='species_a',
            confidence=0.8,
            taxonomic_level='species',
            scientific_name='Species A',
            common_name='Common A'
        )
        
        rolled = taxonomic_rollup.rollup_prediction(prediction, 0.5)
        
        assert rolled.species_name == 'species_a'
        assert rolled.taxonomic_level == 'species'
        assert rolled.confidence == 0.8
    
    def test_rollup_low_confidence_to_genus(self, taxonomic_rollup):
        """Test rollup with low confidence to genus level."""
        prediction = SpeciesPrediction(
            species_name='species_a',
            confidence=0.3,
            taxonomic_level='species',
            scientific_name='Species A',
            common_name='Common A'
        )
        
        rolled = taxonomic_rollup.rollup_prediction(prediction, 0.5)
        
        assert rolled.species_name == 'genus_a'
        assert rolled.taxonomic_level == 'genus'
        assert rolled.confidence == 0.3
        assert 'Genus genus_a' in rolled.common_name
    
    def test_rollup_no_mapping(self, taxonomic_rollup):
        """Test rollup with species not in mapping."""
        prediction = SpeciesPrediction(
            species_name='unknown_species',
            confidence=0.2,
            taxonomic_level='species',
            scientific_name='Unknown Species',
            common_name='Unknown'
        )
        
        rolled = taxonomic_rollup.rollup_prediction(prediction, 0.5)
        
        assert rolled.species_name == 'unknown_species'
        assert rolled.taxonomic_level == 'species'


class TestSoftVotingSystem:
    """Test cases for soft voting system."""
    
    @pytest.fixture
    def soft_voting_config(self):
        """Create soft voting configuration."""
        return SoftVotingConfig(
            confidence_threshold=0.3,
            taxonomic_rollup_threshold=0.2,
            top_k=3,
            weight_by_confidence=True,
            normalize_weights=True
        )
    
    @pytest.fixture
    def soft_voting_system(self, soft_voting_config):
        """Create soft voting system instance."""
        return SoftVotingSystem(soft_voting_config)
    
    @pytest.fixture
    def sample_prediction_results(self):
        """Create sample prediction results."""
        predictions1 = [
            SpeciesPrediction(
                species_name='species_a',
                confidence=0.8,
                taxonomic_level='species',
                scientific_name='Species A',
                common_name='Common A'
            ),
            SpeciesPrediction(
                species_name='species_b',
                confidence=0.6,
                taxonomic_level='species',
                scientific_name='Species B',
                common_name='Common B'
            ),
            SpeciesPrediction(
                species_name='species_c',
                confidence=0.4,
                taxonomic_level='species',
                scientific_name='Species C',
                common_name='Common C'
            )
        ]
        
        result1 = PredictionResult(
            predictions=predictions1,
            processing_time=0.1,
            model_info={'model': 'test'},
            confidence_scores=[0.8, 0.6, 0.4],
            raw_outputs=None
        )
        
        predictions2 = [
            SpeciesPrediction(
                species_name='species_a',
                confidence=0.7,
                taxonomic_level='species',
                scientific_name='Species A',
                common_name='Common A'
            ),
            SpeciesPrediction(
                species_name='species_d',
                confidence=0.5,
                taxonomic_level='species',
                scientific_name='Species D',
                common_name='Common D'
            ),
            SpeciesPrediction(
                species_name='species_b',
                confidence=0.3,
                taxonomic_level='species',
                scientific_name='Species B',
                common_name='Common B'
            )
        ]
        
        result2 = PredictionResult(
            predictions=predictions2,
            processing_time=0.1,
            model_info={'model': 'test'},
            confidence_scores=[0.7, 0.5, 0.3],
            raw_outputs=None
        )
        
        return [result1, result2]
    
    def test_initialization(self, soft_voting_system):
        """Test soft voting system initialization."""
        assert soft_voting_system.config.confidence_threshold == 0.3
        assert soft_voting_system.config.top_k == 3
        assert soft_voting_system.config.weight_by_confidence is True
        assert isinstance(soft_voting_system.taxonomic_rollup, TaxonomicRollup)
    
    def test_aggregate_predictions_success(self, soft_voting_system, sample_prediction_results):
        """Test successful prediction aggregation."""
        result = soft_voting_system.aggregate_predictions(sample_prediction_results)
        
        assert 'aggregated_predictions' in result
        assert 'species_weights' in result
        assert 'diversity_metrics' in result
        assert 'total_predictions' in result
        assert 'unique_species' in result
        
        assert len(result['aggregated_predictions']) > 0
        top_prediction = result['aggregated_predictions'][0]
        assert top_prediction.species_name == 'species_a'
        
        diversity = result['diversity_metrics']
        assert 'species_richness' in diversity
        assert 'shannon_diversity' in diversity
        assert diversity['species_richness'] > 0
    
    def test_aggregate_predictions_empty_input(self, soft_voting_system):
        """Test aggregation with empty input."""
        result = soft_voting_system.aggregate_predictions([])
        
        assert result['aggregated_predictions'] == []
        assert result['species_weights'] == {}
        assert result['diversity_metrics'] == {}
        assert result['total_predictions'] == 0
        assert result['unique_species'] == 0
    
    def test_aggregate_predictions_no_confidence_weighting(self, soft_voting_config):
        """Test aggregation without confidence weighting."""
        soft_voting_config.weight_by_confidence = False
        system = SoftVotingSystem(soft_voting_config)
        
        predictions = [
            SpeciesPrediction(
                species_name='species_a',
                confidence=0.9,
                taxonomic_level='species',
                scientific_name='Species A',
                common_name='Common A'
            )
        ]
        
        result1 = PredictionResult(
            predictions=predictions,
            processing_time=0.1,
            model_info={'model': 'test'},
            confidence_scores=[0.9],
            raw_outputs=None
        )
        
        result = system.aggregate_predictions([result1])
        
        assert len(result['aggregated_predictions']) == 1
        assert result['unique_species'] == 1
    
    def test_compare_with_hard_voting(self, soft_voting_system, sample_prediction_results):
        """Test comparison between soft and hard voting."""
        comparison = soft_voting_system.compare_with_hard_voting(sample_prediction_results)
        
        assert 'soft_voting' in comparison
        assert 'hard_voting' in comparison
        assert 'comparison' in comparison
        
        hard_voting = comparison['hard_voting']
        assert 'predictions' in hard_voting
        assert 'vote_counts' in hard_voting
        assert 'total_votes' in hard_voting
        
        comp_metrics = comparison['comparison']
        assert 'soft_unique_species' in comp_metrics
        assert 'hard_unique_species' in comp_metrics
        assert 'agreement_rate' in comp_metrics
        assert 0 <= comp_metrics['agreement_rate'] <= 1
    
    def test_soft_voting_diversity_calculation(self, soft_voting_system):
        """Test diversity calculation from soft voting weights."""
        species_weights = {
            'species_a': 0.4,
            'species_b': 0.3,
            'species_c': 0.2,
            'species_d': 0.1
        }
        
        diversity = soft_voting_system._calculate_soft_voting_diversity(species_weights)
        
        assert 'species_richness' in diversity
        assert 'shannon_diversity' in diversity
        assert 'pielou_evenness' in diversity
        assert 'simpson_diversity' in diversity
        
        assert diversity['species_richness'] == 4.0
        assert diversity['shannon_diversity'] > 0
        assert 0 <= diversity['pielou_evenness'] <= 1
        assert 0 <= diversity['simpson_diversity'] <= 1
    
    def test_soft_voting_diversity_empty_weights(self, soft_voting_system):
        """Test diversity calculation with empty weights."""
        diversity = soft_voting_system._calculate_soft_voting_diversity({})
        assert diversity == {}
    
    def test_agreement_rate_calculation(self, soft_voting_system):
        """Test agreement rate calculation between voting methods."""
        soft_predictions = [
            SpeciesPrediction(
                species_name='species_a',
                confidence=0.8,
                taxonomic_level='species',
                scientific_name='Species A',
                common_name='Common A'
            ),
            SpeciesPrediction(
                species_name='species_b',
                confidence=0.6,
                taxonomic_level='species',
                scientific_name='Species B',
                common_name='Common B'
            )
        ]
        
        hard_predictions = [
            SpeciesPrediction(
                species_name='species_a',
                confidence=0.7,
                taxonomic_level='species',
                scientific_name='Species A',
                common_name='Common A'
            ),
            SpeciesPrediction(
                species_name='species_c',
                confidence=0.5,
                taxonomic_level='species',
                scientific_name='Species C',
                common_name='Common C'
            )
        ]
        
        agreement_rate = soft_voting_system._calculate_agreement_rate(
            soft_predictions, 
            hard_predictions
        )
        
        assert 0 <= agreement_rate <= 1
        assert agreement_rate > 0  # At least species_a should match
    
    def test_agreement_rate_empty_predictions(self, soft_voting_system):
        """Test agreement rate with empty predictions."""
        agreement_rate = soft_voting_system._calculate_agreement_rate([], [])
        assert agreement_rate == 0.0
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = SoftVotingConfig()
        
        assert config.confidence_threshold == 0.3
        assert config.taxonomic_rollup_threshold == 0.2
        assert config.top_k == 3
        assert config.weight_by_confidence is True
        assert config.normalize_weights is True
