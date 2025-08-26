"""Soft voting system for Top-3 species predictions."""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass

from ..utils.logger import LoggerMixin
from ..utils.data_structures import SpeciesPrediction, PredictionResult


@dataclass
class SoftVotingConfig:
    """Configuration for soft voting system."""
    
    confidence_threshold: float = 0.3
    taxonomic_rollup_threshold: float = 0.2
    top_k: int = 3
    weight_by_confidence: bool = True
    normalize_weights: bool = True


class TaxonomicRollup:
    """Handle taxonomic rollup for low confidence predictions."""
    
    def __init__(self) -> None:
        """Initialize taxonomic rollup system."""
        self.species_to_genus: Dict[str, str] = {}
        self.species_to_family: Dict[str, str] = {}
        self.genus_to_family: Dict[str, str] = {}
    
    def add_taxonomic_mapping(
        self, 
        species: str, 
        genus: Optional[str] = None, 
        family: Optional[str] = None
    ) -> None:
        """Add taxonomic mapping for a species.
        
        Args:
            species: Species name
            genus: Genus name
            family: Family name
        """
        if genus:
            self.species_to_genus[species] = genus
            if family:
                self.genus_to_family[genus] = family
        
        if family:
            self.species_to_family[species] = family
    
    def rollup_prediction(
        self, 
        prediction: SpeciesPrediction, 
        threshold: float
    ) -> SpeciesPrediction:
        """Roll up prediction to higher taxonomic level if confidence is low.
        
        Args:
            prediction: Original species prediction
            threshold: Confidence threshold for rollup
            
        Returns:
            Rolled up prediction
        """
        if prediction.confidence >= threshold:
            return prediction
        
        species = prediction.species_name
        
        if species in self.species_to_genus:
            genus = self.species_to_genus[species]
            return SpeciesPrediction(
                species_name=genus,
                confidence=prediction.confidence,
                taxonomic_level='genus',
                scientific_name=genus,
                common_name=f"Genus {genus}"
            )
        
        if species in self.species_to_family:
            family = self.species_to_family[species]
            return SpeciesPrediction(
                species_name=family,
                confidence=prediction.confidence,
                taxonomic_level='family',
                scientific_name=family,
                common_name=f"Family {family}"
            )
        
        return prediction


class SoftVotingSystem(LoggerMixin):
    """Soft voting system for aggregating Top-3 predictions."""
    
    def __init__(self, config: Optional[SoftVotingConfig] = None):
        """Initialize soft voting system.
        
        Args:
            config: Configuration for soft voting
        """
        self.config = config or SoftVotingConfig()
        self.taxonomic_rollup = TaxonomicRollup()
    
    def aggregate_predictions(
        self, 
        prediction_results: List[PredictionResult]
    ) -> Dict[str, Any]:
        """Aggregate multiple prediction results using soft voting.
        
        Args:
            prediction_results: List of prediction results to aggregate
            
        Returns:
            Dictionary with aggregated results and diversity metrics
        """
        if not prediction_results:
            return self._empty_aggregation()
        
        weighted_predictions = []
        total_weight = 0.0
        
        for result in prediction_results:
            for prediction in result.predictions[:self.config.top_k]:
                rolled_prediction = self.taxonomic_rollup.rollup_prediction(
                    prediction, 
                    self.config.taxonomic_rollup_threshold
                )
                
                if self.config.weight_by_confidence:
                    weight = rolled_prediction.confidence
                else:
                    weight = 1.0
                
                weighted_predictions.append((rolled_prediction, weight))
                total_weight += weight
        
        if total_weight == 0:
            return self._empty_aggregation()
        
        if self.config.normalize_weights:
            weighted_predictions = [
                (pred, weight / total_weight) 
                for pred, weight in weighted_predictions
            ]
        
        species_weights: Dict[str, float] = defaultdict(float)
        species_info: Dict[str, Dict[str, Any]] = {}
        
        for prediction, weight in weighted_predictions:
            species_name = prediction.species_name
            species_weights[species_name] += weight
            
            if species_name not in species_info:
                species_info[species_name] = {
                    'taxonomic_level': prediction.taxonomic_level,
                    'scientific_name': prediction.scientific_name,
                    'common_name': prediction.common_name,
                    'total_weight': 0.0,
                    'prediction_count': 0
                }
            
            species_info[species_name]['total_weight'] = float(species_info[species_name]['total_weight']) + weight
            species_info[species_name]['prediction_count'] = int(species_info[species_name]['prediction_count']) + 1
        
        sorted_species = sorted(
            species_weights.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        aggregated_predictions = []
        for species_name, total_weight in sorted_species:
            info = species_info[species_name]
            
            prediction_count = int(info['prediction_count'])
            avg_confidence = total_weight / prediction_count if prediction_count > 0 else 0.0
            
            aggregated_pred = SpeciesPrediction(
                species_name=species_name,
                confidence=float(avg_confidence),
                taxonomic_level=str(info['taxonomic_level']),
                scientific_name=str(info['scientific_name']) if info['scientific_name'] is not None else None,
                common_name=str(info['common_name']) if info['common_name'] is not None else None
            )
            aggregated_predictions.append(aggregated_pred)
        
        diversity_metrics = self._calculate_soft_voting_diversity(species_weights)
        
        return {
            'aggregated_predictions': aggregated_predictions,
            'species_weights': dict(species_weights),
            'diversity_metrics': diversity_metrics,
            'total_predictions': len(weighted_predictions),
            'unique_species': len(species_weights),
            'aggregation_method': 'soft_voting',
            'config': {
                'confidence_threshold': self.config.confidence_threshold,
                'taxonomic_rollup_threshold': self.config.taxonomic_rollup_threshold,
                'top_k': self.config.top_k,
                'weight_by_confidence': self.config.weight_by_confidence
            }
        }
    
    def compare_with_hard_voting(
        self, 
        prediction_results: List[PredictionResult]
    ) -> Dict[str, Any]:
        """Compare soft voting results with hard voting.
        
        Args:
            prediction_results: List of prediction results
            
        Returns:
            Dictionary with comparison results
        """
        soft_results = self.aggregate_predictions(prediction_results)
        
        hard_votes: Counter[str] = Counter()
        for result in prediction_results:
            if result.predictions:
                top_prediction = result.predictions[0]
                rolled_prediction = self.taxonomic_rollup.rollup_prediction(
                    top_prediction,
                    self.config.taxonomic_rollup_threshold
                )
                hard_votes[rolled_prediction.species_name] += 1
        
        hard_predictions = []
        total_votes = sum(hard_votes.values())
        
        for species, votes in hard_votes.most_common():
            confidence = votes / total_votes if total_votes > 0 else 0.0
            hard_predictions.append(SpeciesPrediction(
                species_name=species,
                confidence=confidence,
                taxonomic_level='species',
                scientific_name=species,
                common_name=""
            ))
        
        return {
            'soft_voting': soft_results,
            'hard_voting': {
                'predictions': hard_predictions,
                'vote_counts': dict(hard_votes),
                'total_votes': total_votes
            },
            'comparison': {
                'soft_unique_species': soft_results['unique_species'],
                'hard_unique_species': len(hard_votes),
                'agreement_rate': self._calculate_agreement_rate(
                    soft_results['aggregated_predictions'],
                    hard_predictions
                )
            }
        }
    
    def _calculate_soft_voting_diversity(
        self, 
        species_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate diversity metrics from soft voting weights.
        
        Args:
            species_weights: Dictionary mapping species to weights
            
        Returns:
            Dictionary with diversity metrics
        """
        if not species_weights:
            return {}
        
        weights = np.array(list(species_weights.values()))
        total_weight = np.sum(weights)
        
        if total_weight == 0:
            return {}
        
        proportions = weights / total_weight
        
        richness = len(species_weights)
        
        shannon = -np.sum(proportions * np.log(proportions + 1e-10))
        
        if richness > 1:
            max_shannon = np.log(richness)
            pielou = shannon / max_shannon
        else:
            pielou = 1.0
        
        simpson = 1.0 - np.sum(proportions ** 2)
        
        return {
            'species_richness': float(richness),
            'shannon_diversity': float(shannon),
            'pielou_evenness': float(pielou),
            'simpson_diversity': float(simpson),
            'total_weight': float(total_weight)
        }
    
    def _calculate_agreement_rate(
        self, 
        soft_predictions: List[SpeciesPrediction],
        hard_predictions: List[SpeciesPrediction]
    ) -> float:
        """Calculate agreement rate between soft and hard voting.
        
        Args:
            soft_predictions: Soft voting predictions
            hard_predictions: Hard voting predictions
            
        Returns:
            Agreement rate (0-1)
        """
        if not soft_predictions or not hard_predictions:
            return 0.0
        
        soft_species = {pred.species_name for pred in soft_predictions[:5]}
        hard_species = {pred.species_name for pred in hard_predictions[:5]}
        
        intersection = len(soft_species & hard_species)
        union = len(soft_species | hard_species)
        
        return intersection / union if union > 0 else 0.0
    
    def _empty_aggregation(self) -> Dict[str, Any]:
        """Return empty aggregation result."""
        return {
            'aggregated_predictions': [],
            'species_weights': {},
            'diversity_metrics': {},
            'total_predictions': 0,
            'unique_species': 0,
            'aggregation_method': 'soft_voting'
        }
