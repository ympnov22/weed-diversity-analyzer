"""CSV output exporter for detailed prediction results."""

import csv
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from ..utils.logger import LoggerMixin
from ..utils.data_structures import PredictionResult, SpeciesPrediction


class CSVExporter(LoggerMixin):
    """Export detailed prediction results to CSV format."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize CSV exporter.
        
        Args:
            output_dir: Directory to save CSV files
        """
        self.output_dir = output_dir or Path("output/csv")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_daily_predictions(
        self,
        date_str: str,
        prediction_results: List[PredictionResult],
        processing_metadata: Dict[str, Any]
    ) -> Path:
        """Export daily prediction results to CSV.
        
        Args:
            date_str: Date in YYYY-MM-DD format
            prediction_results: List of prediction results
            processing_metadata: Processing information and metadata
            
        Returns:
            Path to the exported CSV file
        """
        try:
            base_columns = [
                'date', 'image_id', 'image_path', 'processing_time', 'timestamp',
                'model_name', 'total_predictions', 'model_size', 'lora_enabled', 'device_used'
            ]
            
            for i in range(1, 4):
                base_columns.extend([
                    f'species_name_{i}', f'confidence_{i}', f'inatag_id_{i}', f'taxonomic_level_{i}'
                ])
            
            base_columns.extend([
                'top_confidence', 'mean_confidence', 'confidence_std', 'prediction_entropy'
            ])
            
            rows = []
            
            for result_idx, result in enumerate(prediction_results):
                base_row = {
                    'date': date_str,
                    'image_id': result_idx + 1,
                    'image_path': getattr(result, 'image_path', f'image_{result_idx + 1}'),
                    'processing_time': result.processing_time,
                    'timestamp': getattr(result, 'timestamp', datetime.now().isoformat()),
                    'model_name': getattr(result, 'model_name', 'inatag'),
                    'total_predictions': len(result.predictions)
                }
                
                model_info = getattr(result, 'model_info', {})
                base_row.update({
                    'model_size': model_info.get('size', 'unknown'),
                    'lora_enabled': model_info.get('lora_enabled', False),
                    'device_used': model_info.get('device', 'unknown')
                })
                
                for i in range(3):
                    if i < len(result.predictions):
                        prediction = result.predictions[i]
                        base_row.update({
                            f'species_name_{i+1}': prediction.species_name,
                            f'confidence_{i+1}': prediction.confidence,
                            f'inatag_id_{i+1}': getattr(prediction, 'species_id', ''),
                            f'taxonomic_level_{i+1}': getattr(prediction, 'taxonomic_level', 'species')
                        })
                    else:
                        base_row.update({
                            f'species_name_{i+1}': '',
                            f'confidence_{i+1}': 0.0,
                            f'inatag_id_{i+1}': '',
                            f'taxonomic_level_{i+1}': ''
                        })
                
                if result.predictions:
                    confidences = [p.confidence for p in result.predictions]
                    base_row.update({
                        'top_confidence': max(confidences),
                        'mean_confidence': float(np.mean(confidences)),
                        'confidence_std': float(np.std(confidences)),
                        'prediction_entropy': self._calculate_entropy(confidences)
                    })
                else:
                    base_row.update({
                        'top_confidence': 0.0,
                        'mean_confidence': 0.0,
                        'confidence_std': 0.0,
                        'prediction_entropy': 0.0
                    })
                
                rows.append(base_row)
            
            if rows:
                df = pd.DataFrame(rows)
                string_columns = [col for col in df.columns if 'species_name' in col or 'inatag_id' in col or 'taxonomic_level' in col]
                df[string_columns] = df[string_columns].fillna('')
            else:
                df = pd.DataFrame(columns=base_columns)
            
            output_file = self.output_dir / f"daily_predictions_{date_str}.csv"
            df.to_csv(output_file, index=False, encoding='utf-8')
            
            self.logger.info(f"Exported {len(rows)} prediction records to {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Failed to export daily predictions for {date_str}: {e}")
            raise
    
    def export_aggregated_predictions(
        self,
        daily_results: Dict[str, List[PredictionResult]],
        start_date: str,
        end_date: str
    ) -> Path:
        """Export aggregated prediction results across multiple days.
        
        Args:
            daily_results: Dictionary mapping dates to prediction results
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Path to the exported CSV file
        """
        try:
            all_rows = []
            
            for date_str, prediction_results in daily_results.items():
                for result_idx, result in enumerate(prediction_results):
                    base_row = {
                        'date': date_str,
                        'image_id': f"{date_str}_{result_idx + 1}",
                        'image_path': getattr(result, 'image_path', f'image_{result_idx + 1}'),
                        'processing_time': result.processing_time
                    }
                    
                    for i, prediction in enumerate(result.predictions[:3]):
                        base_row.update({
                            f'species_name_{i+1}': prediction.species_name,
                            f'confidence_{i+1}': prediction.confidence
                        })
                    
                    all_rows.append(base_row)
            
            df = pd.DataFrame(all_rows)
            output_file = self.output_dir / f"aggregated_predictions_{start_date}_to_{end_date}.csv"
            df.to_csv(output_file, index=False, encoding='utf-8')
            
            self.logger.info(f"Exported {len(all_rows)} aggregated records to {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Failed to export aggregated predictions: {e}")
            raise
    
    def export_species_summary(
        self,
        daily_results: Dict[str, List[PredictionResult]],
        output_filename: Optional[str] = None
    ) -> Path:
        """Export species occurrence summary across all days.
        
        Args:
            daily_results: Dictionary mapping dates to prediction results
            output_filename: Custom filename for output
            
        Returns:
            Path to the exported CSV file
        """
        try:
            species_stats: Dict[str, Dict[str, Any]] = {}
            
            for date_str, prediction_results in daily_results.items():
                for result in prediction_results:
                    for prediction in result.predictions:
                        species_name = prediction.species_name
                        confidence = prediction.confidence
                        
                        if species_name not in species_stats:
                            species_stats[species_name] = {
                                'total_occurrences': 0,
                                'days_observed': set(),
                                'confidences': [],
                                'first_observed': date_str,
                                'last_observed': date_str
                            }
                        
                        stats = species_stats[species_name]
                        stats['total_occurrences'] += 1
                        stats['days_observed'].add(date_str)
                        stats['confidences'].append(confidence)
                        
                        if date_str < stats['first_observed']:
                            stats['first_observed'] = date_str
                        if date_str > stats['last_observed']:
                            stats['last_observed'] = date_str
            
            summary_rows = []
            for species_name, stats in species_stats.items():
                confidences = stats['confidences']
                summary_rows.append({
                    'species_name': species_name,
                    'total_occurrences': stats['total_occurrences'],
                    'days_observed': len(stats['days_observed']),
                    'observation_frequency': len(stats['days_observed']) / len(daily_results),
                    'average_confidence': float(np.mean(confidences)),
                    'confidence_std': float(np.std(confidences)),
                    'min_confidence': float(np.min(confidences)),
                    'max_confidence': float(np.max(confidences)),
                    'first_observed': stats['first_observed'],
                    'last_observed': stats['last_observed']
                })
            
            summary_rows.sort(key=lambda x: x['total_occurrences'], reverse=True)
            
            df = pd.DataFrame(summary_rows)
            
            if output_filename is None:
                output_filename = f"species_summary_{min(daily_results.keys())}_to_{max(daily_results.keys())}.csv"
            
            output_file = self.output_dir / output_filename
            df.to_csv(output_file, index=False, encoding='utf-8')
            
            self.logger.info(f"Exported species summary with {len(summary_rows)} species to {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Failed to export species summary: {e}")
            raise
    
    def export_soft_voting_results(
        self,
        date_str: str,
        soft_voting_results: Dict[str, Any],
        comparison_results: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Export soft voting aggregation results.
        
        Args:
            date_str: Date in YYYY-MM-DD format
            soft_voting_results: Results from soft voting system
            comparison_results: Comparison with hard voting (optional)
            
        Returns:
            Path to the exported CSV file
        """
        try:
            rows = []
            
            species_weights = soft_voting_results.get('species_weights', {})
            diversity_metrics = soft_voting_results.get('diversity_metrics', {})
            
            for species_name, weight in species_weights.items():
                row = {
                    'date': date_str,
                    'species_name': species_name,
                    'soft_voting_weight': weight,
                    'normalized_weight': weight / sum(species_weights.values()) if species_weights else 0,
                    'taxonomic_level': soft_voting_results.get('taxonomic_levels', {}).get(species_name, 'species')
                }
                
                if comparison_results:
                    hard_weights = comparison_results.get('hard_voting_weights', {})
                    row['hard_voting_weight'] = hard_weights.get(species_name, 0)
                    row['weight_difference'] = weight - hard_weights.get(species_name, 0)
                
                rows.append(row)
            
            rows.sort(key=lambda x: x['soft_voting_weight'], reverse=True)
            
            if rows:
                summary_row = {
                    'date': date_str,
                    'species_name': 'SUMMARY',
                    'soft_voting_weight': sum(species_weights.values()),
                    'normalized_weight': 1.0,
                    'taxonomic_level': 'summary'
                }
                
                for metric_name, value in diversity_metrics.items():
                    summary_row[f'diversity_{metric_name}'] = value
                
                if comparison_results:
                    summary_row['agreement_rate'] = comparison_results.get('agreement_rate', 0)
                
                rows.append(summary_row)
            
            df = pd.DataFrame(rows)
            output_file = self.output_dir / f"soft_voting_{date_str}.csv"
            df.to_csv(output_file, index=False, encoding='utf-8')
            
            self.logger.info(f"Exported soft voting results to {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Failed to export soft voting results for {date_str}: {e}")
            raise
    
    def _calculate_entropy(self, confidences: List[float]) -> float:
        """Calculate prediction entropy from confidence scores."""
        if not confidences:
            return 0.0
        
        total = sum(confidences)
        if total == 0:
            return 0.0
        
        probabilities = [c / total for c in confidences]
        
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
