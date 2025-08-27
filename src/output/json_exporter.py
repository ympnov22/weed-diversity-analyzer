"""JSON output exporter for daily diversity summaries."""

import json
# import numpy as np  # Removed for minimal deployment
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import asdict

from ..utils.logger import LoggerMixin
from ..utils.data_structures import DiversityMetrics, PredictionResult
from ..analysis.diversity_calculator import DiversityCalculator


class JSONExporter(LoggerMixin):
    """Export diversity analysis results to JSON format."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize JSON exporter.
        
        Args:
            output_dir: Directory to save JSON files
        """
        self.output_dir = output_dir or Path("output/json")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_daily_summary(
        self,
        date_str: str,
        diversity_metrics: DiversityMetrics,
        prediction_results: List[PredictionResult],
        processing_metadata: Dict[str, Any],
        confidence_intervals: Optional[Dict[str, tuple[float, float]]] = None
    ) -> Dict[str, Any]:
        """Export daily diversity summary to JSON.
        
        Args:
            date_str: Date in YYYY-MM-DD format
            diversity_metrics: Calculated diversity metrics
            prediction_results: List of prediction results
            processing_metadata: Processing information and metadata
            confidence_intervals: Bootstrap confidence intervals
            
        Returns:
            Dictionary containing the daily summary
        """
        try:
            top_species = self._extract_top_species(prediction_results)
            
            processing_stats = self._calculate_processing_stats(prediction_results, processing_metadata)
            
            summary = {
                "date": date_str,
                "timestamp": datetime.now().isoformat(),
                "diversity_metrics": self._serialize_diversity_metrics(diversity_metrics),
                "confidence_intervals": confidence_intervals or {},
                "top_species": top_species,
                "processing_info": processing_stats,
                "model_info": processing_metadata.get("model_info", {}),
                "quality_assessment": processing_metadata.get("quality_stats", {}),
                "clustering_info": processing_metadata.get("clustering_stats", {}),
                "metadata": {
                    "total_images_processed": len(prediction_results),
                    "inatag_species_count": 2959,
                    "analysis_version": "1.0.0",
                    "export_timestamp": datetime.now().isoformat()
                }
            }
            
            output_file = self.output_dir / f"daily_summary_{date_str}.json"
            self._save_json(summary, output_file)
            
            self.logger.info(f"Exported daily summary for {date_str} to {output_file}")
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to export daily summary for {date_str}: {e}")
            raise
    
    def export_time_series_data(
        self,
        daily_summaries: List[Dict[str, Any]],
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """Export time series data for visualization.
        
        Args:
            daily_summaries: List of daily summary dictionaries
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Time series data dictionary
        """
        try:
            time_series = {
                "period": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "total_days": len(daily_summaries)
                },
                "diversity_trends": self._extract_diversity_trends(daily_summaries),
                "species_trends": self._extract_species_trends(daily_summaries),
                "processing_trends": self._extract_processing_trends(daily_summaries),
                "summary_statistics": self._calculate_period_statistics(daily_summaries),
                "metadata": {
                    "export_timestamp": datetime.now().isoformat(),
                    "analysis_version": "1.0.0"
                }
            }
            
            output_file = self.output_dir / f"time_series_{start_date}_to_{end_date}.json"
            self._save_json(time_series, output_file)
            
            self.logger.info(f"Exported time series data to {output_file}")
            return time_series
            
        except Exception as e:
            self.logger.error(f"Failed to export time series data: {e}")
            raise
    
    def export_github_calendar_data(
        self,
        daily_summaries: List[Dict[str, Any]],
        metric_name: str = "shannon_diversity"
    ) -> Dict[str, Any]:
        """Export data for GitHub grass-style calendar visualization.
        
        Args:
            daily_summaries: List of daily summary dictionaries
            metric_name: Diversity metric to use for calendar coloring
            
        Returns:
            Calendar data dictionary
        """
        try:
            calendar_data = {
                "metric": metric_name,
                "data": [],
                "scale": self._calculate_metric_scale(daily_summaries, metric_name),
                "metadata": {
                    "total_days": len(daily_summaries),
                    "metric_description": self._get_metric_description(metric_name),
                    "export_timestamp": datetime.now().isoformat()
                }
            }
            
            for summary in daily_summaries:
                diversity_metrics = summary.get("diversity_metrics", {})
                metric_value = diversity_metrics.get(metric_name, 0.0)
                
                calendar_entry = {
                    "date": summary["date"],
                    "value": float(metric_value),
                    "level": self._calculate_intensity_level(float(metric_value), calendar_data["scale"]),
                    "species_count": diversity_metrics.get("species_richness", 0),
                    "total_images": summary.get("metadata", {}).get("total_images_processed", 0)
                }
                calendar_data["data"].append(calendar_entry)
            
            output_file = self.output_dir / f"github_calendar_{metric_name}.json"
            self._save_json(calendar_data, output_file)
            
            self.logger.info(f"Exported GitHub calendar data to {output_file}")
            return calendar_data
            
        except Exception as e:
            self.logger.error(f"Failed to export GitHub calendar data: {e}")
            raise
    
    def _extract_top_species(self, prediction_results: List[PredictionResult], top_k: int = 10) -> List[Dict[str, Any]]:
        """Extract top species from prediction results."""
        species_counts = {}
        species_confidences: Dict[str, List[float]] = {}
        
        for result in prediction_results:
            for prediction in result.predictions:
                species_name = prediction.species_name
                confidence = prediction.confidence
                
                if species_name not in species_counts:
                    species_counts[species_name] = 0
                    species_confidences[species_name] = []
                
                species_counts[species_name] += 1
                species_confidences[species_name].append(confidence)
        
        top_species = []
        for species, count in sorted(species_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            avg_confidence = sum(species_confidences[species]) / len(species_confidences[species])
            top_species.append({
                "species_name": species,
                "count": count,
                "frequency": count / len(prediction_results) if prediction_results else 0,
                "average_confidence": float(avg_confidence)
            })
        
        return top_species
    
    def _calculate_processing_stats(
        self, 
        prediction_results: List[PredictionResult], 
        processing_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate processing statistics."""
        if not prediction_results:
            return {}
        
        processing_times = [result.processing_time for result in prediction_results]
        confidences = []
        for result in prediction_results:
            for prediction in result.predictions:
                confidences.append(prediction.confidence)
        
        import math
        stats = {
            "total_processing_time": float(sum(processing_times)),
            "average_processing_time": float(sum(processing_times) / len(processing_times)) if processing_times else 0.0,
            "min_processing_time": float(min(processing_times)) if processing_times else 0.0,
            "max_processing_time": float(max(processing_times)) if processing_times else 0.0,
            "average_confidence": float(sum(confidences) / len(confidences)) if confidences else 0.0,
            "confidence_std": float(math.sqrt(sum((x - sum(confidences)/len(confidences))**2 for x in confidences) / len(confidences))) if len(confidences) > 1 else 0.0,
            "total_predictions": len(prediction_results),
            "successful_predictions": sum(1 for r in prediction_results if r.predictions)
        }
        
        stats.update(processing_metadata.get("processing_stats", {}))
        
        return stats
    
    def _serialize_diversity_metrics(self, metrics: DiversityMetrics) -> Dict[str, Any]:
        """Serialize diversity metrics to JSON-compatible format."""
        if isinstance(metrics, DiversityMetrics):
            return asdict(metrics)
        elif isinstance(metrics, dict):
            return {k: float(v) if isinstance(v, (int, float)) else v 
                   for k, v in metrics.items()}
        else:
            return {}
    
    def _extract_diversity_trends(self, daily_summaries: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        """Extract diversity metric trends over time."""
        trends: Dict[str, List[Any]] = {
            "dates": [],
            "species_richness": [],
            "shannon_diversity": [],
            "pielou_evenness": [],
            "simpson_diversity": []
        }
        
        for summary in daily_summaries:
            trends["dates"].append(summary["date"])
            diversity_metrics = summary.get("diversity_metrics", {})
            
            trends["species_richness"].append(diversity_metrics.get("species_richness", 0))
            trends["shannon_diversity"].append(diversity_metrics.get("shannon_diversity", 0))
            trends["pielou_evenness"].append(diversity_metrics.get("pielou_evenness", 0))
            trends["simpson_diversity"].append(diversity_metrics.get("simpson_diversity", 0))
        
        return trends
    
    def _extract_species_trends(self, daily_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract species-related trends."""
        all_species = set()
        daily_species = []
        
        for summary in daily_summaries:
            top_species = summary.get("top_species", [])
            day_species = set(species["species_name"] for species in top_species)
            daily_species.append(day_species)
            all_species.update(day_species)
        
        return {
            "total_unique_species": len(all_species),
            "species_list": sorted(list(all_species)),
            "daily_species_counts": [len(day_species) for day_species in daily_species]
        }
    
    def _extract_processing_trends(self, daily_summaries: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        """Extract processing-related trends."""
        trends: Dict[str, List[Any]] = {
            "dates": [],
            "processing_times": [],
            "image_counts": [],
            "average_confidences": []
        }
        
        for summary in daily_summaries:
            trends["dates"].append(summary["date"])
            processing_info = summary.get("processing_info", {})
            
            trends["processing_times"].append(processing_info.get("total_processing_time", 0))
            trends["image_counts"].append(summary.get("metadata", {}).get("total_images_processed", 0))
            trends["average_confidences"].append(processing_info.get("average_confidence", 0))
        
        return trends
    
    def _calculate_period_statistics(self, daily_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics for the entire period."""
        if not daily_summaries:
            return {}
        
        shannon_values = []
        richness_values = []
        
        for summary in daily_summaries:
            diversity_metrics = summary.get("diversity_metrics", {})
            shannon_values.append(diversity_metrics.get("shannon_diversity", 0))
            richness_values.append(diversity_metrics.get("species_richness", 0))
        
        import math
        shannon_mean = sum(shannon_values) / len(shannon_values) if shannon_values else 0.0
        richness_mean = sum(richness_values) / len(richness_values) if richness_values else 0.0
        
        return {
            "shannon_diversity": {
                "mean": float(shannon_mean),
                "std": float(math.sqrt(sum((x - shannon_mean)**2 for x in shannon_values) / len(shannon_values))) if len(shannon_values) > 1 else 0.0,
                "min": float(min(shannon_values)) if shannon_values else 0.0,
                "max": float(max(shannon_values)) if shannon_values else 0.0
            },
            "species_richness": {
                "mean": float(richness_mean),
                "std": float(math.sqrt(sum((x - richness_mean)**2 for x in richness_values) / len(richness_values))) if len(richness_values) > 1 else 0.0,
                "min": float(min(richness_values)) if richness_values else 0.0,
                "max": float(max(richness_values)) if richness_values else 0.0
            },
            "total_days_analyzed": len(daily_summaries)
        }
    
    def _calculate_metric_scale(self, daily_summaries: List[Dict[str, Any]], metric_name: str) -> Dict[str, float]:
        """Calculate scale for metric visualization."""
        values = []
        for summary in daily_summaries:
            diversity_metrics = summary.get("diversity_metrics", {})
            values.append(diversity_metrics.get(metric_name, 0.0))
        
        if not values:
            return {"min": 0.0, "max": 1.0, "q25": 0.0, "q50": 0.5, "q75": 1.0}
        
        return {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "q25": float(np.percentile(values, 25)),
            "q50": float(np.percentile(values, 50)),
            "q75": float(np.percentile(values, 75))
        }
    
    def _calculate_intensity_level(self, value: float, scale: Dict[str, float]) -> int:
        """Calculate intensity level (0-4) for GitHub-style visualization."""
        if value <= scale["q25"]:
            return 1
        elif value <= scale["q50"]:
            return 2
        elif value <= scale["q75"]:
            return 3
        else:
            return 4
    
    def _get_metric_description(self, metric_name: str) -> str:
        """Get description for diversity metric."""
        descriptions = {
            "shannon_diversity": "Shannon diversity index (H') - measures species diversity",
            "species_richness": "Species richness (R) - total number of species",
            "pielou_evenness": "Pielou evenness (J) - measures species distribution uniformity",
            "simpson_diversity": "Simpson diversity - measures dominance and diversity"
        }
        return descriptions.get(metric_name, f"Diversity metric: {metric_name}")
    
    def _save_json(self, data: Dict[str, Any], file_path: Path) -> None:
        """Save data to JSON file with proper formatting."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=self._json_serializer)
    
    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for numpy types and other objects."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (date, datetime)):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
