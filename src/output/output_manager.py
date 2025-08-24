"""Output manager for coordinating JSON and CSV exports."""

from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from ..utils.logger import LoggerMixin
from ..utils.data_structures import PredictionResult, DiversityMetrics
from .json_exporter import JSONExporter
from .csv_exporter import CSVExporter


class OutputManager(LoggerMixin):
    """Manage all output operations for the weed diversity analyzer."""
    
    def __init__(self, output_base_dir: Path = None):
        """Initialize output manager.
        
        Args:
            output_base_dir: Base directory for all outputs
        """
        self.output_base_dir = output_base_dir or Path("output")
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        self.json_exporter = JSONExporter(self.output_base_dir / "json")
        self.csv_exporter = CSVExporter(self.output_base_dir / "csv")
    
    def export_daily_analysis(
        self,
        date_str: str,
        diversity_metrics: DiversityMetrics,
        prediction_results: List[PredictionResult],
        processing_metadata: Dict[str, Any],
        confidence_intervals: Optional[Dict[str, tuple]] = None,
        soft_voting_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Path]:
        """Export complete daily analysis results.
        
        Args:
            date_str: Date in YYYY-MM-DD format
            diversity_metrics: Calculated diversity metrics
            prediction_results: List of prediction results
            processing_metadata: Processing information and metadata
            confidence_intervals: Bootstrap confidence intervals
            soft_voting_results: Soft voting aggregation results
            
        Returns:
            Dictionary mapping output types to file paths
        """
        try:
            output_files = {}
            
            json_summary = self.json_exporter.export_daily_summary(
                date_str=date_str,
                diversity_metrics=diversity_metrics,
                prediction_results=prediction_results,
                processing_metadata=processing_metadata,
                confidence_intervals=confidence_intervals
            )
            
            csv_file = self.csv_exporter.export_daily_predictions(
                date_str=date_str,
                prediction_results=prediction_results,
                processing_metadata=processing_metadata
            )
            
            output_files.update({
                'json_summary': self.json_exporter.output_dir / f"daily_summary_{date_str}.json",
                'csv_predictions': csv_file
            })
            
            if soft_voting_results:
                soft_voting_file = self.csv_exporter.export_soft_voting_results(
                    date_str=date_str,
                    soft_voting_results=soft_voting_results
                )
                output_files['csv_soft_voting'] = soft_voting_file
            
            self.logger.info(f"Exported complete daily analysis for {date_str}")
            return output_files
            
        except Exception as e:
            self.logger.error(f"Failed to export daily analysis for {date_str}: {e}")
            raise
    
    def export_period_analysis(
        self,
        daily_summaries: List[Dict[str, Any]],
        daily_results: Dict[str, List[PredictionResult]],
        start_date: str,
        end_date: str
    ) -> Dict[str, Path]:
        """Export analysis results for a period of days.
        
        Args:
            daily_summaries: List of daily summary dictionaries
            daily_results: Dictionary mapping dates to prediction results
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary mapping output types to file paths
        """
        try:
            output_files = {}
            
            time_series_data = self.json_exporter.export_time_series_data(
                daily_summaries=daily_summaries,
                start_date=start_date,
                end_date=end_date
            )
            output_files['json_time_series'] = self.json_exporter.output_dir / f"time_series_{start_date}_to_{end_date}.json"
            
            calendar_data = self.json_exporter.export_github_calendar_data(
                daily_summaries=daily_summaries,
                metric_name="shannon_diversity"
            )
            output_files['json_github_calendar'] = self.json_exporter.output_dir / "github_calendar_shannon_diversity.json"
            
            aggregated_csv = self.csv_exporter.export_aggregated_predictions(
                daily_results=daily_results,
                start_date=start_date,
                end_date=end_date
            )
            output_files['csv_aggregated'] = aggregated_csv
            
            species_summary = self.csv_exporter.export_species_summary(
                daily_results=daily_results
            )
            output_files['csv_species_summary'] = species_summary
            
            self.logger.info(f"Exported period analysis from {start_date} to {end_date}")
            return output_files
            
        except Exception as e:
            self.logger.error(f"Failed to export period analysis: {e}")
            raise
    
    def export_visualization_data(
        self,
        daily_summaries: List[Dict[str, Any]],
        metrics: List[str] = None
    ) -> Dict[str, Path]:
        """Export data specifically formatted for visualization.
        
        Args:
            daily_summaries: List of daily summary dictionaries
            metrics: List of metrics to export for visualization
            
        Returns:
            Dictionary mapping visualization types to file paths
        """
        try:
            if metrics is None:
                metrics = ["shannon_diversity", "species_richness", "pielou_evenness"]
            
            output_files = {}
            
            for metric in metrics:
                calendar_data = self.json_exporter.export_github_calendar_data(
                    daily_summaries=daily_summaries,
                    metric_name=metric
                )
                output_files[f'github_calendar_{metric}'] = self.json_exporter.output_dir / f"github_calendar_{metric}.json"
            
            self.logger.info(f"Exported visualization data for {len(metrics)} metrics")
            return output_files
            
        except Exception as e:
            self.logger.error(f"Failed to export visualization data: {e}")
            raise
    
    def get_output_summary(self) -> Dict[str, Any]:
        """Get summary of all output files and directories.
        
        Returns:
            Summary of output structure and files
        """
        try:
            summary = {
                "base_directory": str(self.output_base_dir),
                "json_directory": str(self.json_exporter.output_dir),
                "csv_directory": str(self.csv_exporter.output_dir),
                "files": {
                    "json": [],
                    "csv": []
                },
                "total_files": 0,
                "last_updated": datetime.now().isoformat()
            }
            
            if self.json_exporter.output_dir.exists():
                json_files = list(self.json_exporter.output_dir.glob("*.json"))
                summary["files"]["json"] = [str(f.name) for f in json_files]
            
            if self.csv_exporter.output_dir.exists():
                csv_files = list(self.csv_exporter.output_dir.glob("*.csv"))
                summary["files"]["csv"] = [str(f.name) for f in csv_files]
            
            summary["total_files"] = len(summary["files"]["json"]) + len(summary["files"]["csv"])
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get output summary: {e}")
            return {}
    
    def cleanup_old_outputs(self, days_to_keep: int = 30) -> int:
        """Clean up output files older than specified days.
        
        Args:
            days_to_keep: Number of days of files to keep
            
        Returns:
            Number of files deleted
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            deleted_count = 0
            
            for json_file in self.json_exporter.output_dir.glob("*.json"):
                if json_file.stat().st_mtime < cutoff_date.timestamp():
                    json_file.unlink()
                    deleted_count += 1
            
            for csv_file in self.csv_exporter.output_dir.glob("*.csv"):
                if csv_file.stat().st_mtime < cutoff_date.timestamp():
                    csv_file.unlink()
                    deleted_count += 1
            
            self.logger.info(f"Cleaned up {deleted_count} old output files")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old outputs: {e}")
            return 0
