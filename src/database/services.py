"""Database services for replacing file-based storage."""

from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc

from .models import DiversityMetricsModel, AnalysisSessionModel, UserModel, PredictionResultModel, ImageDataModel
from ..utils.data_structures import DiversityMetrics, PredictionResult

class DatabaseService:
    """Service for database operations replacing OutputManager functionality."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def save_daily_analysis(
        self,
        user_id: int,
        session_id: int,
        date_str: str,
        diversity_metrics: DiversityMetrics,
        prediction_results: List[PredictionResult]
    ) -> DiversityMetricsModel:
        """Save daily analysis to database."""
        
        db_metrics = DiversityMetricsModel(
            session_id=session_id,
            date=datetime.fromisoformat(date_str),
            total_images=diversity_metrics.total_images,
            processed_images=diversity_metrics.processed_images,
            species_richness=diversity_metrics.species_richness,
            shannon_diversity=diversity_metrics.shannon_diversity,
            pielou_evenness=diversity_metrics.pielou_evenness,
            simpson_diversity=diversity_metrics.simpson_diversity,
            hill_q0=diversity_metrics.hill_q0,
            hill_q1=diversity_metrics.hill_q1,
            hill_q2=diversity_metrics.hill_q2,
            chao1_estimate=diversity_metrics.chao1_estimate,
            coverage_estimate=diversity_metrics.coverage_estimate,
            species_counts=diversity_metrics.species_counts,
            species_frequencies=diversity_metrics.species_frequencies,
            top_species=diversity_metrics.top_species
        )
        
        self.db.add(db_metrics)
        self.db.commit()
        self.db.refresh(db_metrics)
        
        return db_metrics
    
    def get_daily_summaries(
        self,
        user_id: int,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get daily summaries for visualization."""
        query = self.db.query(DiversityMetricsModel).join(AnalysisSessionModel).filter(
            AnalysisSessionModel.user_id == user_id
        )
        
        if start_date:
            query = query.filter(DiversityMetricsModel.date >= datetime.fromisoformat(start_date))
        if end_date:
            query = query.filter(DiversityMetricsModel.date <= datetime.fromisoformat(end_date))
        
        metrics = query.order_by(DiversityMetricsModel.date).all()
        
        return [self._metrics_to_dict(m) for m in metrics]
    
    def get_calendar_data(self, user_id: int, metric: str) -> Dict[str, Any]:
        """Get calendar data for specified metric."""
        daily_summaries = self.get_daily_summaries(user_id)
        
        calendar_data = {}
        for summary in daily_summaries:
            date_str = summary["date"][:10]
            if metric in summary:
                calendar_data[date_str] = {
                    "value": summary[metric],
                    "level": self._get_intensity_level(summary[metric], metric)
                }
        
        return {
            "data": calendar_data,
            "metric": metric,
            "total_days": len(calendar_data)
        }
    
    def get_time_series_data(
        self,
        user_id: int,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get time series data for visualization."""
        daily_summaries = self.get_daily_summaries(user_id, start_date, end_date)
        
        dates = [s["date"][:10] for s in daily_summaries]
        metrics = {
            "shannon_diversity": [s["shannon_diversity"] for s in daily_summaries],
            "species_richness": [s["species_richness"] for s in daily_summaries],
            "pielou_evenness": [s["pielou_evenness"] for s in daily_summaries]
        }
        
        return {
            "dates": dates,
            "metrics": metrics,
            "total_points": len(dates)
        }
    
    def create_analysis_session(
        self,
        user_id: int,
        session_name: str,
        description: Optional[str] = None
    ) -> AnalysisSessionModel:
        """Create a new analysis session."""
        session = AnalysisSessionModel(
            user_id=user_id,
            session_name=session_name,
            description=description,
            start_date=datetime.now()
        )
        
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        
        return session
    
    def get_user_sessions(self, user_id: int) -> List[AnalysisSessionModel]:
        """Get all analysis sessions for a user."""
        return self.db.query(AnalysisSessionModel).filter(
            AnalysisSessionModel.user_id == user_id
        ).order_by(desc(AnalysisSessionModel.created_at)).all()
    
    def get_species_dashboard_data(self, user_id: int) -> Dict[str, Any]:
        """Get species dashboard data for visualization."""
        daily_summaries = self.get_daily_summaries(user_id)
        
        if not daily_summaries:
            return {"total_species": 0, "total_observations": 0, "species_distribution": []}
        
        all_species_counts = {}
        total_observations = 0
        
        for summary in daily_summaries:
            species_counts = summary.get("species_counts", {})
            total_observations += summary.get("total_images", 0)
            
            for species, count in species_counts.items():
                all_species_counts[species] = all_species_counts.get(species, 0) + count
        
        species_distribution = [
            {"species": species, "count": count, "percentage": (count / total_observations) * 100}
            for species, count in sorted(all_species_counts.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return {
            "total_species": len(all_species_counts),
            "total_observations": total_observations,
            "species_distribution": species_distribution[:20]  # Top 20 species
        }
    
    def _metrics_to_dict(self, metrics: DiversityMetricsModel) -> Dict[str, Any]:
        """Convert DiversityMetricsModel to dictionary."""
        return {
            "date": metrics.date.isoformat(),
            "total_images": metrics.total_images,
            "processed_images": metrics.processed_images,
            "species_richness": metrics.species_richness,
            "shannon_diversity": metrics.shannon_diversity,
            "pielou_evenness": metrics.pielou_evenness,
            "simpson_diversity": metrics.simpson_diversity,
            "hill_numbers": {
                "q0": metrics.hill_q0,
                "q1": metrics.hill_q1,
                "q2": metrics.hill_q2,
            },
            "chao1_estimate": metrics.chao1_estimate,
            "coverage_estimate": metrics.coverage_estimate,
            "species_counts": metrics.species_counts,
            "species_frequencies": metrics.species_frequencies,
            "top_species": metrics.top_species,
        }
    
    def _get_intensity_level(self, value: float, metric: str) -> int:
        """Get intensity level (0-4) for calendar visualization."""
        if value is None:
            return 0
        
        if metric == "shannon_diversity":
            if value >= 2.5:
                return 4
            elif value >= 2.0:
                return 3
            elif value >= 1.5:
                return 2
            elif value >= 1.0:
                return 1
            else:
                return 0
        elif metric == "species_richness":
            if value >= 20:
                return 4
            elif value >= 15:
                return 3
            elif value >= 10:
                return 2
            elif value >= 5:
                return 1
            else:
                return 0
        else:
            return min(4, max(0, int(value * 4)))
