"""Data structures for the Weed Diversity Analyzer."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class ImageData:
    """Represents an image and its metadata."""
    
    path: Path
    date: datetime
    size: Tuple[int, int]  # (width, height)
    file_size_bytes: int
    
    blur_score: Optional[float] = None
    exposure_score: Optional[float] = None
    brightness_mean: Optional[float] = None
    contrast_std: Optional[float] = None
    
    is_processed: bool = False
    is_representative: bool = False  # True if selected as cluster representative
    cluster_id: Optional[int] = None
    similarity_score: Optional[float] = None
    
    def __post_init__(self):
        """Validate and process data after initialization."""
        if isinstance(self.path, str):
            self.path = Path(self.path)
        
        if isinstance(self.date, str):
            self.date = datetime.fromisoformat(self.date)


@dataclass
class SpeciesPrediction:
    """Represents a single species prediction."""
    
    species_name: str
    confidence: float
    taxonomic_level: str = "species"  # species, genus, family, etc.
    scientific_name: Optional[str] = None
    common_name: Optional[str] = None
    
    def __post_init__(self):
        """Validate prediction data."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")


@dataclass
class PredictionResult:
    """Represents the complete prediction result for an image."""
    
    image_path: Path
    predictions: List[SpeciesPrediction]
    model_name: str
    processing_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    top_prediction: Optional[SpeciesPrediction] = None
    mean_confidence: Optional[float] = None
    prediction_entropy: Optional[float] = None
    
    def __post_init__(self):
        """Calculate derived metrics."""
        if isinstance(self.image_path, str):
            self.image_path = Path(self.image_path)
        
        if self.predictions:
            self.top_prediction = max(self.predictions, key=lambda p: p.confidence)
            
            self.mean_confidence = np.mean([p.confidence for p in self.predictions])
            
            confidences = np.array([p.confidence for p in self.predictions])
            if confidences.sum() > 0:
                probs = confidences / confidences.sum()
                self.prediction_entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    def get_top_k(self, k: int = 3) -> List[SpeciesPrediction]:
        """Get top-k predictions sorted by confidence."""
        return sorted(self.predictions, key=lambda p: p.confidence, reverse=True)[:k]
    
    def has_confident_prediction(self, threshold: float = 0.5) -> bool:
        """Check if any prediction exceeds confidence threshold."""
        return any(p.confidence >= threshold for p in self.predictions)


@dataclass
class DiversityMetrics:
    """Represents biodiversity metrics for a set of observations."""
    
    date: datetime
    total_images: int
    processed_images: int
    
    species_richness: int
    shannon_diversity: float
    pielou_evenness: float
    simpson_diversity: float
    
    hill_q0: float  # Species richness
    hill_q1: float  # Exponential of Shannon entropy
    hill_q2: float  # Inverse Simpson index
    
    chao1_estimate: Optional[float] = None
    coverage_estimate: Optional[float] = None
    
    richness_ci: Optional[Tuple[float, float]] = None
    shannon_ci: Optional[Tuple[float, float]] = None
    
    species_counts: Dict[str, int] = field(default_factory=dict)
    species_frequencies: Dict[str, float] = field(default_factory=dict)
    
    top_species: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate and calculate derived metrics."""
        if isinstance(self.date, str):
            self.date = datetime.fromisoformat(self.date)
        
        if self.species_counts and not self.species_frequencies:
            total_count = sum(self.species_counts.values())
            self.species_frequencies = {
                species: count / total_count 
                for species, count in self.species_counts.items()
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "date": self.date.isoformat(),
            "total_images": self.total_images,
            "processed_images": self.processed_images,
            "species_richness": self.species_richness,
            "shannon_diversity": self.shannon_diversity,
            "pielou_evenness": self.pielou_evenness,
            "simpson_diversity": self.simpson_diversity,
            "hill_numbers": {
                "q0": self.hill_q0,
                "q1": self.hill_q1,
                "q2": self.hill_q2,
            },
            "chao1_estimate": self.chao1_estimate,
            "coverage_estimate": self.coverage_estimate,
            "confidence_intervals": {
                "richness": self.richness_ci,
                "shannon": self.shannon_ci,
            },
            "species_counts": self.species_counts,
            "species_frequencies": self.species_frequencies,
            "top_species": self.top_species,
        }


@dataclass
class ProcessingResult:
    """Represents the complete processing result for a day."""
    
    date: datetime
    input_images: List[ImageData]
    processed_images: List[ImageData]
    predictions: List[PredictionResult]
    diversity_metrics: DiversityMetrics
    
    processing_time_total: float
    processing_time_per_image: float
    
    average_confidence: float
    low_confidence_count: int
    failed_predictions: int
    
    clusters_found: int
    redundancy_reduction_ratio: float
    
    def __post_init__(self):
        """Calculate derived statistics."""
        if isinstance(self.date, str):
            self.date = datetime.fromisoformat(self.date)
        
        if self.processed_images:
            self.processing_time_per_image = (
                self.processing_time_total / len(self.processed_images)
            )
        else:
            self.processing_time_per_image = 0.0
        
        if self.input_images:
            self.redundancy_reduction_ratio = (
                len(self.processed_images) / len(self.input_images)
            )
        else:
            self.redundancy_reduction_ratio = 1.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get processing summary for reporting."""
        return {
            "date": self.date.isoformat(),
            "images": {
                "input_count": len(self.input_images),
                "processed_count": len(self.processed_images),
                "redundancy_reduction": f"{(1 - self.redundancy_reduction_ratio) * 100:.1f}%",
            },
            "predictions": {
                "total_predictions": len(self.predictions),
                "average_confidence": f"{self.average_confidence:.3f}",
                "low_confidence_count": self.low_confidence_count,
                "failed_predictions": self.failed_predictions,
            },
            "diversity": {
                "species_richness": self.diversity_metrics.species_richness,
                "shannon_diversity": f"{self.diversity_metrics.shannon_diversity:.3f}",
                "pielou_evenness": f"{self.diversity_metrics.pielou_evenness:.3f}",
            },
            "performance": {
                "total_time": f"{self.processing_time_total:.2f}s",
                "time_per_image": f"{self.processing_time_per_image:.2f}s",
                "clusters_found": self.clusters_found,
            },
        }


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing operations."""
    
    batch_size: int = 16
    max_workers: int = 4
    chunk_size: int = 100
    
    max_memory_gb: float = 8.0
    clear_cache_interval: int = 1000
    
    min_confidence_threshold: float = 0.1
    max_processing_time_per_image: float = 30.0
    
    save_intermediate_results: bool = False
    progress_reporting: bool = True
    log_interval: int = 100


ImageList = List[ImageData]
PredictionList = List[PredictionResult]
DiversityList = List[DiversityMetrics]
ProcessingResultList = List[ProcessingResult]
