"""Stub implementation for similarity clustering to avoid heavy dependencies."""

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

from ..utils.logger import LoggerMixin


@dataclass
class ClusteringConfig:
    """Configuration for clustering operations."""
    n_clusters: int = 5
    similarity_threshold: float = 0.8
    clustering_method: str = "kmeans"


class SimilarityClusteringStub(LoggerMixin):
    """Stub implementation of similarity clustering for minimal deployment."""
    
    def __init__(self, config: Optional[ClusteringConfig] = None):
        super().__init__()
        self.config = config or ClusteringConfig()
        self.logger.info("SimilarityClusteringStub initialized in stub mode (no heavy dependencies)")
    
    def cluster_similar_images(self, 
                              image_paths: List[Path],
                              **kwargs) -> Dict[str, Any]:
        """Stub implementation for image clustering."""
        self.logger.warning("Image clustering not available in minimal mode")
        
        return {
            "clusters": [],
            "cluster_centers": [],
            "labels": [0] * len(image_paths),
            "similarity_matrix": [],
            "metadata": {
                "method": "stub",
                "n_images": len(image_paths),
                "n_clusters": 0
            }
        }
    
    def calculate_similarity_matrix(self, 
                                   features: List[Any]) -> List[List[float]]:
        """Stub implementation for similarity matrix calculation."""
        self.logger.warning("Similarity matrix calculation not available in minimal mode")
        
        n = len(features)
        return [[0.0 for _ in range(n)] for _ in range(n)]
    
    def find_representative_images(self, 
                                  cluster_results: Dict[str, Any]) -> List[Path]:
        """Stub implementation for finding representative images."""
        self.logger.warning("Representative image selection not available in minimal mode")
        return []
