"""Similarity-based clustering for redundancy reduction."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as ssim

from ..utils.logger import LoggerMixin
from ..utils.data_structures import ImageData


@dataclass
class ClusteringConfig:
    """Configuration for similarity clustering."""
    
    similarity_method: str = "ssim"  # ssim, feature_based, histogram
    similarity_threshold: float = 0.85
    clustering_method: str = "kmeans"  # kmeans, hierarchical
    max_clusters: int = 10
    
    feature_detector: str = "orb"  # orb, sift, surf
    max_features: int = 500
    
    hist_bins: int = 256
    hist_channels: List[int] = None  # None means all channels
    
    def __post_init__(self):
        if self.hist_channels is None:
            self.hist_channels = [0, 1, 2]  # BGR channels


class SimilarityClusterer(LoggerMixin):
    """Handles image similarity calculation and clustering for redundancy reduction."""
    
    def __init__(self, config: Optional[ClusteringConfig] = None):
        """Initialize similarity clusterer.
        
        Args:
            config: Clustering configuration. If None, uses defaults.
        """
        self.config = config or ClusteringConfig()
        self.feature_detector = None
        
        if self.config.similarity_method == "feature_based":
            self._init_feature_detector()
    
    def _init_feature_detector(self):
        """Initialize feature detector based on configuration."""
        if self.config.feature_detector == "orb":
            self.feature_detector = cv2.ORB_create(nfeatures=self.config.max_features)
        elif self.config.feature_detector == "sift":
            self.feature_detector = cv2.SIFT_create(nfeatures=self.config.max_features)
        else:
            self.logger.warning(f"Unknown feature detector: {self.config.feature_detector}, using ORB")
            self.feature_detector = cv2.ORB_create(nfeatures=self.config.max_features)
    
    def calculate_ssim_similarity(self, image1: np.ndarray[Any, np.dtype[Any]], image2: np.ndarray[Any, np.dtype[Any]]) -> float:
        """Calculate SSIM (Structural Similarity Index) between two images.
        
        Args:
            image1: First image (BGR format)
            image2: Second image (BGR format)
            
        Returns:
            SSIM similarity score (0-1, higher is more similar)
        """
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        
        if gray1.shape != gray2.shape:
            h, w = min(gray1.shape[0], gray2.shape[0]), min(gray1.shape[1], gray2.shape[1])
            gray1 = cv2.resize(gray1, (w, h))
            gray2 = cv2.resize(gray2, (w, h))
        
        similarity_score = ssim(gray1, gray2)
        return float(similarity_score)
    
    def calculate_histogram_similarity(self, image1: np.ndarray[Any, np.dtype[Any]], image2: np.ndarray[Any, np.dtype[Any]]) -> float:
        """Calculate histogram-based similarity between two images.
        
        Args:
            image1: First image (BGR format)
            image2: Second image (BGR format)
            
        Returns:
            Histogram similarity score (0-1, higher is more similar)
        """
        similarities = []
        
        for channel in self.config.hist_channels:
            if channel < image1.shape[2] and channel < image2.shape[2]:
                hist1 = cv2.calcHist([image1], [channel], None, [self.config.hist_bins], [0, 256])
                hist2 = cv2.calcHist([image2], [channel], None, [self.config.hist_bins], [0, 256])
                
                hist1 = cv2.normalize(hist1, hist1).flatten()
                hist2 = cv2.normalize(hist2, hist2).flatten()
                
                correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                similarities.append(correlation)
        
        return float(np.mean(similarities)) if similarities else 0.0
    
    def calculate_feature_similarity(self, image1: np.ndarray[Any, np.dtype[Any]], image2: np.ndarray[Any, np.dtype[Any]]) -> float:
        """Calculate feature-based similarity between two images.
        
        Args:
            image1: First image (BGR format)
            image2: Second image (BGR format)
            
        Returns:
            Feature similarity score (0-1, higher is more similar)
        """
        if self.feature_detector is None:
            self.logger.error("Feature detector not initialized")
            return 0.0
        
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        
        kp1, desc1 = self.feature_detector.detectAndCompute(gray1, None)
        kp2, desc2 = self.feature_detector.detectAndCompute(gray2, None)
        
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return 0.0
        
        if self.config.feature_detector == "orb":
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        
        matches = matcher.match(desc1, desc2)
        
        if len(matches) == 0:
            return 0.0
        
        matches = sorted(matches, key=lambda x: x.distance)
        
        good_matches = matches[:min(50, len(matches))]
        
        max_features = max(len(desc1), len(desc2))
        similarity = len(good_matches) / max_features
        
        return float(min(similarity, 1.0))
    
    def calculate_similarity(self, image1: np.ndarray[Any, np.dtype[Any]], image2: np.ndarray[Any, np.dtype[Any]]) -> float:
        """Calculate similarity between two images using configured method.
        
        Args:
            image1: First image (BGR format)
            image2: Second image (BGR format)
            
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        if self.config.similarity_method == "ssim":
            return self.calculate_ssim_similarity(image1, image2)
        elif self.config.similarity_method == "histogram":
            return self.calculate_histogram_similarity(image1, image2)
        elif self.config.similarity_method == "feature_based":
            return self.calculate_feature_similarity(image1, image2)
        else:
            self.logger.error(f"Unknown similarity method: {self.config.similarity_method}")
            return 0.0
    
    def build_similarity_matrix(self, images: List[np.ndarray[Any, np.dtype[Any]]]) -> np.ndarray[Any, np.dtype[Any]]:
        """Build similarity matrix for a list of images.
        
        Args:
            images: List of images (BGR format)
            
        Returns:
            Similarity matrix (n x n)
        """
        n = len(images)
        similarity_matrix = np.zeros((n, n))
        
        self.logger.info(f"Building similarity matrix for {n} images")
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    similarity = self.calculate_similarity(images[i], images[j])
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
        
        return similarity_matrix
    
    def cluster_images_kmeans(self, similarity_matrix: np.ndarray[Any, np.dtype[Any]]) -> np.ndarray[Any, np.dtype[Any]]:
        """Cluster images using K-means on similarity matrix.
        
        Args:
            similarity_matrix: Precomputed similarity matrix
            
        Returns:
            Cluster labels for each image
        """
        n_images = similarity_matrix.shape[0]
        
        n_clusters = min(self.config.max_clusters, n_images)
        
        distance_matrix = 1.0 - similarity_matrix
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(distance_matrix)
        
        self.logger.info(f"K-means clustering created {n_clusters} clusters")
        return cluster_labels
    
    def cluster_images_hierarchical(self, similarity_matrix: np.ndarray[Any, np.dtype[Any]]) -> np.ndarray[Any, np.dtype[Any]]:
        """Cluster images using hierarchical clustering.
        
        Args:
            similarity_matrix: Precomputed similarity matrix
            
        Returns:
            Cluster labels for each image
        """
        n_images = similarity_matrix.shape[0]
        
        n_clusters = min(self.config.max_clusters, n_images)
        
        distance_matrix = 1.0 - similarity_matrix
        
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        self.logger.info(f"Hierarchical clustering created {n_clusters} clusters")
        return cluster_labels
    
    def cluster_images(self, images: List[np.ndarray[Any, np.dtype[Any]]]) -> np.ndarray[Any, np.dtype[Any]]:
        """Cluster images using configured method.
        
        Args:
            images: List of images (BGR format)
            
        Returns:
            Cluster labels for each image
        """
        if len(images) <= 1:
            return np.array([0] * len(images))
        
        similarity_matrix = self.build_similarity_matrix(images)
        
        if self.config.clustering_method == "kmeans":
            return self.cluster_images_kmeans(similarity_matrix)
        elif self.config.clustering_method == "hierarchical":
            return self.cluster_images_hierarchical(similarity_matrix)
        else:
            self.logger.error(f"Unknown clustering method: {self.config.clustering_method}")
            return np.array([0] * len(images))
    
    def select_representative_images(
        self, 
        images: List[np.ndarray[Any, np.dtype[Any]]], 
        image_data_list: List[ImageData],
        cluster_labels: np.ndarray[Any, np.dtype[Any]]
    ) -> List[int]:
        """Select representative images from each cluster.
        
        Args:
            images: List of images (BGR format)
            image_data_list: List of corresponding ImageData objects
            cluster_labels: Cluster labels for each image
            
        Returns:
            List of indices of representative images
        """
        representative_indices = []
        unique_clusters = np.unique(cluster_labels)
        
        for cluster_id in unique_clusters:
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            
            if len(cluster_indices) == 1:
                representative_indices.append(cluster_indices[0])
            else:
                best_idx = self._select_best_quality_image(
                    [images[i] for i in cluster_indices],
                    [image_data_list[i] for i in cluster_indices],
                    cluster_indices
                )
                representative_indices.append(best_idx)
        
        self.logger.info(f"Selected {len(representative_indices)} representative images from {len(unique_clusters)} clusters")
        return representative_indices
    
    def _select_best_quality_image(
        self, 
        cluster_images: List[np.ndarray[Any, np.dtype[Any]]],
        cluster_image_data: List[ImageData],
        original_indices: np.ndarray[Any, np.dtype[Any]]
    ) -> int:
        """Select the best quality image from a cluster.
        
        Args:
            cluster_images: Images in the cluster
            cluster_image_data: ImageData objects for cluster images
            original_indices: Original indices of cluster images
            
        Returns:
            Index of best quality image in original list
        """
        best_score = -1
        best_idx = original_indices[0]
        
        for i, (image, image_data) in enumerate(zip(cluster_images, cluster_image_data)):
            score = self._calculate_quality_score(image, image_data)
            
            if score > best_score:
                best_score = score
                best_idx = original_indices[i]
        
        return best_idx
    
    def _calculate_quality_score(self, image: np.ndarray[Any, np.dtype[Any]], image_data: ImageData) -> float:
        """Calculate overall quality score for an image.
        
        Args:
            image: Image (BGR format)
            image_data: ImageData object
            
        Returns:
            Quality score (higher is better)
        """
        score = 0.0
        
        if image_data.blur_score is not None:
            score += min(image_data.blur_score / 1000.0, 1.0) * 0.3
        
        if image_data.exposure_score is not None:
            score += image_data.exposure_score * 0.3
        
        if image_data.contrast_std is not None:
            score += min(image_data.contrast_std / 100.0, 1.0) * 0.2
        
        size_score = min(image_data.file_size_bytes / (5 * 1024 * 1024), 1.0)  # Normalize to 5MB
        score += size_score * 0.1
        
        total_pixels = image_data.size[0] * image_data.size[1]
        resolution_score = min(total_pixels / (1920 * 1080), 1.0)  # Normalize to Full HD
        score += resolution_score * 0.1
        
        return score
    
    def process_image_batch(
        self, 
        image_data_list: List[ImageData],
        load_images: bool = True
    ) -> Tuple[List[int], Dict[str, Any]]:
        """Process a batch of images for similarity clustering.
        
        Args:
            image_data_list: List of ImageData objects
            load_images: Whether to load images for processing
            
        Returns:
            Tuple of (representative_indices, clustering_info)
        """
        if not image_data_list:
            return [], {}
        
        if not load_images or len(image_data_list) == 1:
            return list(range(len(image_data_list))), {"clusters": 1, "method": "no_clustering"}
        
        images = []
        valid_indices = []
        
        for i, image_data in enumerate(image_data_list):
            image = cv2.imread(str(image_data.path))
            if image is not None:
                image = cv2.resize(image, (256, 256))
                images.append(image)
                valid_indices.append(i)
            else:
                self.logger.warning(f"Failed to load image: {image_data.path}")
        
        if len(images) <= 1:
            return valid_indices, {"clusters": len(images), "method": "insufficient_images"}
        
        cluster_labels = self.cluster_images(images)
        
        valid_image_data = [image_data_list[i] for i in valid_indices]
        representative_local_indices = self.select_representative_images(
            images, valid_image_data, cluster_labels
        )
        
        representative_indices = [valid_indices[i] for i in representative_local_indices]
        
        for i, image_data in enumerate(image_data_list):
            if i in valid_indices:
                local_idx = valid_indices.index(i)
                image_data.cluster_id = int(cluster_labels[local_idx])
                image_data.is_representative = i in representative_indices
                
                if not image_data.is_representative:
                    cluster_id = cluster_labels[local_idx]
                    rep_local_idx = None
                    for rep_idx in representative_local_indices:
                        if cluster_labels[rep_idx] == cluster_id:
                            rep_local_idx = rep_idx
                            break
                    
                    if rep_local_idx is not None:
                        similarity = self.calculate_similarity(
                            images[local_idx], images[rep_local_idx]
                        )
                        image_data.similarity_score = similarity
        
        clustering_info = {
            "clusters": len(np.unique(cluster_labels)),
            "method": self.config.clustering_method,
            "similarity_method": self.config.similarity_method,
            "total_images": len(image_data_list),
            "valid_images": len(images),
            "representative_images": len(representative_indices),
            "reduction_ratio": len(representative_indices) / len(image_data_list) if image_data_list else 1.0
        }
        
        self.logger.info(f"Clustering complete: {clustering_info}")
        return representative_indices, clustering_info
