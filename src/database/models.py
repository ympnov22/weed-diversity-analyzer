"""SQLAlchemy models for weed diversity analyzer."""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime

from .database import Base

class UserModel(Base):
    """User model for simple authentication."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    api_key = Column(String(64), unique=True, index=True, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    analysis_sessions = relationship("AnalysisSessionModel", back_populates="user")

class AnalysisSessionModel(Base):
    """Analysis session model to group related data."""
    __tablename__ = "analysis_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_name = Column(String(100), nullable=False)
    description = Column(Text)
    start_date = Column(DateTime(timezone=True), nullable=False)
    end_date = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("UserModel", back_populates="analysis_sessions")
    diversity_metrics = relationship("DiversityMetricsModel", back_populates="session")

class DiversityMetricsModel(Base):
    """Diversity metrics model based on existing DiversityMetrics dataclass."""
    __tablename__ = "diversity_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("analysis_sessions.id"), nullable=False)
    date = Column(DateTime(timezone=True), nullable=False, index=True)
    
    total_images = Column(Integer, nullable=False)
    processed_images = Column(Integer, nullable=False)
    species_richness = Column(Integer, nullable=False)
    shannon_diversity = Column(Float, nullable=False)
    pielou_evenness = Column(Float, nullable=False)
    simpson_diversity = Column(Float, nullable=False)
    
    hill_q0 = Column(Float, nullable=False)
    hill_q1 = Column(Float, nullable=False)
    hill_q2 = Column(Float, nullable=False)
    
    chao1_estimate = Column(Float)
    coverage_estimate = Column(Float)
    
    species_counts = Column(JSON, nullable=False)
    species_frequencies = Column(JSON, nullable=False)
    top_species = Column(JSON, nullable=False)
    
    session = relationship("AnalysisSessionModel", back_populates="diversity_metrics")
    prediction_results = relationship("PredictionResultModel", back_populates="diversity_metrics")

class ImageDataModel(Base):
    """Image data model based on existing ImageData dataclass."""
    __tablename__ = "image_data"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("analysis_sessions.id"), nullable=False)
    path = Column(String(500), nullable=False)
    date = Column(DateTime(timezone=True), nullable=False, index=True)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    
    blur_score = Column(Float)
    exposure_score = Column(Float)
    brightness_mean = Column(Float)
    contrast_std = Column(Float)
    
    is_processed = Column(Boolean, default=False)
    is_representative = Column(Boolean, default=False)
    cluster_id = Column(Integer)
    similarity_score = Column(Float)
    
    prediction_results = relationship("PredictionResultModel", back_populates="image_data")

class PredictionResultModel(Base):
    """Prediction result model based on existing PredictionResult dataclass."""
    __tablename__ = "prediction_results"
    
    id = Column(Integer, primary_key=True, index=True)
    diversity_metrics_id = Column(Integer, ForeignKey("diversity_metrics.id"), nullable=False)
    image_data_id = Column(Integer, ForeignKey("image_data.id"), nullable=False)
    
    model_name = Column(String(100), nullable=False)
    processing_time = Column(Float, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    predictions = Column(JSON, nullable=False)
    top_prediction_species = Column(String(200))
    top_prediction_confidence = Column(Float)
    mean_confidence = Column(Float)
    prediction_entropy = Column(Float)
    
    diversity_metrics = relationship("DiversityMetricsModel", back_populates="prediction_results")
    image_data = relationship("ImageDataModel", back_populates="prediction_results")

class ProcessingResultModel(Base):
    """Processing result model for daily processing summaries."""
    __tablename__ = "processing_results"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("analysis_sessions.id"), nullable=False)
    date = Column(DateTime(timezone=True), nullable=False, index=True)
    
    input_images_count = Column(Integer, nullable=False)
    processed_images_count = Column(Integer, nullable=False)
    predictions_count = Column(Integer, nullable=False)
    
    processing_time_total = Column(Float, nullable=False)
    processing_time_per_image = Column(Float, nullable=False)
    
    average_confidence = Column(Float, nullable=False)
    low_confidence_count = Column(Integer, nullable=False)
    failed_predictions = Column(Integer, nullable=False)
    
    clusters_found = Column(Integer, nullable=False)
    redundancy_reduction_ratio = Column(Float, nullable=False)
    
    processing_metadata = Column(JSON)
