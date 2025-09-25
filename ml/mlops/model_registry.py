"""
Model Registry for Claims Processing MLOps

This module provides a centralized registry for managing ML models including:
- Model versioning and metadata storage
- Model promotion workflows (dev -> staging -> production)
- Model performance tracking
- A/B testing support
- Model rollback capabilities
"""

import os
import json
import joblib
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, String, DateTime, Float, Integer, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
import uuid
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://claims:claims@localhost:5432/claims")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ModelStage(Enum):
    """Model deployment stages"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"

class ModelStatus(Enum):
    """Model status"""
    TRAINING = "training"
    READY = "ready"
    DEPLOYED = "deployed"
    FAILED = "failed"
    DEPRECATED = "deprecated"

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    training_samples: int
    validation_samples: int
    feature_count: int
    training_time_seconds: float
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetrics':
        return cls(**data)

@dataclass
class ModelConfig:
    """Model configuration"""
    model_type: str
    hyperparameters: Dict[str, Any]
    feature_columns: List[str]
    target_column: str
    preprocessing_steps: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        return cls(**data)

# Database Models
class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = {'schema': 'mlops'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(100), nullable=False)
    version = Column(String(20), nullable=False)
    stage = Column(String(20), nullable=False, default=ModelStage.DEVELOPMENT.value)
    status = Column(String(20), nullable=False, default=ModelStatus.TRAINING.value)
    
    # Model artifacts
    model_path = Column(String(500), nullable=False)
    config_path = Column(String(500), nullable=False)
    metadata_path = Column(String(500), nullable=False)
    
    # Performance metrics (stored as JSON)
    metrics = Column(Text)  # JSON string
    config = Column(Text)   # JSON string
    
    # Training information
    training_dataset_hash = Column(String(64))
    training_started_at = Column(DateTime)
    training_completed_at = Column(DateTime)
    trained_by = Column(String(100))
    
    # Deployment information
    deployed_at = Column(DateTime)
    deployed_by = Column(String(100))
    deployment_config = Column(Text)  # JSON string
    
    # Monitoring
    prediction_count = Column(Integer, default=0)
    last_prediction_at = Column(DateTime)
    performance_alerts = Column(Integer, default=0)
    
    # Metadata
    description = Column(Text)
    tags = Column(Text)  # JSON array as string
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ModelExperiment(Base):
    __tablename__ = "model_experiments"
    __table_args__ = {'schema': 'mlops'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_name = Column(String(100), nullable=False)
    model_name = Column(String(100), nullable=False)
    
    # Experiment configuration
    config = Column(Text, nullable=False)  # JSON string
    parameters = Column(Text)  # JSON string
    
    # Results
    metrics = Column(Text)  # JSON string
    artifacts_path = Column(String(500))
    
    # Status
    status = Column(String(20), nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    
    # Metadata
    created_by = Column(String(100))
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class ModelRegistry:
    """Centralized model registry for managing ML models"""
    
    def __init__(self, base_path: str = "/app/models"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create database tables
        Base.metadata.create_all(bind=engine)
        
        logger.info(f"Model registry initialized at {self.base_path}")
    
    def get_db(self) -> Session:
        """Get database session"""
        return SessionLocal()
    
    def register_model(
        self,
        model_name: str,
        model_obj: Any,
        config: ModelConfig,
        metrics: ModelMetrics,
        version: Optional[str] = None,
        stage: ModelStage = ModelStage.DEVELOPMENT,
        description: str = "",
        tags: List[str] = None,
        trained_by: str = "system"
    ) -> str:
        """Register a new model version"""
        
        if version is None:
            version = self._generate_version(model_name)
        
        # Create model directory
        model_dir = self.base_path / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model artifacts
        model_path = model_dir / "model.pkl"
        config_path = model_dir / "config.json"
        metadata_path = model_dir / "metadata.json"
        
        # Save model
        joblib.dump(model_obj, model_path)
        
        # Save configuration
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'version': version,
            'stage': stage.value,
            'metrics': metrics.to_dict(),
            'config': config.to_dict(),
            'created_at': datetime.utcnow().isoformat(),
            'trained_by': trained_by,
            'description': description,
            'tags': tags or []
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Register in database
        db = self.get_db()
        try:
            model_version = ModelVersion(
                model_name=model_name,
                version=version,
                stage=stage.value,
                status=ModelStatus.READY.value,
                model_path=str(model_path),
                config_path=str(config_path),
                metadata_path=str(metadata_path),
                metrics=json.dumps(metrics.to_dict()),
                config=json.dumps(config.to_dict()),
                training_completed_at=datetime.utcnow(),
                trained_by=trained_by,
                description=description,
                tags=json.dumps(tags or [])
            )
            
            db.add(model_version)
            db.commit()
            db.refresh(model_version)
            
            model_id = str(model_version.id)
            logger.info(f"Model registered: {model_name} v{version} (ID: {model_id})")
            
            return model_id
            
        finally:
            db.close()
    
    def load_model(self, model_name: str, version: Optional[str] = None, stage: Optional[ModelStage] = None) -> Tuple[Any, ModelConfig, ModelMetrics]:
        """Load a model from the registry"""
        
        db = self.get_db()
        try:
            query = db.query(ModelVersion).filter(ModelVersion.model_name == model_name)
            
            if version:
                query = query.filter(ModelVersion.version == version)
            elif stage:
                query = query.filter(ModelVersion.stage == stage.value)
            else:
                # Get latest production model, or latest if no production
                prod_model = query.filter(ModelVersion.stage == ModelStage.PRODUCTION.value).first()
                if prod_model:
                    model_version = prod_model
                else:
                    model_version = query.order_by(ModelVersion.created_at.desc()).first()
            
            if not hasattr(locals(), 'model_version'):
                model_version = query.order_by(ModelVersion.created_at.desc()).first()
            
            if not model_version:
                raise ValueError(f"No model found for {model_name}")
            
            # Load model artifacts
            model_obj = joblib.load(model_version.model_path)
            
            with open(model_version.config_path, 'r') as f:
                config_data = json.load(f)
                config = ModelConfig.from_dict(config_data)
            
            metrics_data = json.loads(model_version.metrics)
            metrics = ModelMetrics.from_dict(metrics_data)
            
            logger.info(f"Model loaded: {model_name} v{model_version.version}")
            return model_obj, config, metrics
            
        finally:
            db.close()
    
    def promote_model(self, model_name: str, version: str, target_stage: ModelStage, promoted_by: str = "system") -> bool:
        """Promote model to a different stage"""
        
        db = self.get_db()
        try:
            # Get the model version
            model_version = db.query(ModelVersion).filter(
                ModelVersion.model_name == model_name,
                ModelVersion.version == version
            ).first()
            
            if not model_version:
                raise ValueError(f"Model {model_name} v{version} not found")
            
            # If promoting to production, demote current production model
            if target_stage == ModelStage.PRODUCTION:
                current_prod = db.query(ModelVersion).filter(
                    ModelVersion.model_name == model_name,
                    ModelVersion.stage == ModelStage.PRODUCTION.value
                ).first()
                
                if current_prod:
                    current_prod.stage = ModelStage.ARCHIVED.value
                    current_prod.updated_at = datetime.utcnow()
            
            # Promote the model
            old_stage = model_version.stage
            model_version.stage = target_stage.value
            model_version.deployed_at = datetime.utcnow()
            model_version.deployed_by = promoted_by
            model_version.updated_at = datetime.utcnow()
            
            if target_stage == ModelStage.PRODUCTION:
                model_version.status = ModelStatus.DEPLOYED.value
            
            db.commit()
            
            logger.info(f"Model promoted: {model_name} v{version} from {old_stage} to {target_stage.value}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to promote model: {e}")
            return False
        finally:
            db.close()
    
    def list_models(self, model_name: Optional[str] = None, stage: Optional[ModelStage] = None) -> List[Dict[str, Any]]:
        """List models in the registry"""
        
        db = self.get_db()
        try:
            query = db.query(ModelVersion)
            
            if model_name:
                query = query.filter(ModelVersion.model_name == model_name)
            
            if stage:
                query = query.filter(ModelVersion.stage == stage.value)
            
            models = query.order_by(ModelVersion.created_at.desc()).all()
            
            result = []
            for model in models:
                model_info = {
                    'id': str(model.id),
                    'model_name': model.model_name,
                    'version': model.version,
                    'stage': model.stage,
                    'status': model.status,
                    'metrics': json.loads(model.metrics) if model.metrics else {},
                    'created_at': model.created_at.isoformat(),
                    'trained_by': model.trained_by,
                    'description': model.description,
                    'prediction_count': model.prediction_count,
                    'last_prediction_at': model.last_prediction_at.isoformat() if model.last_prediction_at else None,
                }
                result.append(model_info)
            
            return result
            
        finally:
            db.close()
    
    def get_model_info(self, model_name: str, version: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific model version"""
        
        db = self.get_db()
        try:
            model_version = db.query(ModelVersion).filter(
                ModelVersion.model_name == model_name,
                ModelVersion.version == version
            ).first()
            
            if not model_version:
                return None
            
            # Load metadata from file for complete information
            with open(model_version.metadata_path, 'r') as f:
                file_metadata = json.load(f)
            
            model_info = {
                'id': str(model_version.id),
                'model_name': model_version.model_name,
                'version': model_version.version,
                'stage': model_version.stage,
                'status': model_version.status,
                'metrics': json.loads(model_version.metrics) if model_version.metrics else {},
                'config': json.loads(model_version.config) if model_version.config else {},
                'created_at': model_version.created_at.isoformat(),
                'trained_by': model_version.trained_by,
                'description': model_version.description,
                'tags': json.loads(model_version.tags) if model_version.tags else [],
                'prediction_count': model_version.prediction_count,
                'last_prediction_at': model_version.last_prediction_at.isoformat() if model_version.last_prediction_at else None,
                'performance_alerts': model_version.performance_alerts,
                'file_metadata': file_metadata,
                'model_path': model_version.model_path,
                'config_path': model_version.config_path,
                'metadata_path': model_version.metadata_path
            }
            
            return model_info
            
        finally:
            db.close()
    
    def update_prediction_stats(self, model_name: str, version: str, prediction_count: int = 1):
        """Update model prediction statistics"""
        
        db = self.get_db()
        try:
            model_version = db.query(ModelVersion).filter(
                ModelVersion.model_name == model_name,
                ModelVersion.version == version
            ).first()
            
            if model_version:
                model_version.prediction_count = (model_version.prediction_count or 0) + prediction_count
                model_version.last_prediction_at = datetime.utcnow()
                model_version.updated_at = datetime.utcnow()
                
                db.commit()
                
        except Exception as e:
            logger.error(f"Failed to update prediction stats: {e}")
            db.rollback()
        finally:
            db.close()
    
    def add_performance_alert(self, model_name: str, version: str, alert_type: str, details: str):
        """Add a performance alert for a model"""
        
        db = self.get_db()
        try:
            model_version = db.query(ModelVersion).filter(
                ModelVersion.model_name == model_name,
                ModelVersion.version == version
            ).first()
            
            if model_version:
                model_version.performance_alerts = (model_version.performance_alerts or 0) + 1
                model_version.updated_at = datetime.utcnow()
                
                db.commit()
                
                logger.warning(f"Performance alert added for {model_name} v{version}: {alert_type} - {details}")
                
        except Exception as e:
            logger.error(f"Failed to add performance alert: {e}")
            db.rollback()
        finally:
            db.close()
    
    def delete_model(self, model_name: str, version: str) -> bool:
        """Delete a model version (only if not in production)"""
        
        db = self.get_db()
        try:
            model_version = db.query(ModelVersion).filter(
                ModelVersion.model_name == model_name,
                ModelVersion.version == version
            ).first()
            
            if not model_version:
                return False
            
            if model_version.stage == ModelStage.PRODUCTION.value:
                raise ValueError("Cannot delete production model. Demote first.")
            
            # Delete files
            model_dir = Path(model_version.model_path).parent
            if model_dir.exists():
                shutil.rmtree(model_dir)
            
            # Delete from database
            db.delete(model_version)
            db.commit()
            
            logger.info(f"Model deleted: {model_name} v{version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model: {e}")
            db.rollback()
            return False
        finally:
            db.close()
    
    def _generate_version(self, model_name: str) -> str:
        """Generate a new version number for a model"""
        
        db = self.get_db()
        try:
            # Get the latest version
            latest = db.query(ModelVersion).filter(
                ModelVersion.model_name == model_name
            ).order_by(ModelVersion.created_at.desc()).first()
            
            if not latest:
                return "1.0.0"
            
            # Parse version and increment
            try:
                major, minor, patch = map(int, latest.version.split('.'))
                return f"{major}.{minor}.{patch + 1}"
            except:
                # If version format is unexpected, use timestamp
                return datetime.utcnow().strftime("%Y%m%d.%H%M%S")
                
        finally:
            db.close()
    
    def compare_models(self, model_name: str, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two model versions"""
        
        info1 = self.get_model_info(model_name, version1)
        info2 = self.get_model_info(model_name, version2)
        
        if not info1 or not info2:
            raise ValueError("One or both model versions not found")
        
        metrics1 = info1['metrics']
        metrics2 = info2['metrics']
        
        comparison = {
            'model_name': model_name,
            'version1': version1,
            'version2': version2,
            'metrics_comparison': {},
            'winner': None
        }
        
        # Compare metrics
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']:
            if metric in metrics1 and metric in metrics2:
                diff = metrics2[metric] - metrics1[metric]
                comparison['metrics_comparison'][metric] = {
                    'version1': metrics1[metric],
                    'version2': metrics2[metric],
                    'difference': diff,
                    'improvement': diff > 0
                }
        
        # Determine winner based on F1 score
        if 'f1_score' in metrics1 and 'f1_score' in metrics2:
            if metrics2['f1_score'] > metrics1['f1_score']:
                comparison['winner'] = version2
            elif metrics1['f1_score'] > metrics2['f1_score']:
                comparison['winner'] = version1
            else:
                comparison['winner'] = 'tie'
        
        return comparison

# Global model registry instance
model_registry = ModelRegistry()

def get_model_registry() -> ModelRegistry:
    """Get the global model registry instance"""
    return model_registry 