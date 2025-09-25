"""
MLOps Service API for Claims Processing

This FastAPI service provides REST endpoints for:
- Model registry management
- Model monitoring and alerts
- Automated training pipeline
- A/B testing
- Model deployment workflows
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import pandas as pd
import json
import logging
import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from mlops.model_registry import ModelRegistry, ModelStage, get_model_registry
from mlops.model_monitoring import ModelMonitor, get_model_monitor
from mlops.training_pipeline import TrainingPipeline, TrainingConfig, get_training_pipeline
from models.fraud_detection import FraudDetectionModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic Models
class ModelRegistryRequest(BaseModel):
    model_name: str
    description: str = ""
    tags: List[str] = []

class ModelPromotionRequest(BaseModel):
    model_name: str
    version: str
    target_stage: str
    promoted_by: str = "api"

class TrainingRequest(BaseModel):
    model_name: str
    trigger_reason: str = "manual"
    model_types: List[str] = ["random_forest", "gradient_boosting"]
    hyperparameter_optimization: bool = True
    cross_validation_folds: int = 5
    test_size: float = 0.2
    auto_deploy_threshold: float = 0.7
    experiment_name: Optional[str] = None

class MonitoringRequest(BaseModel):
    model_name: str
    model_version: str
    current_data_days: int = 7
    reference_data_days: int = 30

class PredictionLogRequest(BaseModel):
    model_name: str
    model_version: str
    features: Dict[str, Any]
    prediction: float
    confidence: Optional[float] = None
    request_id: Optional[str] = None
    user_id: Optional[str] = None

class GroundTruthRequest(BaseModel):
    prediction_id: str
    actual_value: float

# Global instances
model_registry = get_model_registry()
model_monitor = get_model_monitor()
training_pipeline = get_training_pipeline()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting MLOps Service")
    # Initialize background monitoring
    asyncio.create_task(background_monitoring_task())
    yield
    logger.info("Shutting down MLOps Service")

# FastAPI app
app = FastAPI(
    title="Claims Processing MLOps Service",
    description="MLOps API for model lifecycle management",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "mlops-service",
        "timestamp": datetime.utcnow().isoformat()
    }

# Model Registry Endpoints
@app.get("/api/v1/models")
async def list_models(
    model_name: Optional[str] = Query(None),
    stage: Optional[str] = Query(None)
):
    """List models in the registry"""
    try:
        stage_enum = ModelStage(stage) if stage else None
        models = model_registry.list_models(model_name=model_name, stage=stage_enum)
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/models/{model_name}/{version}")
async def get_model_info(model_name: str, version: str):
    """Get detailed model information"""
    try:
        model_info = model_registry.get_model_info(model_name, version)
        if not model_info:
            raise HTTPException(status_code=404, detail="Model not found")
        return model_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/models/{model_name}/{version}/promote")
async def promote_model(model_name: str, version: str, request: ModelPromotionRequest):
    """Promote model to different stage"""
    try:
        target_stage = ModelStage(request.target_stage)
        success = model_registry.promote_model(
            model_name, version, target_stage, request.promoted_by
        )
        if not success:
            raise HTTPException(status_code=400, detail="Promotion failed")
        return {"message": f"Model promoted to {target_stage.value}"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1/models/{model_name}/{version}")
async def delete_model(model_name: str, version: str):
    """Delete a model version"""
    try:
        success = model_registry.delete_model(model_name, version)
        if not success:
            raise HTTPException(status_code=400, detail="Deletion failed")
        return {"message": "Model deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/models/{model_name}/compare/{version1}/{version2}")
async def compare_models(model_name: str, version1: str, version2: str):
    """Compare two model versions"""
    try:
        comparison = model_registry.compare_models(model_name, version1, version2)
        return comparison
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Training Pipeline Endpoints
@app.post("/api/v1/training/start")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start automated training pipeline"""
    try:
        # Create training config
        config = TrainingConfig(
            model_name=request.model_name,
            model_types=request.model_types,
            hyperparameter_optimization=request.hyperparameter_optimization,
            cross_validation_folds=request.cross_validation_folds,
            test_size=request.test_size,
            random_state=42,
            max_training_time_minutes=120,
            auto_deploy_threshold=request.auto_deploy_threshold,
            experiment_name=request.experiment_name or f"{request.model_name}_training"
        )
        
        # Generate synthetic training data
        fraud_model = FraudDetectionModel()
        training_data = fraud_model.create_synthetic_data(n_samples=10000)
        
        # Start training in background
        background_tasks.add_task(
            run_training_background,
            training_data,
            config,
            request.trigger_reason
        )
        
        return {
            "message": "Training started",
            "model_name": request.model_name,
            "experiment_name": config.experiment_name
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/training/trigger-retraining")
async def trigger_retraining(
    model_name: str,
    trigger_reason: str = "manual",
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Trigger model retraining"""
    try:
        background_tasks.add_task(
            run_retraining_background,
            model_name,
            trigger_reason
        )
        
        return {
            "message": "Retraining triggered",
            "model_name": model_name,
            "trigger_reason": trigger_reason
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Monitoring Endpoints
@app.post("/api/v1/monitoring/run")
async def run_monitoring(request: MonitoringRequest, background_tasks: BackgroundTasks):
    """Run model monitoring"""
    try:
        background_tasks.add_task(
            run_monitoring_background,
            request.model_name,
            request.model_version,
            request.current_data_days,
            request.reference_data_days
        )
        
        return {
            "message": "Monitoring started",
            "model_name": request.model_name,
            "model_version": request.model_version
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/monitoring/dashboard/{model_name}")
async def get_monitoring_dashboard(
    model_name: str,
    model_version: Optional[str] = Query(None),
    days: int = Query(7)
):
    """Get monitoring dashboard data"""
    try:
        dashboard_data = model_monitor.get_monitoring_dashboard_data(
            model_name, model_version, days
        )
        return dashboard_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/monitoring/alerts")
async def get_alerts(
    model_name: Optional[str] = Query(None),
    resolved: Optional[bool] = Query(None),
    days: int = Query(7)
):
    """Get monitoring alerts"""
    try:
        # This would query the alerts from the database
        # For now, return empty list
        return {"alerts": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/monitoring/log-prediction")
async def log_prediction(request: PredictionLogRequest):
    """Log a model prediction for monitoring"""
    try:
        model_monitor.log_prediction(
            model_name=request.model_name,
            model_version=request.model_version,
            features=request.features,
            prediction=request.prediction,
            confidence=request.confidence,
            request_id=request.request_id,
            user_id=request.user_id
        )
        
        return {"message": "Prediction logged"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/monitoring/add-ground-truth")
async def add_ground_truth(request: GroundTruthRequest):
    """Add ground truth for a logged prediction"""
    try:
        model_monitor.add_ground_truth(
            prediction_id=request.prediction_id,
            actual_value=request.actual_value
        )
        
        return {"message": "Ground truth added"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# A/B Testing Endpoints
@app.post("/api/v1/ab-testing/create")
async def create_ab_test(
    model_name: str,
    version_a: str,
    version_b: str,
    traffic_split: float = 0.5,
    test_name: Optional[str] = None
):
    """Create A/B test between two model versions"""
    try:
        # This would create an A/B test configuration
        # For now, return success message
        return {
            "message": "A/B test created",
            "test_name": test_name or f"{model_name}_ab_test",
            "version_a": version_a,
            "version_b": version_b,
            "traffic_split": traffic_split
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/ab-testing/results/{test_name}")
async def get_ab_test_results(test_name: str):
    """Get A/B test results"""
    try:
        # This would return A/B test results
        # For now, return mock data
        return {
            "test_name": test_name,
            "status": "running",
            "results": {
                "version_a": {"predictions": 100, "avg_confidence": 0.75},
                "version_b": {"predictions": 100, "avg_confidence": 0.78}
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Background Tasks
async def run_training_background(
    training_data: pd.DataFrame,
    config: TrainingConfig,
    trigger_reason: str
):
    """Background task for training pipeline"""
    try:
        logger.info(f"Starting background training for {config.model_name}")
        result = await training_pipeline.run_training_pipeline(
            training_data=training_data,
            config=config,
            trigger_reason=trigger_reason
        )
        logger.info(f"Training completed for {config.model_name}: {result['success']}")
    except Exception as e:
        logger.error(f"Background training failed: {e}")

async def run_retraining_background(model_name: str, trigger_reason: str):
    """Background task for retraining"""
    try:
        logger.info(f"Starting background retraining for {model_name}")
        result = await training_pipeline.trigger_retraining(
            model_name=model_name,
            trigger_reason=trigger_reason
        )
        logger.info(f"Retraining completed for {model_name}: {result['success']}")
    except Exception as e:
        logger.error(f"Background retraining failed: {e}")

async def run_monitoring_background(
    model_name: str,
    model_version: str,
    current_data_days: int,
    reference_data_days: int
):
    """Background task for model monitoring"""
    try:
        logger.info(f"Starting background monitoring for {model_name} v{model_version}")
        
        # Generate synthetic data for monitoring
        fraud_model = FraudDetectionModel()
        current_data = fraud_model.create_synthetic_data(n_samples=1000)
        reference_data = fraud_model.create_synthetic_data(n_samples=5000)
        
        # Add some drift to current data for demonstration
        current_data['estimated_amount'] *= 1.2  # Simulate inflation
        
        feature_columns = ['estimated_amount', 'policy_age_days', 'claim_type']
        
        result = await model_monitor.monitor_model(
            model_name=model_name,
            model_version=model_version,
            current_data=current_data,
            reference_data=reference_data,
            feature_columns=feature_columns
        )
        
        logger.info(f"Monitoring completed for {model_name} v{model_version}: {len(result['alerts'])} alerts")
        
    except Exception as e:
        logger.error(f"Background monitoring failed: {e}")

async def background_monitoring_task():
    """Continuous background monitoring task"""
    while True:
        try:
            # Run monitoring for all production models every hour
            production_models = model_registry.list_models(stage=ModelStage.PRODUCTION)
            
            for model in production_models:
                logger.info(f"Running scheduled monitoring for {model['model_name']} v{model['version']}")
                
                # Run monitoring in background
                asyncio.create_task(run_monitoring_background(
                    model['model_name'],
                    model['version'],
                    current_data_days=1,
                    reference_data_days=7
                ))
            
            # Wait 1 hour before next monitoring cycle
            await asyncio.sleep(3600)
            
        except Exception as e:
            logger.error(f"Background monitoring task failed: {e}")
            await asyncio.sleep(300)  # Wait 5 minutes before retry

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007) 