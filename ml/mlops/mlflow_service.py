"""
Enhanced MLflow Integration Service

Provides comprehensive MLflow integration for:
- Model versioning and lifecycle management
- Experiment tracking with detailed metrics
- Model registry with staging and promotion workflows
- Automated model deployment and rollback
- Performance comparison and A/B testing support
"""

import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.entities import ViewType
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLflowModelManager:
    """Enhanced MLflow model management with versioning and deployment"""
    
    def __init__(self, tracking_uri: str = "http://localhost:5000", registry_uri: str = None):
        """Initialize MLflow client with tracking and registry URIs"""
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri or tracking_uri
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        if self.registry_uri:
            mlflow.set_registry_uri(self.registry_uri)
        
        # Initialize MLflow client
        self.client = MlflowClient(tracking_uri=self.tracking_uri, registry_uri=self.registry_uri)
        
        logger.info(f"MLflow initialized with tracking_uri: {self.tracking_uri}")
    
    def create_experiment(self, experiment_name: str, description: str = "", tags: Dict[str, str] = None) -> str:
        """Create a new MLflow experiment"""
        try:
            experiment_id = mlflow.create_experiment(
                name=experiment_name,
                artifact_location=f"./mlruns/{experiment_name}",
                tags=tags or {}
            )
            
            # Set experiment description
            if description:
                self.client.set_experiment_tag(experiment_id, "description", description)
            
            logger.info(f"Created experiment: {experiment_name} (ID: {experiment_id})")
            return experiment_id
        except Exception as e:
            # Experiment might already exist
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment:
                logger.info(f"Using existing experiment: {experiment_name}")
                return experiment.experiment_id
            raise e
    
    def log_model_training(
        self,
        experiment_name: str,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str,
        model_params: Dict[str, Any],
        metrics: Dict[str, float],
        artifacts: Dict[str, str] = None,
        tags: Dict[str, str] = None
    ) -> str:
        """Log comprehensive model training run to MLflow"""
        
        # Set experiment
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            # Log parameters
            mlflow.log_params(model_params)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log tags
            if tags:
                mlflow.set_tags(tags)
            
            # Log model with signature
            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=X_train.head(5),
                registered_model_name=f"fraud_detection_{model_name.lower().replace(' ', '_')}"
            )
            
            # Log training data info
            mlflow.log_param("training_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("feature_count", X_train.shape[1])
            mlflow.log_param("target_distribution", f"{y_train.mean():.3f}")
            
            # Log feature importance if available
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Save as artifact
                importance_path = "feature_importance.csv"
                feature_importance.to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path)
                os.remove(importance_path)  # Clean up
            
            # Log additional artifacts
            if artifacts:
                for artifact_name, artifact_path in artifacts.items():
                    mlflow.log_artifact(artifact_path, artifact_name)
            
            # Log model metadata
            model_info = {
                "model_type": type(model).__name__,
                "training_date": datetime.now().isoformat(),
                "mlflow_version": mlflow.__version__,
                "run_id": run.info.run_id
            }
            
            with open("model_metadata.json", "w") as f:
                json.dump(model_info, f, indent=2)
            mlflow.log_artifact("model_metadata.json")
            os.remove("model_metadata.json")  # Clean up
            
            logger.info(f"Logged model training run: {run.info.run_id}")
            return run.info.run_id
    
    def register_model(
        self,
        model_name: str,
        run_id: str,
        description: str = "",
        tags: Dict[str, str] = None
    ) -> str:
        """Register a model from a training run"""
        
        model_uri = f"runs:/{run_id}/model"
        
        # Register model
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
            tags=tags
        )
        
        # Update model description
        if description:
            self.client.update_model_version(
                name=model_name,
                version=model_version.version,
                description=description
            )
        
        logger.info(f"Registered model: {model_name} version {model_version.version}")
        return model_version.version
    
    def promote_model(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_existing: bool = True
    ) -> bool:
        """Promote model to a specific stage (Staging, Production, Archived)"""
        
        try:
            # Archive existing models in target stage if requested
            if archive_existing and stage in ["Staging", "Production"]:
                existing_models = self.client.get_latest_versions(
                    name=model_name,
                    stages=[stage]
                )
                
                for model in existing_models:
                    self.client.transition_model_version_stage(
                        name=model_name,
                        version=model.version,
                        stage="Archived",
                        archive_existing_versions=False
                    )
                    logger.info(f"Archived existing model version {model.version}")
            
            # Promote new model to stage
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing
            )
            
            logger.info(f"Promoted {model_name} v{version} to {stage}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            return False
    
    def load_model(self, model_name: str, stage: str = "Production", version: str = None):
        """Load model from MLflow registry"""
        
        if version:
            model_uri = f"models:/{model_name}/{version}"
        else:
            model_uri = f"models:/{model_name}/{stage}"
        
        try:
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Loaded model: {model_name} from {stage}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
    
    def compare_models(
        self,
        model_name: str,
        versions: List[str] = None,
        metrics: List[str] = ["accuracy", "precision", "recall", "f1"]
    ) -> pd.DataFrame:
        """Compare performance of different model versions"""
        
        if not versions:
            # Get all versions
            all_versions = self.client.search_model_versions(f"name='{model_name}'")
            versions = [v.version for v in all_versions]
        
        comparison_data = []
        
        for version in versions:
            try:
                model_version = self.client.get_model_version(model_name, version)
                run_id = model_version.run_id
                
                # Get run metrics
                run = self.client.get_run(run_id)
                run_metrics = run.data.metrics
                
                version_data = {
                    "version": version,
                    "stage": model_version.current_stage,
                    "creation_date": datetime.fromtimestamp(model_version.creation_timestamp / 1000),
                    "run_id": run_id
                }
                
                # Add requested metrics
                for metric in metrics:
                    version_data[metric] = run_metrics.get(metric, None)
                
                comparison_data.append(version_data)
                
            except Exception as e:
                logger.warning(f"Could not get data for version {version}: {e}")
        
        return pd.DataFrame(comparison_data)
    
    def get_model_performance_history(
        self,
        model_name: str,
        days: int = 30,
        metrics: List[str] = ["accuracy", "precision", "recall", "f1"]
    ) -> pd.DataFrame:
        """Get model performance history over time"""
        
        # Get all runs for the model
        experiment_ids = [exp.experiment_id for exp in self.client.search_experiments()]
        
        runs = self.client.search_runs(
            experiment_ids=experiment_ids,
            filter_string=f"tags.model_name = '{model_name}'",
            run_view_type=ViewType.ALL,
            max_results=1000
        )
        
        history_data = []
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for run in runs:
            run_date = datetime.fromtimestamp(run.info.start_time / 1000)
            
            if run_date >= cutoff_date:
                run_data = {
                    "date": run_date,
                    "run_id": run.info.run_id,
                    "status": run.info.status
                }
                
                # Add metrics
                for metric in metrics:
                    run_data[metric] = run.data.metrics.get(metric, None)
                
                history_data.append(run_data)
        
        return pd.DataFrame(history_data).sort_values("date")
    
    def create_model_alias(self, model_name: str, version: str, alias: str) -> bool:
        """Create an alias for a specific model version"""
        try:
            self.client.set_model_version_tag(
                name=model_name,
                version=version,
                key=f"alias_{alias}",
                value="true"
            )
            logger.info(f"Created alias '{alias}' for {model_name} v{version}")
            return True
        except Exception as e:
            logger.error(f"Failed to create alias: {e}")
            return False
    
    def delete_model_version(self, model_name: str, version: str) -> bool:
        """Delete a specific model version"""
        try:
            self.client.delete_model_version(name=model_name, version=version)
            logger.info(f"Deleted {model_name} v{version}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete model version: {e}")
            return False
    
    def get_model_lineage(self, model_name: str, version: str) -> Dict[str, Any]:
        """Get model lineage information including parent runs and datasets"""
        
        try:
            model_version = self.client.get_model_version(model_name, version)
            run = self.client.get_run(model_version.run_id)
            
            lineage = {
                "model_name": model_name,
                "version": version,
                "run_id": model_version.run_id,
                "experiment_id": run.info.experiment_id,
                "parameters": run.data.params,
                "metrics": run.data.metrics,
                "tags": run.data.tags,
                "artifacts": [f.path for f in self.client.list_artifacts(model_version.run_id)],
                "creation_date": datetime.fromtimestamp(model_version.creation_timestamp / 1000),
                "current_stage": model_version.current_stage
            }
            
            return lineage
            
        except Exception as e:
            logger.error(f"Failed to get model lineage: {e}")
            return {}

# Global instance
mlflow_manager = None

def get_mlflow_manager() -> MLflowModelManager:
    """Get global MLflow manager instance"""
    global mlflow_manager
    if mlflow_manager is None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow_manager = MLflowModelManager(tracking_uri=tracking_uri)
    return mlflow_manager

# Example usage and testing functions
if __name__ == "__main__":
    # Initialize MLflow manager
    manager = get_mlflow_manager()
    
    # Create experiment
    experiment_id = manager.create_experiment(
        "fraud_detection_enhanced",
        description="Enhanced fraud detection with MLflow integration",
        tags={"project": "claims_processing", "version": "2.0"}
    )
    
    print(f"✅ MLflow integration ready with experiment: {experiment_id}")
