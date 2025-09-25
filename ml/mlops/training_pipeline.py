"""
Automated Training Pipeline for Claims Processing MLOps

This module provides automated model training including:
- Hyperparameter optimization
- Cross-validation
- Experiment tracking with MLflow
- Automated model deployment
- A/B testing setup
- Performance benchmarking
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import logging
import os
import sys
from pathlib import Path
import hashlib
import asyncio
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import optuna
from optuna.integration.mlflow import MLflowCallback
import joblib

# Add the parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))
from models.fraud_detection import FraudDetectionModel, FraudDetectionFeatureEngineer, ModelMetrics, ModelConfig
from mlops.model_registry import ModelRegistry, ModelStage, get_model_registry
from mlops.model_monitoring import ModelMonitor, get_model_monitor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Training pipeline configuration"""
    model_name: str
    model_types: List[str]
    hyperparameter_optimization: bool
    cross_validation_folds: int
    test_size: float
    random_state: int
    max_training_time_minutes: int
    auto_deploy_threshold: float  # F1 score threshold for auto-deployment
    experiment_name: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ExperimentResult:
    """Training experiment result"""
    experiment_id: str
    run_id: str
    model_type: str
    hyperparameters: Dict[str, Any]
    metrics: ModelMetrics
    model_path: str
    training_time_seconds: float
    cross_val_scores: List[float]
    feature_importance: Optional[Dict[str, float]]
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['metrics'] = self.metrics.to_dict()
        return result

class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna"""
    
    def __init__(self, n_trials: int = 50, timeout_minutes: int = 60):
        self.n_trials = n_trials
        self.timeout_seconds = timeout_minutes * 60
        
    def optimize_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Optimize Random Forest hyperparameters"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
                'random_state': 42
            }
            
            model = RandomForestClassifier(**params)
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout_seconds)
        
        return study.best_params
    
    def optimize_gradient_boosting(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Optimize Gradient Boosting hyperparameters"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': 42
            }
            
            model = GradientBoostingClassifier(**params)
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout_seconds)
        
        return study.best_params
    
    def optimize_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Optimize Logistic Regression hyperparameters"""
        
        def objective(trial):
            params = {
                'C': trial.suggest_float('C', 0.01, 100, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
                'solver': 'saga',  # Supports all penalties
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
                'max_iter': 1000,
                'random_state': 42
            }
            
            # ElasticNet requires l1_ratio
            if params['penalty'] == 'elasticnet':
                params['l1_ratio'] = trial.suggest_float('l1_ratio', 0, 1)
            
            model = LogisticRegression(**params)
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout_seconds)
        
        return study.best_params

class TrainingPipeline:
    """Automated model training pipeline"""
    
    def __init__(self, mlflow_tracking_uri: str = "http://localhost:5000"):
        self.model_registry = get_model_registry()
        self.model_monitor = get_model_monitor()
        self.optimizer = HyperparameterOptimizer()
        
        # Configure MLflow
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        logger.info(f"Training pipeline initialized with MLflow at {mlflow_tracking_uri}")
    
    async def run_training_pipeline(
        self,
        training_data: pd.DataFrame,
        config: TrainingConfig,
        target_column: str = 'is_fraud',
        trigger_reason: str = "manual",
        baseline_model_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run complete training pipeline"""
        
        pipeline_start_time = datetime.utcnow()
        
        logger.info(f"Starting training pipeline for {config.model_name}")
        logger.info(f"Trigger reason: {trigger_reason}")
        logger.info(f"Training data shape: {training_data.shape}")
        
        # Set MLflow experiment
        mlflow.set_experiment(config.experiment_name)
        
        # Create data hash for tracking
        data_hash = self._create_data_hash(training_data)
        
        pipeline_results = {
            'pipeline_id': str(hash(f"{config.model_name}_{pipeline_start_time}")),
            'config': config.to_dict(),
            'trigger_reason': trigger_reason,
            'data_hash': data_hash,
            'data_shape': training_data.shape,
            'started_at': pipeline_start_time.isoformat(),
            'experiments': [],
            'best_model': None,
            'deployment_decision': None,
            'completed_at': None
        }
        
        try:
            # 1. Data preparation
            logger.info("Preparing training data...")
            X_train, X_test, y_train, y_test = await self._prepare_data(
                training_data, target_column, config.test_size, config.random_state
            )
            
            # 2. Feature engineering
            logger.info("Running feature engineering...")
            feature_engineer = FraudDetectionFeatureEngineer()
            X_train_features = feature_engineer.fit(X_train).transform(X_train)
            X_test_features = feature_engineer.transform(X_test)
            
            feature_columns = list(X_train_features.columns)
            
            # 3. Train multiple models
            logger.info(f"Training {len(config.model_types)} model types...")
            
            for model_type in config.model_types:
                logger.info(f"Training {model_type}...")
                
                experiment_result = await self._train_model(
                    model_type=model_type,
                    X_train=X_train_features,
                    X_test=X_test_features,
                    y_train=y_train,
                    y_test=y_test,
                    feature_columns=feature_columns,
                    config=config,
                    feature_engineer=feature_engineer
                )
                
                pipeline_results['experiments'].append(experiment_result.to_dict())
            
            # 4. Select best model
            logger.info("Selecting best model...")
            best_experiment = max(
                pipeline_results['experiments'],
                key=lambda x: x['metrics']['f1_score']
            )
            pipeline_results['best_model'] = best_experiment
            
            logger.info(f"Best model: {best_experiment['model_type']} (F1: {best_experiment['metrics']['f1_score']:.4f})")
            
            # 5. Model comparison and deployment decision
            deployment_decision = await self._make_deployment_decision(
                best_experiment,
                config,
                baseline_model_version
            )
            pipeline_results['deployment_decision'] = deployment_decision
            
            # 6. Deploy if decision is positive
            if deployment_decision['should_deploy']:
                logger.info("Deploying new model...")
                deployment_result = await self._deploy_model(
                    best_experiment,
                    config,
                    feature_engineer,
                    deployment_decision['target_stage']
                )
                pipeline_results['deployment_result'] = deployment_result
            else:
                logger.info(f"Model not deployed: {deployment_decision['reason']}")
            
            pipeline_results['completed_at'] = datetime.utcnow().isoformat()
            pipeline_results['success'] = True
            
            logger.info("Training pipeline completed successfully")
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            pipeline_results['completed_at'] = datetime.utcnow().isoformat()
            pipeline_results['success'] = False
            pipeline_results['error'] = str(e)
            raise
    
    async def _prepare_data(
        self,
        data: pd.DataFrame,
        target_column: str,
        test_size: float,
        random_state: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare training and test data"""
        
        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        logger.info(f"Fraud rate - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}")
        
        return X_train, X_test, y_train, y_test
    
    async def _train_model(
        self,
        model_type: str,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        feature_columns: List[str],
        config: TrainingConfig,
        feature_engineer: FraudDetectionFeatureEngineer
    ) -> ExperimentResult:
        """Train a single model with hyperparameter optimization"""
        
        training_start_time = datetime.utcnow()
        
        with mlflow.start_run(run_name=f"{config.model_name}_{model_type}"):
            # Log basic info
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("training_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("feature_count", len(feature_columns))
            
            # Hyperparameter optimization
            if config.hyperparameter_optimization:
                logger.info(f"Optimizing hyperparameters for {model_type}...")
                
                if model_type == 'random_forest':
                    best_params = self.optimizer.optimize_random_forest(X_train, y_train)
                    model = RandomForestClassifier(**best_params)
                elif model_type == 'gradient_boosting':
                    best_params = self.optimizer.optimize_gradient_boosting(X_train, y_train)
                    model = GradientBoostingClassifier(**best_params)
                elif model_type == 'logistic_regression':
                    best_params = self.optimizer.optimize_logistic_regression(X_train, y_train)
                    model = LogisticRegression(**best_params)
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")
                
                mlflow.log_params(best_params)
            else:
                # Use default parameters
                if model_type == 'random_forest':
                    model = RandomForestClassifier(n_estimators=100, random_state=config.random_state)
                    best_params = model.get_params()
                elif model_type == 'gradient_boosting':
                    model = GradientBoostingClassifier(n_estimators=100, random_state=config.random_state)
                    best_params = model.get_params()
                elif model_type == 'logistic_regression':
                    model = LogisticRegression(random_state=config.random_state, max_iter=1000)
                    best_params = model.get_params()
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")
            
            # Cross-validation
            logger.info(f"Running {config.cross_validation_folds}-fold cross-validation...")
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=config.cross_validation_folds,
                scoring='f1'
            )
            
            mlflow.log_metric("cv_f1_mean", cv_scores.mean())
            mlflow.log_metric("cv_f1_std", cv_scores.std())
            
            # Train final model
            logger.info("Training final model...")
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            training_time = (datetime.utcnow() - training_start_time).total_seconds()
            
            metrics = ModelMetrics(
                accuracy=accuracy_score(y_test, y_pred),
                precision=precision_score(y_test, y_pred, zero_division=0),
                recall=recall_score(y_test, y_pred, zero_division=0),
                f1_score=f1_score(y_test, y_pred, zero_division=0),
                auc_score=roc_auc_score(y_test, y_pred_proba),
                training_samples=len(X_train),
                validation_samples=len(X_test),
                feature_count=len(feature_columns),
                training_time_seconds=training_time
            )
            
            # Log metrics to MLflow
            mlflow.log_metrics(metrics.to_dict())
            
            # Feature importance
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(feature_columns, model.feature_importances_))
                
                # Log top 10 most important features
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                for feature, importance in top_features:
                    mlflow.log_metric(f"feature_importance_{feature}", importance)
            
            # Save model artifact
            model_path = f"models/{model_type}_{mlflow.active_run().info.run_id}"
            mlflow.sklearn.log_model(model, model_path)
            
            # Create model config
            model_config = ModelConfig(
                model_type=model_type,
                hyperparameters=best_params,
                feature_columns=feature_columns,
                target_column='is_fraud',
                preprocessing_steps=['feature_engineering']
            )
            
            # Log config
            mlflow.log_dict(model_config.to_dict(), "model_config.json")
            
            experiment_result = ExperimentResult(
                experiment_id=mlflow.active_run().info.experiment_id,
                run_id=mlflow.active_run().info.run_id,
                model_type=model_type,
                hyperparameters=best_params,
                metrics=metrics,
                model_path=model_path,
                training_time_seconds=training_time,
                cross_val_scores=cv_scores.tolist(),
                feature_importance=feature_importance
            )
            
            logger.info(f"Model training completed - F1: {metrics.f1_score:.4f}, AUC: {metrics.auc_score:.4f}")
            
            return experiment_result
    
    async def _make_deployment_decision(
        self,
        best_experiment: Dict[str, Any],
        config: TrainingConfig,
        baseline_model_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Make deployment decision based on model performance"""
        
        decision = {
            'should_deploy': False,
            'reason': '',
            'target_stage': ModelStage.STAGING,
            'comparison_results': None
        }
        
        best_f1 = best_experiment['metrics']['f1_score']
        
        # Check minimum performance threshold
        if best_f1 < config.auto_deploy_threshold:
            decision['reason'] = f"Performance below threshold ({best_f1:.4f} < {config.auto_deploy_threshold:.4f})"
            return decision
        
        # Compare with baseline model if provided
        if baseline_model_version:
            try:
                baseline_info = self.model_registry.get_model_info(
                    config.model_name, baseline_model_version
                )
                
                if baseline_info:
                    baseline_f1 = baseline_info['metrics'].get('f1_score', 0)
                    improvement = best_f1 - baseline_f1
                    
                    decision['comparison_results'] = {
                        'baseline_f1': baseline_f1,
                        'new_f1': best_f1,
                        'improvement': improvement,
                        'relative_improvement': improvement / baseline_f1 if baseline_f1 > 0 else float('inf')
                    }
                    
                    # Require at least 1% improvement
                    if improvement < 0.01:
                        decision['reason'] = f"Insufficient improvement ({improvement:.4f} < 0.01)"
                        return decision
                    
                    # Deploy to production if improvement is significant (>5%)
                    if improvement > 0.05:
                        decision['target_stage'] = ModelStage.PRODUCTION
                        
            except Exception as e:
                logger.warning(f"Could not compare with baseline model: {e}")
        
        # Deploy the model
        decision['should_deploy'] = True
        decision['reason'] = f"Performance meets criteria (F1: {best_f1:.4f})"
        
        return decision
    
    async def _deploy_model(
        self,
        experiment: Dict[str, Any],
        config: TrainingConfig,
        feature_engineer: FraudDetectionFeatureEngineer,
        target_stage: ModelStage
    ) -> Dict[str, Any]:
        """Deploy the best model to the registry"""
        
        try:
            # Load model from MLflow
            model_uri = f"runs:/{experiment['run_id']}/{experiment['model_path']}"
            model = mlflow.sklearn.load_model(model_uri)
            
            # Create model config and metrics objects
            model_config = ModelConfig.from_dict(experiment['hyperparameters'])
            model_config.model_type = experiment['model_type']
            model_config.feature_columns = experiment.get('feature_columns', [])
            model_config.target_column = 'is_fraud'
            model_config.preprocessing_steps = ['feature_engineering']
            
            model_metrics = ModelMetrics.from_dict(experiment['metrics'])
            
            # Register model
            model_id = self.model_registry.register_model(
                model_name=config.model_name,
                model_obj=model,
                config=model_config,
                metrics=model_metrics,
                stage=target_stage,
                description=f"Automated training - {experiment['model_type']} - F1: {model_metrics.f1_score:.4f}",
                tags=['automated', experiment['model_type']],
                trained_by='training_pipeline'
            )
            
            # Get version from registry
            model_versions = self.model_registry.list_models(model_name=config.model_name)
            latest_version = max(model_versions, key=lambda x: x['created_at'])['version']
            
            deployment_result = {
                'model_id': model_id,
                'model_version': latest_version,
                'stage': target_stage.value,
                'mlflow_run_id': experiment['run_id'],
                'deployment_timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Model deployed: {config.model_name} v{latest_version} to {target_stage.value}")
            
            return deployment_result
            
        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            raise
    
    def _create_data_hash(self, data: pd.DataFrame) -> str:
        """Create a hash of the training data for tracking"""
        data_string = pd.util.hash_pandas_object(data).sum()
        return hashlib.md5(str(data_string).encode()).hexdigest()
    
    async def trigger_retraining(
        self,
        model_name: str,
        trigger_reason: str,
        training_data: Optional[pd.DataFrame] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Trigger model retraining"""
        
        logger.info(f"Retraining triggered for {model_name}: {trigger_reason}")
        
        # Default training config
        default_config = TrainingConfig(
            model_name=model_name,
            model_types=['random_forest', 'gradient_boosting', 'logistic_regression'],
            hyperparameter_optimization=True,
            cross_validation_folds=5,
            test_size=0.2,
            random_state=42,
            max_training_time_minutes=120,
            auto_deploy_threshold=0.7,
            experiment_name=f"{model_name}_retraining"
        )
        
        # Apply config overrides
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(default_config, key):
                    setattr(default_config, key, value)
        
        # Generate training data if not provided
        if training_data is None:
            logger.info("Generating synthetic training data for retraining")
            fraud_model = FraudDetectionModel()
            training_data = fraud_model.create_synthetic_data(n_samples=10000)
        
        # Get current production model for comparison
        try:
            production_models = self.model_registry.list_models(
                model_name=model_name,
                stage=ModelStage.PRODUCTION
            )
            baseline_version = production_models[0]['version'] if production_models else None
        except:
            baseline_version = None
        
        # Run training pipeline
        return await self.run_training_pipeline(
            training_data=training_data,
            config=default_config,
            trigger_reason=trigger_reason,
            baseline_model_version=baseline_version
        )

# Global training pipeline instance
training_pipeline = TrainingPipeline()

def get_training_pipeline() -> TrainingPipeline:
    """Get the global training pipeline instance"""
    return training_pipeline 