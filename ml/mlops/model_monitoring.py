"""
Model Monitoring System for Claims Processing MLOps

This module provides comprehensive monitoring for ML models including:
- Data drift detection
- Performance monitoring
- Alert system
- Automated retraining triggers
- Model health dashboards
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import logging
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
from evidently.test_suite import TestSuite
try:
    from evidently.tests import TestNumberOfColumnsWithMissingValues, TestNumberOfRowsWithMissingValues, TestNumberOfConstantColumns
except ImportError:
    # Fallback for different evidently versions - define dummy classes
    TestNumberOfColumnsWithMissingValues = None
    TestNumberOfRowsWithMissingValues = None  
    TestNumberOfConstantColumns = None
import asyncio
import httpx
from sqlalchemy import create_engine, Column, String, DateTime, Float, Integer, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
import uuid
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://claims:claims@localhost:5432/claims")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

@dataclass
class DriftReport:
    """Data drift analysis results"""
    timestamp: datetime
    model_name: str
    model_version: str
    drift_detected: bool
    drift_score: float
    drifted_features: List[str]
    feature_drift_scores: Dict[str, float]
    dataset_size: int
    reference_size: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'model_name': self.model_name,
            'model_version': self.model_version,
            'drift_detected': self.drift_detected,
            'drift_score': self.drift_score,
            'drifted_features': self.drifted_features,
            'feature_drift_scores': self.feature_drift_scores,
            'dataset_size': self.dataset_size,
            'reference_size': self.reference_size
        }

@dataclass
class PerformanceReport:
    """Model performance monitoring results"""
    timestamp: datetime
    model_name: str
    model_version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: Optional[float]
    prediction_count: int
    avg_confidence: float
    performance_degraded: bool
    degradation_threshold: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'model_name': self.model_name,
            'model_version': self.model_version,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'auc_score': self.auc_score,
            'prediction_count': self.prediction_count,
            'avg_confidence': self.avg_confidence,
            'performance_degraded': self.performance_degraded,
            'degradation_threshold': self.degradation_threshold
        }

@dataclass
class ModelAlert:
    """Model monitoring alert"""
    alert_id: str
    timestamp: datetime
    model_name: str
    model_version: str
    alert_type: str  # drift, performance, data_quality, prediction_volume
    severity: str    # low, medium, high, critical
    message: str
    details: Dict[str, Any]
    resolved: bool = False
    resolved_at: Optional[datetime] = None

# Database Models
class MonitoringRun(Base):
    __tablename__ = "monitoring_runs"
    __table_args__ = {'schema': 'mlops'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(20), nullable=False)
    run_type = Column(String(50), nullable=False)  # drift, performance, quality
    
    # Results
    results = Column(Text)  # JSON string
    drift_detected = Column(Boolean)
    performance_degraded = Column(Boolean)
    alerts_generated = Column(Integer, default=0)
    
    # Execution info
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    status = Column(String(20), default='running')  # running, completed, failed
    error_message = Column(Text)
    
    # Data info
    dataset_size = Column(Integer)
    reference_size = Column(Integer)
    
    created_at = Column(DateTime, default=datetime.utcnow)

class ModelAlert(Base):
    __tablename__ = "model_alerts"
    __table_args__ = {'schema': 'mlops'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(20), nullable=False)
    
    alert_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)
    message = Column(Text, nullable=False)
    details = Column(Text)  # JSON string
    
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime)
    resolved_by = Column(String(100))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    __table_args__ = {'schema': 'mlops'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(20), nullable=False)
    
    # Input features (JSON)
    features = Column(Text, nullable=False)
    
    # Prediction results
    prediction = Column(Float, nullable=False)
    confidence = Column(Float)
    prediction_class = Column(String(50))
    
    # Metadata
    request_id = Column(String(100))
    user_id = Column(String(100))
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Ground truth (for performance monitoring)
    actual_value = Column(Float)
    feedback_timestamp = Column(DateTime)

class ModelMonitor:
    """Comprehensive model monitoring system"""
    
    def __init__(self):
        # Create database tables
        Base.metadata.create_all(bind=engine)
        
        # Default thresholds
        self.drift_threshold = 0.1
        self.performance_degradation_threshold = 0.05
        self.min_samples_for_monitoring = 100
        
        # Notification settings
        self.notification_webhook = os.getenv("MONITORING_WEBHOOK_URL")
        
        logger.info("Model monitoring system initialized")
    
    def get_db(self) -> Session:
        """Get database session"""
        return SessionLocal()
    
    async def monitor_model(
        self,
        model_name: str,
        model_version: str,
        current_data: pd.DataFrame,
        reference_data: pd.DataFrame,
        feature_columns: List[str],
        target_column: Optional[str] = None,
        prediction_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run comprehensive model monitoring"""
        
        monitoring_results = {
            'model_name': model_name,
            'model_version': model_version,
            'timestamp': datetime.utcnow().isoformat(),
            'drift_report': None,
            'performance_report': None,
            'data_quality_report': None,
            'alerts': []
        }
        
        db = self.get_db()
        
        try:
            # Start monitoring run
            monitoring_run = MonitoringRun(
                model_name=model_name,
                model_version=model_version,
                run_type='comprehensive',
                dataset_size=len(current_data),
                reference_size=len(reference_data)
            )
            db.add(monitoring_run)
            db.commit()
            db.refresh(monitoring_run)
            
            # 1. Data Drift Detection
            logger.info(f"Running drift detection for {model_name} v{model_version}")
            drift_report = await self._detect_data_drift(
                current_data, reference_data, feature_columns
            )
            monitoring_results['drift_report'] = drift_report.to_dict()
            
            if drift_report.drift_detected:
                alert = await self._create_alert(
                    model_name, model_version,
                    alert_type='drift',
                    severity='high' if drift_report.drift_score > 0.2 else 'medium',
                    message=f"Data drift detected (score: {drift_report.drift_score:.3f})",
                    details=drift_report.to_dict()
                )
                monitoring_results['alerts'].append(alert)
            
            # 2. Performance Monitoring
            if target_column and prediction_column:
                logger.info(f"Running performance monitoring for {model_name} v{model_version}")
                performance_report = await self._monitor_performance(
                    current_data, target_column, prediction_column
                )
                monitoring_results['performance_report'] = performance_report.to_dict()
                
                if performance_report.performance_degraded:
                    alert = await self._create_alert(
                        model_name, model_version,
                        alert_type='performance',
                        severity='high',
                        message=f"Performance degradation detected (F1: {performance_report.f1_score:.3f})",
                        details=performance_report.to_dict()
                    )
                    monitoring_results['alerts'].append(alert)
            
            # 3. Data Quality Monitoring
            logger.info(f"Running data quality monitoring for {model_name} v{model_version}")
            quality_issues = await self._check_data_quality(current_data, feature_columns)
            monitoring_results['data_quality_report'] = quality_issues
            
            if quality_issues['critical_issues'] > 0:
                alert = await self._create_alert(
                    model_name, model_version,
                    alert_type='data_quality',
                    severity='high',
                    message=f"Critical data quality issues detected ({quality_issues['critical_issues']} issues)",
                    details=quality_issues
                )
                monitoring_results['alerts'].append(alert)
            
            # 4. Prediction Volume Monitoring
            prediction_stats = await self._monitor_prediction_volume(model_name, model_version)
            if prediction_stats['volume_anomaly']:
                alert = await self._create_alert(
                    model_name, model_version,
                    alert_type='prediction_volume',
                    severity='medium',
                    message=f"Unusual prediction volume: {prediction_stats['current_volume']}",
                    details=prediction_stats
                )
                monitoring_results['alerts'].append(alert)
            
            # Update monitoring run
            monitoring_run.completed_at = datetime.utcnow()
            monitoring_run.status = 'completed'
            monitoring_run.results = json.dumps(monitoring_results, default=str)
            monitoring_run.drift_detected = drift_report.drift_detected
            monitoring_run.performance_degraded = (
                monitoring_results['performance_report']['performance_degraded'] 
                if monitoring_results['performance_report'] else False
            )
            monitoring_run.alerts_generated = len(monitoring_results['alerts'])
            
            db.commit()
            
            # Send notifications for critical alerts
            for alert in monitoring_results['alerts']:
                if alert['severity'] in ['high', 'critical']:
                    await self._send_notification(alert)
            
            logger.info(f"Monitoring completed for {model_name} v{model_version}: {len(monitoring_results['alerts'])} alerts")
            
            return monitoring_results
            
        except Exception as e:
            logger.error(f"Monitoring failed for {model_name} v{model_version}: {e}")
            monitoring_run.status = 'failed'
            monitoring_run.error_message = str(e)
            monitoring_run.completed_at = datetime.utcnow()
            db.commit()
            raise
        finally:
            db.close()
    
    async def _detect_data_drift(
        self,
        current_data: pd.DataFrame,
        reference_data: pd.DataFrame,
        feature_columns: List[str]
    ) -> DriftReport:
        """Detect data drift using statistical tests"""
        
        drift_scores = {}
        drifted_features = []
        
        for feature in feature_columns:
            if feature not in current_data.columns or feature not in reference_data.columns:
                continue
            
            current_values = current_data[feature].dropna()
            reference_values = reference_data[feature].dropna()
            
            if len(current_values) == 0 or len(reference_values) == 0:
                continue
            
            # Use appropriate statistical test based on data type
            if pd.api.types.is_numeric_dtype(current_values):
                # Kolmogorov-Smirnov test for numerical features
                statistic, p_value = stats.ks_2samp(reference_values, current_values)
                drift_score = statistic
            else:
                # Chi-square test for categorical features
                try:
                    current_counts = current_values.value_counts()
                    reference_counts = reference_values.value_counts()
                    
                    # Align categories
                    all_categories = set(current_counts.index) | set(reference_counts.index)
                    current_aligned = [current_counts.get(cat, 0) for cat in all_categories]
                    reference_aligned = [reference_counts.get(cat, 0) for cat in all_categories]
                    
                    if sum(current_aligned) > 0 and sum(reference_aligned) > 0:
                        chi2, p_value = stats.chisquare(current_aligned, reference_aligned)
                        drift_score = min(chi2 / len(all_categories), 1.0)  # Normalize
                    else:
                        drift_score = 0.0
                        p_value = 1.0
                except:
                    drift_score = 0.0
                    p_value = 1.0
            
            drift_scores[feature] = drift_score
            
            if drift_score > self.drift_threshold:
                drifted_features.append(feature)
        
        # Overall drift score (average of feature drift scores)
        overall_drift_score = np.mean(list(drift_scores.values())) if drift_scores else 0.0
        drift_detected = overall_drift_score > self.drift_threshold
        
        return DriftReport(
            timestamp=datetime.utcnow(),
            model_name="",  # Will be set by caller
            model_version="",  # Will be set by caller
            drift_detected=drift_detected,
            drift_score=overall_drift_score,
            drifted_features=drifted_features,
            feature_drift_scores=drift_scores,
            dataset_size=len(current_data),
            reference_size=len(reference_data)
        )
    
    async def _monitor_performance(
        self,
        data: pd.DataFrame,
        target_column: str,
        prediction_column: str
    ) -> PerformanceReport:
        """Monitor model performance metrics"""
        
        y_true = data[target_column]
        y_pred = data[prediction_column]
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred_binary)
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        
        try:
            auc = roc_auc_score(y_true, y_pred)
        except:
            auc = None
        
        avg_confidence = y_pred.mean()
        prediction_count = len(data)
        
        # Check for performance degradation (would compare against baseline)
        # For now, use simple threshold
        performance_degraded = f1 < (1.0 - self.performance_degradation_threshold)
        
        return PerformanceReport(
            timestamp=datetime.utcnow(),
            model_name="",  # Will be set by caller
            model_version="",  # Will be set by caller
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_score=auc,
            prediction_count=prediction_count,
            avg_confidence=avg_confidence,
            performance_degraded=performance_degraded,
            degradation_threshold=self.performance_degradation_threshold
        )
    
    async def _check_data_quality(
        self,
        data: pd.DataFrame,
        feature_columns: List[str]
    ) -> Dict[str, Any]:
        """Check data quality issues"""
        
        quality_report = {
            'total_samples': len(data),
            'missing_values': {},
            'outliers': {},
            'data_types': {},
            'critical_issues': 0,
            'warnings': 0,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        for feature in feature_columns:
            if feature not in data.columns:
                quality_report['critical_issues'] += 1
                continue
            
            column_data = data[feature]
            
            # Missing values
            missing_pct = column_data.isnull().sum() / len(data)
            quality_report['missing_values'][feature] = missing_pct
            
            if missing_pct > 0.1:  # More than 10% missing
                quality_report['warnings'] += 1
            if missing_pct > 0.5:  # More than 50% missing
                quality_report['critical_issues'] += 1
            
            # Data type consistency
            quality_report['data_types'][feature] = str(column_data.dtype)
            
            # Outlier detection for numerical features
            if pd.api.types.is_numeric_dtype(column_data):
                Q1 = column_data.quantile(0.25)
                Q3 = column_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((column_data < lower_bound) | (column_data > upper_bound)).sum()
                outlier_pct = outliers / len(data)
                quality_report['outliers'][feature] = outlier_pct
                
                if outlier_pct > 0.1:  # More than 10% outliers
                    quality_report['warnings'] += 1
        
        return quality_report
    
    async def _monitor_prediction_volume(
        self,
        model_name: str,
        model_version: str
    ) -> Dict[str, Any]:
        """Monitor prediction volume for anomalies"""
        
        db = self.get_db()
        try:
            # Get prediction counts for last 24 hours
            last_24h = datetime.utcnow() - timedelta(hours=24)
            
            recent_predictions = db.query(PredictionLog).filter(
                PredictionLog.model_name == model_name,
                PredictionLog.model_version == model_version,
                PredictionLog.timestamp >= last_24h
            ).count()
            
            # Get historical average (last 7 days, excluding last 24h)
            week_ago = datetime.utcnow() - timedelta(days=7)
            
            historical_predictions = db.query(PredictionLog).filter(
                PredictionLog.model_name == model_name,
                PredictionLog.model_version == model_version,
                PredictionLog.timestamp >= week_ago,
                PredictionLog.timestamp < last_24h
            ).count()
            
            historical_daily_avg = historical_predictions / 6  # 6 days
            
            # Detect anomalies
            volume_anomaly = False
            anomaly_type = None
            
            if historical_daily_avg > 0:
                ratio = recent_predictions / historical_daily_avg
                if ratio < 0.5:  # 50% drop
                    volume_anomaly = True
                    anomaly_type = 'low_volume'
                elif ratio > 2.0:  # 100% increase
                    volume_anomaly = True
                    anomaly_type = 'high_volume'
            
            return {
                'current_volume': recent_predictions,
                'historical_avg': historical_daily_avg,
                'volume_anomaly': volume_anomaly,
                'anomaly_type': anomaly_type,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        finally:
            db.close()
    
    async def _create_alert(
        self,
        model_name: str,
        model_version: str,
        alert_type: str,
        severity: str,
        message: str,
        details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a monitoring alert"""
        
        db = self.get_db()
        try:
            alert = ModelAlert(
                model_name=model_name,
                model_version=model_version,
                alert_type=alert_type,
                severity=severity,
                message=message,
                details=json.dumps(details, default=str)
            )
            
            db.add(alert)
            db.commit()
            db.refresh(alert)
            
            alert_dict = {
                'id': str(alert.id),
                'model_name': model_name,
                'model_version': model_version,
                'alert_type': alert_type,
                'severity': severity,
                'message': message,
                'details': details,
                'timestamp': alert.created_at.isoformat()
            }
            
            logger.warning(f"Alert created: {alert_type} for {model_name} v{model_version} - {message}")
            
            return alert_dict
            
        finally:
            db.close()
    
    async def _send_notification(self, alert: Dict[str, Any]):
        """Send alert notification"""
        
        if not self.notification_webhook:
            logger.info("No notification webhook configured, skipping notification")
            return
        
        try:
            notification_payload = {
                'alert_id': alert['id'],
                'model': f"{alert['model_name']} v{alert['model_version']}",
                'type': alert['alert_type'],
                'severity': alert['severity'],
                'message': alert['message'],
                'timestamp': alert['timestamp']
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.notification_webhook,
                    json=notification_payload,
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    logger.info(f"Notification sent for alert {alert['id']}")
                else:
                    logger.error(f"Failed to send notification: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
    
    def log_prediction(
        self,
        model_name: str,
        model_version: str,
        features: Dict[str, Any],
        prediction: float,
        confidence: Optional[float] = None,
        prediction_class: Optional[str] = None,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """Log a model prediction for monitoring"""
        
        db = self.get_db()
        try:
            prediction_log = PredictionLog(
                model_name=model_name,
                model_version=model_version,
                features=json.dumps(features, default=str),
                prediction=prediction,
                confidence=confidence,
                prediction_class=prediction_class,
                request_id=request_id,
                user_id=user_id
            )
            
            db.add(prediction_log)
            db.commit()
            
        except Exception as e:
            logger.error(f"Failed to log prediction: {e}")
            db.rollback()
        finally:
            db.close()
    
    def add_ground_truth(
        self,
        prediction_id: str,
        actual_value: float
    ):
        """Add ground truth for a logged prediction"""
        
        db = self.get_db()
        try:
            prediction_log = db.query(PredictionLog).filter(
                PredictionLog.id == prediction_id
            ).first()
            
            if prediction_log:
                prediction_log.actual_value = actual_value
                prediction_log.feedback_timestamp = datetime.utcnow()
                db.commit()
                
        except Exception as e:
            logger.error(f"Failed to add ground truth: {e}")
            db.rollback()
        finally:
            db.close()
    
    def get_monitoring_dashboard_data(
        self,
        model_name: str,
        model_version: Optional[str] = None,
        days: int = 7
    ) -> Dict[str, Any]:
        """Get monitoring data for dashboard"""
        
        db = self.get_db()
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # Base query
            query = db.query(MonitoringRun).filter(
                MonitoringRun.model_name == model_name,
                MonitoringRun.created_at >= start_date
            )
            
            if model_version:
                query = query.filter(MonitoringRun.model_version == model_version)
            
            monitoring_runs = query.order_by(MonitoringRun.created_at.desc()).all()
            
            # Get alerts
            alert_query = db.query(ModelAlert).filter(
                ModelAlert.model_name == model_name,
                ModelAlert.created_at >= start_date
            )
            
            if model_version:
                alert_query = alert_query.filter(ModelAlert.model_version == model_version)
            
            alerts = alert_query.order_by(ModelAlert.created_at.desc()).all()
            
            # Get prediction volume
            prediction_query = db.query(PredictionLog).filter(
                PredictionLog.model_name == model_name,
                PredictionLog.timestamp >= start_date
            )
            
            if model_version:
                prediction_query = prediction_query.filter(PredictionLog.model_version == model_version)
            
            prediction_count = prediction_query.count()
            
            dashboard_data = {
                'model_name': model_name,
                'model_version': model_version,
                'period_days': days,
                'monitoring_runs': len(monitoring_runs),
                'active_alerts': len([a for a in alerts if not a.resolved]),
                'total_alerts': len(alerts),
                'prediction_volume': prediction_count,
                'last_monitoring_run': monitoring_runs[0].created_at.isoformat() if monitoring_runs else None,
                'recent_alerts': [
                    {
                        'id': str(alert.id),
                        'type': alert.alert_type,
                        'severity': alert.severity,
                        'message': alert.message,
                        'timestamp': alert.created_at.isoformat(),
                        'resolved': alert.resolved
                    }
                    for alert in alerts[:10]  # Last 10 alerts
                ]
            }
            
            return dashboard_data
            
        finally:
            db.close()

# Global monitor instance
model_monitor = ModelMonitor()

def get_model_monitor() -> ModelMonitor:
    """Get the global model monitor instance"""
    return model_monitor 