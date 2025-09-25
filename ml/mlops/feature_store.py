"""
Sophisticated Feature Store for ML Pipeline

Provides comprehensive feature management including:
- Feature definition and versioning
- Real-time feature serving
- Feature lineage and governance
- Data quality monitoring
- Feature transformation pipelines
- Historical feature point-in-time lookups
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import hashlib
from pathlib import Path
import sqlite3
import redis
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureType(Enum):
    """Feature data types"""
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    CATEGORICAL = "categorical"
    EMBEDDING = "embedding"

class FeatureStatus(Enum):
    """Feature lifecycle status"""
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"

@dataclass
class FeatureDefinition:
    """Comprehensive feature definition"""
    name: str
    feature_type: FeatureType
    description: str
    owner: str
    version: str = "1.0"
    status: FeatureStatus = FeatureStatus.DRAFT
    tags: List[str] = field(default_factory=list)
    transformation_logic: str = ""
    dependencies: List[str] = field(default_factory=list)
    data_sources: List[str] = field(default_factory=list)
    refresh_frequency: str = "daily"  # daily, hourly, real-time
    retention_days: int = 365
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FeatureValue:
    """Feature value with metadata"""
    feature_name: str
    entity_id: str
    value: Any
    timestamp: datetime
    version: str = "1.0"
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class FeatureStore(ABC):
    """Abstract base class for feature stores"""
    
    @abstractmethod
    def register_feature(self, feature_def: FeatureDefinition) -> bool:
        """Register a new feature definition"""
        pass
    
    @abstractmethod
    def get_feature(self, feature_name: str, entity_id: str, timestamp: Optional[datetime] = None) -> Optional[FeatureValue]:
        """Get feature value for entity at specific timestamp"""
        pass
    
    @abstractmethod
    def batch_get_features(self, feature_names: List[str], entity_ids: List[str], timestamp: Optional[datetime] = None) -> pd.DataFrame:
        """Get multiple features for multiple entities"""
        pass
    
    @abstractmethod
    def write_features(self, features: List[FeatureValue]) -> bool:
        """Write feature values to store"""
        pass

class InMemoryFeatureStore(FeatureStore):
    """In-memory feature store implementation for development/testing"""
    
    def __init__(self):
        self.feature_definitions: Dict[str, FeatureDefinition] = {}
        self.feature_values: Dict[str, List[FeatureValue]] = {}  # feature_name -> list of values
        self.entity_index: Dict[str, Dict[str, List[FeatureValue]]] = {}  # entity_id -> feature_name -> values
    
    def register_feature(self, feature_def: FeatureDefinition) -> bool:
        """Register feature definition"""
        try:
            self.feature_definitions[feature_def.name] = feature_def
            if feature_def.name not in self.feature_values:
                self.feature_values[feature_def.name] = []
            
            logger.info(f"Registered feature: {feature_def.name} v{feature_def.version}")
            return True
        except Exception as e:
            logger.error(f"Failed to register feature {feature_def.name}: {e}")
            return False
    
    def get_feature(self, feature_name: str, entity_id: str, timestamp: Optional[datetime] = None) -> Optional[FeatureValue]:
        """Get feature value for entity at timestamp"""
        
        if feature_name not in self.feature_definitions:
            return None
        
        if entity_id not in self.entity_index:
            return None
        
        if feature_name not in self.entity_index[entity_id]:
            return None
        
        values = self.entity_index[entity_id][feature_name]
        
        if timestamp is None:
            # Return latest value
            return max(values, key=lambda x: x.timestamp) if values else None
        
        # Point-in-time lookup
        valid_values = [v for v in values if v.timestamp <= timestamp]
        return max(valid_values, key=lambda x: x.timestamp) if valid_values else None
    
    def batch_get_features(self, feature_names: List[str], entity_ids: List[str], timestamp: Optional[datetime] = None) -> pd.DataFrame:
        """Get multiple features for multiple entities"""
        
        results = []
        
        for entity_id in entity_ids:
            row = {"entity_id": entity_id}
            
            for feature_name in feature_names:
                feature_value = self.get_feature(feature_name, entity_id, timestamp)
                row[feature_name] = feature_value.value if feature_value else None
                row[f"{feature_name}_timestamp"] = feature_value.timestamp if feature_value else None
                row[f"{feature_name}_confidence"] = feature_value.confidence if feature_value else None
            
            results.append(row)
        
        return pd.DataFrame(results)
    
    def write_features(self, features: List[FeatureValue]) -> bool:
        """Write feature values to store"""
        
        try:
            for feature_value in features:
                feature_name = feature_value.feature_name
                entity_id = feature_value.entity_id
                
                # Add to feature values list
                if feature_name not in self.feature_values:
                    self.feature_values[feature_name] = []
                self.feature_values[feature_name].append(feature_value)
                
                # Add to entity index
                if entity_id not in self.entity_index:
                    self.entity_index[entity_id] = {}
                if feature_name not in self.entity_index[entity_id]:
                    self.entity_index[entity_id][feature_name] = []
                self.entity_index[entity_id][feature_name].append(feature_value)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to write features: {e}")
            return False

class SQLiteFeatureStore(FeatureStore):
    """SQLite-based feature store for persistent storage"""
    
    def __init__(self, db_path: str = "feature_store.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database schema"""
        
        with sqlite3.connect(self.db_path) as conn:
            # Feature definitions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_definitions (
                    name TEXT PRIMARY KEY,
                    feature_type TEXT NOT NULL,
                    description TEXT,
                    owner TEXT,
                    version TEXT,
                    status TEXT,
                    tags TEXT,
                    transformation_logic TEXT,
                    dependencies TEXT,
                    data_sources TEXT,
                    refresh_frequency TEXT,
                    retention_days INTEGER,
                    validation_rules TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            # Feature values table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_values (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feature_name TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    value TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    version TEXT,
                    confidence REAL,
                    metadata TEXT,
                    FOREIGN KEY (feature_name) REFERENCES feature_definitions (name)
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feature_entity_time ON feature_values (feature_name, entity_id, timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_time ON feature_values (entity_id, timestamp)")
            
            conn.commit()
    
    def register_feature(self, feature_def: FeatureDefinition) -> bool:
        """Register feature definition"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO feature_definitions 
                    (name, feature_type, description, owner, version, status, tags, 
                     transformation_logic, dependencies, data_sources, refresh_frequency, 
                     retention_days, validation_rules, created_at, updated_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    feature_def.name,
                    feature_def.feature_type.value,
                    feature_def.description,
                    feature_def.owner,
                    feature_def.version,
                    feature_def.status.value,
                    json.dumps(feature_def.tags),
                    feature_def.transformation_logic,
                    json.dumps(feature_def.dependencies),
                    json.dumps(feature_def.data_sources),
                    feature_def.refresh_frequency,
                    feature_def.retention_days,
                    json.dumps(feature_def.validation_rules),
                    feature_def.created_at,
                    feature_def.updated_at,
                    json.dumps(feature_def.metadata)
                ))
                conn.commit()
            
            logger.info(f"Registered feature: {feature_def.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register feature: {e}")
            return False
    
    def get_feature(self, feature_name: str, entity_id: str, timestamp: Optional[datetime] = None) -> Optional[FeatureValue]:
        """Get feature value for entity at timestamp"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                if timestamp is None:
                    # Get latest value
                    cursor = conn.execute("""
                        SELECT * FROM feature_values 
                        WHERE feature_name = ? AND entity_id = ?
                        ORDER BY timestamp DESC LIMIT 1
                    """, (feature_name, entity_id))
                else:
                    # Point-in-time lookup
                    cursor = conn.execute("""
                        SELECT * FROM feature_values 
                        WHERE feature_name = ? AND entity_id = ? AND timestamp <= ?
                        ORDER BY timestamp DESC LIMIT 1
                    """, (feature_name, entity_id, timestamp))
                
                row = cursor.fetchone()
                
                if row:
                    return FeatureValue(
                        feature_name=row['feature_name'],
                        entity_id=row['entity_id'],
                        value=json.loads(row['value']) if row['value'].startswith(('[', '{')) else row['value'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        version=row['version'],
                        confidence=row['confidence'],
                        metadata=json.loads(row['metadata']) if row['metadata'] else {}
                    )
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get feature: {e}")
            return None
    
    def batch_get_features(self, feature_names: List[str], entity_ids: List[str], timestamp: Optional[datetime] = None) -> pd.DataFrame:
        """Get multiple features for multiple entities"""
        
        results = []
        
        for entity_id in entity_ids:
            row = {"entity_id": entity_id}
            
            for feature_name in feature_names:
                feature_value = self.get_feature(feature_name, entity_id, timestamp)
                row[feature_name] = feature_value.value if feature_value else None
                row[f"{feature_name}_timestamp"] = feature_value.timestamp if feature_value else None
                row[f"{feature_name}_confidence"] = feature_value.confidence if feature_value else None
            
            results.append(row)
        
        return pd.DataFrame(results)
    
    def write_features(self, features: List[FeatureValue]) -> bool:
        """Write feature values to store"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                for feature_value in features:
                    value_json = json.dumps(feature_value.value) if isinstance(feature_value.value, (dict, list)) else str(feature_value.value)
                    
                    conn.execute("""
                        INSERT INTO feature_values 
                        (feature_name, entity_id, value, timestamp, version, confidence, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        feature_value.feature_name,
                        feature_value.entity_id,
                        value_json,
                        feature_value.timestamp.isoformat(),
                        feature_value.version,
                        feature_value.confidence,
                        json.dumps(feature_value.metadata)
                    ))
                
                conn.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to write features: {e}")
            return False

class FeatureTransformationEngine:
    """Engine for applying feature transformations"""
    
    def __init__(self, feature_store: FeatureStore):
        self.feature_store = feature_store
        self.transformations: Dict[str, callable] = {}
    
    def register_transformation(self, name: str, func: callable):
        """Register a transformation function"""
        self.transformations[name] = func
        logger.info(f"Registered transformation: {name}")
    
    def apply_transformation(self, transformation_name: str, data: pd.DataFrame) -> pd.DataFrame:
        """Apply transformation to data"""
        
        if transformation_name not in self.transformations:
            logger.error(f"Transformation {transformation_name} not found")
            return data
        
        try:
            return self.transformations[transformation_name](data)
        except Exception as e:
            logger.error(f"Failed to apply transformation {transformation_name}: {e}")
            return data

class FeatureQualityMonitor:
    """Monitor feature quality and data drift"""
    
    def __init__(self, feature_store: FeatureStore):
        self.feature_store = feature_store
        self.quality_metrics: Dict[str, Dict[str, Any]] = {}
    
    def calculate_quality_metrics(self, feature_name: str, entity_ids: List[str]) -> Dict[str, Any]:
        """Calculate quality metrics for a feature"""
        
        # Get recent feature values
        df = self.feature_store.batch_get_features([feature_name], entity_ids)
        
        if df.empty or df[feature_name].isna().all():
            return {"error": "No valid feature values found"}
        
        values = df[feature_name].dropna()
        
        metrics = {
            "completeness": len(values) / len(df),
            "null_rate": df[feature_name].isna().sum() / len(df),
            "unique_values": len(values.unique()) if len(values) > 0 else 0,
            "calculated_at": datetime.now().isoformat()
        }
        
        # Numeric metrics
        if pd.api.types.is_numeric_dtype(values):
            metrics.update({
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "median": float(values.median()),
                "q25": float(values.quantile(0.25)),
                "q75": float(values.quantile(0.75))
            })
        
        # Categorical metrics
        elif pd.api.types.is_object_dtype(values):
            value_counts = values.value_counts()
            metrics.update({
                "most_common_value": value_counts.index[0] if len(value_counts) > 0 else None,
                "most_common_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                "cardinality": len(value_counts)
            })
        
        self.quality_metrics[feature_name] = metrics
        return metrics
    
    def detect_drift(self, feature_name: str, reference_metrics: Dict[str, Any], current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Detect data drift between reference and current metrics"""
        
        drift_results = {
            "feature_name": feature_name,
            "drift_detected": False,
            "drift_score": 0.0,
            "alerts": []
        }
        
        # Check completeness drift
        completeness_diff = abs(current_metrics.get("completeness", 0) - reference_metrics.get("completeness", 0))
        if completeness_diff > 0.1:  # 10% threshold
            drift_results["alerts"].append(f"Completeness drift: {completeness_diff:.2%}")
            drift_results["drift_detected"] = True
        
        # Check distribution drift for numeric features
        if "mean" in reference_metrics and "mean" in current_metrics:
            mean_diff = abs(current_metrics["mean"] - reference_metrics["mean"])
            std_diff = abs(current_metrics["std"] - reference_metrics["std"])
            
            # Normalize by reference standard deviation
            ref_std = reference_metrics.get("std", 1)
            if ref_std > 0:
                normalized_mean_diff = mean_diff / ref_std
                if normalized_mean_diff > 2:  # 2 standard deviations
                    drift_results["alerts"].append(f"Mean drift: {normalized_mean_diff:.2f} std devs")
                    drift_results["drift_detected"] = True
                    drift_results["drift_score"] = max(drift_results["drift_score"], normalized_mean_diff / 2)
        
        return drift_results

class FeatureLineageTracker:
    """Track feature lineage and dependencies"""
    
    def __init__(self):
        self.lineage_graph: Dict[str, Dict[str, List[str]]] = {
            "upstream": {},    # feature -> list of upstream dependencies
            "downstream": {}   # feature -> list of downstream dependents
        }
    
    def add_dependency(self, feature_name: str, dependency: str):
        """Add a dependency relationship"""
        
        # Add upstream dependency
        if feature_name not in self.lineage_graph["upstream"]:
            self.lineage_graph["upstream"][feature_name] = []
        self.lineage_graph["upstream"][feature_name].append(dependency)
        
        # Add downstream dependent
        if dependency not in self.lineage_graph["downstream"]:
            self.lineage_graph["downstream"][dependency] = []
        self.lineage_graph["downstream"][dependency].append(feature_name)
    
    def get_upstream_dependencies(self, feature_name: str) -> List[str]:
        """Get all upstream dependencies for a feature"""
        return self.lineage_graph["upstream"].get(feature_name, [])
    
    def get_downstream_dependents(self, feature_name: str) -> List[str]:
        """Get all downstream dependents for a feature"""
        return self.lineage_graph["downstream"].get(feature_name, [])
    
    def get_full_lineage(self, feature_name: str) -> Dict[str, Any]:
        """Get complete lineage information for a feature"""
        
        def get_all_upstream(feature: str, visited: set = None) -> List[str]:
            if visited is None:
                visited = set()
            
            if feature in visited:
                return []
            
            visited.add(feature)
            upstream = self.get_upstream_dependencies(feature)
            all_upstream = upstream.copy()
            
            for dep in upstream:
                all_upstream.extend(get_all_upstream(dep, visited.copy()))
            
            return list(set(all_upstream))
        
        def get_all_downstream(feature: str, visited: set = None) -> List[str]:
            if visited is None:
                visited = set()
            
            if feature in visited:
                return []
            
            visited.add(feature)
            downstream = self.get_downstream_dependents(feature)
            all_downstream = downstream.copy()
            
            for dep in downstream:
                all_downstream.extend(get_all_downstream(dep, visited.copy()))
            
            return list(set(all_downstream))
        
        return {
            "feature_name": feature_name,
            "direct_upstream": self.get_upstream_dependencies(feature_name),
            "direct_downstream": self.get_downstream_dependents(feature_name),
            "all_upstream": get_all_upstream(feature_name),
            "all_downstream": get_all_downstream(feature_name)
        }

# Example fraud detection features
def create_fraud_detection_features() -> List[FeatureDefinition]:
    """Create comprehensive fraud detection features"""
    
    features = [
        FeatureDefinition(
            name="claim_amount_log",
            feature_type=FeatureType.FLOAT,
            description="Log-transformed claim amount to handle skewness",
            owner="ml_team",
            transformation_logic="np.log1p(claim_amount)",
            dependencies=["claim_amount"],
            validation_rules={"min_value": 0, "max_value": 20}
        ),
        FeatureDefinition(
            name="days_since_policy_start",
            feature_type=FeatureType.INTEGER,
            description="Number of days between policy start and claim date",
            owner="ml_team",
            transformation_logic="(claim_date - policy_start_date).days",
            dependencies=["claim_date", "policy_start_date"],
            validation_rules={"min_value": 0, "max_value": 3650}
        ),
        FeatureDefinition(
            name="customer_claim_frequency",
            feature_type=FeatureType.FLOAT,
            description="Customer's historical claim frequency (claims per year)",
            owner="ml_team",
            transformation_logic="count(claims) / years_as_customer",
            dependencies=["customer_id", "historical_claims"],
            refresh_frequency="daily"
        ),
        FeatureDefinition(
            name="location_risk_score",
            feature_type=FeatureType.FLOAT,
            description="Risk score based on claim location",
            owner="risk_team",
            transformation_logic="location_risk_mapping[location]",
            dependencies=["claim_location"],
            validation_rules={"min_value": 0, "max_value": 1}
        ),
        FeatureDefinition(
            name="time_based_features",
            feature_type=FeatureType.EMBEDDING,
            description="Time-based features (hour, day of week, month)",
            owner="ml_team",
            transformation_logic="extract_time_features(claim_timestamp)",
            dependencies=["claim_timestamp"]
        )
    ]
    
    return features

# Global instances
feature_store = None
transformation_engine = None
quality_monitor = None
lineage_tracker = None

def get_feature_store() -> FeatureStore:
    """Get global feature store instance"""
    global feature_store
    if feature_store is None:
        feature_store = SQLiteFeatureStore()
    return feature_store

def get_transformation_engine() -> FeatureTransformationEngine:
    """Get global transformation engine instance"""
    global transformation_engine
    if transformation_engine is None:
        transformation_engine = FeatureTransformationEngine(get_feature_store())
    return transformation_engine

def get_quality_monitor() -> FeatureQualityMonitor:
    """Get global quality monitor instance"""
    global quality_monitor
    if quality_monitor is None:
        quality_monitor = FeatureQualityMonitor(get_feature_store())
    return quality_monitor

def get_lineage_tracker() -> FeatureLineageTracker:
    """Get global lineage tracker instance"""
    global lineage_tracker
    if lineage_tracker is None:
        lineage_tracker = FeatureLineageTracker()
    return lineage_tracker

# Example usage
if __name__ == "__main__":
    # Initialize feature store
    store = get_feature_store()
    
    # Register fraud detection features
    fraud_features = create_fraud_detection_features()
    for feature in fraud_features:
        store.register_feature(feature)
    
    # Create sample feature values
    sample_features = [
        FeatureValue(
            feature_name="claim_amount_log",
            entity_id="claim_001",
            value=9.21,
            timestamp=datetime.now()
        ),
        FeatureValue(
            feature_name="days_since_policy_start",
            entity_id="claim_001",
            value=45,
            timestamp=datetime.now()
        )
    ]
    
    # Write features
    success = store.write_features(sample_features)
    print(f"✅ Feature store initialized with {len(fraud_features)} features")
    print(f"✅ Sample features written: {success}")
