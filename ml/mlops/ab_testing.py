"""
A/B Testing Framework for ML Models

Provides comprehensive A/B testing capabilities for:
- Model comparison and performance evaluation
- Traffic splitting and routing
- Statistical significance testing
- Real-time monitoring and alerting
- Automated rollback on performance degradation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import asyncio
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import hashlib
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestStatus(Enum):
    """A/B test status enumeration"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

class TrafficSplitStrategy(Enum):
    """Traffic splitting strategies"""
    RANDOM = "random"
    HASH_BASED = "hash_based"
    GEOGRAPHIC = "geographic"
    TIME_BASED = "time_based"

@dataclass
class ModelVariant:
    """Represents a model variant in A/B test"""
    name: str
    model_name: str
    model_version: str
    traffic_percentage: float
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ABTestConfig:
    """A/B test configuration"""
    test_id: str
    test_name: str
    description: str
    variants: List[ModelVariant]
    traffic_split_strategy: TrafficSplitStrategy
    start_date: datetime
    end_date: Optional[datetime] = None
    min_sample_size: int = 1000
    significance_level: float = 0.05
    success_metrics: List[str] = field(default_factory=lambda: ["accuracy", "precision", "recall", "f1"])
    monitoring_metrics: List[str] = field(default_factory=lambda: ["response_time", "error_rate"])
    auto_rollback: bool = True
    rollback_threshold: float = 0.05  # 5% performance drop triggers rollback

@dataclass
class TestResult:
    """A/B test result for a single prediction"""
    test_id: str
    variant_name: str
    user_id: str
    prediction: Any
    actual_outcome: Optional[Any] = None
    prediction_time: datetime = field(default_factory=datetime.now)
    response_time_ms: float = 0.0
    confidence_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class ABTestManager:
    """Manages A/B tests for ML models"""
    
    def __init__(self):
        self.active_tests: Dict[str, ABTestConfig] = {}
        self.test_results: Dict[str, List[TestResult]] = {}
        self.model_cache: Dict[str, Any] = {}
        
    def create_test(self, config: ABTestConfig) -> bool:
        """Create a new A/B test"""
        
        # Validate configuration
        if not self._validate_config(config):
            return False
        
        # Check traffic percentages sum to 100%
        total_traffic = sum(variant.traffic_percentage for variant in config.variants)
        if abs(total_traffic - 100.0) > 0.01:
            logger.error(f"Traffic percentages must sum to 100%, got {total_traffic}")
            return False
        
        # Store test configuration
        self.active_tests[config.test_id] = config
        self.test_results[config.test_id] = []
        
        logger.info(f"Created A/B test: {config.test_name} ({config.test_id})")
        return True
    
    def _validate_config(self, config: ABTestConfig) -> bool:
        """Validate A/B test configuration"""
        
        if len(config.variants) < 2:
            logger.error("A/B test must have at least 2 variants")
            return False
        
        if config.min_sample_size < 100:
            logger.error("Minimum sample size should be at least 100")
            return False
        
        if not (0.01 <= config.significance_level <= 0.1):
            logger.error("Significance level should be between 0.01 and 0.1")
            return False
        
        return True
    
    def assign_variant(self, test_id: str, user_id: str) -> Optional[str]:
        """Assign user to a test variant based on traffic splitting strategy"""
        
        if test_id not in self.active_tests:
            return None
        
        config = self.active_tests[test_id]
        
        # Check if test is active
        now = datetime.now()
        if now < config.start_date or (config.end_date and now > config.end_date):
            return None
        
        # Apply traffic splitting strategy
        if config.traffic_split_strategy == TrafficSplitStrategy.HASH_BASED:
            return self._hash_based_assignment(config, user_id)
        elif config.traffic_split_strategy == TrafficSplitStrategy.RANDOM:
            return self._random_assignment(config)
        else:
            # Default to hash-based
            return self._hash_based_assignment(config, user_id)
    
    def _hash_based_assignment(self, config: ABTestConfig, user_id: str) -> str:
        """Assign variant using consistent hash-based splitting"""
        
        # Create consistent hash
        hash_input = f"{config.test_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        percentage = (hash_value % 100) + 1  # 1-100
        
        # Assign based on cumulative traffic percentages
        cumulative_percentage = 0
        for variant in config.variants:
            cumulative_percentage += variant.traffic_percentage
            if percentage <= cumulative_percentage:
                return variant.name
        
        # Fallback to first variant
        return config.variants[0].name
    
    def _random_assignment(self, config: ABTestConfig) -> str:
        """Assign variant using random splitting"""
        
        rand_value = random.random() * 100
        
        cumulative_percentage = 0
        for variant in config.variants:
            cumulative_percentage += variant.traffic_percentage
            if rand_value <= cumulative_percentage:
                return variant.name
        
        # Fallback to first variant
        return config.variants[0].name
    
    def get_model_for_variant(self, test_id: str, variant_name: str) -> Optional[Any]:
        """Get model instance for a specific variant"""
        
        if test_id not in self.active_tests:
            return None
        
        config = self.active_tests[test_id]
        variant = next((v for v in config.variants if v.name == variant_name), None)
        
        if not variant:
            return None
        
        # Check model cache
        cache_key = f"{variant.model_name}:{variant.model_version}"
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]
        
        # Load model (this would integrate with MLflow or model registry)
        # For now, return a placeholder
        logger.info(f"Loading model {variant.model_name} v{variant.model_version}")
        return None  # Replace with actual model loading logic
    
    def record_result(self, result: TestResult):
        """Record A/B test result"""
        
        if result.test_id in self.test_results:
            self.test_results[result.test_id].append(result)
            logger.debug(f"Recorded result for test {result.test_id}, variant {result.variant_name}")
    
    def get_test_statistics(self, test_id: str) -> Dict[str, Any]:
        """Calculate comprehensive test statistics"""
        
        if test_id not in self.test_results:
            return {}
        
        results = self.test_results[test_id]
        config = self.active_tests[test_id]
        
        # Group results by variant
        variant_results = {}
        for result in results:
            if result.variant_name not in variant_results:
                variant_results[result.variant_name] = []
            variant_results[result.variant_name].append(result)
        
        statistics = {
            "test_id": test_id,
            "test_name": config.test_name,
            "total_samples": len(results),
            "variants": {},
            "statistical_significance": {},
            "recommendations": []
        }
        
        # Calculate statistics for each variant
        for variant_name, variant_results_list in variant_results.items():
            variant_stats = self._calculate_variant_statistics(variant_results_list)
            statistics["variants"][variant_name] = variant_stats
        
        # Calculate statistical significance
        if len(variant_results) >= 2:
            statistics["statistical_significance"] = self._calculate_statistical_significance(
                variant_results, config.significance_level
            )
        
        # Generate recommendations
        statistics["recommendations"] = self._generate_recommendations(statistics, config)
        
        return statistics
    
    def _calculate_variant_statistics(self, results: List[TestResult]) -> Dict[str, Any]:
        """Calculate statistics for a single variant"""
        
        if not results:
            return {}
        
        # Filter results with actual outcomes
        completed_results = [r for r in results if r.actual_outcome is not None]
        
        stats = {
            "total_predictions": len(results),
            "completed_predictions": len(completed_results),
            "avg_response_time_ms": np.mean([r.response_time_ms for r in results]),
            "avg_confidence_score": np.mean([r.confidence_score for r in results if r.confidence_score > 0]),
        }
        
        if completed_results:
            # Calculate performance metrics (assuming binary classification)
            y_true = [r.actual_outcome for r in completed_results]
            y_pred = [r.prediction for r in completed_results]
            
            if len(set(y_true)) > 1:  # Ensure we have both classes
                stats.update({
                    "accuracy": accuracy_score(y_true, y_pred),
                    "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
                    "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
                    "f1": f1_score(y_true, y_pred, average='weighted', zero_division=0)
                })
        
        return stats
    
    def _calculate_statistical_significance(
        self, 
        variant_results: Dict[str, List[TestResult]], 
        significance_level: float
    ) -> Dict[str, Any]:
        """Calculate statistical significance between variants"""
        
        significance_results = {}
        variant_names = list(variant_results.keys())
        
        # Pairwise comparison
        for i in range(len(variant_names)):
            for j in range(i + 1, len(variant_names)):
                variant_a = variant_names[i]
                variant_b = variant_names[j]
                
                # Get completed results for both variants
                results_a = [r for r in variant_results[variant_a] if r.actual_outcome is not None]
                results_b = [r for r in variant_results[variant_b] if r.actual_outcome is not None]
                
                if len(results_a) < 30 or len(results_b) < 30:
                    continue  # Need minimum sample size for reliable testing
                
                # Perform t-test on accuracy/success rate
                success_a = [1 if r.prediction == r.actual_outcome else 0 for r in results_a]
                success_b = [1 if r.prediction == r.actual_outcome else 0 for r in results_b]
                
                t_stat, p_value = stats.ttest_ind(success_a, success_b)
                
                comparison_key = f"{variant_a}_vs_{variant_b}"
                significance_results[comparison_key] = {
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "is_significant": p_value < significance_level,
                    "confidence_level": 1 - significance_level,
                    "sample_size_a": len(results_a),
                    "sample_size_b": len(results_b),
                    "mean_performance_a": np.mean(success_a),
                    "mean_performance_b": np.mean(success_b)
                }
        
        return significance_results
    
    def _generate_recommendations(self, statistics: Dict[str, Any], config: ABTestConfig) -> List[str]:
        """Generate actionable recommendations based on test results"""
        
        recommendations = []
        
        # Check sample size
        if statistics["total_samples"] < config.min_sample_size:
            recommendations.append(
                f"Insufficient sample size ({statistics['total_samples']} < {config.min_sample_size}). "
                "Continue test to reach statistical power."
            )
        
        # Check for clear winner
        significance_results = statistics.get("statistical_significance", {})
        if significance_results:
            significant_comparisons = [
                comp for comp, result in significance_results.items() 
                if result["is_significant"]
            ]
            
            if significant_comparisons:
                # Find best performing variant
                variant_performance = {}
                for variant_name, variant_stats in statistics["variants"].items():
                    if "accuracy" in variant_stats:
                        variant_performance[variant_name] = variant_stats["accuracy"]
                
                if variant_performance:
                    best_variant = max(variant_performance.keys(), key=lambda x: variant_performance[x])
                    recommendations.append(
                        f"Statistically significant difference found. "
                        f"Recommend promoting '{best_variant}' variant to production."
                    )
        
        # Check for performance issues
        for variant_name, variant_stats in statistics["variants"].items():
            if variant_stats.get("avg_response_time_ms", 0) > 1000:  # 1 second threshold
                recommendations.append(
                    f"Variant '{variant_name}' has high response time "
                    f"({variant_stats['avg_response_time_ms']:.0f}ms). Consider optimization."
                )
        
        return recommendations
    
    def should_rollback(self, test_id: str) -> Tuple[bool, str]:
        """Check if test should be rolled back due to performance issues"""
        
        if test_id not in self.active_tests:
            return False, "Test not found"
        
        config = self.active_tests[test_id]
        if not config.auto_rollback:
            return False, "Auto-rollback disabled"
        
        statistics = self.get_test_statistics(test_id)
        
        # Check if any variant is performing significantly worse
        for variant_name, variant_stats in statistics["variants"].items():
            if "accuracy" in variant_stats:
                # Compare with baseline (assuming first variant is control)
                control_variant = config.variants[0].name
                control_stats = statistics["variants"].get(control_variant, {})
                
                if "accuracy" in control_stats:
                    performance_drop = control_stats["accuracy"] - variant_stats["accuracy"]
                    if performance_drop > config.rollback_threshold:
                        return True, f"Variant '{variant_name}' performance dropped by {performance_drop:.1%}"
        
        return False, "No rollback needed"
    
    def pause_test(self, test_id: str) -> bool:
        """Pause an active A/B test"""
        if test_id in self.active_tests:
            # In a real implementation, this would update test status
            logger.info(f"Paused A/B test: {test_id}")
            return True
        return False
    
    def stop_test(self, test_id: str) -> bool:
        """Stop an A/B test and finalize results"""
        if test_id in self.active_tests:
            config = self.active_tests[test_id]
            config.end_date = datetime.now()
            logger.info(f"Stopped A/B test: {test_id}")
            return True
        return False
    
    def export_results(self, test_id: str) -> Dict[str, Any]:
        """Export comprehensive test results for reporting"""
        
        if test_id not in self.test_results:
            return {}
        
        config = self.active_tests[test_id]
        statistics = self.get_test_statistics(test_id)
        
        export_data = {
            "test_configuration": {
                "test_id": config.test_id,
                "test_name": config.test_name,
                "description": config.description,
                "start_date": config.start_date.isoformat(),
                "end_date": config.end_date.isoformat() if config.end_date else None,
                "variants": [
                    {
                        "name": v.name,
                        "model_name": v.model_name,
                        "model_version": v.model_version,
                        "traffic_percentage": v.traffic_percentage
                    }
                    for v in config.variants
                ]
            },
            "test_results": statistics,
            "raw_data_sample": [
                {
                    "variant": r.variant_name,
                    "prediction_time": r.prediction_time.isoformat(),
                    "response_time_ms": r.response_time_ms,
                    "confidence_score": r.confidence_score
                }
                for r in self.test_results[test_id][:100]  # Sample of raw data
            ]
        }
        
        return export_data

# Global instance
ab_test_manager = None

def get_ab_test_manager() -> ABTestManager:
    """Get global A/B test manager instance"""
    global ab_test_manager
    if ab_test_manager is None:
        ab_test_manager = ABTestManager()
    return ab_test_manager

# Example usage
if __name__ == "__main__":
    # Initialize A/B test manager
    manager = get_ab_test_manager()
    
    # Create test configuration
    config = ABTestConfig(
        test_id="fraud_model_comparison_v1",
        test_name="Random Forest vs Gradient Boosting",
        description="Compare performance of RF and GB models for fraud detection",
        variants=[
            ModelVariant(
                name="control_rf",
                model_name="fraud_detection_random_forest",
                model_version="1.0",
                traffic_percentage=50.0,
                description="Random Forest baseline model"
            ),
            ModelVariant(
                name="treatment_gb",
                model_name="fraud_detection_gradient_boosting",
                model_version="1.0",
                traffic_percentage=50.0,
                description="Gradient Boosting experimental model"
            )
        ],
        traffic_split_strategy=TrafficSplitStrategy.HASH_BASED,
        start_date=datetime.now(),
        min_sample_size=1000,
        significance_level=0.05
    )
    
    # Create test
    success = manager.create_test(config)
    print(f"✅ A/B test created: {success}")
    
    # Simulate variant assignment
    user_id = "user_123"
    assigned_variant = manager.assign_variant(config.test_id, user_id)
    print(f"✅ User {user_id} assigned to variant: {assigned_variant}")
