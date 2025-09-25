"""
Fraud Detection Model for Insurance Claims

This module contains a complete fraud detection system including:
- Feature engineering
- Model training
- Model inference
- Model evaluation and monitoring
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import joblib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudDetectionFeatureEngineer:
    """Feature engineering for fraud detection"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def fit(self, df: pd.DataFrame) -> 'FraudDetectionFeatureEngineer':
        """Fit feature engineering components"""
        features_df = self.transform(df, fit=True)
        self.feature_names = list(features_df.columns)
        return self
    
    def transform(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Transform raw claim data into ML features"""
        df = df.copy()
        
        # Basic numerical features
        features = {
            'claim_amount_log': np.log1p(df.get('estimated_amount', 0)),
            'claim_amount_normalized': df.get('estimated_amount', 0) / 100000,
            'policy_age_days': df.get('policy_age_days', 0),
            'policy_age_years': df.get('policy_age_days', 0) / 365.25,
            'description_length': df.get('description', '').astype(str).str.len(),
            'description_word_count': df.get('description', '').astype(str).str.split().str.len(),
        }
        
        # Time-based features
        if 'incident_date' in df.columns:
            incident_dates = pd.to_datetime(df['incident_date'], errors='coerce')
            features.update({
                'incident_weekday': incident_dates.dt.dayofweek,
                'incident_month': incident_dates.dt.month,
                'incident_hour': incident_dates.dt.hour,
                'is_weekend': (incident_dates.dt.dayofweek >= 5).astype(int),
                'is_holiday_season': incident_dates.dt.month.isin([11, 12, 1]).astype(int),
            })
        
        # Reporting delay
        if 'reported_date' in df.columns and 'incident_date' in df.columns:
            incident_dates = pd.to_datetime(df['incident_date'], errors='coerce')
            reported_dates = pd.to_datetime(df['reported_date'], errors='coerce')
            reporting_delay = (reported_dates - incident_dates).dt.days
            features.update({
                'reporting_delay_days': reporting_delay.fillna(0),
                'is_immediate_report': (reporting_delay <= 1).astype(int),
                'is_delayed_report': (reporting_delay > 7).astype(int),
            })
        
        # Categorical features
        categorical_features = ['claim_type']
        for col in categorical_features:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    features[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        encoded_values = []
                        for value in df[col].astype(str):
                            if value in self.label_encoders[col].classes_:
                                encoded_values.append(self.label_encoders[col].transform([value])[0])
                            else:
                                encoded_values.append(-1)  # Unknown category
                        features[f'{col}_encoded'] = encoded_values
                    else:
                        features[f'{col}_encoded'] = 0
        
        # Customer-based features (simplified for demo)
        if 'customer_id' in df.columns:
            customer_hash = df['customer_id'].astype(str).apply(lambda x: hash(x) % 1000)
            features.update({
                'customer_hash_mod_100': customer_hash % 100,
                'customer_risk_bucket': customer_hash % 10,
            })
        
        # Risk indicators
        features.update({
            'high_amount_flag': (df.get('estimated_amount', 0) > 50000).astype(int),
            'new_policy_flag': (df.get('policy_age_days', 365) < 90).astype(int),
            'round_amount_flag': (df.get('estimated_amount', 0) % 1000 == 0).astype(int),
            'short_description_flag': (df.get('description', '').astype(str).str.len() < 50).astype(int),
        })
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features)
        
        # Fill NaN values
        features_df = features_df.fillna(0)
        
        return features_df

class FraudDetectionModel:
    """Complete fraud detection model"""
    
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.feature_engineer = FraudDetectionFeatureEngineer()
        self.threshold = 0.5
        self.model_metadata = {}
        
        # Initialize model based on type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def create_synthetic_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """Create synthetic training data for demonstration"""
        np.random.seed(42)
        
        # Generate base features
        data = {
            'estimated_amount': np.random.lognormal(8, 1.5, n_samples),
            'policy_age_days': np.random.exponential(365, n_samples),
            'description': [f"Incident description {i}" for i in range(n_samples)],
            'claim_type': np.random.choice(['auto', 'home', 'health'], n_samples, p=[0.6, 0.3, 0.1]),
            'customer_id': [f"customer_{i % 1000}" for i in range(n_samples)],
        }
        
        # Generate dates
        base_date = datetime.now() - timedelta(days=365)
        incident_dates = [base_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_samples)]
        reported_dates = [
            incident + timedelta(days=max(0, np.random.poisson(2)))
            for incident in incident_dates
        ]
        
        data['incident_date'] = incident_dates
        data['reported_date'] = reported_dates
        
        df = pd.DataFrame(data)
        
        # Create fraud labels based on suspicious patterns
        fraud_probability = (
            (df['estimated_amount'] > df['estimated_amount'].quantile(0.9)) * 0.3 +
            (df['policy_age_days'] < 30) * 0.25 +
            ((df['reported_date'] - df['incident_date']).dt.days > 7) * 0.15 +
            (df['estimated_amount'] % 1000 == 0) * 0.1 +
            (df['claim_type'] == 'auto') * 0.05 +
            np.random.random(n_samples) * 0.15
        )
        
        df['is_fraud'] = (fraud_probability > 0.4).astype(int)
        
        # Add some noise to make it more realistic
        noise_indices = np.random.choice(n_samples, int(n_samples * 0.05), replace=False)
        df.loc[noise_indices, 'is_fraud'] = 1 - df.loc[noise_indices, 'is_fraud']
        
        return df
    
    def train(self, df: pd.DataFrame, target_column: str = 'is_fraud') -> Dict[str, Any]:
        """Train the fraud detection model"""
        logger.info(f"Training {self.model_type} model on {len(df)} samples")
        
        # Feature engineering
        X = self.feature_engineer.fit(df).transform(df)
        y = df[target_column]
        
        logger.info(f"Generated {X.shape[1]} features: {list(X.columns)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        # Predictions for detailed evaluation
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        
        # Find optimal threshold
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_threshold_idx = np.argmax(f1_scores)
        self.threshold = thresholds[optimal_threshold_idx]
        
        # Store model metadata
        self.model_metadata = {
            'model_type': self.model_type,
            'training_samples': len(df),
            'features': list(X.columns),
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'auc_score': auc_score,
            'optimal_threshold': self.threshold,
            'precision': classification_rep['1']['precision'],
            'recall': classification_rep['1']['recall'],
            'f1_score': classification_rep['1']['f1-score'],
            'fraud_rate': y.mean(),
            'training_date': datetime.now().isoformat(),
        }
        
        # Feature importance (for tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, self.model.feature_importances_))
            self.model_metadata['feature_importance'] = feature_importance
        
        logger.info(f"Model training complete. AUC: {auc_score:.4f}, F1: {classification_rep['1']['f1-score']:.4f}")
        
        return self.model_metadata
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Predict fraud probability for claims"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Feature engineering
        X = self.feature_engineer.transform(df)
        
        # Ensure all required features are present
        missing_features = set(self.feature_engineer.feature_names) - set(X.columns)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}. Filling with zeros.")
            for feature in missing_features:
                X[feature] = 0
        
        # Reorder columns to match training
        X = X[self.feature_engineer.feature_names]
        
        # Predict
        fraud_probabilities = self.model.predict_proba(X)[:, 1]
        predictions = (fraud_probabilities > self.threshold).astype(int)
        
        # Generate explanations
        explanations = []
        risk_factors = []
        
        for i, (_, row) in enumerate(df.iterrows()):
            explanation, factors = self._generate_explanation(row, fraud_probabilities[i])
            explanations.append(explanation)
            risk_factors.append(factors)
        
        return {
            'fraud_probabilities': fraud_probabilities.tolist(),
            'predictions': predictions.tolist(),
            'threshold': self.threshold,
            'explanations': explanations,
            'risk_factors': risk_factors,
            'model_metadata': self.model_metadata
        }
    
    def _generate_explanation(self, claim_data: pd.Series, fraud_prob: float) -> Tuple[str, List[str]]:
        """Generate human-readable explanation for prediction"""
        factors = []
        
        # Check various risk factors
        if claim_data.get('estimated_amount', 0) > 50000:
            factors.append("High claim amount")
        
        if claim_data.get('policy_age_days', 365) < 90:
            factors.append("Recently purchased policy")
        
        if len(str(claim_data.get('description', ''))) < 50:
            factors.append("Insufficient incident description")
        
        # Time-based factors
        if 'incident_date' in claim_data and 'reported_date' in claim_data:
            try:
                incident_date = pd.to_datetime(claim_data['incident_date'])
                reported_date = pd.to_datetime(claim_data['reported_date'])
                delay = (reported_date - incident_date).days
                
                if delay > 7:
                    factors.append("Delayed reporting")
                if incident_date.weekday() >= 5:
                    factors.append("Weekend incident")
            except:
                pass
        
        # Round numbers
        if claim_data.get('estimated_amount', 0) % 1000 == 0:
            factors.append("Round number claim amount")
        
        # Generate explanation
        if fraud_prob > 0.7:
            explanation = f"High fraud risk (score: {fraud_prob:.2f}). "
            if factors:
                explanation += f"Key concerns: {', '.join(factors[:3])}. "
            explanation += "Recommend manual review by adjuster."
        elif fraud_prob > 0.4:
            explanation = f"Moderate fraud risk (score: {fraud_prob:.2f}). "
            if factors:
                explanation += f"Some suspicious indicators: {', '.join(factors[:2])}. "
            explanation += "Consider additional verification."
        else:
            explanation = f"Low fraud risk (score: {fraud_prob:.2f}). Claim appears legitimate."
        
        return explanation, factors
    
    def save_model(self, model_dir: str) -> str:
        """Save the trained model and components"""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, 'fraud_detection_model.pkl')
        joblib.dump(self.model, model_path)
        
        # Save feature engineer
        feature_engineer_path = os.path.join(model_dir, 'feature_engineer.pkl')
        joblib.dump(self.feature_engineer, feature_engineer_path)
        
        # Save metadata
        metadata_path = os.path.join(model_dir, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.model_metadata, f, indent=2, default=str)
        
        # Save threshold
        config_path = os.path.join(model_dir, 'model_config.json')
        config = {
            'threshold': self.threshold,
            'model_type': self.model_type,
            'feature_names': self.feature_engineer.feature_names
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model saved to {model_dir}")
        return model_dir
    
    @classmethod
    def load_model(cls, model_dir: str) -> 'FraudDetectionModel':
        """Load a trained model"""
        # Load config
        config_path = os.path.join(model_dir, 'model_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create instance
        instance = cls(model_type=config['model_type'])
        instance.threshold = config['threshold']
        
        # Load model
        model_path = os.path.join(model_dir, 'fraud_detection_model.pkl')
        instance.model = joblib.load(model_path)
        
        # Load feature engineer
        feature_engineer_path = os.path.join(model_dir, 'feature_engineer.pkl')
        instance.feature_engineer = joblib.load(feature_engineer_path)
        
        # Load metadata
        metadata_path = os.path.join(model_dir, 'model_metadata.json')
        with open(metadata_path, 'r') as f:
            instance.model_metadata = json.load(f)
        
        logger.info(f"Model loaded from {model_dir}")
        return instance

def train_fraud_detection_model():
    """Train and save a fraud detection model"""
    logger.info("Starting fraud detection model training")
    
    # Create model
    model = FraudDetectionModel(model_type='random_forest')
    
    # Generate synthetic data
    logger.info("Generating synthetic training data")
    df = model.create_synthetic_data(n_samples=10000)
    
    # Train model
    training_results = model.train(df)
    logger.info(f"Training results: {training_results}")
    
    # Save model
    model_dir = '../models/fraud_detection'
    model.save_model(model_dir)
    
    return model, training_results

if __name__ == "__main__":
    # Train the model
    trained_model, results = train_fraud_detection_model()
    
    # Test the model with sample data
    test_data = pd.DataFrame({
        'estimated_amount': [75000, 5000, 25000],
        'policy_age_days': [15, 365, 180],
        'description': ['Car accident', 'Minor fender bender on highway during rush hour', 'Home damage'],
        'claim_type': ['auto', 'auto', 'home'],
        'customer_id': ['customer_1', 'customer_2', 'customer_3'],
        'incident_date': [datetime.now() - timedelta(days=1)] * 3,
        'reported_date': [datetime.now()] * 3,
    })
    
    predictions = trained_model.predict(test_data)
    
    print("\nSample Predictions:")
    for i, (prob, pred, explanation) in enumerate(zip(
        predictions['fraud_probabilities'],
        predictions['predictions'],
        predictions['explanations']
    )):
        print(f"Claim {i+1}: Fraud Prob={prob:.3f}, Prediction={'Fraud' if pred else 'Legitimate'}")
        print(f"  Explanation: {explanation}")
        print() 