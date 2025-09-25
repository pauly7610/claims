from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import joblib
import json
import os
import logging
from datetime import datetime, timedelta
import asyncio
from contextlib import asynccontextmanager
import structlog
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# ML Libraries
import torch
from transformers import pipeline, AutoTokenizer, AutoModel
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import cv2
from PIL import Image
import easyocr
import re

# Configure logging
logger = structlog.get_logger()

# Metrics
FRAUD_PREDICTIONS = Counter('fraud_predictions_total', 'Total fraud predictions', ['prediction'])
DOCUMENT_ANALYSES = Counter('document_analyses_total', 'Total document analyses', ['document_type'])
MODEL_INFERENCE_TIME = Histogram('model_inference_duration_seconds', 'Model inference time', ['model_type'])

# Pydantic Models
class FraudAnalysisRequest(BaseModel):
    claim_type: str
    estimated_amount: float
    description: str
    incident_date: str
    policy_age_days: int
    customer_id: str

class FraudAnalysisResponse(BaseModel):
    fraud_score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    risk_factors: List[str]
    explanation: str

class DocumentAnalysisRequest(BaseModel):
    document_path: str
    document_type: Optional[str] = "unknown"

class DocumentAnalysisResponse(BaseModel):
    extracted_data: Dict[str, Any]
    confidence: float
    document_type: str
    text_content: str

class DamageAssessmentRequest(BaseModel):
    image_path: str
    claim_type: str

class DamageAssessmentResponse(BaseModel):
    damage_severity: str  # low, medium, high
    estimated_cost: float
    damage_areas: List[str]
    confidence: float

# AI Models Manager
class AIModelsManager:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.ocr_reader = None
        self.nlp_pipeline = None
        
    async def load_models(self):
        """Load all AI models"""
        try:
            logger.info("Loading AI models...")
            
            # Load fraud detection model
            await self._load_fraud_model()
            
            # Load NLP models
            await self._load_nlp_models()
            
            # Load OCR model
            await self._load_ocr_model()
            
            logger.info("All AI models loaded successfully")
            
        except Exception as e:
            logger.error("Failed to load AI models", error=str(e))
            # Create fallback models for development
            await self._create_fallback_models()
    
    async def _load_fraud_model(self):
        """Load fraud detection model"""
        try:
            model_path = "/app/models/fraud_detection_model.pkl"
            if os.path.exists(model_path):
                self.models['fraud_detection'] = joblib.load(model_path)
                logger.info("Loaded fraud detection model from file")
            else:
                # Create and train a simple fraud detection model
                self.models['fraud_detection'] = self._create_fraud_model()
                logger.info("Created new fraud detection model")
                
        except Exception as e:
            logger.error("Failed to load fraud model", error=str(e))
            self.models['fraud_detection'] = self._create_fraud_model()
    
    def _create_fraud_model(self):
        """Create a simple fraud detection model for demonstration"""
        from sklearn.ensemble import RandomForestClassifier
        
        # Create synthetic training data
        np.random.seed(42)
        n_samples = 1000
        
        # Features: claim_amount, policy_age, previous_claims, days_to_report, etc.
        X = np.random.rand(n_samples, 8)
        
        # Create fraud labels based on suspicious patterns
        fraud_probability = (
            (X[:, 0] > 0.8) * 0.3 +  # High claim amount
            (X[:, 1] < 0.1) * 0.2 +  # New policy
            (X[:, 2] > 0.7) * 0.2 +  # Many previous claims
            (X[:, 3] > 0.8) * 0.1 +  # Delayed reporting
            np.random.random(n_samples) * 0.2
        )
        y = (fraud_probability > 0.4).astype(int)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Create scaler
        scaler = StandardScaler()
        scaler.fit(X)
        self.scalers['fraud_detection'] = scaler
        
        return model
    
    async def _load_nlp_models(self):
        """Load NLP models for text analysis"""
        try:
            # Load sentiment analysis pipeline
            self.nlp_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                return_all_scores=True
            )
            logger.info("Loaded NLP models")
        except Exception as e:
            logger.error("Failed to load NLP models", error=str(e))
            self.nlp_pipeline = None
    
    async def _load_ocr_model(self):
        """Load OCR model"""
        try:
            self.ocr_reader = easyocr.Reader(['en'])
            logger.info("Loaded OCR model")
        except Exception as e:
            logger.error("Failed to load OCR model", error=str(e))
            self.ocr_reader = None
    
    async def _create_fallback_models(self):
        """Create simple fallback models for development"""
        logger.info("Creating fallback models for development")
        if 'fraud_detection' not in self.models:
            self.models['fraud_detection'] = self._create_fraud_model()

# AI Service Implementation
class FraudDetectionService:
    def __init__(self, models_manager: AIModelsManager):
        self.models = models_manager
    
    async def analyze_fraud(self, request: FraudAnalysisRequest) -> FraudAnalysisResponse:
        """Analyze claim for fraud indicators"""
        with MODEL_INFERENCE_TIME.labels(model_type='fraud_detection').time():
            try:
                # Extract features from request
                features = self._extract_fraud_features(request)
                
                # Get model prediction
                model = self.models.models.get('fraud_detection')
                if not model:
                    raise HTTPException(status_code=503, detail="Fraud detection model not available")
                
                # Scale features if scaler available
                if 'fraud_detection' in self.models.scalers:
                    features = self.models.scalers['fraud_detection'].transform([features])
                else:
                    features = np.array([features])
                
                # Predict
                fraud_probability = model.predict_proba(features)[0][1]  # Probability of fraud
                prediction = model.predict(features)[0]
                
                # Analyze risk factors
                risk_factors = self._identify_risk_factors(request, fraud_probability)
                
                # Generate explanation
                explanation = self._generate_explanation(request, fraud_probability, risk_factors)
                
                FRAUD_PREDICTIONS.labels(prediction='fraud' if prediction else 'legitimate').inc()
                
                return FraudAnalysisResponse(
                    fraud_score=float(fraud_probability),
                    confidence=min(0.95, max(0.6, abs(fraud_probability - 0.5) * 2)),
                    risk_factors=risk_factors,
                    explanation=explanation
                )
                
            except Exception as e:
                logger.error("Fraud analysis failed", error=str(e))
                # Return safe default
                return FraudAnalysisResponse(
                    fraud_score=0.0,
                    confidence=0.0,
                    risk_factors=[],
                    explanation="Analysis failed - manual review required"
                )
    
    def _extract_fraud_features(self, request: FraudAnalysisRequest) -> List[float]:
        """Extract numerical features for fraud detection"""
        # Convert request to numerical features
        features = [
            min(request.estimated_amount / 100000, 1.0),  # Normalized claim amount
            min(request.policy_age_days / 365, 5.0),      # Policy age in years (capped at 5)
            len(request.description) / 1000,              # Description length
            1.0 if request.claim_type == 'auto' else 0.0, # Claim type indicators
            1.0 if request.claim_type == 'home' else 0.0,
            1.0 if request.claim_type == 'health' else 0.0,
            # Add more features based on incident date, customer history, etc.
            self._get_time_feature(request.incident_date),
            hash(request.customer_id) % 100 / 100.0      # Customer hash feature
        ]
        return features
    
    def _get_time_feature(self, incident_date: str) -> float:
        """Extract time-based features"""
        try:
            date = datetime.fromisoformat(incident_date.replace('Z', '+00:00'))
            # Weekend indicator
            return 1.0 if date.weekday() >= 5 else 0.0
        except:
            return 0.0
    
    def _identify_risk_factors(self, request: FraudAnalysisRequest, fraud_score: float) -> List[str]:
        """Identify specific risk factors"""
        risk_factors = []
        
        if request.estimated_amount > 50000:
            risk_factors.append("High claim amount")
        
        if request.policy_age_days < 30:
            risk_factors.append("Recently purchased policy")
        
        if len(request.description) < 50:
            risk_factors.append("Insufficient incident description")
        
        # Time-based risk factors
        try:
            incident_date = datetime.fromisoformat(request.incident_date.replace('Z', '+00:00'))
            if incident_date.weekday() >= 5:
                risk_factors.append("Weekend incident")
            
            if (datetime.now() - incident_date).days > 30:
                risk_factors.append("Delayed reporting")
        except:
            risk_factors.append("Invalid incident date")
        
        # NLP analysis of description
        if self.models.nlp_pipeline:
            try:
                sentiment = self.models.nlp_pipeline(request.description)[0]
                if sentiment['label'] == 'NEGATIVE' and sentiment['score'] > 0.8:
                    risk_factors.append("Suspicious language patterns")
            except:
                pass
        
        return risk_factors
    
    def _generate_explanation(self, request: FraudAnalysisRequest, fraud_score: float, risk_factors: List[str]) -> str:
        """Generate human-readable explanation"""
        if fraud_score > 0.7:
            explanation = f"High fraud risk detected (score: {fraud_score:.2f}). "
            explanation += f"Key concerns: {', '.join(risk_factors[:3])}. Recommend manual review."
        elif fraud_score > 0.4:
            explanation = f"Moderate fraud risk (score: {fraud_score:.2f}). "
            explanation += f"Some suspicious indicators present. Consider additional verification."
        else:
            explanation = f"Low fraud risk (score: {fraud_score:.2f}). Claim appears legitimate."
        
        return explanation

class DocumentAnalysisService:
    def __init__(self, models_manager: AIModelsManager):
        self.models = models_manager
    
    async def analyze_document(self, request: DocumentAnalysisRequest) -> DocumentAnalysisResponse:
        """Analyze document and extract information"""
        with MODEL_INFERENCE_TIME.labels(model_type='document_analysis').time():
            try:
                # For demo, simulate document analysis
                # In production, this would load and process actual images
                
                extracted_data = await self._extract_document_data(request.document_path)
                document_type = self._classify_document_type(extracted_data)
                
                DOCUMENT_ANALYSES.labels(document_type=document_type).inc()
                
                return DocumentAnalysisResponse(
                    extracted_data=extracted_data,
                    confidence=0.85,
                    document_type=document_type,
                    text_content=extracted_data.get('raw_text', '')
                )
                
            except Exception as e:
                logger.error("Document analysis failed", error=str(e))
                return DocumentAnalysisResponse(
                    extracted_data={},
                    confidence=0.0,
                    document_type="unknown",
                    text_content=""
                )
    
    async def _extract_document_data(self, document_path: str) -> Dict[str, Any]:
        """Extract data from document using OCR"""
        # Simulate OCR extraction
        # In production, this would use the actual OCR model
        
        extracted_data = {
            "raw_text": "Sample extracted text from document",
            "policy_number": "POL-2024-001234",
            "date": "2024-01-15",
            "amount": 5000.00,
            "incident_type": "Vehicle damage",
            "confidence_scores": {
                "policy_number": 0.95,
                "date": 0.90,
                "amount": 0.85
            }
        }
        
        # If OCR reader is available, use it
        if self.models.ocr_reader and os.path.exists(document_path):
            try:
                results = self.models.ocr_reader.readtext(document_path)
                text_content = ' '.join([result[1] for result in results])
                extracted_data['raw_text'] = text_content
                
                # Extract structured data using regex patterns
                extracted_data.update(self._parse_structured_data(text_content))
                
            except Exception as e:
                logger.error("OCR processing failed", error=str(e))
        
        return extracted_data
    
    def _parse_structured_data(self, text: str) -> Dict[str, Any]:
        """Parse structured data from OCR text"""
        data = {}
        
        # Extract policy number
        policy_match = re.search(r'POL-\d{4}-\d{6}', text)
        if policy_match:
            data['policy_number'] = policy_match.group()
        
        # Extract dates
        date_match = re.search(r'\d{4}-\d{2}-\d{2}', text)
        if date_match:
            data['date'] = date_match.group()
        
        # Extract amounts
        amount_matches = re.findall(r'\$[\d,]+\.?\d*', text)
        if amount_matches:
            data['amounts'] = amount_matches
        
        return data
    
    def _classify_document_type(self, extracted_data: Dict[str, Any]) -> str:
        """Classify document type based on extracted data"""
        text = extracted_data.get('raw_text', '').lower()
        
        if 'police report' in text or 'incident report' in text:
            return 'police_report'
        elif 'medical' in text or 'hospital' in text:
            return 'medical_report'
        elif 'estimate' in text or 'repair' in text:
            return 'repair_estimate'
        elif 'receipt' in text or 'invoice' in text:
            return 'receipt'
        else:
            return 'other'

class DamageAssessmentService:
    def __init__(self, models_manager: AIModelsManager):
        self.models = models_manager
    
    async def assess_damage(self, request: DamageAssessmentRequest) -> DamageAssessmentResponse:
        """Assess damage from images"""
        with MODEL_INFERENCE_TIME.labels(model_type='damage_assessment').time():
            try:
                # Simulate damage assessment
                # In production, this would use computer vision models
                
                damage_analysis = await self._analyze_damage_image(request.image_path)
                
                return DamageAssessmentResponse(
                    damage_severity=damage_analysis['severity'],
                    estimated_cost=damage_analysis['cost'],
                    damage_areas=damage_analysis['areas'],
                    confidence=damage_analysis['confidence']
                )
                
            except Exception as e:
                logger.error("Damage assessment failed", error=str(e))
                return DamageAssessmentResponse(
                    damage_severity="unknown",
                    estimated_cost=0.0,
                    damage_areas=[],
                    confidence=0.0
                )
    
    async def _analyze_damage_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze damage from image"""
        # Simulate computer vision analysis
        # In production, this would use actual CV models
        
        analysis = {
            'severity': 'medium',
            'cost': 3500.0,
            'areas': ['front_bumper', 'headlight'],
            'confidence': 0.75
        }
        
        # If image exists, do basic analysis
        if os.path.exists(image_path):
            try:
                image = cv2.imread(image_path)
                if image is not None:
                    # Simple analysis based on image properties
                    height, width = image.shape[:2]
                    
                    # Simulate more sophisticated analysis
                    if width * height > 1000000:  # High resolution
                        analysis['confidence'] = 0.85
                    
            except Exception as e:
                logger.error("Image processing failed", error=str(e))
        
        return analysis

# Initialize models manager
models_manager = AIModelsManager()
fraud_service = FraudDetectionService(models_manager)
document_service = DocumentAnalysisService(models_manager)
damage_service = DamageAssessmentService(models_manager)

# FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI Service")
    await models_manager.load_models()
    yield
    logger.info("Shutting down AI Service")

app = FastAPI(
    title="AI Service",
    description="AI/ML models for claims processing",
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

# Routes
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "ai-service",
        "models_loaded": len(models_manager.models)
    }

@app.get("/metrics")
async def metrics():
    from fastapi import Response
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/analyze-fraud", response_model=FraudAnalysisResponse)
async def analyze_fraud(request: FraudAnalysisRequest):
    """Analyze claim for fraud indicators"""
    return await fraud_service.analyze_fraud(request)

@app.post("/analyze-document", response_model=DocumentAnalysisResponse)
async def analyze_document(request: DocumentAnalysisRequest):
    """Analyze document and extract information"""
    return await document_service.analyze_document(request)

@app.post("/assess-damage", response_model=DamageAssessmentResponse)
async def assess_damage(request: DamageAssessmentRequest):
    """Assess damage from images"""
    return await damage_service.assess_damage(request)

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """Upload document for analysis"""
    try:
        # Save uploaded file
        upload_dir = "/app/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Analyze document
        request = DocumentAnalysisRequest(document_path=file_path)
        result = await document_service.analyze_document(request)
        
        return {
            "filename": file.filename,
            "file_path": file_path,
            "analysis": result
        }
        
    except Exception as e:
        logger.error("Document upload failed", error=str(e))
        raise HTTPException(status_code=500, detail="Document upload failed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 