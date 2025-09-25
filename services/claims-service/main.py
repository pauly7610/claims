from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, String, DateTime, Decimal, Integer, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.dialects.postgresql import UUID
from pydantic import BaseModel, Field
from datetime import datetime, date
from typing import List, Optional
import uuid
import os
from contextlib import asynccontextmanager
import httpx
import structlog
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time

# Configure logging
logger = structlog.get_logger()

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://claims:claims@localhost:5432/claims")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Metrics
CLAIMS_PROCESSED = Counter('claims_processed_total', 'Total claims processed', ['status'])
CLAIM_PROCESSING_TIME = Histogram('claim_processing_duration_seconds', 'Time spent processing claims')

# Database Models
class Policy(Base):
    __tablename__ = "policies"
    __table_args__ = {'schema': 'claims'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    policy_number = Column(String(50), unique=True, nullable=False)
    customer_id = Column(UUID(as_uuid=True), nullable=False)
    policy_type = Column(String(50), nullable=False)
    coverage_amount = Column(Decimal(12, 2), nullable=False)
    deductible = Column(Decimal(10, 2), nullable=False)
    premium = Column(Decimal(10, 2), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    status = Column(String(20), default='active')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    claims = relationship("Claim", back_populates="policy")

class Claim(Base):
    __tablename__ = "claims"
    __table_args__ = {'schema': 'claims'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    claim_number = Column(String(50), unique=True, nullable=False)
    policy_id = Column(UUID(as_uuid=True), ForeignKey('claims.policies.id'), nullable=False)
    customer_id = Column(UUID(as_uuid=True), nullable=False)
    incident_date = Column(DateTime, nullable=False)
    reported_date = Column(DateTime, default=datetime.utcnow)
    claim_type = Column(String(50), nullable=False)
    description = Column(Text)
    estimated_amount = Column(Decimal(12, 2))
    approved_amount = Column(Decimal(12, 2))
    status = Column(String(20), default='submitted')
    priority = Column(String(10), default='medium')
    assigned_adjuster_id = Column(UUID(as_uuid=True))
    fraud_score = Column(Decimal(3, 2), default=0.0)
    ai_confidence = Column(Decimal(3, 2), default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    policy = relationship("Policy", back_populates="claims")
    documents = relationship("ClaimDocument", back_populates="claim")
    history = relationship("ClaimHistory", back_populates="claim")

class ClaimDocument(Base):
    __tablename__ = "claim_documents"
    __table_args__ = {'schema': 'claims'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    claim_id = Column(UUID(as_uuid=True), ForeignKey('claims.claims.id'), nullable=False)
    file_name = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_type = Column(String(50), nullable=False)
    file_size = Column(Integer, nullable=False)
    document_type = Column(String(50), nullable=False)
    ai_extracted_data = Column(Text)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)
    
    claim = relationship("Claim", back_populates="documents")

class ClaimHistory(Base):
    __tablename__ = "claim_history"
    __table_args__ = {'schema': 'claims'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    claim_id = Column(UUID(as_uuid=True), ForeignKey('claims.claims.id'), nullable=False)
    status_from = Column(String(20))
    status_to = Column(String(20), nullable=False)
    changed_by = Column(UUID(as_uuid=True), nullable=False)
    change_reason = Column(Text)
    ai_decision = Column(Boolean, default=False)
    metadata = Column(Text)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)
    
    claim = relationship("Claim", back_populates="history")

# Pydantic Models
class ClaimCreate(BaseModel):
    policy_number: str
    incident_date: date
    claim_type: str
    description: str
    estimated_amount: Optional[float] = None

class ClaimUpdate(BaseModel):
    status: Optional[str] = None
    approved_amount: Optional[float] = None
    assigned_adjuster_id: Optional[str] = None
    priority: Optional[str] = None

class ClaimResponse(BaseModel):
    id: str
    claim_number: str
    policy_id: str
    customer_id: str
    incident_date: date
    reported_date: datetime
    claim_type: str
    description: Optional[str]
    estimated_amount: Optional[float]
    approved_amount: Optional[float]
    status: str
    priority: str
    fraud_score: float
    ai_confidence: float
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class PolicyResponse(BaseModel):
    id: str
    policy_number: str
    customer_id: str
    policy_type: str
    coverage_amount: float
    deductible: float
    status: str
    
    class Config:
        from_attributes = True

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# AI Service client
class AIServiceClient:
    def __init__(self, base_url: str = "http://ai-service:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient()
    
    async def analyze_fraud(self, claim_data: dict) -> dict:
        """Call AI service for fraud detection"""
        try:
            response = await self.client.post(
                f"{self.base_url}/analyze-fraud",
                json=claim_data,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("Fraud analysis failed", error=str(e))
            return {"fraud_score": 0.0, "confidence": 0.0, "risk_factors": []}
    
    async def analyze_document(self, document_path: str) -> dict:
        """Call AI service for document analysis"""
        try:
            response = await self.client.post(
                f"{self.base_url}/analyze-document",
                json={"document_path": document_path},
                timeout=60.0
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("Document analysis failed", error=str(e))
            return {"extracted_data": {}, "confidence": 0.0}

ai_client = AIServiceClient()

# Business Logic
class ClaimsService:
    def __init__(self, db: Session):
        self.db = db
    
    def generate_claim_number(self) -> str:
        """Generate unique claim number"""
        timestamp = datetime.now().strftime("%Y%m%d")
        # In production, this would be more sophisticated
        import random
        suffix = f"{random.randint(1000, 9999)}"
        return f"CLM-{timestamp}-{suffix}"
    
    async def create_claim(self, claim_data: ClaimCreate, customer_id: str) -> Claim:
        """Create a new claim with AI analysis"""
        with CLAIM_PROCESSING_TIME.time():
            # Verify policy exists and is active
            policy = self.db.query(Policy).filter(
                Policy.policy_number == claim_data.policy_number,
                Policy.status == 'active'
            ).first()
            
            if not policy:
                raise HTTPException(status_code=404, detail="Policy not found or inactive")
            
            # Check policy coverage
            if policy.customer_id != uuid.UUID(customer_id):
                raise HTTPException(status_code=403, detail="Policy does not belong to customer")
            
            # Create claim
            claim = Claim(
                claim_number=self.generate_claim_number(),
                policy_id=policy.id,
                customer_id=uuid.UUID(customer_id),
                incident_date=claim_data.incident_date,
                claim_type=claim_data.claim_type,
                description=claim_data.description,
                estimated_amount=claim_data.estimated_amount,
                status='submitted'
            )
            
            self.db.add(claim)
            self.db.commit()
            self.db.refresh(claim)
            
            # Add initial history entry
            history = ClaimHistory(
                claim_id=claim.id,
                status_to='submitted',
                changed_by=uuid.UUID(customer_id),
                change_reason='Initial claim submission'
            )
            self.db.add(history)
            
            # Trigger AI analysis
            await self._analyze_claim_async(claim)
            
            self.db.commit()
            CLAIMS_PROCESSED.labels(status='created').inc()
            
            logger.info("Claim created", claim_id=str(claim.id), claim_number=claim.claim_number)
            return claim
    
    async def _analyze_claim_async(self, claim: Claim):
        """Perform AI analysis on claim"""
        try:
            # Prepare data for AI analysis
            claim_data = {
                "claim_type": claim.claim_type,
                "estimated_amount": float(claim.estimated_amount) if claim.estimated_amount else 0,
                "description": claim.description,
                "incident_date": claim.incident_date.isoformat(),
                "policy_age_days": (datetime.now() - claim.policy.created_at).days,
                "customer_id": str(claim.customer_id)
            }
            
            # Call AI service for fraud analysis
            fraud_result = await ai_client.analyze_fraud(claim_data)
            
            # Update claim with AI results
            claim.fraud_score = fraud_result.get('fraud_score', 0.0)
            claim.ai_confidence = fraud_result.get('confidence', 0.0)
            
            # Determine next status based on AI analysis
            if claim.fraud_score > 0.7:
                claim.status = 'adjuster_review'
                claim.priority = 'high'
            elif claim.fraud_score > 0.4:
                claim.status = 'ai_processing'
                claim.priority = 'medium'
            else:
                claim.status = 'under_review'
                claim.priority = 'low'
            
            # Add history entry for AI analysis
            history = ClaimHistory(
                claim_id=claim.id,
                status_from='submitted',
                status_to=claim.status,
                changed_by=claim.customer_id,  # System user in production
                change_reason=f'AI analysis complete - fraud score: {claim.fraud_score}',
                ai_decision=True
            )
            self.db.add(history)
            
            logger.info("AI analysis complete", 
                       claim_id=str(claim.id), 
                       fraud_score=claim.fraud_score,
                       new_status=claim.status)
            
        except Exception as e:
            logger.error("AI analysis failed", claim_id=str(claim.id), error=str(e))
            # Continue processing without AI analysis
    
    def get_claim(self, claim_id: str, customer_id: str) -> Claim:
        """Get claim by ID"""
        claim = self.db.query(Claim).filter(
            Claim.id == claim_id,
            Claim.customer_id == customer_id
        ).first()
        
        if not claim:
            raise HTTPException(status_code=404, detail="Claim not found")
        
        return claim
    
    def list_claims(self, customer_id: str, skip: int = 0, limit: int = 100) -> List[Claim]:
        """List claims for customer"""
        return self.db.query(Claim).filter(
            Claim.customer_id == customer_id
        ).offset(skip).limit(limit).all()
    
    def update_claim_status(self, claim_id: str, status: str, user_id: str, reason: str = None) -> Claim:
        """Update claim status"""
        claim = self.db.query(Claim).filter(Claim.id == claim_id).first()
        if not claim:
            raise HTTPException(status_code=404, detail="Claim not found")
        
        old_status = claim.status
        claim.status = status
        claim.updated_at = datetime.utcnow()
        
        # Add history entry
        history = ClaimHistory(
            claim_id=claim.id,
            status_from=old_status,
            status_to=status,
            changed_by=uuid.UUID(user_id),
            change_reason=reason
        )
        self.db.add(history)
        self.db.commit()
        
        CLAIMS_PROCESSED.labels(status=status).inc()
        logger.info("Claim status updated", claim_id=str(claim.id), old_status=old_status, new_status=status)
        
        return claim

# FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Claims Service")
    # Create tables
    Base.metadata.create_all(bind=engine)
    yield
    logger.info("Shutting down Claims Service")

app = FastAPI(
    title="Claims Processing Service",
    description="Core claims processing business logic",
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

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# API Routes
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "claims-service"}

@app.get("/metrics")
async def metrics():
    from fastapi import Response
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/claims", response_model=ClaimResponse)
async def create_claim(
    claim_data: ClaimCreate,
    customer_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Create a new insurance claim"""
    service = ClaimsService(db)
    claim = await service.create_claim(claim_data, customer_id)
    return claim

@app.get("/claims/{claim_id}", response_model=ClaimResponse)
async def get_claim(claim_id: str, customer_id: str, db: Session = Depends(get_db)):
    """Get claim by ID"""
    service = ClaimsService(db)
    claim = service.get_claim(claim_id, customer_id)
    return claim

@app.get("/claims", response_model=List[ClaimResponse])
async def list_claims(
    customer_id: str,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """List claims for customer"""
    service = ClaimsService(db)
    claims = service.list_claims(customer_id, skip, limit)
    return claims

@app.put("/claims/{claim_id}/status")
async def update_claim_status(
    claim_id: str,
    status: str,
    user_id: str,
    reason: str = None,
    db: Session = Depends(get_db)
):
    """Update claim status (for adjusters/admins)"""
    service = ClaimsService(db)
    claim = service.update_claim_status(claim_id, status, user_id, reason)
    return {"message": "Claim status updated", "claim_id": str(claim.id), "new_status": claim.status}

@app.get("/policies/{policy_number}", response_model=PolicyResponse)
async def get_policy(policy_number: str, db: Session = Depends(get_db)):
    """Get policy by number"""
    policy = db.query(Policy).filter(Policy.policy_number == policy_number).first()
    if not policy:
        raise HTTPException(status_code=404, detail="Policy not found")
    return policy

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 