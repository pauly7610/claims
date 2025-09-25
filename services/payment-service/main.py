from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import os
import asyncio
from datetime import datetime, timedelta
import structlog
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
import json
import httpx
from enum import Enum
import uuid
from decimal import Decimal

# Configure logging
logger = structlog.get_logger()

# Metrics
PAYMENTS_PROCESSED = Counter('payments_processed_total', 'Total payments processed', ['method', 'status'])
PAYMENT_PROCESSING_TIME = Histogram('payment_processing_duration_seconds', 'Payment processing time', ['method'])
PAYMENT_AMOUNTS = Histogram('payment_amounts', 'Payment amounts processed', buckets=[100, 500, 1000, 5000, 10000, 50000, 100000])

app = FastAPI(
    title="Payment Service",
    description="Payment processing service for insurance claims",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enums
class PaymentMethod(str, Enum):
    BANK_TRANSFER = "bank_transfer"
    CHECK = "check"
    CREDIT_CARD = "credit_card"
    DIGITAL_WALLET = "digital_wallet"

class PaymentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"

class TransactionType(str, Enum):
    CLAIM_PAYMENT = "claim_payment"
    PREMIUM_PAYMENT = "premium_payment"
    REFUND = "refund"
    ADJUSTMENT = "adjustment"

# Pydantic Models
class BankAccount(BaseModel):
    account_holder_name: str
    account_number: str = Field(..., regex=r'^\d{8,17}$')
    routing_number: str = Field(..., regex=r'^\d{9}$')
    account_type: str = Field(default="checking")  # checking, savings

class PaymentRequest(BaseModel):
    claim_id: str
    customer_id: str
    amount: float = Field(..., gt=0)
    currency: str = Field(default="USD")
    payment_method: PaymentMethod
    description: str
    bank_account: Optional[BankAccount] = None
    check_address: Optional[Dict[str, str]] = None
    metadata: Optional[Dict[str, Any]] = None

class PaymentResponse(BaseModel):
    payment_id: str
    status: PaymentStatus
    amount: float
    currency: str
    payment_method: PaymentMethod
    transaction_id: Optional[str] = None
    estimated_completion: Optional[datetime] = None
    created_at: datetime
    message: str

class PaymentStatusUpdate(BaseModel):
    payment_id: str
    status: PaymentStatus
    transaction_id: Optional[str] = None
    failure_reason: Optional[str] = None
    completed_at: Optional[datetime] = None

class RefundRequest(BaseModel):
    payment_id: str
    amount: Optional[float] = None  # If None, refund full amount
    reason: str

# In-memory storage for demo (use database in production)
payments_db = {}
transactions_db = {}

class PaymentProcessor:
    def __init__(self):
        self.stripe_key = os.getenv("STRIPE_SECRET_KEY", "")
        self.bank_api_key = os.getenv("BANK_API_KEY", "")
        self.notification_service_url = os.getenv("NOTIFICATION_SERVICE_URL", "http://notification-service:8000")
        
    async def process_payment(self, request: PaymentRequest) -> PaymentResponse:
        """Process payment based on method"""
        payment_id = f"pay_{uuid.uuid4().hex[:12]}"
        
        with PAYMENT_PROCESSING_TIME.labels(method=request.payment_method.value).time():
            try:
                # Create payment record
                payment_record = {
                    "payment_id": payment_id,
                    "claim_id": request.claim_id,
                    "customer_id": request.customer_id,
                    "amount": request.amount,
                    "currency": request.currency,
                    "payment_method": request.payment_method.value,
                    "description": request.description,
                    "status": PaymentStatus.PENDING.value,
                    "created_at": datetime.utcnow(),
                    "metadata": request.metadata or {}
                }
                
                payments_db[payment_id] = payment_record
                
                # Process based on payment method
                if request.payment_method == PaymentMethod.BANK_TRANSFER:
                    result = await self._process_bank_transfer(payment_id, request)
                elif request.payment_method == PaymentMethod.CHECK:
                    result = await self._process_check_payment(payment_id, request)
                elif request.payment_method == PaymentMethod.CREDIT_CARD:
                    result = await self._process_credit_card(payment_id, request)
                elif request.payment_method == PaymentMethod.DIGITAL_WALLET:
                    result = await self._process_digital_wallet(payment_id, request)
                else:
                    raise HTTPException(status_code=400, detail="Unsupported payment method")
                
                # Update payment record
                payments_db[payment_id].update(result)
                
                # Record metrics
                PAYMENTS_PROCESSED.labels(
                    method=request.payment_method.value,
                    status=result["status"]
                ).inc()
                PAYMENT_AMOUNTS.observe(request.amount)
                
                # Send notification
                await self._send_payment_notification(payment_id, result["status"])
                
                return PaymentResponse(
                    payment_id=payment_id,
                    status=PaymentStatus(result["status"]),
                    amount=request.amount,
                    currency=request.currency,
                    payment_method=request.payment_method,
                    transaction_id=result.get("transaction_id"),
                    estimated_completion=result.get("estimated_completion"),
                    created_at=payment_record["created_at"],
                    message=result.get("message", "Payment processed successfully")
                )
                
            except Exception as e:
                logger.error("Payment processing failed", error=str(e), payment_id=payment_id)
                PAYMENTS_PROCESSED.labels(
                    method=request.payment_method.value,
                    status="failed"
                ).inc()
                
                # Update payment record
                if payment_id in payments_db:
                    payments_db[payment_id]["status"] = PaymentStatus.FAILED.value
                    payments_db[payment_id]["failure_reason"] = str(e)
                
                raise HTTPException(status_code=500, detail=f"Payment processing failed: {str(e)}")
    
    async def _process_bank_transfer(self, payment_id: str, request: PaymentRequest) -> Dict[str, Any]:
        """Process bank transfer payment"""
        if not request.bank_account:
            raise ValueError("Bank account information required for bank transfer")
        
        # Simulate bank API call
        await asyncio.sleep(1)  # Simulate processing time
        
        # In production, integrate with actual bank API
        transaction_id = f"ach_{uuid.uuid4().hex[:16]}"
        
        logger.info("Bank transfer initiated", 
                   payment_id=payment_id,
                   transaction_id=transaction_id,
                   amount=request.amount)
        
        return {
            "status": PaymentStatus.PROCESSING.value,
            "transaction_id": transaction_id,
            "estimated_completion": datetime.utcnow() + timedelta(days=3),
            "message": "Bank transfer initiated. Funds will be available in 3-5 business days."
        }
    
    async def _process_check_payment(self, payment_id: str, request: PaymentRequest) -> Dict[str, Any]:
        """Process check payment"""
        if not request.check_address:
            raise ValueError("Mailing address required for check payment")
        
        # Simulate check processing
        await asyncio.sleep(0.5)
        
        check_number = f"CHK{datetime.utcnow().strftime('%Y%m%d')}{payment_id[-6:]}"
        
        logger.info("Check payment processed",
                   payment_id=payment_id,
                   check_number=check_number,
                   amount=request.amount)
        
        return {
            "status": PaymentStatus.PROCESSING.value,
            "transaction_id": check_number,
            "estimated_completion": datetime.utcnow() + timedelta(days=7),
            "message": f"Check #{check_number} will be mailed within 2 business days."
        }
    
    async def _process_credit_card(self, payment_id: str, request: PaymentRequest) -> Dict[str, Any]:
        """Process credit card payment (refunds)"""
        # This would typically be for refunding to a credit card
        await asyncio.sleep(0.3)
        
        transaction_id = f"cc_{uuid.uuid4().hex[:16]}"
        
        logger.info("Credit card refund processed",
                   payment_id=payment_id,
                   transaction_id=transaction_id,
                   amount=request.amount)
        
        return {
            "status": PaymentStatus.COMPLETED.value,
            "transaction_id": transaction_id,
            "estimated_completion": datetime.utcnow(),
            "message": "Credit card refund processed successfully."
        }
    
    async def _process_digital_wallet(self, payment_id: str, request: PaymentRequest) -> Dict[str, Any]:
        """Process digital wallet payment"""
        await asyncio.sleep(0.2)
        
        transaction_id = f"dw_{uuid.uuid4().hex[:16]}"
        
        logger.info("Digital wallet payment processed",
                   payment_id=payment_id,
                   transaction_id=transaction_id,
                   amount=request.amount)
        
        return {
            "status": PaymentStatus.COMPLETED.value,
            "transaction_id": transaction_id,
            "estimated_completion": datetime.utcnow(),
            "message": "Digital wallet payment completed successfully."
        }
    
    async def _send_payment_notification(self, payment_id: str, status: str):
        """Send payment notification"""
        try:
            payment = payments_db.get(payment_id)
            if not payment:
                return
            
            # Prepare notification data
            notification_data = {
                "user_id": payment["customer_id"],
                "notification_type": "payment_processed",
                "channel": "email",
                "data": {
                    "claim_id": payment["claim_id"],
                    "payment_id": payment_id,
                    "amount": payment["amount"],
                    "status": status,
                    "payment_method": payment["payment_method"],
                    "email": "customer@example.com"  # Would get from user service
                }
            }
            
            # Send to notification service
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.notification_service_url}/send-notification",
                    json=notification_data,
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    logger.info("Payment notification sent", payment_id=payment_id)
                else:
                    logger.warning("Payment notification failed", 
                                 payment_id=payment_id,
                                 status_code=response.status_code)
                    
        except Exception as e:
            logger.error("Failed to send payment notification", 
                        error=str(e), 
                        payment_id=payment_id)

# Initialize processor
payment_processor = PaymentProcessor()

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "payment-service",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Process payment
@app.post("/process-payment", response_model=PaymentResponse)
async def process_payment(request: PaymentRequest):
    """Process a payment"""
    return await payment_processor.process_payment(request)

# Get payment status
@app.get("/payments/{payment_id}")
async def get_payment_status(payment_id: str):
    """Get payment status"""
    if payment_id not in payments_db:
        raise HTTPException(status_code=404, detail="Payment not found")
    
    payment = payments_db[payment_id]
    return {
        "payment_id": payment_id,
        "status": payment["status"],
        "amount": payment["amount"],
        "currency": payment["currency"],
        "payment_method": payment["payment_method"],
        "created_at": payment["created_at"].isoformat(),
        "transaction_id": payment.get("transaction_id"),
        "estimated_completion": payment.get("estimated_completion").isoformat() if payment.get("estimated_completion") else None
    }

# Update payment status (for external webhooks)
@app.post("/payments/{payment_id}/status")
async def update_payment_status(payment_id: str, update: PaymentStatusUpdate):
    """Update payment status (webhook endpoint)"""
    if payment_id not in payments_db:
        raise HTTPException(status_code=404, detail="Payment not found")
    
    payment = payments_db[payment_id]
    payment["status"] = update.status.value
    
    if update.transaction_id:
        payment["transaction_id"] = update.transaction_id
    
    if update.failure_reason:
        payment["failure_reason"] = update.failure_reason
    
    if update.completed_at:
        payment["completed_at"] = update.completed_at
    
    logger.info("Payment status updated",
               payment_id=payment_id,
               new_status=update.status.value)
    
    # Send notification about status change
    await payment_processor._send_payment_notification(payment_id, update.status.value)
    
    return {"status": "updated", "payment_id": payment_id}

# Process refund
@app.post("/refunds")
async def process_refund(request: RefundRequest):
    """Process a refund"""
    if request.payment_id not in payments_db:
        raise HTTPException(status_code=404, detail="Original payment not found")
    
    original_payment = payments_db[request.payment_id]
    refund_amount = request.amount or original_payment["amount"]
    
    if refund_amount > original_payment["amount"]:
        raise HTTPException(status_code=400, detail="Refund amount cannot exceed original payment")
    
    refund_id = f"ref_{uuid.uuid4().hex[:12]}"
    
    # Create refund record
    refund_record = {
        "refund_id": refund_id,
        "original_payment_id": request.payment_id,
        "amount": refund_amount,
        "reason": request.reason,
        "status": PaymentStatus.PROCESSING.value,
        "created_at": datetime.utcnow()
    }
    
    # In production, process actual refund through payment gateway
    # For demo, mark as completed
    refund_record["status"] = PaymentStatus.COMPLETED.value
    refund_record["completed_at"] = datetime.utcnow()
    
    transactions_db[refund_id] = refund_record
    
    logger.info("Refund processed",
               refund_id=refund_id,
               original_payment_id=request.payment_id,
               amount=refund_amount)
    
    return {
        "refund_id": refund_id,
        "status": refund_record["status"],
        "amount": refund_amount,
        "original_payment_id": request.payment_id,
        "created_at": refund_record["created_at"].isoformat()
    }

# Get payment history
@app.get("/payments")
async def get_payment_history(
    customer_id: Optional[str] = None,
    claim_id: Optional[str] = None,
    status: Optional[PaymentStatus] = None,
    limit: int = 50,
    offset: int = 0
):
    """Get payment history with filters"""
    payments = list(payments_db.values())
    
    # Apply filters
    if customer_id:
        payments = [p for p in payments if p["customer_id"] == customer_id]
    
    if claim_id:
        payments = [p for p in payments if p["claim_id"] == claim_id]
    
    if status:
        payments = [p for p in payments if p["status"] == status.value]
    
    # Sort by created_at descending
    payments.sort(key=lambda x: x["created_at"], reverse=True)
    
    # Paginate
    total = len(payments)
    payments = payments[offset:offset + limit]
    
    # Format response
    formatted_payments = []
    for payment in payments:
        formatted_payment = payment.copy()
        formatted_payment["created_at"] = payment["created_at"].isoformat()
        if "estimated_completion" in payment and payment["estimated_completion"]:
            formatted_payment["estimated_completion"] = payment["estimated_completion"].isoformat()
        formatted_payments.append(formatted_payment)
    
    return {
        "payments": formatted_payments,
        "total": total,
        "limit": limit,
        "offset": offset
    }

# Webhook endpoint for external payment providers
@app.post("/webhook")
async def payment_webhook(payload: Dict[str, Any]):
    """Webhook endpoint for external payment providers"""
    logger.info("Payment webhook received", payload=payload)
    
    # Process webhook payload from Stripe, bank APIs, etc.
    # Update payment statuses accordingly
    
    return {"status": "received", "timestamp": datetime.utcnow().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 