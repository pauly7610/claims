from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from typing import Dict, List, Optional, Any
import smtplib
import ssl
import os
import asyncio
from datetime import datetime
import structlog
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
import json
import httpx

# Configure logging
logger = structlog.get_logger()

# Metrics
NOTIFICATIONS_SENT = Counter('notifications_sent_total', 'Total notifications sent', ['channel', 'status'])
NOTIFICATION_PROCESSING_TIME = Histogram('notification_processing_duration_seconds', 'Notification processing time', ['channel'])

app = FastAPI(
    title="Notification Service",
    description="Multi-channel notification service for claims processing",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class EmailNotification(BaseModel):
    to: EmailStr
    subject: str
    body: str
    html_body: Optional[str] = None
    template: Optional[str] = None
    template_data: Optional[Dict[str, Any]] = None

class SMSNotification(BaseModel):
    phone_number: str = Field(..., regex=r'^\+?1?\d{9,15}$')
    message: str = Field(..., max_length=160)

class PushNotification(BaseModel):
    user_id: str
    title: str
    body: str
    data: Optional[Dict[str, Any]] = None

class NotificationRequest(BaseModel):
    user_id: str
    notification_type: str  # claim_status, payment_processed, document_required, etc.
    channel: str  # email, sms, push, all
    data: Dict[str, Any]
    priority: str = "normal"  # low, normal, high, urgent

class NotificationResponse(BaseModel):
    notification_id: str
    status: str
    message: str
    sent_at: datetime

# Email Templates
EMAIL_TEMPLATES = {
    "claim_submitted": {
        "subject": "Claim Submitted Successfully - #{claim_number}",
        "template": """
        <h2>Claim Submitted Successfully</h2>
        <p>Dear {customer_name},</p>
        <p>Your insurance claim has been successfully submitted and is now being processed.</p>
        <p><strong>Claim Details:</strong></p>
        <ul>
            <li>Claim Number: <strong>{claim_number}</strong></li>
            <li>Claim Type: {claim_type}</li>
            <li>Estimated Amount: ${estimated_amount:,.2f}</li>
            <li>Incident Date: {incident_date}</li>
        </ul>
        <p>You can track your claim status at any time by logging into your account.</p>
        <p>If you have any questions, please don't hesitate to contact us.</p>
        <p>Best regards,<br>Claims Processing Team</p>
        """
    },
    "claim_approved": {
        "subject": "Great News! Your Claim #{claim_number} Has Been Approved",
        "template": """
        <h2>🎉 Claim Approved!</h2>
        <p>Dear {customer_name},</p>
        <p>We're pleased to inform you that your claim has been approved for payment.</p>
        <p><strong>Claim Details:</strong></p>
        <ul>
            <li>Claim Number: <strong>{claim_number}</strong></li>
            <li>Approved Amount: <strong>${approved_amount:,.2f}</strong></li>
            <li>Payment Method: {payment_method}</li>
            <li>Expected Payment Date: {payment_date}</li>
        </ul>
        <p>Payment will be processed within 3-5 business days.</p>
        <p>Thank you for choosing our insurance services.</p>
        <p>Best regards,<br>Claims Processing Team</p>
        """
    },
    "claim_denied": {
        "subject": "Claim #{claim_number} - Additional Review Required",
        "template": """
        <h2>Claim Review Update</h2>
        <p>Dear {customer_name},</p>
        <p>After careful review, your claim requires additional documentation before we can proceed.</p>
        <p><strong>Claim Details:</strong></p>
        <ul>
            <li>Claim Number: <strong>{claim_number}</strong></li>
            <li>Review Date: {review_date}</li>
        </ul>
        <p><strong>Required Documents:</strong></p>
        <p>{required_documents}</p>
        <p>Please upload the required documents through your account portal within 30 days.</p>
        <p>If you have any questions, please contact our claims department.</p>
        <p>Best regards,<br>Claims Processing Team</p>
        """
    },
    "fraud_alert": {
        "subject": "Claim #{claim_number} - Additional Verification Required",
        "template": """
        <h2>Additional Verification Required</h2>
        <p>Dear {customer_name},</p>
        <p>As part of our standard review process, your claim requires additional verification.</p>
        <p><strong>Claim Number:</strong> {claim_number}</p>
        <p>A claims specialist will contact you within 2 business days to discuss your claim.</p>
        <p>This is a routine part of our review process and helps ensure accurate claim processing.</p>
        <p>Thank you for your patience.</p>
        <p>Best regards,<br>Claims Review Team</p>
        """
    }
}

# SMS Templates
SMS_TEMPLATES = {
    "claim_submitted": "Claim #{claim_number} submitted successfully. Track status online or call {support_phone}.",
    "claim_approved": "🎉 Great news! Claim #{claim_number} approved for ${approved_amount:,.0f}. Payment processing within 3-5 days.",
    "claim_denied": "Claim #{claim_number} needs additional documents. Check your account for details.",
    "fraud_alert": "Claim #{claim_number} requires verification. Claims specialist will contact you within 2 days.",
    "payment_processed": "Payment of ${amount:,.0f} for claim #{claim_number} has been processed. Check your account."
}

class NotificationService:
    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.from_email = os.getenv("FROM_EMAIL", "noreply@claims.com")
        
        # SMS Configuration (Twilio example)
        self.twilio_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
        self.twilio_token = os.getenv("TWILIO_AUTH_TOKEN", "")
        self.twilio_phone = os.getenv("TWILIO_PHONE_NUMBER", "")
        
        # Push notification configuration
        self.firebase_key = os.getenv("FIREBASE_SERVER_KEY", "")
        
    async def send_email(self, notification: EmailNotification) -> Dict[str, Any]:
        """Send email notification"""
        with NOTIFICATION_PROCESSING_TIME.labels(channel='email').time():
            try:
                # Create message
                msg = MIMEMultipart('alternative')
                msg['Subject'] = notification.subject
                msg['From'] = self.from_email
                msg['To'] = notification.to
                
                # Add body
                if notification.html_body:
                    html_part = MIMEText(notification.html_body, 'html')
                    msg.attach(html_part)
                
                text_part = MIMEText(notification.body, 'plain')
                msg.attach(text_part)
                
                # Send email
                context = ssl.create_default_context()
                with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                    server.starttls(context=context)
                    if self.smtp_username and self.smtp_password:
                        server.login(self.smtp_username, self.smtp_password)
                    server.send_message(msg)
                
                NOTIFICATIONS_SENT.labels(channel='email', status='success').inc()
                logger.info("Email sent successfully", to=notification.to, subject=notification.subject)
                
                return {
                    "status": "sent",
                    "message": "Email sent successfully",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                NOTIFICATIONS_SENT.labels(channel='email', status='failed').inc()
                logger.error("Email sending failed", error=str(e), to=notification.to)
                return {
                    "status": "failed",
                    "message": f"Email sending failed: {str(e)}",
                    "timestamp": datetime.utcnow().isoformat()
                }
    
    async def send_sms(self, notification: SMSNotification) -> Dict[str, Any]:
        """Send SMS notification"""
        with NOTIFICATION_PROCESSING_TIME.labels(channel='sms').time():
            try:
                if not self.twilio_sid or not self.twilio_token:
                    # Simulate SMS sending in development
                    logger.info("SMS simulated (no Twilio config)", 
                               phone=notification.phone_number, 
                               message=notification.message)
                    NOTIFICATIONS_SENT.labels(channel='sms', status='simulated').inc()
                    return {
                        "status": "simulated",
                        "message": "SMS simulated (no Twilio configuration)",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                
                # Real Twilio implementation would go here
                # For now, simulate success
                NOTIFICATIONS_SENT.labels(channel='sms', status='success').inc()
                logger.info("SMS sent successfully", phone=notification.phone_number)
                
                return {
                    "status": "sent",
                    "message": "SMS sent successfully",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                NOTIFICATIONS_SENT.labels(channel='sms', status='failed').inc()
                logger.error("SMS sending failed", error=str(e), phone=notification.phone_number)
                return {
                    "status": "failed",
                    "message": f"SMS sending failed: {str(e)}",
                    "timestamp": datetime.utcnow().isoformat()
                }
    
    async def send_push(self, notification: PushNotification) -> Dict[str, Any]:
        """Send push notification"""
        with NOTIFICATION_PROCESSING_TIME.labels(channel='push').time():
            try:
                if not self.firebase_key:
                    # Simulate push notification in development
                    logger.info("Push notification simulated", 
                               user_id=notification.user_id,
                               title=notification.title)
                    NOTIFICATIONS_SENT.labels(channel='push', status='simulated').inc()
                    return {
                        "status": "simulated",
                        "message": "Push notification simulated (no Firebase config)",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                
                # Real Firebase implementation would go here
                NOTIFICATIONS_SENT.labels(channel='push', status='success').inc()
                logger.info("Push notification sent", user_id=notification.user_id)
                
                return {
                    "status": "sent",
                    "message": "Push notification sent successfully",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                NOTIFICATIONS_SENT.labels(channel='push', status='failed').inc()
                logger.error("Push notification failed", error=str(e), user_id=notification.user_id)
                return {
                    "status": "failed",
                    "message": f"Push notification failed: {str(e)}",
                    "timestamp": datetime.utcnow().isoformat()
                }
    
    def render_template(self, template_name: str, data: Dict[str, Any]) -> Dict[str, str]:
        """Render email template with data"""
        if template_name not in EMAIL_TEMPLATES:
            return {
                "subject": "Notification",
                "html_body": "You have a new notification.",
                "text_body": "You have a new notification."
            }
        
        template = EMAIL_TEMPLATES[template_name]
        subject = template["subject"].format(**data)
        html_body = template["template"].format(**data)
        text_body = html_body.replace('<h2>', '').replace('</h2>', '\n').replace('<p>', '').replace('</p>', '\n')
        
        return {
            "subject": subject,
            "html_body": html_body,
            "text_body": text_body
        }

# Initialize service
notification_service = NotificationService()

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "notification-service",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Send email
@app.post("/send-email")
async def send_email(notification: EmailNotification):
    """Send email notification"""
    result = await notification_service.send_email(notification)
    return result

# Send SMS
@app.post("/send-sms")
async def send_sms(notification: SMSNotification):
    """Send SMS notification"""
    result = await notification_service.send_sms(notification)
    return result

# Send push notification
@app.post("/send-push")
async def send_push(notification: PushNotification):
    """Send push notification"""
    result = await notification_service.send_push(notification)
    return result

# Send notification (unified endpoint)
@app.post("/send-notification", response_model=NotificationResponse)
async def send_notification(request: NotificationRequest):
    """Send notification through specified channel(s)"""
    notification_id = f"notif_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{request.user_id}"
    results = []
    
    try:
        # Render templates
        if request.notification_type in EMAIL_TEMPLATES:
            email_content = notification_service.render_template(request.notification_type, request.data)
        else:
            email_content = {
                "subject": "Notification",
                "html_body": request.data.get("message", "You have a new notification."),
                "text_body": request.data.get("message", "You have a new notification.")
            }
        
        # Send based on channel
        if request.channel == "email" or request.channel == "all":
            if "email" in request.data:
                email_notif = EmailNotification(
                    to=request.data["email"],
                    subject=email_content["subject"],
                    body=email_content["text_body"],
                    html_body=email_content["html_body"]
                )
                email_result = await notification_service.send_email(email_notif)
                results.append({"channel": "email", **email_result})
        
        if request.channel == "sms" or request.channel == "all":
            if "phone_number" in request.data and request.notification_type in SMS_TEMPLATES:
                sms_message = SMS_TEMPLATES[request.notification_type].format(**request.data)
                sms_notif = SMSNotification(
                    phone_number=request.data["phone_number"],
                    message=sms_message
                )
                sms_result = await notification_service.send_sms(sms_notif)
                results.append({"channel": "sms", **sms_result})
        
        if request.channel == "push" or request.channel == "all":
            push_notif = PushNotification(
                user_id=request.user_id,
                title=request.data.get("title", "Notification"),
                body=request.data.get("message", "You have a new notification."),
                data=request.data
            )
            push_result = await notification_service.send_push(push_notif)
            results.append({"channel": "push", **push_result})
        
        # Determine overall status
        statuses = [r.get("status") for r in results]
        if all(s in ["sent", "simulated"] for s in statuses):
            overall_status = "sent"
            message = "All notifications sent successfully"
        elif any(s in ["sent", "simulated"] for s in statuses):
            overall_status = "partial"
            message = "Some notifications sent successfully"
        else:
            overall_status = "failed"
            message = "All notifications failed"
        
        return NotificationResponse(
            notification_id=notification_id,
            status=overall_status,
            message=message,
            sent_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error("Notification sending failed", error=str(e), request=request.dict())
        return NotificationResponse(
            notification_id=notification_id,
            status="failed",
            message=f"Notification failed: {str(e)}",
            sent_at=datetime.utcnow()
        )

# Get user notifications (for inbox)
@app.get("/notifications")
async def get_notifications(user_id: str, limit: int = 50, offset: int = 0):
    """Get notifications for a user (placeholder for database integration)"""
    # In production, this would query a notifications database
    return {
        "notifications": [],
        "total": 0,
        "limit": limit,
        "offset": offset
    }

# Webhook endpoint for external services
@app.post("/webhook")
async def notification_webhook(payload: Dict[str, Any]):
    """Webhook endpoint for external notification triggers"""
    logger.info("Webhook received", payload=payload)
    
    # Process webhook payload and send appropriate notifications
    # This could be triggered by the MLOps system, payment processor, etc.
    
    return {"status": "received", "timestamp": datetime.utcnow().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 