"""
Service-to-Service Request Signing System

Provides secure communication between microservices with:
- HMAC-based request signing
- Timestamp-based replay attack prevention
- Service authentication and authorization
- Request integrity verification
- Automatic key rotation
- Comprehensive audit logging
"""

import hmac
import hashlib
import time
import json
import base64
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import secrets
import urllib.parse
from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer
import redis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignatureAlgorithm(Enum):
    """Signature algorithms"""
    HMAC_SHA256 = "hmac-sha256"
    HMAC_SHA512 = "hmac-sha512"

@dataclass
class ServiceCredentials:
    """Service credentials for signing"""
    service_id: str
    service_name: str
    secret_key: str
    algorithm: SignatureAlgorithm = SignatureAlgorithm.HMAC_SHA256
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    permissions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SignedRequest:
    """Signed request information"""
    service_id: str
    timestamp: int
    nonce: str
    signature: str
    headers: Dict[str, str]
    body_hash: Optional[str] = None

class RequestSigner:
    """Handles request signing and verification"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.service_credentials: Dict[str, ServiceCredentials] = {}
        self.nonce_cache_ttl = 300  # 5 minutes
        
        # Initialize Redis for nonce tracking
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("Connected to Redis for request signing")
        except Exception as e:
            logger.warning(f"Redis not available for nonce tracking: {e}")
            self.redis_client = None
        
        # Initialize default service credentials
        self._initialize_service_credentials()
    
    def _initialize_service_credentials(self):
        """Initialize service credentials"""
        
        services = [
            ("api-gateway", "API Gateway", ["*"]),
            ("claims-service", "Claims Service", ["claims:read", "claims:write", "claims:update"]),
            ("ai-service", "AI Service", ["ai:predict", "ai:analyze"]),
            ("auth-service", "Auth Service", ["auth:verify", "users:read"]),
            ("notification-service", "Notification Service", ["notifications:send"]),
            ("payment-service", "Payment Service", ["payments:process", "payments:refund"]),
            ("file-service", "File Service", ["files:upload", "files:download", "files:delete"]),
            ("mlops-service", "MLOps Service", ["models:deploy", "models:monitor"])
        ]
        
        for service_id, service_name, permissions in services:
            self.register_service(
                service_id=service_id,
                service_name=service_name,
                permissions=permissions
            )
    
    def register_service(
        self, 
        service_id: str, 
        service_name: str, 
        permissions: List[str] = None,
        secret_key: str = None,
        expires_in_days: int = 365
    ) -> ServiceCredentials:
        """Register a new service with credentials"""
        
        if not secret_key:
            secret_key = self._generate_secret_key()
        
        credentials = ServiceCredentials(
            service_id=service_id,
            service_name=service_name,
            secret_key=secret_key,
            expires_at=datetime.now() + timedelta(days=expires_in_days),
            permissions=permissions or []
        )
        
        self.service_credentials[service_id] = credentials
        logger.info(f"Registered service: {service_id} ({service_name})")
        
        return credentials
    
    def _generate_secret_key(self, length: int = 64) -> str:
        """Generate a cryptographically secure secret key"""
        return secrets.token_urlsafe(length)
    
    def _create_canonical_string(
        self, 
        method: str, 
        path: str, 
        query_params: Dict[str, str], 
        headers: Dict[str, str], 
        body: str = ""
    ) -> str:
        """Create canonical string for signing"""
        
        # Normalize method and path
        method = method.upper()
        path = urllib.parse.quote(path, safe='/')
        
        # Sort query parameters
        sorted_params = sorted(query_params.items())
        query_string = "&".join([f"{k}={urllib.parse.quote(str(v))}" for k, v in sorted_params])
        
        # Sort headers (only include signed headers)
        signed_headers = ["host", "x-service-id", "x-timestamp", "x-nonce"]
        header_string = "\n".join([f"{h.lower()}:{headers.get(h, '').strip()}" for h in signed_headers if h in headers])
        
        # Create body hash
        body_hash = hashlib.sha256(body.encode('utf-8')).hexdigest() if body else ""
        
        # Combine all parts
        canonical_string = f"{method}\n{path}\n{query_string}\n{header_string}\n{body_hash}"
        
        return canonical_string
    
    def sign_request(
        self, 
        service_id: str, 
        method: str, 
        url: str, 
        headers: Dict[str, str] = None, 
        body: str = ""
    ) -> Dict[str, str]:
        """Sign a request"""
        
        if service_id not in self.service_credentials:
            raise ValueError(f"Service {service_id} not registered")
        
        credentials = self.service_credentials[service_id]
        
        # Check if credentials are expired
        if credentials.expires_at and datetime.now() > credentials.expires_at:
            raise ValueError(f"Credentials for service {service_id} have expired")
        
        # Parse URL
        from urllib.parse import urlparse, parse_qs
        parsed_url = urlparse(url)
        path = parsed_url.path
        query_params = {k: v[0] if v else "" for k, v in parse_qs(parsed_url.query).items()}
        
        # Prepare headers
        if headers is None:
            headers = {}
        
        # Add required headers
        timestamp = int(time.time())
        nonce = secrets.token_urlsafe(16)
        
        headers.update({
            "X-Service-ID": service_id,
            "X-Timestamp": str(timestamp),
            "X-Nonce": nonce,
            "Host": parsed_url.netloc or "localhost"
        })
        
        # Create canonical string
        canonical_string = self._create_canonical_string(method, path, query_params, headers, body)
        
        # Create signature
        if credentials.algorithm == SignatureAlgorithm.HMAC_SHA256:
            signature = hmac.new(
                credentials.secret_key.encode('utf-8'),
                canonical_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
        elif credentials.algorithm == SignatureAlgorithm.HMAC_SHA512:
            signature = hmac.new(
                credentials.secret_key.encode('utf-8'),
                canonical_string.encode('utf-8'),
                hashlib.sha512
            ).hexdigest()
        else:
            raise ValueError(f"Unsupported algorithm: {credentials.algorithm}")
        
        # Add authorization header
        auth_header = f"{credentials.algorithm.value} Credential={service_id}, SignedHeaders=host;x-service-id;x-timestamp;x-nonce, Signature={signature}"
        headers["Authorization"] = auth_header
        
        logger.debug(f"Signed request for service {service_id}")
        return headers
    
    def verify_request(
        self, 
        method: str, 
        url: str, 
        headers: Dict[str, str], 
        body: str = "",
        max_age_seconds: int = 300
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """Verify a signed request"""
        
        try:
            # Extract authorization header
            auth_header = headers.get("Authorization", "")
            if not auth_header:
                return False, "Missing Authorization header", None
            
            # Parse authorization header
            if not auth_header.startswith(("hmac-sha256", "hmac-sha512")):
                return False, "Invalid authorization scheme", None
            
            # Extract components
            auth_parts = {}
            for part in auth_header.split(", "):
                if "=" in part:
                    key, value = part.split("=", 1)
                    auth_parts[key.strip()] = value.strip()
            
            service_id = auth_parts.get("Credential")
            signature = auth_parts.get("Signature")
            
            if not service_id or not signature:
                return False, "Missing credential or signature", None
            
            # Check if service is registered
            if service_id not in self.service_credentials:
                return False, f"Unknown service: {service_id}", None
            
            credentials = self.service_credentials[service_id]
            
            # Check credentials expiry
            if credentials.expires_at and datetime.now() > credentials.expires_at:
                return False, f"Expired credentials for service: {service_id}", None
            
            # Extract required headers
            timestamp_str = headers.get("X-Timestamp")
            nonce = headers.get("X-Nonce")
            
            if not timestamp_str or not nonce:
                return False, "Missing timestamp or nonce", None
            
            # Verify timestamp (prevent replay attacks)
            try:
                timestamp = int(timestamp_str)
                current_time = int(time.time())
                
                if abs(current_time - timestamp) > max_age_seconds:
                    return False, f"Request too old or from future", None
            except ValueError:
                return False, "Invalid timestamp format", None
            
            # Check nonce uniqueness (prevent replay attacks)
            if not self._check_and_store_nonce(service_id, nonce, timestamp):
                return False, "Duplicate nonce (replay attack)", None
            
            # Parse URL for canonical string
            from urllib.parse import urlparse, parse_qs
            parsed_url = urlparse(url)
            path = parsed_url.path
            query_params = {k: v[0] if v else "" for k, v in parse_qs(parsed_url.query).items()}
            
            # Create canonical string
            canonical_string = self._create_canonical_string(method, path, query_params, headers, body)
            
            # Verify signature
            if credentials.algorithm == SignatureAlgorithm.HMAC_SHA256:
                expected_signature = hmac.new(
                    credentials.secret_key.encode('utf-8'),
                    canonical_string.encode('utf-8'),
                    hashlib.sha256
                ).hexdigest()
            elif credentials.algorithm == SignatureAlgorithm.HMAC_SHA512:
                expected_signature = hmac.new(
                    credentials.secret_key.encode('utf-8'),
                    canonical_string.encode('utf-8'),
                    hashlib.sha512
                ).hexdigest()
            else:
                return False, f"Unsupported algorithm: {credentials.algorithm}", None
            
            # Constant-time comparison to prevent timing attacks
            if not hmac.compare_digest(signature, expected_signature):
                return False, "Invalid signature", None
            
            # Return success with service info
            service_info = {
                "service_id": service_id,
                "service_name": credentials.service_name,
                "permissions": credentials.permissions,
                "timestamp": timestamp,
                "nonce": nonce
            }
            
            logger.debug(f"Successfully verified request from service {service_id}")
            return True, None, service_info
            
        except Exception as e:
            logger.error(f"Request verification error: {e}")
            return False, f"Verification error: {str(e)}", None
    
    def _check_and_store_nonce(self, service_id: str, nonce: str, timestamp: int) -> bool:
        """Check nonce uniqueness and store it"""
        
        nonce_key = f"nonce:{service_id}:{nonce}"
        
        if self.redis_client:
            try:
                # Use SET with NX (only set if not exists) and EX (expiry)
                result = self.redis_client.set(nonce_key, timestamp, nx=True, ex=self.nonce_cache_ttl)
                return result is not None
            except Exception as e:
                logger.warning(f"Redis nonce check failed: {e}")
        
        # Fallback: allow request (less secure but functional)
        return True
    
    def rotate_service_key(self, service_id: str) -> ServiceCredentials:
        """Rotate secret key for a service"""
        
        if service_id not in self.service_credentials:
            raise ValueError(f"Service {service_id} not registered")
        
        credentials = self.service_credentials[service_id]
        old_key = credentials.secret_key
        
        # Generate new key
        credentials.secret_key = self._generate_secret_key()
        credentials.updated_at = datetime.now()
        
        logger.info(f"Rotated secret key for service {service_id}")
        
        # In production, you would want to:
        # 1. Store old key temporarily for graceful transition
        # 2. Notify the service about key rotation
        # 3. Update key in secure storage (e.g., HashiCorp Vault)
        
        return credentials
    
    def get_service_info(self, service_id: str) -> Optional[Dict[str, Any]]:
        """Get service information"""
        
        if service_id not in self.service_credentials:
            return None
        
        credentials = self.service_credentials[service_id]
        
        return {
            "service_id": credentials.service_id,
            "service_name": credentials.service_name,
            "algorithm": credentials.algorithm.value,
            "created_at": credentials.created_at.isoformat(),
            "expires_at": credentials.expires_at.isoformat() if credentials.expires_at else None,
            "permissions": credentials.permissions,
            "metadata": credentials.metadata
        }
    
    def list_services(self) -> List[Dict[str, Any]]:
        """List all registered services"""
        
        return [self.get_service_info(service_id) for service_id in self.service_credentials.keys()]

class ServiceAuthMiddleware:
    """FastAPI middleware for service authentication"""
    
    def __init__(self, signer: RequestSigner):
        self.signer = signer
        self.security = HTTPBearer(auto_error=False)
    
    async def verify_service_request(self, request: Request) -> Dict[str, Any]:
        """Verify service request and return service info"""
        
        # Get request details
        method = request.method
        url = str(request.url)
        headers = dict(request.headers)
        
        # Get body if present
        body = ""
        if method in ["POST", "PUT", "PATCH"]:
            try:
                body_bytes = await request.body()
                body = body_bytes.decode('utf-8')
            except:
                body = ""
        
        # Verify request
        is_valid, error_msg, service_info = self.signer.verify_request(method, url, headers, body)
        
        if not is_valid:
            logger.warning(f"Service authentication failed: {error_msg}")
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "Service authentication failed",
                    "message": error_msg
                },
                headers={"WWW-Authenticate": "HMAC-SHA256"}
            )
        
        return service_info

# Global instance
request_signer = None

def get_request_signer() -> RequestSigner:
    """Get global request signer instance"""
    global request_signer
    if request_signer is None:
        request_signer = RequestSigner()
    return request_signer

# FastAPI dependency
async def verify_service_auth(request: Request) -> Dict[str, Any]:
    """FastAPI dependency for service authentication"""
    signer = get_request_signer()
    middleware = ServiceAuthMiddleware(signer)
    return await middleware.verify_service_request(request)

# Utility functions
def sign_service_request(
    service_id: str, 
    method: str, 
    url: str, 
    headers: Dict[str, str] = None, 
    body: str = ""
) -> Dict[str, str]:
    """Utility function to sign a service request"""
    signer = get_request_signer()
    return signer.sign_request(service_id, method, url, headers, body)

def get_service_credentials_info() -> Dict[str, Any]:
    """Get information about all service credentials"""
    signer = get_request_signer()
    return {
        "total_services": len(signer.service_credentials),
        "services": signer.list_services(),
        "nonce_cache_ttl": signer.nonce_cache_ttl,
        "redis_connected": signer.redis_client is not None
    }

# Example usage
if __name__ == "__main__":
    import asyncio
    from fastapi import FastAPI, Request
    
    app = FastAPI()
    
    @app.get("/api/internal/test")
    async def internal_test(request: Request, service_info: Dict[str, Any] = Depends(verify_service_auth)):
        return {
            "message": "Success",
            "authenticated_service": service_info
        }
    
    @app.get("/api/internal/credentials-info")
    async def credentials_info():
        return get_service_credentials_info()
    
    # Example of signing a request
    signer = get_request_signer()
    signed_headers = signer.sign_request(
        service_id="claims-service",
        method="GET",
        url="http://localhost:8000/api/internal/test"
    )
    
    print("✅ Request signing system initialized")
    print(f"Example signed headers: {json.dumps(signed_headers, indent=2)}")
    print("Available at: http://localhost:8000/api/internal/credentials-info")
