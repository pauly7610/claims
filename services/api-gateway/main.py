from fastapi import FastAPI, HTTPException, Depends, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import structlog
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
import time
import httpx
import os
from typing import Dict, Any, Optional
import json

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status', 'service'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency', ['service'])
SERVICE_ERRORS = Counter('service_errors_total', 'Total service errors', ['service', 'error_type'])

# Security
security = HTTPBearer(auto_error=False)

# Service Registry
SERVICE_REGISTRY = {
    "claims": {
        "url": os.getenv("CLAIMS_SERVICE_URL", "http://claims-service:8000"),
        "health_endpoint": "/health"
    },
    "ai": {
        "url": os.getenv("AI_SERVICE_URL", "http://ai-service:8000"),
        "health_endpoint": "/health"
    },
    "auth": {
        "url": os.getenv("AUTH_SERVICE_URL", "http://auth-service:8000"),
        "health_endpoint": "/health"
    },
    "notifications": {
        "url": os.getenv("NOTIFICATION_SERVICE_URL", "http://notification-service:8000"),
        "health_endpoint": "/health"
    },
    "payments": {
        "url": os.getenv("PAYMENT_SERVICE_URL", "http://payment-service:8000"),
        "health_endpoint": "/health"
    },
    "files": {
        "url": os.getenv("FILE_SERVICE_URL", "http://file-service:8000"),
        "health_endpoint": "/health"
    }
}

# HTTP Client for service communication
http_client = httpx.AsyncClient(timeout=30.0)

# Authentication Service Client
class AuthServiceClient:
    def __init__(self):
        self.base_url = SERVICE_REGISTRY["auth"]["url"]
    
    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify token with auth service"""
        try:
            response = await http_client.post(
                f"{self.base_url}/verify-token",
                headers={"Authorization": f"Bearer {token}"}
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error("Token verification failed", error=str(e))
            return None

auth_client = AuthServiceClient()

# Service Proxy
class ServiceProxy:
    def __init__(self):
        self.services = SERVICE_REGISTRY
    
    async def proxy_request(
        self,
        service_name: str,
        path: str,
        method: str,
        headers: Dict[str, str] = None,
        params: Dict[str, Any] = None,
        json_data: Dict[str, Any] = None,
        data: Any = None
    ) -> Response:
        """Proxy request to downstream service"""
        
        if service_name not in self.services:
            raise HTTPException(
                status_code=404,
                detail=f"Service '{service_name}' not found"
            )
        
        service_url = self.services[service_name]["url"]
        target_url = f"{service_url}{path}"
        
        # Prepare headers
        proxy_headers = {}
        if headers:
            # Forward important headers
            for key, value in headers.items():
                if key.lower() in ['authorization', 'content-type', 'user-agent']:
                    proxy_headers[key] = value
        
        try:
            with REQUEST_LATENCY.labels(service=service_name).time():
                # Make request to downstream service
                response = await http_client.request(
                    method=method,
                    url=target_url,
                    headers=proxy_headers,
                    params=params,
                    json=json_data,
                    data=data
                )
                
                # Create FastAPI response
                fastapi_response = Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.headers.get('content-type', 'application/json')
                )
                
                REQUEST_COUNT.labels(
                    method=method,
                    endpoint=path,
                    status=response.status_code,
                    service=service_name
                ).inc()
                
                return fastapi_response
                
        except httpx.RequestError as e:
            SERVICE_ERRORS.labels(service=service_name, error_type='connection').inc()
            logger.error("Service request failed", service=service_name, error=str(e))
            raise HTTPException(
                status_code=503,
                detail=f"Service '{service_name}' unavailable"
            )
        except Exception as e:
            SERVICE_ERRORS.labels(service=service_name, error_type='unknown').inc()
            logger.error("Proxy request failed", service=service_name, error=str(e))
            raise HTTPException(
                status_code=500,
                detail="Internal server error"
            )
    
    async def check_service_health(self, service_name: str) -> bool:
        """Check if service is healthy"""
        if service_name not in self.services:
            return False
        
        try:
            service_config = self.services[service_name]
            health_url = f"{service_config['url']}{service_config['health_endpoint']}"
            
            response = await http_client.get(health_url, timeout=5.0)
            return response.status_code == 200
            
        except Exception:
            return False

service_proxy = ServiceProxy()

# Authentication Middleware
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[Dict[str, Any]]:
    """Get current user from token"""
    if not credentials:
        return None
    
    user_info = await auth_client.verify_token(credentials.credentials)
    return user_info

# Route Protection
def require_auth():
    """Require authentication for route"""
    async def auth_dependency(user_info: Optional[Dict[str, Any]] = Depends(get_current_user)):
        if not user_info:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        return user_info
    return auth_dependency

def require_role(allowed_roles: list):
    """Require specific role for route"""
    async def role_dependency(user_info: Dict[str, Any] = Depends(require_auth())):
        user_role = user_info.get("role")
        if user_role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return user_info
    return role_dependency

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting API Gateway service")
    
    # Check service health on startup
    for service_name in SERVICE_REGISTRY:
        is_healthy = await service_proxy.check_service_health(service_name)
        logger.info(f"Service health check", service=service_name, healthy=is_healthy)
    
    yield
    
    # Cleanup
    await http_client.aclose()
    logger.info("Shutting down API Gateway service")

# Initialize FastAPI app
app = FastAPI(
    title="Claims Processing API Gateway",
    description="AI-Powered Insurance Claims Processing System",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "api-gateway"]
)

# Add OpenTelemetry instrumentation
FastAPIInstrumentor.instrument_app(app)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate process time
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log request
    logger.info(
        "Request processed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        process_time=process_time,
        client_ip=request.client.host if request.client else None
    )
    
    return response

@app.get("/")
async def root():
    return {
        "message": "Claims Processing API Gateway",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": time.time(),
        "services": list(SERVICE_REGISTRY.keys())
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check including downstream services"""
    service_health = {}
    
    for service_name in SERVICE_REGISTRY:
        service_health[service_name] = await service_proxy.check_service_health(service_name)
    
    overall_health = all(service_health.values())
    
    return {
        "status": "healthy" if overall_health else "degraded",
        "timestamp": time.time(),
        "version": "1.0.0",
        "services": service_health
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    from fastapi import Response
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Authentication Routes (direct proxy to auth service)
@app.post("/api/v1/auth/register")
async def register(request: Request):
    """Register new user"""
    body = await request.body()
    return await service_proxy.proxy_request(
        service_name="auth",
        path="/register",
        method="POST",
        headers=dict(request.headers),
        data=body
    )

@app.post("/api/v1/auth/login")
async def login(request: Request):
    """User login"""
    body = await request.body()
    return await service_proxy.proxy_request(
        service_name="auth",
        path="/login",
        method="POST",
        headers=dict(request.headers),
        data=body
    )

@app.post("/api/v1/auth/refresh")
async def refresh_token(request: Request):
    """Refresh access token"""
    body = await request.body()
    return await service_proxy.proxy_request(
        service_name="auth",
        path="/refresh",
        method="POST",
        headers=dict(request.headers),
        data=body
    )

@app.post("/api/v1/auth/logout")
async def logout(request: Request, user_info: Dict[str, Any] = Depends(require_auth())):
    """User logout"""
    body = await request.body()
    return await service_proxy.proxy_request(
        service_name="auth",
        path="/logout",
        method="POST",
        headers=dict(request.headers),
        data=body
    )

@app.get("/api/v1/auth/me")
async def get_user_profile(request: Request, user_info: Dict[str, Any] = Depends(require_auth())):
    """Get current user profile"""
    return await service_proxy.proxy_request(
        service_name="auth",
        path="/me",
        method="GET",
        headers=dict(request.headers)
    )

# Claims Routes (protected)
@app.post("/api/v1/claims")
async def create_claim(request: Request, user_info: Dict[str, Any] = Depends(require_auth())):
    """Create new insurance claim"""
    body = await request.body()
    
    # Add customer ID from authenticated user
    try:
        request_data = json.loads(body) if body else {}
        customer_id = user_info.get("user_id")
        
        return await service_proxy.proxy_request(
            service_name="claims",
            path="/claims",
            method="POST",
            headers=dict(request.headers),
            params={"customer_id": customer_id},
            data=body
        )
    except Exception as e:
        logger.error("Create claim failed", error=str(e))
        raise HTTPException(status_code=400, detail="Invalid request data")

@app.get("/api/v1/claims")
async def list_claims(request: Request, user_info: Dict[str, Any] = Depends(require_auth())):
    """List claims for authenticated user"""
    customer_id = user_info.get("user_id")
    
    return await service_proxy.proxy_request(
        service_name="claims",
        path="/claims",
        method="GET",
        headers=dict(request.headers),
        params={"customer_id": customer_id, **dict(request.query_params)}
    )

@app.get("/api/v1/claims/{claim_id}")
async def get_claim(claim_id: str, request: Request, user_info: Dict[str, Any] = Depends(require_auth())):
    """Get specific claim"""
    customer_id = user_info.get("user_id")
    
    return await service_proxy.proxy_request(
        service_name="claims",
        path=f"/claims/{claim_id}",
        method="GET",
        headers=dict(request.headers),
        params={"customer_id": customer_id}
    )

@app.put("/api/v1/claims/{claim_id}/status")
async def update_claim_status(
    claim_id: str,
    request: Request,
    user_info: Dict[str, Any] = Depends(require_role(["adjuster", "admin"]))
):
    """Update claim status (adjusters/admins only)"""
    body = await request.body()
    user_id = user_info.get("user_id")
    
    return await service_proxy.proxy_request(
        service_name="claims",
        path=f"/claims/{claim_id}/status",
        method="PUT",
        headers=dict(request.headers),
        params={"user_id": user_id},
        data=body
    )

# AI Service Routes (protected)
@app.post("/api/v1/ai/analyze-fraud")
async def analyze_fraud(request: Request, user_info: Dict[str, Any] = Depends(require_role(["adjuster", "admin", "system"]))):
    """Analyze claim for fraud (internal use)"""
    body = await request.body()
    return await service_proxy.proxy_request(
        service_name="ai",
        path="/analyze-fraud",
        method="POST",
        headers=dict(request.headers),
        data=body
    )

@app.post("/api/v1/ai/analyze-document")
async def analyze_document(request: Request, user_info: Dict[str, Any] = Depends(require_auth())):
    """Analyze uploaded document"""
    body = await request.body()
    return await service_proxy.proxy_request(
        service_name="ai",
        path="/analyze-document",
        method="POST",
        headers=dict(request.headers),
        data=body
    )

@app.post("/api/v1/ai/upload-document")
async def upload_document(request: Request, user_info: Dict[str, Any] = Depends(require_auth())):
    """Upload document for analysis"""
    # This would handle file upload and proxy to AI service
    return await service_proxy.proxy_request(
        service_name="ai",
        path="/upload-document",
        method="POST",
        headers=dict(request.headers)
    )

# File Service Routes (protected)
@app.post("/api/v1/files/upload")
async def upload_file(request: Request, user_info: Dict[str, Any] = Depends(require_auth())):
    """Upload file"""
    return await service_proxy.proxy_request(
        service_name="files",
        path="/upload",
        method="POST",
        headers=dict(request.headers)
    )

@app.get("/api/v1/files/{file_id}")
async def get_file(file_id: str, request: Request, user_info: Dict[str, Any] = Depends(require_auth())):
    """Get file"""
    return await service_proxy.proxy_request(
        service_name="files",
        path=f"/files/{file_id}",
        method="GET",
        headers=dict(request.headers)
    )

# Notification Routes (protected)
@app.get("/api/v1/notifications")
async def get_notifications(request: Request, user_info: Dict[str, Any] = Depends(require_auth())):
    """Get user notifications"""
    user_id = user_info.get("user_id")
    
    return await service_proxy.proxy_request(
        service_name="notifications",
        path="/notifications",
        method="GET",
        headers=dict(request.headers),
        params={"user_id": user_id, **dict(request.query_params)}
    )

# Admin Routes (admin only)
@app.get("/api/v1/admin/users")
async def list_users(request: Request, user_info: Dict[str, Any] = Depends(require_role(["admin"]))):
    """List all users (admin only)"""
    return await service_proxy.proxy_request(
        service_name="auth",
        path="/users",
        method="GET",
        headers=dict(request.headers),
        params=dict(request.query_params)
    )

@app.get("/api/v1/admin/claims")
async def list_all_claims(request: Request, user_info: Dict[str, Any] = Depends(require_role(["admin", "adjuster"]))):
    """List all claims (admin/adjuster only)"""
    return await service_proxy.proxy_request(
        service_name="claims",
        path="/admin/claims",
        method="GET",
        headers=dict(request.headers),
        params=dict(request.query_params)
    )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error("HTTP exception", path=request.url.path, status_code=exc.status_code, detail=exc.detail)
    return {
        "error": {
            "code": exc.status_code,
            "message": exc.detail,
            "timestamp": time.time()
        }
    }

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception", path=request.url.path, error=str(exc))
    return Response(
        content=json.dumps({
            "error": {
                "code": 500,
                "message": "Internal server error",
                "timestamp": time.time()
            }
        }),
        status_code=500,
        media_type="application/json"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 