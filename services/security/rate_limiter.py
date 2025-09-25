"""
Advanced API Rate Limiting System

Provides comprehensive rate limiting with:
- Per-user and per-IP rate limiting
- Multiple rate limiting algorithms (token bucket, sliding window)
- Distributed rate limiting with Redis
- Dynamic rate limit adjustment based on user tier
- Rate limiting bypass for trusted services
- Detailed monitoring and alerting
"""

import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
import redis
from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RateLimitAlgorithm(Enum):
    """Rate limiting algorithms"""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"

class UserTier(Enum):
    """User tier for different rate limits"""
    FREE = "free"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"
    ADMIN = "admin"

@dataclass
class RateLimitRule:
    """Rate limiting rule definition"""
    name: str
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_allowance: int = 10  # Additional requests allowed in burst
    user_tiers: List[UserTier] = None
    endpoints: List[str] = None  # Specific endpoints this rule applies to
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW

@dataclass
class RateLimitStatus:
    """Current rate limit status"""
    allowed: bool
    remaining_requests: int
    reset_time: datetime
    retry_after_seconds: int
    rule_name: str
    current_usage: int

class RedisRateLimiter:
    """Redis-based distributed rate limiter"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            logger.info("Connected to Redis for rate limiting")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    def _get_key(self, identifier: str, rule_name: str, window: str) -> str:
        """Generate Redis key for rate limiting"""
        return f"rate_limit:{rule_name}:{identifier}:{window}"
    
    def _get_current_window(self, window_size_seconds: int) -> int:
        """Get current time window"""
        return int(time.time()) // window_size_seconds
    
    async def check_rate_limit_sliding_window(
        self, 
        identifier: str, 
        rule: RateLimitRule
    ) -> RateLimitStatus:
        """Check rate limit using sliding window algorithm"""
        
        if not self.redis_client:
            # Fallback to allowing request if Redis is unavailable
            return RateLimitStatus(
                allowed=True,
                remaining_requests=rule.requests_per_minute,
                reset_time=datetime.now() + timedelta(minutes=1),
                retry_after_seconds=0,
                rule_name=rule.name,
                current_usage=0
            )
        
        now = time.time()
        minute_window = int(now // 60)
        hour_window = int(now // 3600)
        day_window = int(now // 86400)
        
        # Keys for different time windows
        minute_key = self._get_key(identifier, rule.name, f"minute:{minute_window}")
        hour_key = self._get_key(identifier, rule.name, f"hour:{hour_window}")
        day_key = self._get_key(identifier, rule.name, f"day:{day_window}")
        
        try:
            # Use pipeline for atomic operations
            pipe = self.redis_client.pipeline()
            
            # Get current counts
            pipe.get(minute_key)
            pipe.get(hour_key)
            pipe.get(day_key)
            
            results = pipe.execute()
            
            minute_count = int(results[0] or 0)
            hour_count = int(results[1] or 0)
            day_count = int(results[2] or 0)
            
            # Check limits
            if minute_count >= rule.requests_per_minute:
                return RateLimitStatus(
                    allowed=False,
                    remaining_requests=0,
                    reset_time=datetime.fromtimestamp((minute_window + 1) * 60),
                    retry_after_seconds=60 - int(now % 60),
                    rule_name=rule.name,
                    current_usage=minute_count
                )
            
            if hour_count >= rule.requests_per_hour:
                return RateLimitStatus(
                    allowed=False,
                    remaining_requests=0,
                    reset_time=datetime.fromtimestamp((hour_window + 1) * 3600),
                    retry_after_seconds=3600 - int(now % 3600),
                    rule_name=rule.name,
                    current_usage=hour_count
                )
            
            if day_count >= rule.requests_per_day:
                return RateLimitStatus(
                    allowed=False,
                    remaining_requests=0,
                    reset_time=datetime.fromtimestamp((day_window + 1) * 86400),
                    retry_after_seconds=86400 - int(now % 86400),
                    rule_name=rule.name,
                    current_usage=day_count
                )
            
            # Increment counters
            pipe = self.redis_client.pipeline()
            pipe.incr(minute_key)
            pipe.expire(minute_key, 120)  # Keep for 2 minutes
            pipe.incr(hour_key)
            pipe.expire(hour_key, 7200)  # Keep for 2 hours
            pipe.incr(day_key)
            pipe.expire(day_key, 172800)  # Keep for 2 days
            pipe.execute()
            
            return RateLimitStatus(
                allowed=True,
                remaining_requests=min(
                    rule.requests_per_minute - minute_count - 1,
                    rule.requests_per_hour - hour_count - 1,
                    rule.requests_per_day - day_count - 1
                ),
                reset_time=datetime.fromtimestamp((minute_window + 1) * 60),
                retry_after_seconds=0,
                rule_name=rule.name,
                current_usage=max(minute_count, hour_count, day_count)
            )
            
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            # Fallback to allowing request
            return RateLimitStatus(
                allowed=True,
                remaining_requests=rule.requests_per_minute,
                reset_time=datetime.now() + timedelta(minutes=1),
                retry_after_seconds=0,
                rule_name=rule.name,
                current_usage=0
            )
    
    async def check_rate_limit_token_bucket(
        self, 
        identifier: str, 
        rule: RateLimitRule
    ) -> RateLimitStatus:
        """Check rate limit using token bucket algorithm"""
        
        if not self.redis_client:
            return RateLimitStatus(
                allowed=True,
                remaining_requests=rule.requests_per_minute,
                reset_time=datetime.now() + timedelta(minutes=1),
                retry_after_seconds=0,
                rule_name=rule.name,
                current_usage=0
            )
        
        bucket_key = self._get_key(identifier, rule.name, "bucket")
        now = time.time()
        
        try:
            # Get current bucket state
            bucket_data = self.redis_client.hgetall(bucket_key)
            
            if not bucket_data:
                # Initialize bucket
                tokens = rule.requests_per_minute + rule.burst_allowance
                last_refill = now
            else:
                tokens = float(bucket_data.get("tokens", 0))
                last_refill = float(bucket_data.get("last_refill", now))
            
            # Calculate tokens to add based on time elapsed
            time_elapsed = now - last_refill
            tokens_to_add = time_elapsed * (rule.requests_per_minute / 60.0)
            tokens = min(tokens + tokens_to_add, rule.requests_per_minute + rule.burst_allowance)
            
            if tokens >= 1:
                # Allow request and consume token
                tokens -= 1
                
                # Update bucket state
                self.redis_client.hset(bucket_key, mapping={
                    "tokens": tokens,
                    "last_refill": now
                })
                self.redis_client.expire(bucket_key, 3600)  # Expire after 1 hour of inactivity
                
                return RateLimitStatus(
                    allowed=True,
                    remaining_requests=int(tokens),
                    reset_time=datetime.now() + timedelta(seconds=60/rule.requests_per_minute),
                    retry_after_seconds=0,
                    rule_name=rule.name,
                    current_usage=rule.requests_per_minute + rule.burst_allowance - int(tokens)
                )
            else:
                # Rate limited
                retry_after = (1 - tokens) / (rule.requests_per_minute / 60.0)
                
                return RateLimitStatus(
                    allowed=False,
                    remaining_requests=0,
                    reset_time=datetime.now() + timedelta(seconds=retry_after),
                    retry_after_seconds=int(retry_after),
                    rule_name=rule.name,
                    current_usage=rule.requests_per_minute + rule.burst_allowance
                )
                
        except Exception as e:
            logger.error(f"Token bucket error: {e}")
            return RateLimitStatus(
                allowed=True,
                remaining_requests=rule.requests_per_minute,
                reset_time=datetime.now() + timedelta(minutes=1),
                retry_after_seconds=0,
                rule_name=rule.name,
                current_usage=0
            )

class RateLimitManager:
    """Main rate limiting manager"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.limiter = RedisRateLimiter(redis_url)
        self.rules: Dict[str, RateLimitRule] = {}
        self.trusted_services: List[str] = []
        self.security = HTTPBearer(auto_error=False)
        
        # Initialize default rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default rate limiting rules"""
        
        # Free tier users
        self.add_rule(RateLimitRule(
            name="free_tier",
            requests_per_minute=60,
            requests_per_hour=1000,
            requests_per_day=10000,
            burst_allowance=20,
            user_tiers=[UserTier.FREE]
        ))
        
        # Premium users
        self.add_rule(RateLimitRule(
            name="premium_tier",
            requests_per_minute=300,
            requests_per_hour=10000,
            requests_per_day=100000,
            burst_allowance=100,
            user_tiers=[UserTier.PREMIUM]
        ))
        
        # Enterprise users
        self.add_rule(RateLimitRule(
            name="enterprise_tier",
            requests_per_minute=1000,
            requests_per_hour=50000,
            requests_per_day=1000000,
            burst_allowance=500,
            user_tiers=[UserTier.ENTERPRISE]
        ))
        
        # Admin users
        self.add_rule(RateLimitRule(
            name="admin_tier",
            requests_per_minute=2000,
            requests_per_hour=100000,
            requests_per_day=2000000,
            burst_allowance=1000,
            user_tiers=[UserTier.ADMIN]
        ))
        
        # AI/ML endpoints (more restrictive due to computational cost)
        self.add_rule(RateLimitRule(
            name="ml_endpoints",
            requests_per_minute=30,
            requests_per_hour=500,
            requests_per_day=2000,
            burst_allowance=10,
            endpoints=["/api/ai/predict", "/api/ai/fraud-detection", "/api/ai/document-analysis"],
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET
        ))
        
        logger.info(f"Initialized {len(self.rules)} rate limiting rules")
    
    def add_rule(self, rule: RateLimitRule):
        """Add a rate limiting rule"""
        self.rules[rule.name] = rule
        logger.info(f"Added rate limiting rule: {rule.name}")
    
    def add_trusted_service(self, service_identifier: str):
        """Add a trusted service that bypasses rate limiting"""
        self.trusted_services.append(service_identifier)
        logger.info(f"Added trusted service: {service_identifier}")
    
    def _get_user_tier(self, token: str) -> UserTier:
        """Extract user tier from JWT token"""
        try:
            # In production, verify JWT signature
            payload = jwt.decode(token, options={"verify_signature": False})
            tier_str = payload.get("tier", "free")
            return UserTier(tier_str.lower())
        except Exception as e:
            logger.warning(f"Failed to decode token: {e}")
            return UserTier.FREE
    
    def _get_applicable_rules(self, user_tier: UserTier, endpoint: str) -> List[RateLimitRule]:
        """Get applicable rate limiting rules for user and endpoint"""
        applicable_rules = []
        
        for rule in self.rules.values():
            # Check user tier
            if rule.user_tiers and user_tier not in rule.user_tiers:
                continue
            
            # Check endpoint
            if rule.endpoints and not any(endpoint.startswith(ep) for ep in rule.endpoints):
                continue
            
            applicable_rules.append(rule)
        
        return applicable_rules
    
    def _get_client_identifier(self, request: Request, user_id: Optional[str] = None) -> str:
        """Get unique client identifier for rate limiting"""
        if user_id:
            return f"user:{user_id}"
        
        # Use IP address as fallback
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"
        
        return f"ip:{client_ip}"
    
    def _is_trusted_service(self, request: Request, credentials: Optional[HTTPAuthorizationCredentials]) -> bool:
        """Check if request is from a trusted service"""
        
        # Check service token
        if credentials:
            try:
                payload = jwt.decode(credentials.credentials, options={"verify_signature": False})
                service_id = payload.get("service_id")
                if service_id in self.trusted_services:
                    return True
            except:
                pass
        
        # Check service header
        service_header = request.headers.get("X-Service-ID")
        if service_header in self.trusted_services:
            return True
        
        return False
    
    async def check_rate_limit(self, request: Request) -> RateLimitStatus:
        """Check rate limit for incoming request"""
        
        # Get credentials
        credentials = await self.security(request)
        
        # Check if trusted service
        if self._is_trusted_service(request, credentials):
            return RateLimitStatus(
                allowed=True,
                remaining_requests=999999,
                reset_time=datetime.now() + timedelta(hours=1),
                retry_after_seconds=0,
                rule_name="trusted_service",
                current_usage=0
            )
        
        # Get user information
        user_id = None
        user_tier = UserTier.FREE
        
        if credentials:
            try:
                payload = jwt.decode(credentials.credentials, options={"verify_signature": False})
                user_id = payload.get("sub") or payload.get("user_id")
                tier_str = payload.get("tier", "free")
                user_tier = UserTier(tier_str.lower())
            except Exception as e:
                logger.warning(f"Failed to decode token: {e}")
        
        # Get client identifier
        client_id = self._get_client_identifier(request, user_id)
        
        # Get endpoint
        endpoint = request.url.path
        
        # Get applicable rules
        applicable_rules = self._get_applicable_rules(user_tier, endpoint)
        
        if not applicable_rules:
            # No rules apply, allow request
            return RateLimitStatus(
                allowed=True,
                remaining_requests=1000,
                reset_time=datetime.now() + timedelta(hours=1),
                retry_after_seconds=0,
                rule_name="no_limit",
                current_usage=0
            )
        
        # Check each applicable rule
        for rule in applicable_rules:
            if rule.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
                status = await self.limiter.check_rate_limit_token_bucket(client_id, rule)
            else:
                status = await self.limiter.check_rate_limit_sliding_window(client_id, rule)
            
            if not status.allowed:
                # Log rate limit violation
                logger.warning(f"Rate limit exceeded for {client_id} on rule {rule.name}")
                return status
        
        # All rules passed, allow request
        # Return status from most restrictive rule
        most_restrictive = min(applicable_rules, key=lambda r: r.requests_per_minute)
        return await self.limiter.check_rate_limit_sliding_window(client_id, most_restrictive)

# Global instance
rate_limit_manager = None

def get_rate_limit_manager() -> RateLimitManager:
    """Get global rate limit manager instance"""
    global rate_limit_manager
    if rate_limit_manager is None:
        rate_limit_manager = RateLimitManager()
    return rate_limit_manager

# FastAPI dependency for rate limiting
async def rate_limit_dependency(request: Request) -> RateLimitStatus:
    """FastAPI dependency for rate limiting"""
    manager = get_rate_limit_manager()
    status = await manager.check_rate_limit(request)
    
    if not status.allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "rule": status.rule_name,
                "retry_after": status.retry_after_seconds,
                "reset_time": status.reset_time.isoformat()
            },
            headers={
                "X-RateLimit-Limit": str(status.current_usage + status.remaining_requests),
                "X-RateLimit-Remaining": str(status.remaining_requests),
                "X-RateLimit-Reset": str(int(status.reset_time.timestamp())),
                "Retry-After": str(status.retry_after_seconds)
            }
        )
    
    return status

# Utility functions for monitoring
def get_rate_limit_stats() -> Dict[str, Any]:
    """Get rate limiting statistics"""
    manager = get_rate_limit_manager()
    
    return {
        "active_rules": len(manager.rules),
        "trusted_services": len(manager.trusted_services),
        "redis_connected": manager.limiter.redis_client is not None,
        "rules": [
            {
                "name": rule.name,
                "requests_per_minute": rule.requests_per_minute,
                "requests_per_hour": rule.requests_per_hour,
                "requests_per_day": rule.requests_per_day,
                "algorithm": rule.algorithm.value,
                "user_tiers": [tier.value for tier in rule.user_tiers] if rule.user_tiers else None,
                "endpoints": rule.endpoints
            }
            for rule in manager.rules.values()
        ]
    }

# Example usage
if __name__ == "__main__":
    import asyncio
    from fastapi import FastAPI, Request
    
    app = FastAPI()
    
    @app.get("/api/test")
    async def test_endpoint(request: Request, rate_limit: RateLimitStatus = Depends(rate_limit_dependency)):
        return {
            "message": "Success",
            "rate_limit_info": {
                "remaining": rate_limit.remaining_requests,
                "reset_time": rate_limit.reset_time.isoformat()
            }
        }
    
    @app.get("/api/rate-limit-stats")
    async def rate_limit_stats():
        return get_rate_limit_stats()
    
    print("✅ Rate limiting system initialized")
    print("Available at: http://localhost:8000/api/rate-limit-stats")
