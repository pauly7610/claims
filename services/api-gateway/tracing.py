from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.resources import Resource
import os

def setup_tracing(app, service_name: str = "api-gateway"):
    """Setup distributed tracing with Jaeger"""
    
    # Create resource with service information
    resource = Resource.create({
        "service.name": service_name,
        "service.version": "1.0.0",
        "deployment.environment": os.getenv("ENVIRONMENT", "development")
    })
    
    # Set up tracer provider
    trace.set_tracer_provider(TracerProvider(resource=resource))
    tracer = trace.get_tracer(__name__)
    
    # Configure Jaeger exporter
    jaeger_exporter = JaegerExporter(
        agent_host_name=os.getenv("JAEGER_AGENT_HOST", "jaeger"),
        agent_port=int(os.getenv("JAEGER_AGENT_PORT", "6831")),
    )
    
    # Add span processor
    span_processor = BatchSpanProcessor(jaeger_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    
    # Instrument FastAPI
    FastAPIInstrumentor.instrument_app(app, tracer_provider=trace.get_tracer_provider())
    
    # Instrument HTTP client
    HTTPXClientInstrumentor().instrument()
    
    # Instrument SQLAlchemy if used
    try:
        SQLAlchemyInstrumentor().instrument()
    except Exception:
        pass  # SQLAlchemy not used in API Gateway
    
    return tracer

def create_custom_span(tracer, name: str, operation_type: str = "internal"):
    """Create a custom span for business logic tracing"""
    span = tracer.start_span(name)
    span.set_attribute("operation.type", operation_type)
    return span

def add_business_attributes(span, **attributes):
    """Add business-specific attributes to a span"""
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(f"business.{key}", str(value))

def trace_service_call(tracer, service_name: str, endpoint: str, method: str = "GET"):
    """Create a span for service-to-service calls"""
    span = tracer.start_span(f"{service_name}.{endpoint}")
    span.set_attribute("service.name", service_name)
    span.set_attribute("http.method", method)
    span.set_attribute("http.url", endpoint)
    span.set_attribute("span.kind", "client")
    return span

def trace_business_operation(tracer, operation: str, user_id: str = None, claim_id: str = None):
    """Create a span for business operations"""
    span = tracer.start_span(f"business.{operation}")
    span.set_attribute("operation.type", "business")
    
    if user_id:
        span.set_attribute("user.id", user_id)
    if claim_id:
        span.set_attribute("claim.id", claim_id)
    
    return span 