from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
from opentelemetry.sdk.resources import Resource
import os

def setup_tracing(app, service_name: str = "claims-service"):
    """Setup distributed tracing with Jaeger for Claims Service"""
    
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
    
    # Instrument HTTP client for AI service calls
    HTTPXClientInstrumentor().instrument()
    
    # Instrument database
    SQLAlchemyInstrumentor().instrument()
    Psycopg2Instrumentor().instrument()
    
    return tracer

def trace_claim_operation(tracer, operation: str, claim_id: str, user_id: str = None):
    """Create a span for claim-specific operations"""
    span = tracer.start_span(f"claim.{operation}")
    span.set_attribute("operation.type", "claim_processing")
    span.set_attribute("claim.id", claim_id)
    
    if user_id:
        span.set_attribute("user.id", user_id)
    
    return span

def trace_ai_analysis(tracer, claim_id: str, analysis_type: str):
    """Create a span for AI analysis operations"""
    span = tracer.start_span(f"ai.{analysis_type}")
    span.set_attribute("operation.type", "ai_analysis")
    span.set_attribute("claim.id", claim_id)
    span.set_attribute("ai.analysis_type", analysis_type)
    return span

def trace_database_operation(tracer, operation: str, table: str, record_id: str = None):
    """Create a span for database operations"""
    span = tracer.start_span(f"db.{operation}")
    span.set_attribute("operation.type", "database")
    span.set_attribute("db.operation", operation)
    span.set_attribute("db.table", table)
    
    if record_id:
        span.set_attribute("db.record_id", record_id)
    
    return span

def add_claim_attributes(span, claim_data: dict):
    """Add claim-specific attributes to a span"""
    if "claim_type" in claim_data:
        span.set_attribute("claim.type", claim_data["claim_type"])
    if "estimated_amount" in claim_data:
        span.set_attribute("claim.amount", str(claim_data["estimated_amount"]))
    if "status" in claim_data:
        span.set_attribute("claim.status", claim_data["status"])
    if "customer_id" in claim_data:
        span.set_attribute("customer.id", claim_data["customer_id"]) 