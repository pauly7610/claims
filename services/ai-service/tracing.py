from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import Resource
import os
import time

def setup_tracing(app, service_name: str = "ai-service"):
    """Setup distributed tracing with Jaeger for AI Service"""
    
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
    
    return tracer

def trace_model_inference(tracer, model_name: str, model_version: str = None):
    """Create a span for ML model inference"""
    span = tracer.start_span(f"ml.inference.{model_name}")
    span.set_attribute("operation.type", "ml_inference")
    span.set_attribute("ml.model.name", model_name)
    
    if model_version:
        span.set_attribute("ml.model.version", model_version)
    
    return span

def trace_fraud_analysis(tracer, claim_id: str):
    """Create a span for fraud analysis"""
    span = tracer.start_span("fraud.analysis")
    span.set_attribute("operation.type", "fraud_detection")
    span.set_attribute("claim.id", claim_id)
    return span

def trace_document_processing(tracer, document_type: str, processing_type: str):
    """Create a span for document processing operations"""
    span = tracer.start_span(f"document.{processing_type}")
    span.set_attribute("operation.type", "document_processing")
    span.set_attribute("document.type", document_type)
    span.set_attribute("processing.type", processing_type)
    return span

def trace_feature_extraction(tracer, feature_count: int):
    """Create a span for feature extraction"""
    span = tracer.start_span("ml.feature_extraction")
    span.set_attribute("operation.type", "feature_engineering")
    span.set_attribute("ml.feature_count", feature_count)
    return span

def add_model_metrics(span, prediction: float, confidence: float, inference_time: float):
    """Add ML model metrics to a span"""
    span.set_attribute("ml.prediction", str(prediction))
    span.set_attribute("ml.confidence", str(confidence))
    span.set_attribute("ml.inference_time_ms", str(inference_time * 1000))

def add_fraud_analysis_results(span, fraud_score: float, risk_factors: list, explanation: str):
    """Add fraud analysis results to a span"""
    span.set_attribute("fraud.score", str(fraud_score))
    span.set_attribute("fraud.risk_factors_count", str(len(risk_factors)))
    span.set_attribute("fraud.explanation_length", str(len(explanation)))
    
    # Add risk factors as individual attributes
    for i, factor in enumerate(risk_factors[:5]):  # Limit to top 5
        span.set_attribute(f"fraud.risk_factor_{i+1}", factor)

def add_document_analysis_results(span, extracted_text_length: int, confidence: float, document_type: str):
    """Add document analysis results to a span"""
    span.set_attribute("document.extracted_text_length", str(extracted_text_length))
    span.set_attribute("document.confidence", str(confidence))
    span.set_attribute("document.detected_type", document_type)

class MLTracer:
    """Context manager for ML operation tracing"""
    
    def __init__(self, tracer, operation_name: str, **attributes):
        self.tracer = tracer
        self.operation_name = operation_name
        self.attributes = attributes
        self.span = None
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.span = self.tracer.start_span(self.operation_name)
        
        # Add initial attributes
        for key, value in self.attributes.items():
            if value is not None:
                self.span.set_attribute(key, str(value))
        
        return self.span
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            # Add execution time
            execution_time = time.time() - self.start_time
            self.span.set_attribute("execution_time_ms", str(execution_time * 1000))
            
            # Add error information if exception occurred
            if exc_type:
                self.span.set_attribute("error", True)
                self.span.set_attribute("error.type", exc_type.__name__)
                self.span.set_attribute("error.message", str(exc_val))
            
            self.span.end() 