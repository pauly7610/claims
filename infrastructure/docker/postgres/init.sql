-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS claims;
CREATE SCHEMA IF NOT EXISTS auth;
CREATE SCHEMA IF NOT EXISTS payments;
CREATE SCHEMA IF NOT EXISTS files;
CREATE SCHEMA IF NOT EXISTS notifications;
CREATE SCHEMA IF NOT EXISTS audit;
CREATE SCHEMA IF NOT EXISTS mlops;

-- Claims tables
CREATE TABLE IF NOT EXISTS claims.policies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    policy_number VARCHAR(50) UNIQUE NOT NULL,
    customer_id UUID NOT NULL,
    policy_type VARCHAR(50) NOT NULL,
    coverage_amount DECIMAL(12,2) NOT NULL,
    deductible DECIMAL(10,2) NOT NULL,
    premium DECIMAL(10,2) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS claims.claims (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    claim_number VARCHAR(50) UNIQUE NOT NULL,
    policy_id UUID REFERENCES claims.policies(id),
    customer_id UUID NOT NULL,
    incident_date DATE NOT NULL,
    reported_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    claim_type VARCHAR(50) NOT NULL,
    description TEXT,
    estimated_amount DECIMAL(12,2),
    approved_amount DECIMAL(12,2),
    status VARCHAR(20) DEFAULT 'submitted',
    priority VARCHAR(10) DEFAULT 'medium',
    assigned_adjuster_id UUID,
    fraud_score DECIMAL(3,2) DEFAULT 0.0,
    ai_confidence DECIMAL(3,2) DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS claims.claim_documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    claim_id UUID REFERENCES claims.claims(id) ON DELETE CASCADE,
    file_name VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    file_size INTEGER NOT NULL,
    document_type VARCHAR(50) NOT NULL,
    ai_extracted_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS claims.claim_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    claim_id UUID REFERENCES claims.claims(id) ON DELETE CASCADE,
    status_from VARCHAR(20),
    status_to VARCHAR(20) NOT NULL,
    changed_by UUID NOT NULL,
    change_reason TEXT,
    ai_decision BOOLEAN DEFAULT false,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Auth tables
CREATE TABLE IF NOT EXISTS auth.users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'customer',
    phone VARCHAR(20),
    is_active BOOLEAN DEFAULT true,
    email_verified BOOLEAN DEFAULT false,
    last_login TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS auth.user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Payments tables
CREATE TABLE IF NOT EXISTS payments.payments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    claim_id UUID REFERENCES claims.claims(id),
    amount DECIMAL(12,2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    payment_method VARCHAR(50) NOT NULL,
    payment_status VARCHAR(20) DEFAULT 'pending',
    external_payment_id VARCHAR(255),
    processed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Notifications tables
CREATE TABLE IF NOT EXISTS notifications.notifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    claim_id UUID,
    type VARCHAR(50) NOT NULL,
    title VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    channel VARCHAR(20) NOT NULL, -- email, sms, push, in_app
    status VARCHAR(20) DEFAULT 'pending',
    sent_at TIMESTAMP,
    read_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Audit tables
CREATE TABLE IF NOT EXISTS audit.audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_type VARCHAR(50) NOT NULL,
    entity_id UUID NOT NULL,
    action VARCHAR(50) NOT NULL,
    user_id UUID,
    changes JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- MLOps tables
CREATE TABLE IF NOT EXISTS mlops.model_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    version VARCHAR(20) NOT NULL,
    stage VARCHAR(20) NOT NULL DEFAULT 'development',
    status VARCHAR(20) NOT NULL DEFAULT 'training',
    
    -- Model artifacts
    model_path VARCHAR(500) NOT NULL,
    config_path VARCHAR(500) NOT NULL,
    metadata_path VARCHAR(500) NOT NULL,
    
    -- Performance metrics (stored as JSON)
    metrics TEXT,
    config TEXT,
    
    -- Training information
    training_dataset_hash VARCHAR(64),
    training_started_at TIMESTAMP,
    training_completed_at TIMESTAMP,
    trained_by VARCHAR(100),
    
    -- Deployment information
    deployed_at TIMESTAMP,
    deployed_by VARCHAR(100),
    deployment_config TEXT,
    
    -- Monitoring
    prediction_count INTEGER DEFAULT 0,
    last_prediction_at TIMESTAMP,
    performance_alerts INTEGER DEFAULT 0,
    
    -- Metadata
    description TEXT,
    tags TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS mlops.model_experiments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_name VARCHAR(100) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    
    -- Experiment configuration
    config TEXT NOT NULL,
    parameters TEXT,
    
    -- Results
    metrics TEXT,
    artifacts_path VARCHAR(500),
    
    -- Status
    status VARCHAR(20) NOT NULL,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    
    -- Metadata
    created_by VARCHAR(100),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS mlops.monitoring_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    run_type VARCHAR(50) NOT NULL,
    
    -- Results
    results TEXT,
    drift_detected BOOLEAN,
    performance_degraded BOOLEAN,
    alerts_generated INTEGER DEFAULT 0,
    
    -- Execution info
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    status VARCHAR(20) DEFAULT 'running',
    error_message TEXT,
    
    -- Data info
    dataset_size INTEGER,
    reference_size INTEGER,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS mlops.model_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    details TEXT,
    
    resolved BOOLEAN DEFAULT false,
    resolved_at TIMESTAMP,
    resolved_by VARCHAR(100),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS mlops.prediction_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    
    -- Input features (JSON)
    features TEXT NOT NULL,
    
    -- Prediction results
    prediction DECIMAL(10,6) NOT NULL,
    confidence DECIMAL(10,6),
    prediction_class VARCHAR(50),
    
    -- Metadata
    request_id VARCHAR(100),
    user_id VARCHAR(100),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Ground truth (for performance monitoring)
    actual_value DECIMAL(10,6),
    feedback_timestamp TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_claims_policy_id ON claims.claims(policy_id);
CREATE INDEX IF NOT EXISTS idx_claims_customer_id ON claims.claims(customer_id);
CREATE INDEX IF NOT EXISTS idx_claims_status ON claims.claims(status);
CREATE INDEX IF NOT EXISTS idx_claims_created_at ON claims.claims(created_at);
CREATE INDEX IF NOT EXISTS idx_claim_documents_claim_id ON claims.claim_documents(claim_id);
CREATE INDEX IF NOT EXISTS idx_claim_history_claim_id ON claims.claim_history(claim_id);
CREATE INDEX IF NOT EXISTS idx_users_email ON auth.users(email);
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON auth.user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_payments_claim_id ON payments.payments(claim_id);
CREATE INDEX IF NOT EXISTS idx_notifications_user_id ON notifications.notifications(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_entity ON audit.audit_logs(entity_type, entity_id);

-- MLOps indexes
CREATE INDEX IF NOT EXISTS idx_model_versions_name ON mlops.model_versions(model_name);
CREATE INDEX IF NOT EXISTS idx_model_versions_stage ON mlops.model_versions(stage);
CREATE INDEX IF NOT EXISTS idx_model_versions_status ON mlops.model_versions(status);
CREATE INDEX IF NOT EXISTS idx_model_experiments_name ON mlops.model_experiments(model_name);
CREATE INDEX IF NOT EXISTS idx_monitoring_runs_model ON mlops.monitoring_runs(model_name, model_version);
CREATE INDEX IF NOT EXISTS idx_model_alerts_model ON mlops.model_alerts(model_name, model_version);
CREATE INDEX IF NOT EXISTS idx_model_alerts_resolved ON mlops.model_alerts(resolved);
CREATE INDEX IF NOT EXISTS idx_prediction_logs_model ON mlops.prediction_logs(model_name, model_version);
CREATE INDEX IF NOT EXISTS idx_prediction_logs_timestamp ON mlops.prediction_logs(timestamp);

-- Insert sample data
INSERT INTO auth.users (id, email, password_hash, first_name, last_name, role) VALUES
('550e8400-e29b-41d4-a716-446655440000', 'admin@claims.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj8xCl1LQvjO', 'Admin', 'User', 'admin'),
('550e8400-e29b-41d4-a716-446655440001', 'adjuster@claims.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj8xCl1LQvjO', 'Claims', 'Adjuster', 'adjuster'),
('550e8400-e29b-41d4-a716-446655440002', 'customer@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj8xCl1LQvjO', 'John', 'Doe', 'customer'),
('550e8400-e29b-41d4-a716-446655440003', 'mlops@claims.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj8xCl1LQvjO', 'MLOps', 'Engineer', 'admin')
ON CONFLICT (email) DO NOTHING; 