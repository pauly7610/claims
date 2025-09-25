# Getting Started with Claims Processing System

This guide will help you set up and run the AI-powered insurance claims processing system locally for development.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Node.js** (v18 or higher) - [Download here](https://nodejs.org/)
- **Python** (v3.9 or higher) - [Download here](https://python.org/)
- **Docker & Docker Compose** - [Download here](https://docker.com/)
- **Git** - [Download here](https://git-scm.com/)

## Quick Start

### 1. Clone and Install

```bash
# Clone the repository
git clone <repository-url>
cd claims

# Install dependencies
npm install

# Copy environment variables
cp env.example .env
```

### 2. Configure Environment

Edit the `.env` file with your configuration:

```bash
# Required: Database configuration
DATABASE_URL=postgresql://claims:claims@localhost:5432/claims
REDIS_URL=redis://localhost:6379

# Required: JWT secret (generate a secure random string)
JWT_SECRET=your-super-secure-jwt-secret-key-here

# Optional: External service API keys
OPENAI_API_KEY=your-openai-api-key
SENDGRID_API_KEY=your-sendgrid-api-key
STRIPE_SECRET_KEY=your-stripe-secret-key
```

### 3. Start the Development Environment

```bash
# Start all services with Docker
npm run docker:up

# This will start:
# - PostgreSQL database (port 5432)
# - Redis cache (port 6379)
# - Prometheus monitoring (port 9090)
# - Grafana dashboards (port 3003)
# - Jaeger tracing (port 16686)
```

### 4. Start the Applications

In separate terminal windows:

```bash
# Terminal 1: Start customer portal
npm run dev --filter=@claims/customer-portal

# Terminal 2: Start adjuster dashboard  
npm run dev --filter=@claims/adjuster-dashboard

# Terminal 3: Start admin panel
npm run dev --filter=@claims/admin-panel

# Terminal 4: Start API gateway
cd services/api-gateway
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### 5. Access the Applications

Once everything is running, you can access:

- **Customer Portal**: http://localhost:3000
- **Adjuster Dashboard**: http://localhost:3001  
- **Admin Panel**: http://localhost:3002
- **API Gateway**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Grafana Dashboards**: http://localhost:3003 (admin/admin)
- **Prometheus Metrics**: http://localhost:9090
- **Jaeger Tracing**: http://localhost:16686

## Development Workflow

### Working with the Monorepo

This project uses [Turbo](https://turbo.build/) for efficient monorepo management:

```bash
# Build all packages and apps
npm run build

# Run tests across all packages
npm run test

# Lint all code
npm run lint

# Type check all TypeScript
npm run type-check

# Clean all build artifacts
npm run clean
```

### Working with Individual Packages

```bash
# Work on the design system
cd packages/design-system
npm run storybook  # Start Storybook on port 6006

# Work on a specific app
npm run dev --filter=@claims/customer-portal

# Build a specific package
npm run build --filter=@claims/shared-types
```

### Working with AI/ML Models

```bash
# Set up Python environment for ML work
cd ml
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Start Jupyter Lab
jupyter lab

# Or use the Docker environment
docker-compose up jupyter
# Access at http://localhost:8888
```

## Database Management

### Initial Setup

The database will be automatically initialized with the schema when you start Docker Compose. Sample users are created:

- **Admin**: admin@claims.com / password
- **Adjuster**: adjuster@claims.com / password  
- **Customer**: customer@example.com / password

### Manual Database Operations

```bash
# Connect to the database
docker exec -it claims_postgres_1 psql -U claims -d claims

# Run migrations (when available)
cd services/api-gateway
alembic upgrade head

# Reset database (WARNING: destroys all data)
docker-compose down -v
docker-compose up postgres
```

## Monitoring and Observability

### Grafana Dashboards

Access Grafana at http://localhost:3003 with credentials `admin/admin`:

- **System Overview**: General system health and performance
- **Claims Processing**: Claims-specific metrics and KPIs
- **AI Model Performance**: Model accuracy, drift, and usage metrics
- **Business Metrics**: Customer satisfaction, processing times, fraud detection

### Prometheus Metrics

Key metrics available at http://localhost:9090:

- `http_requests_total` - Total HTTP requests by service
- `http_request_duration_seconds` - Request latency
- `claims_processed_total` - Total claims processed
- `fraud_detection_rate` - Fraud detection accuracy
- `model_prediction_confidence` - AI model confidence scores

### Distributed Tracing

View request traces at http://localhost:16686:

- End-to-end request tracing across microservices
- Performance bottleneck identification
- Error tracking and debugging

## Testing

### Running Tests

```bash
# Run all tests
npm run test

# Run tests for specific package
npm run test --filter=@claims/shared-types

# Run tests in watch mode
npm run test -- --watch
```

### Test Coverage

```bash
# Generate coverage report
npm run test -- --coverage

# View coverage report
open coverage/lcov-report/index.html
```

## Common Development Tasks

### Adding a New Component

```bash
# Navigate to design system
cd packages/design-system

# Create component files
mkdir src/components/NewComponent
touch src/components/NewComponent/NewComponent.tsx
touch src/components/NewComponent/NewComponent.stories.tsx
touch src/components/NewComponent/index.ts

# Add to main export
# Edit src/index.ts to export the new component
```

### Adding a New API Endpoint

```bash
# Navigate to appropriate service
cd services/claims-service

# Add route handler
# Edit main.py or create new route file

# Update OpenAPI schema
# Edit openapi.yaml

# Regenerate API client
cd packages/api-client
npm run generate
```

### Creating a New ML Model

```bash
# Navigate to ML directory
cd ml/notebooks

# Create new notebook
jupyter lab

# Follow the pattern in existing notebooks:
# 1. Data exploration
# 2. Feature engineering  
# 3. Model training
# 4. Evaluation
# 5. Model saving
# 6. Monitoring setup
```

## Troubleshooting

### Common Issues

**Port conflicts:**
```bash
# Check what's using a port
lsof -i :3000

# Kill process using port
kill -9 <PID>
```

**Database connection issues:**
```bash
# Check if PostgreSQL is running
docker ps | grep postgres

# View database logs
docker logs claims_postgres_1

# Reset database connection
docker-compose restart postgres
```

**Node.js module issues:**
```bash
# Clear npm cache
npm cache clean --force

# Remove node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

**Python environment issues:**
```bash
# Recreate virtual environment
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Getting Help

- Check the [Architecture Documentation](./architecture/)
- Review [API Documentation](./api/)
- Look at existing [examples and tests](../packages/)
- Open an issue on GitHub

## Next Steps

Once you have the system running:

1. **Explore the codebase** - Start with the shared types and design system
2. **Review the sample data** - Check the database schema and sample claims
3. **Run the ML notebooks** - Understand the AI models and training process
4. **Customize the UI** - Modify components in the design system
5. **Add new features** - Follow the established patterns and architecture

For production deployment, see the [Deployment Guide](./deployment/).

Happy coding! 🚀 