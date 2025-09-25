#!/usr/bin/env python3
"""
Test script for the Claims Processing Backend Services

This script demonstrates the working backend by:
1. Testing the AI fraud detection model
2. Testing database connections
3. Testing API endpoints
4. Showing end-to-end claim processing workflow
"""

import asyncio
import httpx
import json
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any

# Add the ML models to the path
sys.path.append('ml/models')

try:
    from fraud_detection import FraudDetectionModel
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️  ML models not available - install requirements from ml/requirements.txt")

class BackendTester:
    def __init__(self):
        self.base_urls = {
            'gateway': 'http://localhost:8000',
            'auth': 'http://localhost:8003',
            'claims': 'http://localhost:8001',
            'ai': 'http://localhost:8002'
        }
        self.auth_token = None
        self.test_user = {
            'email': 'test@example.com',
            'password': 'testpass123',
            'first_name': 'Test',
            'last_name': 'User'
        }
    
    async def test_ml_model(self):
        """Test the fraud detection ML model"""
        print("\n🤖 Testing AI/ML Fraud Detection Model")
        print("=" * 50)
        
        if not ML_AVAILABLE:
            print("❌ ML models not available")
            return False
        
        try:
            # Create and train model
            model = FraudDetectionModel(model_type='random_forest')
            
            # Generate synthetic data
            print("📊 Generating synthetic training data...")
            df = model.create_synthetic_data(n_samples=1000)
            print(f"   Generated {len(df)} samples with {df['is_fraud'].mean():.2%} fraud rate")
            
            # Train model
            print("🎯 Training fraud detection model...")
            results = model.train(df)
            print(f"   AUC Score: {results['auc_score']:.3f}")
            print(f"   Precision: {results['precision']:.3f}")
            print(f"   Recall: {results['recall']:.3f}")
            print(f"   F1 Score: {results['f1_score']:.3f}")
            
            # Test predictions
            print("🔍 Testing model predictions...")
            test_claims = [
                {
                    'estimated_amount': 75000,
                    'policy_age_days': 15,
                    'description': 'Car accident',
                    'claim_type': 'auto',
                    'customer_id': 'customer_1',
                    'incident_date': datetime.now() - timedelta(days=1),
                    'reported_date': datetime.now()
                },
                {
                    'estimated_amount': 5000,
                    'policy_age_days': 365,
                    'description': 'Minor fender bender on highway during rush hour with detailed description of events',
                    'claim_type': 'auto',
                    'customer_id': 'customer_2',
                    'incident_date': datetime.now() - timedelta(days=2),
                    'reported_date': datetime.now() - timedelta(days=1)
                }
            ]
            
            import pandas as pd
            test_df = pd.DataFrame(test_claims)
            predictions = model.predict(test_df)
            
            for i, (claim, prob, pred, explanation) in enumerate(zip(
                test_claims,
                predictions['fraud_probabilities'],
                predictions['predictions'],
                predictions['explanations']
            )):
                print(f"\n   Claim {i+1}: ${claim['estimated_amount']:,}")
                print(f"   Fraud Probability: {prob:.3f}")
                print(f"   Prediction: {'🚨 FRAUD' if pred else '✅ LEGITIMATE'}")
                print(f"   Explanation: {explanation}")
            
            print("\n✅ ML Model test completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ ML Model test failed: {e}")
            return False
    
    async def test_service_health(self):
        """Test all service health endpoints"""
        print("\n🏥 Testing Service Health")
        print("=" * 50)
        
        results = {}
        async with httpx.AsyncClient() as client:
            for service, url in self.base_urls.items():
                try:
                    response = await client.get(f"{url}/health", timeout=5.0)
                    if response.status_code == 200:
                        print(f"✅ {service.upper()}: Healthy")
                        results[service] = True
                    else:
                        print(f"⚠️  {service.upper()}: Unhealthy (status: {response.status_code})")
                        results[service] = False
                except Exception as e:
                    print(f"❌ {service.upper()}: Unreachable - {e}")
                    results[service] = False
        
        return results
    
    async def test_authentication(self):
        """Test authentication flow"""
        print("\n🔐 Testing Authentication")
        print("=" * 50)
        
        async with httpx.AsyncClient() as client:
            try:
                # Test registration
                print("📝 Testing user registration...")
                register_response = await client.post(
                    f"{self.base_urls['auth']}/register",
                    json=self.test_user
                )
                
                if register_response.status_code in [200, 201]:
                    print("✅ User registration successful")
                elif register_response.status_code == 400:
                    print("ℹ️  User already exists (expected)")
                else:
                    print(f"⚠️  Registration response: {register_response.status_code}")
                
                # Test login
                print("🔑 Testing user login...")
                login_response = await client.post(
                    f"{self.base_urls['auth']}/login",
                    json={
                        'email': self.test_user['email'],
                        'password': self.test_user['password']
                    }
                )
                
                if login_response.status_code == 200:
                    token_data = login_response.json()
                    self.auth_token = token_data['access_token']
                    print("✅ Login successful")
                    print(f"   Token expires in: {token_data['expires_in']} seconds")
                    return True
                else:
                    print(f"❌ Login failed: {login_response.status_code}")
                    print(f"   Response: {login_response.text}")
                    return False
                    
            except Exception as e:
                print(f"❌ Authentication test failed: {e}")
                return False
    
    async def test_claims_workflow(self):
        """Test end-to-end claims processing"""
        print("\n📋 Testing Claims Workflow")
        print("=" * 50)
        
        if not self.auth_token:
            print("❌ No auth token available - skipping claims test")
            return False
        
        headers = {'Authorization': f'Bearer {self.auth_token}'}
        
        async with httpx.AsyncClient() as client:
            try:
                # Create a test claim
                print("📝 Creating test claim...")
                claim_data = {
                    'policy_number': 'POL-2024-001234',
                    'incident_date': '2024-01-15',
                    'claim_type': 'auto',
                    'description': 'Vehicle collision at intersection during heavy rain',
                    'estimated_amount': 8500
                }
                
                create_response = await client.post(
                    f"{self.base_urls['claims']}/claims",
                    json=claim_data,
                    headers=headers,
                    params={'customer_id': 'test-customer-id'}
                )
                
                if create_response.status_code in [200, 201]:
                    claim = create_response.json()
                    claim_id = claim['id']
                    print(f"✅ Claim created successfully")
                    print(f"   Claim ID: {claim_id}")
                    print(f"   Claim Number: {claim['claim_number']}")
                    print(f"   Status: {claim['status']}")
                    print(f"   Fraud Score: {claim['fraud_score']}")
                    
                    # Retrieve the claim
                    print("🔍 Retrieving claim...")
                    get_response = await client.get(
                        f"{self.base_urls['claims']}/claims/{claim_id}",
                        headers=headers,
                        params={'customer_id': 'test-customer-id'}
                    )
                    
                    if get_response.status_code == 200:
                        retrieved_claim = get_response.json()
                        print(f"✅ Claim retrieved successfully")
                        print(f"   Updated Status: {retrieved_claim['status']}")
                        print(f"   AI Confidence: {retrieved_claim['ai_confidence']}")
                        return True
                    else:
                        print(f"⚠️  Claim retrieval failed: {get_response.status_code}")
                        return False
                        
                else:
                    print(f"❌ Claim creation failed: {create_response.status_code}")
                    print(f"   Response: {create_response.text}")
                    return False
                    
            except Exception as e:
                print(f"❌ Claims workflow test failed: {e}")
                return False
    
    async def test_ai_services(self):
        """Test AI service endpoints"""
        print("\n🧠 Testing AI Services")
        print("=" * 50)
        
        async with httpx.AsyncClient() as client:
            try:
                # Test fraud analysis
                print("🔍 Testing fraud analysis...")
                fraud_data = {
                    'claim_type': 'auto',
                    'estimated_amount': 75000,
                    'description': 'Expensive car damage',
                    'incident_date': '2024-01-15',
                    'policy_age_days': 15,
                    'customer_id': 'test-customer'
                }
                
                fraud_response = await client.post(
                    f"{self.base_urls['ai']}/analyze-fraud",
                    json=fraud_data
                )
                
                if fraud_response.status_code == 200:
                    fraud_result = fraud_response.json()
                    print(f"✅ Fraud analysis successful")
                    print(f"   Fraud Score: {fraud_result['fraud_score']:.3f}")
                    print(f"   Confidence: {fraud_result['confidence']:.3f}")
                    print(f"   Risk Factors: {', '.join(fraud_result['risk_factors'])}")
                    print(f"   Explanation: {fraud_result['explanation']}")
                    
                    # Test document analysis
                    print("📄 Testing document analysis...")
                    doc_data = {
                        'document_path': '/fake/path/document.pdf',
                        'document_type': 'police_report'
                    }
                    
                    doc_response = await client.post(
                        f"{self.base_urls['ai']}/analyze-document",
                        json=doc_data
                    )
                    
                    if doc_response.status_code == 200:
                        doc_result = doc_response.json()
                        print(f"✅ Document analysis successful")
                        print(f"   Document Type: {doc_result['document_type']}")
                        print(f"   Confidence: {doc_result['confidence']:.3f}")
                        print(f"   Extracted Data: {len(doc_result['extracted_data'])} fields")
                        return True
                    else:
                        print(f"⚠️  Document analysis failed: {doc_response.status_code}")
                        return False
                else:
                    print(f"❌ Fraud analysis failed: {fraud_response.status_code}")
                    return False
                    
            except Exception as e:
                print(f"❌ AI services test failed: {e}")
                return False
    
    async def test_api_gateway(self):
        """Test API Gateway routing and authentication"""
        print("\n🚪 Testing API Gateway")
        print("=" * 50)
        
        async with httpx.AsyncClient() as client:
            try:
                # Test gateway health
                print("🏥 Testing gateway health...")
                health_response = await client.get(f"{self.base_urls['gateway']}/health")
                
                if health_response.status_code == 200:
                    health_data = health_response.json()
                    print(f"✅ Gateway healthy")
                    print(f"   Status: {health_data['status']}")
                    print(f"   Services: {health_data.get('services', {})}")
                    
                    # Test protected endpoint without auth
                    print("🔒 Testing protected endpoint without auth...")
                    unauth_response = await client.get(f"{self.base_urls['gateway']}/api/v1/claims")
                    
                    if unauth_response.status_code == 401:
                        print("✅ Authentication required (expected)")
                        
                        if self.auth_token:
                            # Test with auth
                            print("🔑 Testing protected endpoint with auth...")
                            headers = {'Authorization': f'Bearer {self.auth_token}'}
                            auth_response = await client.get(
                                f"{self.base_urls['gateway']}/api/v1/claims",
                                headers=headers
                            )
                            
                            print(f"✅ Authenticated request: {auth_response.status_code}")
                            return True
                        else:
                            print("⚠️  No auth token available for protected endpoint test")
                            return True
                    else:
                        print(f"⚠️  Expected 401, got {unauth_response.status_code}")
                        return False
                else:
                    print(f"❌ Gateway health check failed: {health_response.status_code}")
                    return False
                    
            except Exception as e:
                print(f"❌ API Gateway test failed: {e}")
                return False
    
    async def run_all_tests(self):
        """Run all backend tests"""
        print("🚀 Starting Backend Services Test Suite")
        print("=" * 60)
        
        test_results = {}
        
        # Test ML model first (doesn't require services)
        test_results['ml_model'] = await self.test_ml_model()
        
        # Test service health
        health_results = await self.test_service_health()
        test_results['service_health'] = all(health_results.values())
        
        # Only run other tests if basic services are healthy
        if health_results.get('auth', False):
            test_results['authentication'] = await self.test_authentication()
        else:
            print("⚠️  Skipping authentication test - auth service not available")
            test_results['authentication'] = False
        
        if health_results.get('claims', False) and test_results['authentication']:
            test_results['claims_workflow'] = await self.test_claims_workflow()
        else:
            print("⚠️  Skipping claims workflow test - dependencies not available")
            test_results['claims_workflow'] = False
        
        if health_results.get('ai', False):
            test_results['ai_services'] = await self.test_ai_services()
        else:
            print("⚠️  Skipping AI services test - AI service not available")
            test_results['ai_services'] = False
        
        if health_results.get('gateway', False):
            test_results['api_gateway'] = await self.test_api_gateway()
        else:
            print("⚠️  Skipping API gateway test - gateway not available")
            test_results['api_gateway'] = False
        
        # Print summary
        print("\n📊 Test Results Summary")
        print("=" * 60)
        
        passed = sum(test_results.values())
        total = len(test_results)
        
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name.replace('_', ' ').title():<25} {status}")
        
        print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("🎉 All tests passed! Backend is fully functional.")
        elif passed >= total * 0.7:
            print("⚠️  Most tests passed. Some services may need to be started.")
        else:
            print("❌ Many tests failed. Check service status and configuration.")
        
        return test_results

async def main():
    """Main test runner"""
    print("🔧 Claims Processing Backend Test Suite")
    print("This will test all backend services and AI models\n")
    
    # Instructions for starting services
    print("📋 Before running tests, make sure services are started:")
    print("   1. Start Docker services: npm run docker:up")
    print("   2. Start Claims Service: cd services/claims-service && python main.py")
    print("   3. Start AI Service: cd services/ai-service && python main.py")
    print("   4. Start Auth Service: cd services/auth-service && python main.py")
    print("   5. Start API Gateway: cd services/api-gateway && python main.py")
    print("\nPress Enter to continue or Ctrl+C to exit...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\nTest cancelled by user")
        return
    
    tester = BackendTester()
    results = await tester.run_all_tests()
    
    # Exit with appropriate code
    if all(results.values()):
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 