#!/usr/bin/env python3
"""
Service Starter Script for Claims Processing Backend

This script helps start all backend services in the correct order
and provides helpful information about the system status.
"""

import subprocess
import sys
import time
import os
import signal
from typing import List, Dict
import asyncio
import httpx

class ServiceManager:
    def __init__(self):
        self.services = {
            'auth-service': {
                'path': 'services/auth-service',
                'command': ['python', 'main.py'],
                'port': 8003,
                'health_endpoint': '/health'
            },
            'claims-service': {
                'path': 'services/claims-service', 
                'command': ['python', 'main.py'],
                'port': 8001,
                'health_endpoint': '/health'
            },
            'ai-service': {
                'path': 'services/ai-service',
                'command': ['python', 'main.py'],
                'port': 8002,
                'health_endpoint': '/health'
            },
            'api-gateway': {
                'path': 'services/api-gateway',
                'command': ['python', 'main.py'],
                'port': 8000,
                'health_endpoint': '/health'
            }
        }
        self.processes: Dict[str, subprocess.Popen] = {}
        self.running = True
    
    def check_requirements(self):
        """Check if Python and required directories exist"""
        print("🔍 Checking requirements...")
        
        # Check Python
        try:
            result = subprocess.run([sys.executable, '--version'], 
                                  capture_output=True, text=True)
            print(f"✅ Python: {result.stdout.strip()}")
        except Exception as e:
            print(f"❌ Python not found: {e}")
            return False
        
        # Check service directories
        missing_dirs = []
        for service_name, config in self.services.items():
            service_path = config['path']
            main_file = os.path.join(service_path, 'main.py')
            
            if not os.path.exists(service_path):
                missing_dirs.append(service_path)
            elif not os.path.exists(main_file):
                missing_dirs.append(f"{service_path}/main.py")
            else:
                print(f"✅ {service_name}: Found at {service_path}")
        
        if missing_dirs:
            print(f"❌ Missing directories/files: {missing_dirs}")
            return False
        
        return True
    
    def install_dependencies(self):
        """Install Python dependencies for all services"""
        print("\n📦 Installing Python dependencies...")
        
        for service_name, config in self.services.items():
            service_path = config['path']
            requirements_file = os.path.join(service_path, 'requirements.txt')
            
            if os.path.exists(requirements_file):
                print(f"Installing dependencies for {service_name}...")
                try:
                    subprocess.run([
                        sys.executable, '-m', 'pip', 'install', '-r', requirements_file
                    ], check=True, cwd=service_path)
                    print(f"✅ {service_name} dependencies installed")
                except subprocess.CalledProcessError as e:
                    print(f"⚠️  Failed to install {service_name} dependencies: {e}")
            else:
                print(f"⚠️  No requirements.txt found for {service_name}")
    
    def start_service(self, service_name: str):
        """Start a single service"""
        config = self.services[service_name]
        service_path = config['path']
        command = config['command']
        
        print(f"🚀 Starting {service_name} on port {config['port']}...")
        
        try:
            # Start the service
            process = subprocess.Popen(
                command,
                cwd=service_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes[service_name] = process
            print(f"✅ {service_name} started (PID: {process.pid})")
            
            # Give it a moment to start
            time.sleep(2)
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start {service_name}: {e}")
            return False
    
    async def check_service_health(self, service_name: str) -> bool:
        """Check if a service is healthy"""
        config = self.services[service_name]
        health_url = f"http://localhost:{config['port']}{config['health_endpoint']}"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(health_url, timeout=5.0)
                return response.status_code == 200
        except:
            return False
    
    async def wait_for_service(self, service_name: str, timeout: int = 30):
        """Wait for a service to become healthy"""
        print(f"⏳ Waiting for {service_name} to be ready...")
        
        for i in range(timeout):
            if await self.check_service_health(service_name):
                print(f"✅ {service_name} is ready!")
                return True
            
            # Check if process is still running
            if service_name in self.processes:
                if self.processes[service_name].poll() is not None:
                    print(f"❌ {service_name} process died")
                    return False
            
            await asyncio.sleep(1)
        
        print(f"⏰ {service_name} didn't become ready within {timeout} seconds")
        return False
    
    def start_all_services(self):
        """Start all services in the correct order"""
        print("🚀 Starting all backend services...")
        
        # Start services in dependency order
        service_order = ['auth-service', 'claims-service', 'ai-service', 'api-gateway']
        
        for service_name in service_order:
            if not self.start_service(service_name):
                print(f"❌ Failed to start {service_name}, stopping...")
                return False
        
        return True
    
    async def verify_all_services(self):
        """Verify all services are healthy"""
        print("\n🏥 Verifying service health...")
        
        all_healthy = True
        for service_name in self.services:
            is_healthy = await self.wait_for_service(service_name, timeout=10)
            if not is_healthy:
                all_healthy = False
        
        return all_healthy
    
    def show_service_status(self):
        """Show current status of all services"""
        print("\n📊 Service Status:")
        print("=" * 50)
        
        for service_name, config in self.services.items():
            if service_name in self.processes:
                process = self.processes[service_name]
                if process.poll() is None:
                    status = f"✅ Running (PID: {process.pid})"
                else:
                    status = f"❌ Stopped (Exit code: {process.poll()})"
            else:
                status = "⚪ Not started"
            
            print(f"{service_name:<20} Port {config['port']:<6} {status}")
    
    def show_urls(self):
        """Show service URLs"""
        print("\n🌐 Service URLs:")
        print("=" * 50)
        
        urls = {
            'API Gateway': 'http://localhost:8000',
            'API Documentation': 'http://localhost:8000/docs',
            'Auth Service': 'http://localhost:8003',
            'Claims Service': 'http://localhost:8001', 
            'AI Service': 'http://localhost:8002',
            'Health Check': 'http://localhost:8000/health'
        }
        
        for name, url in urls.items():
            print(f"{name:<20} {url}")
    
    def stop_all_services(self):
        """Stop all running services"""
        print("\n🛑 Stopping all services...")
        
        for service_name, process in self.processes.items():
            if process.poll() is None:  # Still running
                print(f"Stopping {service_name}...")
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=5)
                    print(f"✅ {service_name} stopped gracefully")
                except subprocess.TimeoutExpired:
                    print(f"⚠️  Force killing {service_name}...")
                    process.kill()
                    process.wait()
    
    def handle_interrupt(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print("\n\n🛑 Interrupt received, shutting down...")
        self.running = False
        self.stop_all_services()
        sys.exit(0)

async def main():
    """Main function"""
    print("🔧 Claims Processing Backend Service Manager")
    print("=" * 60)
    
    manager = ServiceManager()
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, manager.handle_interrupt)
    
    # Check requirements
    if not manager.check_requirements():
        print("\n❌ Requirements check failed. Please ensure all service directories exist.")
        return
    
    # Ask about dependency installation
    print("\n📦 Do you want to install Python dependencies? (y/n): ", end="")
    if input().lower().startswith('y'):
        manager.install_dependencies()
    
    # Start services
    print("\n🚀 Starting services...")
    if not manager.start_all_services():
        print("❌ Failed to start services")
        return
    
    # Verify services are healthy
    if await manager.verify_all_services():
        print("\n🎉 All services started successfully!")
        manager.show_service_status()
        manager.show_urls()
        
        print("\n" + "=" * 60)
        print("🎯 Backend is ready! You can now:")
        print("   • Test the API at http://localhost:8000/docs")
        print("   • Run the test suite: python test_backend.py")
        print("   • Start building the frontend applications")
        print("\n⏹️  Press Ctrl+C to stop all services")
        
        # Keep running until interrupted
        try:
            while manager.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        print("\n❌ Some services failed to start properly")
        manager.show_service_status()
        
        print("\n🔧 Troubleshooting:")
        print("   • Check if ports 8000-8003 are available")
        print("   • Ensure Docker services are running: npm run docker:up")
        print("   • Check service logs for errors")
        
        manager.stop_all_services()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0) 