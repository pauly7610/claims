'use client';

import React from 'react';
import { ModernButton } from '../ui/modern-button';

const PlusIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
  </svg>
);

const PlayIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.828 14.828a4 4 0 01-5.656 0M9 10h1m4 0h1m-6 4h.01M19 10a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
);

const FloatingElements = () => (
  <div className="absolute inset-0 overflow-hidden pointer-events-none">
    <div className="absolute top-1/4 left-1/4 w-64 h-64 bg-blue-200 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse"></div>
    <div className="absolute top-1/3 right-1/4 w-72 h-72 bg-purple-200 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse" style={{ animationDelay: '2s' }}></div>
    <div className="absolute bottom-1/4 left-1/3 w-80 h-80 bg-pink-200 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse" style={{ animationDelay: '4s' }}></div>
  </div>
);

export const HeroSection: React.FC = () => {
  return (
    <section className="relative min-h-screen flex items-center overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0 bg-gradient-to-br from-blue-50 via-white to-purple-50">
        <div className="absolute inset-0 bg-grid-pattern opacity-[0.02]" />
        <FloatingElements />
      </div>
      
      {/* Content */}
      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 w-full">
        <div className="text-center animate-fade-in">
          <h1 className="text-5xl md:text-7xl font-bold mb-6">
            <span className="bg-gradient-to-r from-gray-900 via-blue-800 to-purple-600 bg-clip-text text-transparent">
              Claims Made
            </span>
            <span className="block text-blue-600 mt-2">
              Incredibly Simple
            </span>
          </h1>
          
          <p className="mt-6 text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
            Submit, track, and resolve your insurance claims with AI-powered assistance. 
            Get faster approvals and transparent communication every step of the way.
          </p>
          
          {/* CTA Buttons */}
          <div className="mt-10 flex flex-col sm:flex-row gap-4 justify-center">
            <ModernButton 
              size="xl" 
              variant="primary"
              icon={<PlusIcon />}
              className="animate-slide-in"
            >
              Start Your Claim
            </ModernButton>
            <ModernButton 
              variant="secondary" 
              size="xl" 
              icon={<PlayIcon />}
              className="animate-slide-in"
              style={{ animationDelay: '0.2s' }}
            >
              Watch Demo
            </ModernButton>
          </div>

          {/* Trust Indicators */}
          <div className="mt-16 grid grid-cols-2 md:grid-cols-4 gap-8 animate-fade-in" style={{ animationDelay: '0.4s' }}>
            <div className="text-center">
              <div className="text-3xl font-bold text-green-600">99.7%</div>
              <div className="text-sm text-gray-600 mt-1">Uptime</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-purple-600">18.5hrs</div>
              <div className="text-sm text-gray-600 mt-1">Avg Resolution</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-yellow-600">97.2%</div>
              <div className="text-sm text-gray-600 mt-1">Customer Satisfaction</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-red-600">$950K</div>
              <div className="text-sm text-gray-600 mt-1">Fraud Prevented</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
