import React from 'react'
import Link from 'next/link'
import { ArrowRight, Shield, Clock, Smartphone, CheckCircle, Play, Sparkles } from 'lucide-react'
import { HeroSection } from '../components/landing/hero-section'
import { InteractiveCard, CardContent } from '../components/ui/interactive-card'
import { ModernButton } from '../components/ui/modern-button'

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50">
      {/* Header */}
      <header className="border-b bg-white/80 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Shield className="h-8 w-8 text-blue-600" />
            <span className="text-xl font-bold text-gray-900">ClaimsAI</span>
          </div>
          <div className="flex items-center space-x-4">
            <Link href="/login" className="px-4 py-2 text-gray-600 hover:text-gray-900">
              Sign In
            </Link>
            <Link href="/register" className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
              Get Started
            </Link>
          </div>
        </div>
      </header>

      {/* Modern Hero Section */}
      <HeroSection />

      {/* Features Section */}
      <section className="py-20 bg-white">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              Why Choose Our Platform?
            </h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Experience the future of insurance claims processing with our AI-powered platform
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            <InteractiveCard className="group">
              <CardContent className="text-center p-8">
                <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-blue-600 rounded-2xl flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform duration-300 shadow-lg">
                  <Clock className="h-8 w-8 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-4">Lightning Fast Processing</h3>
                <p className="text-gray-600 leading-relaxed mb-4">
                  AI-powered analysis processes claims in minutes, not days. Get faster approvals and payments with our intelligent automation.
                </p>
                <div className="inline-flex items-center text-sm font-medium text-blue-600 bg-blue-50 px-3 py-1 rounded-full">
                  <Sparkles className="w-4 h-4 mr-1" />
                  Average: 18.5 hours
                </div>
              </CardContent>
            </InteractiveCard>

            <InteractiveCard className="group">
              <CardContent className="text-center p-8">
                <div className="w-16 h-16 bg-gradient-to-br from-green-500 to-emerald-600 rounded-2xl flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform duration-300 shadow-lg">
                  <Shield className="h-8 w-8 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-4">Advanced Fraud Protection</h3>
                <p className="text-gray-600 leading-relaxed mb-4">
                  ML-powered fraud detection ensures legitimate claims are processed quickly while protecting against fraudulent activities.
                </p>
                <div className="inline-flex items-center text-sm font-medium text-green-600 bg-green-50 px-3 py-1 rounded-full">
                  <Shield className="w-4 h-4 mr-1" />
                  $950K Fraud Prevented
                </div>
              </CardContent>
            </InteractiveCard>

            <InteractiveCard className="group">
              <CardContent className="text-center p-8">
                <div className="w-16 h-16 bg-gradient-to-br from-purple-500 to-pink-600 rounded-2xl flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform duration-300 shadow-lg">
                  <Smartphone className="h-8 w-8 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-4">Mobile-First Experience</h3>
                <p className="text-gray-600 leading-relaxed mb-4">
                  Submit claims on-the-go with our mobile-optimized interface. Upload photos and documents with ease.
                </p>
                <div className="inline-flex items-center text-sm font-medium text-purple-600 bg-purple-50 px-3 py-1 rounded-full">
                  <Smartphone className="w-4 h-4 mr-1" />
                  97.2% Mobile Satisfaction
                </div>
              </CardContent>
            </InteractiveCard>
          </div>
        </div>
      </section>

      {/* Process Section */}
      <section className="py-20 bg-gray-50">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              Simple 3-Step Process
            </h2>
            <p className="text-lg text-gray-600">
              Filing a claim has never been easier
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            <div className="relative">
              <div className="bg-white p-8 rounded-lg shadow-sm border-t-4 border-blue-500">
                <div className="w-10 h-10 bg-blue-500 text-white rounded-full flex items-center justify-center text-lg font-bold mb-4">
                  1
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">Submit Your Claim</h3>
                <p className="text-gray-600 mb-4">
                  Fill out our simple form and upload supporting documents. Our AI will guide you through the process.
                </p>
                <ul className="space-y-2">
                  <li className="flex items-center text-sm text-gray-600">
                    <CheckCircle className="h-4 w-4 text-green-500 mr-2" />
                    Upload photos and documents
                  </li>
                  <li className="flex items-center text-sm text-gray-600">
                    <CheckCircle className="h-4 w-4 text-green-500 mr-2" />
                    AI-powered form assistance
                  </li>
                </ul>
              </div>
            </div>

            <div className="relative">
              <div className="bg-white p-8 rounded-lg shadow-sm border-t-4 border-green-500">
                <div className="w-10 h-10 bg-green-500 text-white rounded-full flex items-center justify-center text-lg font-bold mb-4">
                  2
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">AI Analysis</h3>
                <p className="text-gray-600 mb-4">
                  Our AI analyzes your claim for completeness and authenticity, ensuring fast and fair processing.
                </p>
                <ul className="space-y-2">
                  <li className="flex items-center text-sm text-gray-600">
                    <CheckCircle className="h-4 w-4 text-green-500 mr-2" />
                    Automated fraud detection
                  </li>
                  <li className="flex items-center text-sm text-gray-600">
                    <CheckCircle className="h-4 w-4 text-green-500 mr-2" />
                    Document verification
                  </li>
                </ul>
              </div>
            </div>

            <div className="relative">
              <div className="bg-white p-8 rounded-lg shadow-sm border-t-4 border-yellow-500">
                <div className="w-10 h-10 bg-yellow-500 text-white rounded-full flex items-center justify-center text-lg font-bold mb-4">
                  3
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">Get Paid</h3>
                <p className="text-gray-600 mb-4">
                  Receive approval and payment quickly. Track your claim status in real-time.
                </p>
                <ul className="space-y-2">
                  <li className="flex items-center text-sm text-gray-600">
                    <CheckCircle className="h-4 w-4 text-green-500 mr-2" />
                    Real-time status updates
                  </li>
                  <li className="flex items-center text-sm text-gray-600">
                    <CheckCircle className="h-4 w-4 text-green-500 mr-2" />
                    Fast payment processing
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-r from-blue-600 to-blue-700 text-white">
        <div className="container mx-auto px-4 text-center">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            Ready to File Your Claim?
          </h2>
          <p className="text-xl mb-8 opacity-90">
            Join thousands of customers who trust our AI-powered claims processing
          </p>
          <Link href="/register" className="inline-flex items-center px-8 py-4 bg-white text-blue-700 font-semibold rounded-lg hover:bg-gray-100 transition-colors">
            Get Started Today
            <ArrowRight className="ml-2 h-5 w-5" />
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12">
        <div className="container mx-auto px-4">
          <div className="grid md:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center space-x-2 mb-4">
                <Shield className="h-6 w-6" />
                <span className="text-lg font-bold">ClaimsAI</span>
              </div>
              <p className="text-gray-400">
                AI-powered insurance claims processing platform
              </p>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Product</h4>
              <ul className="space-y-2 text-gray-400">
                <li><Link href="/features">Features</Link></li>
                <li><Link href="/pricing">Pricing</Link></li>
                <li><Link href="/demo">Demo</Link></li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Support</h4>
              <ul className="space-y-2 text-gray-400">
                <li><Link href="/help">Help Center</Link></li>
                <li><Link href="/contact">Contact Us</Link></li>
                <li><Link href="/status">Status</Link></li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Legal</h4>
              <ul className="space-y-2 text-gray-400">
                <li><Link href="/privacy">Privacy</Link></li>
                <li><Link href="/terms">Terms</Link></li>
                <li><Link href="/security">Security</Link></li>
              </ul>
            </div>
          </div>
          <div className="border-t border-gray-800 mt-8 pt-8 text-center text-gray-400">
            <p>&copy; 2024 ClaimsAI. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  )
} 