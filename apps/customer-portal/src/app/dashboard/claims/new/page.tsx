'use client'

import React from 'react'
import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { useMutation } from '@tanstack/react-query'
// Simple UI Components (inline)
const Card = ({ children, className = '' }: { children: React.ReactNode, className?: string }) => 
  <div className={`bg-white rounded-lg border shadow-sm ${className}`}>{children}</div>
const CardContent = ({ children, className = '' }: { children: React.ReactNode, className?: string }) => 
  <div className={className}>{children}</div>
const CardHeader = ({ children }: { children: React.ReactNode }) => 
  <div className="p-6 pb-0">{children}</div>
const CardTitle = ({ children, className = '' }: { children: React.ReactNode, className?: string }) => 
  <h3 className={`text-lg font-semibold ${className}`}>{children}</h3>
interface ButtonProps {
  children: React.ReactNode
  variant?: 'default' | 'outline' | 'ghost'
  size?: 'default' | 'sm'
  onClick?: () => void
  className?: string
  type?: 'button' | 'submit'
  loading?: boolean
  disabled?: boolean
}

const Button = ({ children, variant = 'default', size = 'default', onClick, className = '', type = 'button', loading = false, disabled = false }: ButtonProps) => 
  <button type={type} onClick={onClick} disabled={disabled || loading} className={`px-4 py-2 rounded-md font-medium ${variant === 'outline' ? 'border border-gray-300 bg-white text-gray-700 hover:bg-gray-50' : variant === 'ghost' ? 'bg-transparent text-gray-700 hover:bg-gray-100' : 'bg-blue-600 text-white hover:bg-blue-700'} ${size === 'sm' ? 'px-3 py-1 text-sm' : ''} ${disabled ? 'opacity-50 cursor-not-allowed' : ''} ${className}`}>{loading ? 'Loading...' : children}</button>
interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string
  leftIcon?: React.ReactNode
  className?: string
}

const Input = ({ label, leftIcon, className = '', ...props }: InputProps) => (
  <div className="space-y-1">
    {label && <label className="block text-sm font-medium text-gray-700">{label}</label>}
    <div className="relative">
      {leftIcon && <div className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400">{leftIcon}</div>}
      <input className={`w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent ${leftIcon ? 'pl-10' : ''} ${className}`} {...props} />
    </div>
  </div>
)

interface AlertProps {
  title: string
  description: string
  variant?: 'default' | 'destructive' | 'info'
  className?: string
}

const Alert = ({ title, description, variant = 'default', className = '' }: AlertProps) => 
  <div className={`p-4 rounded-md ${variant === 'destructive' ? 'bg-red-50 border border-red-200' : variant === 'info' ? 'bg-blue-50 border border-blue-200' : 'bg-gray-50 border border-gray-200'} ${className}`}>
    <h4 className="font-medium">{title}</h4>
    <p className="text-sm mt-1">{description}</p>
  </div>
import { 
  Upload, 
  FileText, 
  Camera, 
  AlertCircle, 
  CheckCircle,
  ArrowLeft,
  Bot,
  Sparkles
} from 'lucide-react'
import { claimsApi } from '@/lib/api-client'
import { useDropzone } from 'react-dropzone'

interface ClaimFormData {
  claim_type: string
  description: string
  estimated_amount: number
  incident_date: string
  location: string
  policy_number: string
}

const claimTypes = [
  { value: 'auto', label: 'Auto Insurance', description: 'Vehicle accidents, theft, or damage' },
  { value: 'home', label: 'Home Insurance', description: 'Property damage, theft, or natural disasters' },
  { value: 'health', label: 'Health Insurance', description: 'Medical expenses and treatments' },
  { value: 'life', label: 'Life Insurance', description: 'Life insurance claims' },
  { value: 'travel', label: 'Travel Insurance', description: 'Trip cancellation, lost luggage, or medical emergencies' },
  { value: 'other', label: 'Other', description: 'Other types of insurance claims' },
]

export default function NewClaimPage() {
  const router = useRouter()
  const [step, setStep] = useState(1)
  const [formData, setFormData] = useState<ClaimFormData>({
    claim_type: '',
    description: '',
    estimated_amount: 0,
    incident_date: '',
    location: '',
    policy_number: '',
  })
  const [files, setFiles] = useState<File[]>([])
  const [aiSuggestions, setAiSuggestions] = useState<string[]>([])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif'],
      'application/pdf': ['.pdf'],
      'application/msword': ['.doc'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
    },
    onDrop: (acceptedFiles) => {
      setFiles([...files, ...acceptedFiles])
    },
  })

  const submitClaim = useMutation({
    mutationFn: async (data: ClaimFormData & { files: File[] }) => {
      // First create the claim
      const claimResponse = await claimsApi.createClaim({
        claim_type: data.claim_type,
        description: data.description,
        estimated_amount: data.estimated_amount,
        incident_date: data.incident_date,
        location: data.location,
        policy_number: data.policy_number,
      })

      const claimId = claimResponse.data.claim.id

      // Upload files if any
      if (data.files.length > 0) {
        await Promise.all(
          data.files.map(file => claimsApi.uploadFile(file, claimId))
        )
      }

      return claimResponse.data
    },
    onSuccess: (data) => {
      router.push(`/dashboard/claims/${data.claim.id}`)
    },
  })

  const handleInputChange = (field: keyof ClaimFormData, value: any) => {
    setFormData({ ...formData, [field]: value })
    
    // AI suggestions based on input
    if (field === 'description' && value.length > 20) {
      // Simulate AI suggestions
      setAiSuggestions([
        'Consider including the time of incident',
        'Add details about weather conditions',
        'Include any witnesses present',
      ])
    }
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    submitClaim.mutate({ ...formData, files })
  }

  const removeFile = (index: number) => {
    setFiles(files.filter((_, i) => i !== index))
  }

  return (
    <div className="min-h-screen bg-neutral-50">
      {/* Header */}
      <header className="bg-white border-b">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center space-x-4">
            <Button
              variant="ghost"
              onClick={() => router.back()}
            >
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back
            </Button>
            <div>
              <h1 className="text-2xl font-bold text-neutral-900">Submit New Claim</h1>
              <p className="text-neutral-600">Our AI will help guide you through the process</p>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8 max-w-4xl">
        {/* Progress Steps */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            {[1, 2, 3].map((stepNumber) => (
              <div key={stepNumber} className="flex items-center">
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
                    step >= stepNumber
                      ? 'bg-primary-600 text-white'
                      : 'bg-neutral-200 text-neutral-600'
                  }`}
                >
                  {stepNumber}
                </div>
                <div className="ml-3">
                  <p className={`text-sm font-medium ${
                    step >= stepNumber ? 'text-primary-600' : 'text-neutral-500'
                  }`}>
                    {stepNumber === 1 && 'Claim Details'}
                    {stepNumber === 2 && 'Upload Documents'}
                    {stepNumber === 3 && 'Review & Submit'}
                  </p>
                </div>
                {stepNumber < 3 && (
                  <div className={`w-16 h-0.5 ml-4 ${
                    step > stepNumber ? 'bg-primary-600' : 'bg-neutral-200'
                  }`} />
                )}
              </div>
            ))}
          </div>
        </div>

        <form onSubmit={handleSubmit}>
          {/* Step 1: Claim Details */}
          {step === 1 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <FileText className="h-5 w-5 mr-2" />
                  Claim Information
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Claim Type Selection */}
                <div>
                  <label className="block text-sm font-medium text-neutral-700 mb-3">
                    What type of claim are you filing? *
                  </label>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {claimTypes.map((type) => (
                      <div
                        key={type.value}
                        className={`border rounded-lg p-4 cursor-pointer transition-colors ${
                          formData.claim_type === type.value
                            ? 'border-primary-500 bg-primary-50'
                            : 'border-neutral-200 hover:border-neutral-300'
                        }`}
                        onClick={() => handleInputChange('claim_type', type.value)}
                      >
                        <h3 className="font-medium text-neutral-900">{type.label}</h3>
                        <p className="text-sm text-neutral-600">{type.description}</p>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <Input
                    label="Policy Number"
                    value={formData.policy_number}
                    onChange={(e) => handleInputChange('policy_number', e.target.value)}
                    placeholder="Enter your policy number"
                    required
                  />

                  <Input
                    label="Estimated Amount"
                    type="number"
                    value={formData.estimated_amount || ''}
                    onChange={(e) => handleInputChange('estimated_amount', parseFloat(e.target.value) || 0)}
                    placeholder="0.00"
                    leftIcon={<span className="text-sm">$</span>}
                    required
                  />

                  <Input
                    label="Incident Date"
                    type="date"
                    value={formData.incident_date}
                    onChange={(e) => handleInputChange('incident_date', e.target.value)}
                    required
                  />

                  <Input
                    label="Location"
                    value={formData.location}
                    onChange={(e) => handleInputChange('location', e.target.value)}
                    placeholder="Where did the incident occur?"
                    required
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-neutral-700 mb-2">
                    Describe what happened *
                  </label>
                  <textarea
                    className="w-full min-h-[120px] px-3 py-2 border border-neutral-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                    value={formData.description}
                    onChange={(e) => handleInputChange('description', e.target.value)}
                    placeholder="Provide a detailed description of the incident..."
                    required
                  />
                  
                  {/* AI Suggestions */}
                  {aiSuggestions.length > 0 && (
                    <div className="mt-3 p-3 bg-primary-50 rounded-lg border border-primary-200">
                      <div className="flex items-center mb-2">
                        <Bot className="h-4 w-4 text-primary-600 mr-2" />
                        <span className="text-sm font-medium text-primary-700">AI Suggestions</span>
                      </div>
                      <ul className="space-y-1">
                        {aiSuggestions.map((suggestion, index) => (
                          <li key={index} className="text-sm text-primary-600 flex items-center">
                            <Sparkles className="h-3 w-3 mr-2" />
                            {suggestion}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>

                <div className="flex justify-end">
                  <Button
                    type="button"
                    onClick={() => setStep(2)}
                    disabled={!formData.claim_type || !formData.description || !formData.policy_number}
                  >
                    Next: Upload Documents
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Step 2: File Upload */}
          {step === 2 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Upload className="h-5 w-5 mr-2" />
                  Upload Supporting Documents
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <Alert
                  variant="info"
                  title="Upload Tips"
                  description="Include photos of damage, receipts, police reports, and any other relevant documents. Supported formats: JPG, PNG, PDF, DOC, DOCX"
                />

                {/* File Drop Zone */}
                <div
                  {...getRootProps()}
                  className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                    isDragActive
                      ? 'border-primary-500 bg-primary-50'
                      : 'border-neutral-300 hover:border-primary-400 hover:bg-neutral-50'
                  }`}
                >
                  <input {...getInputProps()} />
                  <Upload className="h-12 w-12 text-neutral-400 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-neutral-900 mb-2">
                    {isDragActive ? 'Drop files here' : 'Upload your documents'}
                  </h3>
                  <p className="text-neutral-600">
                    Drag and drop files here, or click to select files
                  </p>
                  <p className="text-sm text-neutral-500 mt-2">
                    Maximum file size: 10MB per file
                  </p>
                </div>

                {/* Uploaded Files */}
                {files.length > 0 && (
                  <div className="space-y-3">
                    <h4 className="font-medium text-neutral-900">Uploaded Files</h4>
                    {files.map((file, index) => (
                      <div key={index} className="flex items-center justify-between p-3 bg-neutral-50 rounded-lg">
                        <div className="flex items-center">
                          <FileText className="h-5 w-5 text-neutral-500 mr-3" />
                          <div>
                            <p className="font-medium text-neutral-900">{file.name}</p>
                            <p className="text-sm text-neutral-500">
                              {(file.size / 1024 / 1024).toFixed(2)} MB
                            </p>
                          </div>
                        </div>
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          onClick={() => removeFile(index)}
                        >
                          Remove
                        </Button>
                      </div>
                    ))}
                  </div>
                )}

                <div className="flex justify-between">
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => setStep(1)}
                  >
                    Back
                  </Button>
                  <Button
                    type="button"
                    onClick={() => setStep(3)}
                  >
                    Next: Review
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Step 3: Review & Submit */}
          {step === 3 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <CheckCircle className="h-5 w-5 mr-2" />
                  Review Your Claim
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-medium text-neutral-900 mb-2">Claim Type</h4>
                    <p className="text-neutral-600 capitalize">
                      {claimTypes.find(t => t.value === formData.claim_type)?.label}
                    </p>
                  </div>
                  <div>
                    <h4 className="font-medium text-neutral-900 mb-2">Policy Number</h4>
                    <p className="text-neutral-600">{formData.policy_number}</p>
                  </div>
                  <div>
                    <h4 className="font-medium text-neutral-900 mb-2">Estimated Amount</h4>
                    <p className="text-neutral-600">${formData.estimated_amount.toLocaleString()}</p>
                  </div>
                  <div>
                    <h4 className="font-medium text-neutral-900 mb-2">Incident Date</h4>
                    <p className="text-neutral-600">{formData.incident_date}</p>
                  </div>
                  <div>
                    <h4 className="font-medium text-neutral-900 mb-2">Location</h4>
                    <p className="text-neutral-600">{formData.location}</p>
                  </div>
                  <div>
                    <h4 className="font-medium text-neutral-900 mb-2">Documents</h4>
                    <p className="text-neutral-600">{files.length} files uploaded</p>
                  </div>
                </div>

                <div>
                  <h4 className="font-medium text-neutral-900 mb-2">Description</h4>
                  <p className="text-neutral-600 bg-neutral-50 p-3 rounded-lg">
                    {formData.description}
                  </p>
                </div>

                {submitClaim.error && (
                  <Alert
                    variant="destructive"
                    title="Submission Failed"
                    description={submitClaim.error.message || 'Please try again.'}
                  />
                )}

                <div className="flex justify-between">
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => setStep(2)}
                  >
                    Back
                  </Button>
                  <Button
                    type="submit"
                    loading={submitClaim.isPending}
                    disabled={submitClaim.isPending}
                  >
                    {submitClaim.isPending ? 'Submitting...' : 'Submit Claim'}
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}
        </form>
      </div>
    </div>
  )
} 