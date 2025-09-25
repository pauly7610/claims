'use client'
import React from 'react'
import { useState } from 'react'
import Link from 'next/link'
import { 
  Plus, 
  FileText, 
  Clock, 
  CheckCircle, 
  XCircle, 
  AlertCircle,
  DollarSign,
  Calendar,
  TrendingUp
} from 'lucide-react'

interface Claim {
  id: string
  claim_number: string
  claim_type: string
  status: string
  estimated_amount: number
  created_at: string
  description: string
  last_updated: string
}

// Simple UI Components
const Card = ({ children, className = '' }: { children: React.ReactNode, className?: string }) => 
  <div className={`bg-white rounded-lg border shadow-sm ${className}`}>{children}</div>
const CardContent = ({ children, className = '' }: { children: React.ReactNode, className?: string }) => 
  <div className={className}>{children}</div>
const CardHeader = ({ children }: { children: React.ReactNode }) => 
  <div className="p-6 pb-0">{children}</div>
const CardTitle = ({ children }: { children: React.ReactNode }) => 
  <h3 className="text-lg font-semibold">{children}</h3>
const Button = ({ children, variant = 'default', size = 'default', onClick, className = '' }: any) => 
  <button onClick={onClick} className={`px-4 py-2 rounded-md font-medium ${variant === 'outline' ? 'border border-gray-300 bg-white text-gray-700 hover:bg-gray-50' : 'bg-blue-600 text-white hover:bg-blue-700'} ${size === 'sm' ? 'px-3 py-1 text-sm' : ''} ${className}`}>{children}</button>
const Badge = ({ children, variant = 'default' }: any) => 
  <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${variant === 'success' ? 'bg-green-100 text-green-800' : variant === 'warning' ? 'bg-yellow-100 text-yellow-800' : variant === 'destructive' ? 'bg-red-100 text-red-800' : 'bg-gray-100 text-gray-800'}`}>{children}</span>
const Alert = ({ title, description, variant = 'default', className = '' }: any) => 
  <div className={`p-4 rounded-md ${variant === 'destructive' ? 'bg-red-50 border border-red-200' : 'bg-blue-50 border border-blue-200'} ${className}`}>
    <h4 className="font-medium">{title}</h4>
    <p className="text-sm mt-1">{description}</p>
  </div>

const statusConfig = {
  submitted: { label: 'Submitted', icon: Clock, variant: 'secondary' as const, color: 'text-blue-600' },
  under_review: { label: 'Under Review', icon: AlertCircle, variant: 'warning' as const, color: 'text-yellow-600' },
  approved: { label: 'Approved', icon: CheckCircle, variant: 'success' as const, color: 'text-green-600' },
  denied: { label: 'Denied', icon: XCircle, variant: 'destructive' as const, color: 'text-red-600' },
  paid: { label: 'Paid', icon: DollarSign, variant: 'success' as const, color: 'text-green-600' },
}

const formatDate = (dateString: string) => {
  const date = new Date(dateString)
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
}

const formatDateShort = (dateString: string) => {
  const date = new Date(dateString)
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

export default function DashboardPage() {
  const user = { first_name: 'John' } // Mock user
  const [selectedStatus, setSelectedStatus] = useState<string>('all')

  // Mock data
  const claims: Claim[] = [
    {
      id: '1',
      claim_number: 'CLM-2024-001',
      claim_type: 'auto',
      status: 'submitted',
      estimated_amount: 5000,
      created_at: '2024-01-15T10:00:00Z',
      description: 'Vehicle collision on Highway 101',
      last_updated: '2024-01-15T10:00:00Z'
    },
    {
      id: '2',
      claim_number: 'CLM-2024-002',
      claim_type: 'home',
      status: 'approved',
      estimated_amount: 12000,
      created_at: '2024-01-10T14:30:00Z',
      description: 'Water damage from burst pipe',
      last_updated: '2024-01-20T09:15:00Z'
    }
  ]
  const isLoading = false
  const error = null

  const filteredClaims = claims.filter((claim: Claim) => 
    selectedStatus === 'all' || claim.status === selectedStatus
  )

  // Calculate statistics
  const stats = {
    total: claims.length,
    pending: claims.filter((c: Claim) => ['submitted', 'under_review'].includes(c.status)).length,
    approved: claims.filter((c: Claim) => c.status === 'approved').length,
    totalValue: claims.reduce((sum: number, c: Claim) => sum + c.estimated_amount, 0),
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading your claims...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                Welcome back, {user?.first_name}!
              </h1>
              <p className="text-gray-600">Manage your insurance claims</p>
            </div>
            <Link href="/dashboard/claims/new">
              <Button>
                <Plus className="h-4 w-4 mr-2" />
                New Claim
              </Button>
            </Link>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8">
        {error && (
          <Alert 
            variant="destructive" 
            title="Error Loading Claims" 
            description="We couldn't load your claims. Please try again."
            className="mb-6"
          />
        )}

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <FileText className="h-8 w-8 text-blue-600" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">Total Claims</p>
                  <p className="text-2xl font-bold text-gray-900">{stats.total}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <Clock className="h-8 w-8 text-yellow-600" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">Pending</p>
                  <p className="text-2xl font-bold text-gray-900">{stats.pending}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <CheckCircle className="h-8 w-8 text-green-600" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">Approved</p>
                  <p className="text-2xl font-bold text-gray-900">{stats.approved}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <DollarSign className="h-8 w-8 text-green-600" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">Total Value</p>
                  <p className="text-2xl font-bold text-gray-900">
                    ${stats.totalValue.toLocaleString()}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Claims Section */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Your Claims</CardTitle>
              <div className="flex space-x-2">
                {['all', 'submitted', 'under_review', 'approved', 'denied'].map((status) => (
                  <Button
                    key={status}
                    variant={selectedStatus === status ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setSelectedStatus(status)}
                  >
                    {status === 'all' ? 'All' : status.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </Button>
                ))}
              </div>
            </div>
          </CardHeader>
          <CardContent>
            {filteredClaims.length === 0 ? (
              <div className="text-center py-12">
                <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No claims found</h3>
                <p className="text-gray-600 mb-4">
                  {selectedStatus === 'all' 
                    ? "You haven't submitted any claims yet." 
                    : `No claims with status "${selectedStatus.replace('_', ' ')}" found.`}
                </p>
                <Link href="/dashboard/claims/new">
                  <Button>
                    <Plus className="h-4 w-4 mr-2" />
                    Submit Your First Claim
                  </Button>
                </Link>
              </div>
            ) : (
              <div className="space-y-4 p-6">
                {filteredClaims.map((claim: Claim) => {
                  const config = statusConfig[claim.status as keyof typeof statusConfig] || statusConfig.submitted
                  const StatusIcon = config.icon
                  
                  return (
                    <div
                      key={claim.id}
                      className="border rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer"
                    >
                      <Link href={`/dashboard/claims/${claim.id}`}>
                        <div className="flex items-center justify-between">
                          <div className="flex-1">
                            <div className="flex items-center space-x-3 mb-2">
                              <h3 className="font-semibold text-gray-900">
                                {claim.claim_number}
                              </h3>
                              <Badge variant={config.variant}>
                                <StatusIcon className="h-3 w-3 mr-1" />
                                {config.label}
                              </Badge>
                              <span className="text-sm text-gray-500 capitalize">
                                {claim.claim_type.replace('_', ' ')}
                              </span>
                            </div>
                            <p className="text-gray-600 text-sm mb-2">{claim.description}</p>
                            <div className="flex items-center space-x-4 text-sm text-gray-500">
                              <span className="flex items-center">
                                <DollarSign className="h-4 w-4 mr-1" />
                                ${claim.estimated_amount.toLocaleString()}
                              </span>
                              <span className="flex items-center">
                                <Calendar className="h-4 w-4 mr-1" />
                                {formatDate(claim.created_at)}
                              </span>
                              <span className="flex items-center">
                                <TrendingUp className="h-4 w-4 mr-1" />
                                Updated {formatDateShort(claim.last_updated)}
                              </span>
                            </div>
                          </div>
                          <div className={`text-right ${config.color}`}>
                            <StatusIcon className="h-6 w-6 ml-4" />
                          </div>
                        </div>
                      </Link>
                    </div>
                  )
                })}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Quick Actions */}
        {claims && claims.length > 0 && (
          <div className="mt-8">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Link href="/dashboard/claims/new">
                <Card className="hover:shadow-md transition-shadow cursor-pointer">
                  <CardContent className="p-6 text-center">
                    <Plus className="h-8 w-8 text-blue-600 mx-auto mb-2" />
                    <h3 className="font-medium text-gray-900">Submit New Claim</h3>
                    <p className="text-sm text-gray-600">Start a new insurance claim</p>
                  </CardContent>
                </Card>
              </Link>

              <Link href="/dashboard/documents">
                <Card className="hover:shadow-md transition-shadow cursor-pointer">
                  <CardContent className="p-6 text-center">
                    <FileText className="h-8 w-8 text-green-600 mx-auto mb-2" />
                    <h3 className="font-medium text-gray-900">Upload Documents</h3>
                    <p className="text-sm text-gray-600">Add supporting documents</p>
                  </CardContent>
                </Card>
              </Link>

              <Link href="/dashboard/profile">
                <Card className="hover:shadow-md transition-shadow cursor-pointer">
                  <CardContent className="p-6 text-center">
                    <AlertCircle className="h-8 w-8 text-yellow-600 mx-auto mb-2" />
                    <h3 className="font-medium text-gray-900">Update Profile</h3>
                    <p className="text-sm text-gray-600">Manage your account settings</p>
                  </CardContent>
                </Card>
              </Link>
            </div>
          </div>
        )}
      </div>
    </div>
  )
} 