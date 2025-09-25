'use client'
import React from 'react'
import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
// Simple UI Components (inline)
const Card = ({ children, className = '' }: { children: React.ReactNode, className?: string }) => 
  <div className={`bg-white rounded-lg border shadow-sm ${className}`}>{children}</div>
const CardContent = ({ children, className = '' }: { children: React.ReactNode, className?: string }) => 
  <div className={className}>{children}</div>
const CardHeader = ({ children }: { children: React.ReactNode }) => 
  <div className="p-6 pb-0">{children}</div>
const CardTitle = ({ children }: { children: React.ReactNode }) => 
  <h3 className="text-lg font-semibold">{children}</h3>
interface ButtonProps {
  children: React.ReactNode
  variant?: 'default' | 'outline'
  size?: 'default' | 'sm'
  onClick?: () => void
  className?: string
}

const Button = ({ children, variant = 'default', size = 'default', onClick, className = '' }: ButtonProps) => 
  <button onClick={onClick} className={`px-4 py-2 rounded-md font-medium ${variant === 'outline' ? 'border border-gray-300 bg-white text-gray-700 hover:bg-gray-50' : 'bg-blue-600 text-white hover:bg-blue-700'} ${size === 'sm' ? 'px-3 py-1 text-sm' : ''} ${className}`}>{children}</button>

interface InputProps {
  placeholder?: string
  value: string
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void
  leftIcon?: React.ReactNode
  className?: string
}

const Input = ({ placeholder, value, onChange, leftIcon, className = '' }: InputProps) => (
  <div className="relative">
    {leftIcon && <div className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400">{leftIcon}</div>}
    <input placeholder={placeholder} value={value} onChange={onChange} className={`w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent ${leftIcon ? 'pl-10' : ''} ${className}`} />
  </div>
)
import { 
  Search,
  Filter,
  FileText,
  Clock,
  CheckCircle,
  XCircle,
  AlertTriangle,
  User,
  Calendar,
  DollarSign,
  BarChart3,
  TrendingUp,
  Eye,
  MessageSquare
} from 'lucide-react'
import { format } from 'date-fns'

interface Claim {
  id: string
  claim_number: string
  customer_name: string
  claim_type: string
  status: string
  priority: 'low' | 'medium' | 'high' | 'urgent'
  estimated_amount: number
  created_at: string
  assigned_adjuster: string
  fraud_score: number
  description: string
}

const statusConfig = {
  submitted: { label: 'New', icon: Clock, variant: 'secondary' as const, color: 'bg-blue-100 text-blue-800' },
  under_review: { label: 'In Review', icon: AlertTriangle, variant: 'warning' as const, color: 'bg-yellow-100 text-yellow-800' },
  approved: { label: 'Approved', icon: CheckCircle, variant: 'success' as const, color: 'bg-green-100 text-green-800' },
  denied: { label: 'Denied', icon: XCircle, variant: 'destructive' as const, color: 'bg-red-100 text-red-800' },
}

const priorityConfig = {
  low: { label: 'Low', color: 'bg-gray-100 text-gray-800' },
  medium: { label: 'Medium', color: 'bg-blue-100 text-blue-800' },
  high: { label: 'High', color: 'bg-orange-100 text-orange-800' },
  urgent: { label: 'Urgent', color: 'bg-red-100 text-red-800' },
}

// Mock data - in real app this would come from API
const mockClaims: Claim[] = [
  {
    id: '1',
    claim_number: 'CLM-2024-001',
    customer_name: 'John Smith',
    claim_type: 'auto',
    status: 'submitted',
    priority: 'high',
    estimated_amount: 15000,
    created_at: '2024-01-15T10:30:00Z',
    assigned_adjuster: 'Sarah Johnson',
    fraud_score: 0.15,
    description: 'Vehicle collision on Highway 101'
  },
  {
    id: '2',
    claim_number: 'CLM-2024-002',
    customer_name: 'Mary Johnson',
    claim_type: 'home',
    status: 'under_review',
    priority: 'medium',
    estimated_amount: 25000,
    created_at: '2024-01-14T14:20:00Z',
    assigned_adjuster: 'Mike Chen',
    fraud_score: 0.05,
    description: 'Water damage from burst pipe'
  },
  {
    id: '3',
    claim_number: 'CLM-2024-003',
    customer_name: 'Robert Davis',
    claim_type: 'auto',
    status: 'approved',
    priority: 'low',
    estimated_amount: 8500,
    created_at: '2024-01-13T09:15:00Z',
    assigned_adjuster: 'Sarah Johnson',
    fraud_score: 0.02,
    description: 'Minor fender bender in parking lot'
  },
]

export default function AdjusterDashboard() {
  const [searchTerm, setSearchTerm] = useState('')
  const [statusFilter, setStatusFilter] = useState('all')
  const [priorityFilter, setPriorityFilter] = useState('all')

  // In real app, this would be an API call
  const claims = mockClaims.filter(claim => {
    const matchesSearch = claim.claim_number.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         claim.customer_name.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesStatus = statusFilter === 'all' || claim.status === statusFilter
    const matchesPriority = priorityFilter === 'all' || claim.priority === priorityFilter
    
    return matchesSearch && matchesStatus && matchesPriority
  })

  // Calculate statistics
  const stats = {
    total: mockClaims.length,
    pending: mockClaims.filter(c => ['submitted', 'under_review'].includes(c.status)).length,
    approved: mockClaims.filter(c => c.status === 'approved').length,
    highPriority: mockClaims.filter(c => ['high', 'urgent'].includes(c.priority)).length,
    totalValue: mockClaims.reduce((sum, c) => sum + c.estimated_amount, 0),
    avgFraudScore: mockClaims.reduce((sum, c) => sum + c.fraud_score, 0) / mockClaims.length,
  }

  return (
    <div className="min-h-screen bg-neutral-50">
      {/* Header */}
      <header className="bg-white border-b">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-neutral-900">Claims Dashboard</h1>
              <p className="text-neutral-600">Manage and review insurance claims</p>
            </div>
            <div className="flex items-center space-x-4">
              <Button variant="outline">
                <Filter className="h-4 w-4 mr-2" />
                Advanced Filters
              </Button>
              <Button>
                <BarChart3 className="h-4 w-4 mr-2" />
                Analytics
              </Button>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8">
        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-6 gap-6 mb-8">
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <FileText className="h-8 w-8 text-primary-600" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-neutral-600">Total Claims</p>
                  <p className="text-2xl font-bold text-neutral-900">{stats.total}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <Clock className="h-8 w-8 text-yellow-600" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-neutral-600">Pending</p>
                  <p className="text-2xl font-bold text-neutral-900">{stats.pending}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <CheckCircle className="h-8 w-8 text-green-600" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-neutral-600">Approved</p>
                  <p className="text-2xl font-bold text-neutral-900">{stats.approved}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <AlertTriangle className="h-8 w-8 text-red-600" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-neutral-600">High Priority</p>
                  <p className="text-2xl font-bold text-neutral-900">{stats.highPriority}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <DollarSign className="h-8 w-8 text-green-600" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-neutral-600">Total Value</p>
                  <p className="text-2xl font-bold text-neutral-900">
                    ${(stats.totalValue / 1000).toFixed(0)}K
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <TrendingUp className="h-8 w-8 text-blue-600" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-neutral-600">Avg Fraud Score</p>
                  <p className="text-2xl font-bold text-neutral-900">
                    {(stats.avgFraudScore * 100).toFixed(1)}%
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Filters and Search */}
        <Card className="mb-6">
          <CardContent className="p-6">
            <div className="flex flex-col md:flex-row gap-4">
              <div className="flex-1">
                <Input
                  placeholder="Search claims by number or customer name..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  leftIcon={<Search className="h-4 w-4" />}
                />
              </div>
              <div className="flex gap-2">
                <select
                  className="px-3 py-2 border border-neutral-300 rounded-md text-sm"
                  value={statusFilter}
                  onChange={(e) => setStatusFilter(e.target.value)}
                >
                  <option value="all">All Status</option>
                  <option value="submitted">New</option>
                  <option value="under_review">In Review</option>
                  <option value="approved">Approved</option>
                  <option value="denied">Denied</option>
                </select>
                <select
                  className="px-3 py-2 border border-neutral-300 rounded-md text-sm"
                  value={priorityFilter}
                  onChange={(e) => setPriorityFilter(e.target.value)}
                >
                  <option value="all">All Priority</option>
                  <option value="urgent">Urgent</option>
                  <option value="high">High</option>
                  <option value="medium">Medium</option>
                  <option value="low">Low</option>
                </select>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Claims Table */}
        <Card>
          <CardHeader>
            <CardTitle>Claims Overview</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-neutral-200">
                    <th className="text-left py-3 px-4 font-medium text-neutral-900">Claim</th>
                    <th className="text-left py-3 px-4 font-medium text-neutral-900">Customer</th>
                    <th className="text-left py-3 px-4 font-medium text-neutral-900">Type</th>
                    <th className="text-left py-3 px-4 font-medium text-neutral-900">Status</th>
                    <th className="text-left py-3 px-4 font-medium text-neutral-900">Priority</th>
                    <th className="text-left py-3 px-4 font-medium text-neutral-900">Amount</th>
                    <th className="text-left py-3 px-4 font-medium text-neutral-900">Fraud Risk</th>
                    <th className="text-left py-3 px-4 font-medium text-neutral-900">Date</th>
                    <th className="text-left py-3 px-4 font-medium text-neutral-900">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {claims.map((claim) => {
                    const statusConf = statusConfig[claim.status as keyof typeof statusConfig]
                    const priorityConf = priorityConfig[claim.priority]
                    const StatusIcon = statusConf.icon
                    
                    return (
                      <tr key={claim.id} className="border-b border-neutral-100 hover:bg-neutral-50">
                        <td className="py-4 px-4">
                          <div>
                            <p className="font-medium text-neutral-900">{claim.claim_number}</p>
                            <p className="text-sm text-neutral-500 truncate max-w-[200px]">
                              {claim.description}
                            </p>
                          </div>
                        </td>
                        <td className="py-4 px-4">
                          <div className="flex items-center">
                            <User className="h-4 w-4 text-neutral-400 mr-2" />
                            <span className="text-neutral-900">{claim.customer_name}</span>
                          </div>
                        </td>
                        <td className="py-4 px-4">
                          <span className="capitalize text-neutral-600">
                            {claim.claim_type}
                          </span>
                        </td>
                        <td className="py-4 px-4">
                          <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${statusConf.color}`}>
                            <StatusIcon className="h-3 w-3 mr-1" />
                            {statusConf.label}
                          </div>
                        </td>
                        <td className="py-4 px-4">
                          <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${priorityConf.color}`}>
                            {priorityConf.label}
                          </div>
                        </td>
                        <td className="py-4 px-4">
                          <span className="font-medium text-neutral-900">
                            ${claim.estimated_amount.toLocaleString()}
                          </span>
                        </td>
                        <td className="py-4 px-4">
                          <div className="flex items-center">
                            <div className={`w-2 h-2 rounded-full mr-2 ${
                              claim.fraud_score > 0.3 ? 'bg-red-500' :
                              claim.fraud_score > 0.1 ? 'bg-yellow-500' : 'bg-green-500'
                            }`} />
                            <span className={`text-sm font-medium ${
                              claim.fraud_score > 0.3 ? 'text-red-600' :
                              claim.fraud_score > 0.1 ? 'text-yellow-600' : 'text-green-600'
                            }`}>
                              {(claim.fraud_score * 100).toFixed(1)}%
                            </span>
                          </div>
                        </td>
                        <td className="py-4 px-4">
                          <div className="flex items-center text-sm text-neutral-500">
                            <Calendar className="h-4 w-4 mr-1" />
                            {format(new Date(claim.created_at), 'MMM d')}
                          </div>
                        </td>
                        <td className="py-4 px-4">
                          <div className="flex items-center space-x-2">
                            <Button size="sm" variant="outline">
                              <Eye className="h-4 w-4" />
                            </Button>
                            <Button size="sm" variant="outline">
                              <MessageSquare className="h-4 w-4" />
                            </Button>
                          </div>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
} 