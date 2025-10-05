'use client';

import React, { useState } from 'react';
import { InteractiveCard, CardContent, CardHeader, CardTitle } from '../ui/interactive-card';
import { ModernButton } from '../ui/modern-button';
import { 
  FileText, 
  Clock, 
  ShieldAlert, 
  Star, 
  TrendingUp, 
  Bell, 
  Filter, 
  Search,
  Eye,
  MessageSquare,
  AlertTriangle,
  CheckCircle,
  XCircle,
  DollarSign,
  Calendar,
  User,
  Brain
} from 'lucide-react';

interface Claim {
  id: string;
  claim_number: string;
  customer_name: string;
  description: string;
  status: 'submitted' | 'under_review' | 'approved' | 'denied';
  priority: 'low' | 'medium' | 'high' | 'urgent';
  estimated_amount: number;
  fraud_score: number;
  created_at: string;
  ai_insights?: string;
}

interface KPICardProps {
  title: string;
  value: string;
  change: string;
  trend: 'up' | 'down';
  icon: React.ReactNode;
  color: 'blue' | 'green' | 'red' | 'yellow';
}

const KPICard: React.FC<KPICardProps> = ({ title, value, change, trend, icon, color }) => {
  const colorClasses = {
    blue: 'from-blue-500 to-blue-600',
    green: 'from-green-500 to-emerald-600',
    red: 'from-red-500 to-pink-600',
    yellow: 'from-yellow-500 to-orange-500'
  };

  const trendColor = trend === 'up' ? 'text-green-600' : 'text-red-600';

  return (
    <InteractiveCard className="group">
      <CardContent className="p-6">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-gray-600 mb-1">{title}</p>
            <p className="text-2xl font-bold text-gray-900">{value}</p>
            <div className="flex items-center mt-2">
              <TrendingUp className={`h-3 w-3 mr-1 ${trendColor}`} />
              <span className={`text-xs font-medium ${trendColor}`}>{change}</span>
              <span className="text-xs text-gray-500 ml-1">vs last month</span>
            </div>
          </div>
          <div className={`w-12 h-12 bg-gradient-to-br ${colorClasses[color]} rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform duration-200`}>
            <div className="text-white">
              {icon}
            </div>
          </div>
        </div>
      </CardContent>
    </InteractiveCard>
  );
};

const FraudRiskIndicator: React.FC<{ score: number }> = ({ score }) => {
  const getRiskLevel = (score: number) => {
    if (score >= 80) return { level: 'High', color: 'bg-red-500', textColor: 'text-red-600' };
    if (score >= 60) return { level: 'Medium', color: 'bg-yellow-500', textColor: 'text-yellow-600' };
    return { level: 'Low', color: 'bg-green-500', textColor: 'text-green-600' };
  };

  const risk = getRiskLevel(score);

  return (
    <div className="text-right">
      <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${risk.textColor} bg-opacity-10`} style={{ backgroundColor: `${risk.color.replace('bg-', '')}20` }}>
        <div className={`w-2 h-2 rounded-full mr-1 ${risk.color}`}></div>
        {risk.level} Risk
      </div>
      <div className="text-sm text-gray-500 mt-1">{score}% confidence</div>
    </div>
  );
};

const PriorityBadge: React.FC<{ priority: string }> = ({ priority }) => {
  const priorityConfig = {
    low: 'bg-gray-100 text-gray-800',
    medium: 'bg-blue-100 text-blue-800',
    high: 'bg-orange-100 text-orange-800',
    urgent: 'bg-red-100 text-red-800'
  };

  return (
    <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${priorityConfig[priority as keyof typeof priorityConfig]}`}>
      {priority.toUpperCase()}
    </span>
  );
};

const StatusBadge: React.FC<{ status: string }> = ({ status }) => {
  const statusConfig = {
    submitted: { color: 'bg-blue-100 text-blue-800', icon: Clock },
    under_review: { color: 'bg-yellow-100 text-yellow-800', icon: AlertTriangle },
    approved: { color: 'bg-green-100 text-green-800', icon: CheckCircle },
    denied: { color: 'bg-red-100 text-red-800', icon: XCircle }
  };

  const config = statusConfig[status as keyof typeof statusConfig] || statusConfig.submitted;
  const StatusIcon = config.icon;

  return (
    <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${config.color}`}>
      <StatusIcon className="w-3 h-3 mr-1" />
      {status.replace('_', ' ').toUpperCase()}
    </span>
  );
};

export const AdjusterCommandCenter: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');

  // Mock data
  const stats = {
    total: 156,
    avgResolutionTime: '18.5 hrs',
    fraudDetectionRate: '85.6%',
    customerSatisfaction: '97.2%'
  };

  const claims: Claim[] = [
    {
      id: '1',
      claim_number: 'CLM-2024-001',
      customer_name: 'John Smith',
      description: 'Vehicle collision on Highway 101',
      status: 'under_review',
      priority: 'high',
      estimated_amount: 15000,
      fraud_score: 25,
      created_at: '2024-01-15T10:00:00Z',
      ai_insights: 'Photos show consistent damage pattern. Customer history clean.'
    },
    {
      id: '2',
      claim_number: 'CLM-2024-002',
      customer_name: 'Sarah Johnson',
      description: 'Water damage from burst pipe',
      status: 'submitted',
      priority: 'urgent',
      estimated_amount: 8500,
      fraud_score: 85,
      created_at: '2024-01-14T14:30:00Z',
      ai_insights: 'Multiple similar claims in area. Requires additional verification.'
    },
    {
      id: '3',
      claim_number: 'CLM-2024-003',
      customer_name: 'Mike Davis',
      description: 'Theft of personal belongings',
      status: 'approved',
      priority: 'medium',
      estimated_amount: 3200,
      fraud_score: 15,
      created_at: '2024-01-13T09:15:00Z'
    }
  ];

  const recentActions = [
    { id: '1', action: 'Approved claim CLM-2024-003', time: '5 minutes ago', type: 'approval' },
    { id: '2', action: 'Requested additional docs for CLM-2024-002', time: '1 hour ago', type: 'request' },
    { id: '3', action: 'Flagged CLM-2024-004 for fraud review', time: '2 hours ago', type: 'flag' }
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navigation Header */}
      <header className="bg-white shadow-sm border-b backdrop-blur-sm bg-white/95 sticky top-0 z-40">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Claims Command Center</h1>
              <p className="text-gray-600">Intelligent claim processing and fraud detection</p>
            </div>
            <div className="flex items-center space-x-4">
              <button className="relative p-2 text-gray-400 hover:text-gray-600 transition-colors">
                <Bell className="h-6 w-6" />
                <span className="absolute top-0 right-0 block h-2 w-2 rounded-full bg-red-400"></span>
              </button>
              <div className="flex items-center space-x-2">
                <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                  <User className="h-4 w-4 text-white" />
                </div>
                <span className="text-sm font-medium text-gray-700">Sarah Wilson</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="p-6">
        {/* KPI Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <KPICard
            title="Active Claims"
            value={stats.total.toString()}
            change="+12%"
            trend="up"
            icon={<FileText className="h-6 w-6" />}
            color="blue"
          />
          <KPICard
            title="Avg Resolution Time"
            value={stats.avgResolutionTime}
            change="-8%"
            trend="down"
            icon={<Clock className="h-6 w-6" />}
            color="green"
          />
          <KPICard
            title="Fraud Detection Rate"
            value={stats.fraudDetectionRate}
            change="+2.1%"
            trend="up"
            icon={<ShieldAlert className="h-6 w-6" />}
            color="red"
          />
          <KPICard
            title="Customer Satisfaction"
            value={stats.customerSatisfaction}
            change="+0.8%"
            trend="up"
            icon={<Star className="h-6 w-6" />}
            color="yellow"
          />
        </div>

        {/* Advanced Filters & Search */}
        <InteractiveCard className="mb-6">
          <CardContent className="p-6">
            <div className="flex flex-wrap items-center gap-4">
              <div className="flex-1 min-w-80">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-5 w-5" />
                  <input
                    type="text"
                    placeholder="Search by claim number, customer name, or description..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="w-full pl-10 pr-4 py-3 border-2 border-gray-200 rounded-lg focus:border-blue-500 focus:ring-4 focus:ring-blue-500/20 transition-all duration-200"
                  />
                </div>
              </div>
              <select 
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="px-4 py-3 border-2 border-gray-200 rounded-lg focus:border-blue-500 focus:ring-4 focus:ring-blue-500/20 transition-all duration-200"
              >
                <option value="all">All Status</option>
                <option value="submitted">Submitted</option>
                <option value="under_review">Under Review</option>
                <option value="approved">Approved</option>
                <option value="denied">Denied</option>
              </select>
              <ModernButton variant="secondary" icon={<Filter className="h-4 w-4" />}>
                More Filters
              </ModernButton>
            </div>
          </CardContent>
        </InteractiveCard>

        {/* Claims Grid with Analytics */}
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          {/* Claims List - 2/3 width */}
          <div className="xl:col-span-2">
            <InteractiveCard>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="flex items-center">
                    <FileText className="mr-2 h-5 w-5 text-blue-600" />
                    Claims Queue
                  </CardTitle>
                  <div className="flex items-center space-x-2">
                    <span className="text-sm text-gray-500">{claims.length} claims</span>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="p-0">
                <div className="space-y-4 p-6">
                  {claims.map((claim) => (
                    <InteractiveCard key={claim.id} className="group hover:shadow-lg transition-all duration-200 relative border-l-4 border-l-blue-500">
                      <CardContent className="p-6">
                        <div className="flex items-start justify-between mb-4">
                          <div className="flex-1">
                            <div className="flex items-center space-x-3 mb-2">
                              <h3 className="font-semibold text-lg text-gray-900 group-hover:text-blue-600 transition-colors">
                                {claim.claim_number}
                              </h3>
                              <StatusBadge status={claim.status} />
                              <PriorityBadge priority={claim.priority} />
                            </div>
                            <p className="text-gray-600 mb-2">{claim.description}</p>
                            <div className="flex items-center text-sm text-gray-500 space-x-4">
                              <span className="flex items-center">
                                <User className="mr-1 h-4 w-4" />
                                {claim.customer_name}
                              </span>
                              <span className="flex items-center">
                                <Calendar className="mr-1 h-4 w-4" />
                                {new Date(claim.created_at).toLocaleDateString()}
                              </span>
                              <span className="flex items-center">
                                <DollarSign className="mr-1 h-4 w-4" />
                                ${claim.estimated_amount.toLocaleString()}
                              </span>
                            </div>
                          </div>
                          
                          {/* Fraud Risk Indicator */}
                          <FraudRiskIndicator score={claim.fraud_score} />
                        </div>

                        {/* AI Insights */}
                        {claim.ai_insights && (
                          <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                            <div className="flex items-start">
                              <Brain className="h-4 w-4 text-blue-600 mr-2 mt-0.5 flex-shrink-0" />
                              <div>
                                <div className="text-sm font-medium text-blue-800 mb-1">AI Insight</div>
                                <div className="text-sm text-blue-700">{claim.ai_insights}</div>
                              </div>
                            </div>
                          </div>
                        )}

                        {/* Action Buttons */}
                        <div className="flex items-center justify-between pt-4 border-t border-gray-100">
                          <div className="flex space-x-2">
                            <ModernButton size="sm" variant="secondary" icon={<Eye className="h-4 w-4" />}>
                              Review
                            </ModernButton>
                            <ModernButton size="sm" variant="secondary" icon={<MessageSquare className="h-4 w-4" />}>
                              Contact
                            </ModernButton>
                          </div>
                          <div className="flex space-x-2">
                            <ModernButton size="sm" variant="danger">
                              Deny
                            </ModernButton>
                            <ModernButton size="sm" variant="success">
                              Approve
                            </ModernButton>
                          </div>
                        </div>
                      </CardContent>
                    </InteractiveCard>
                  ))}
                </div>
              </CardContent>
            </InteractiveCard>
          </div>

          {/* Sidebar Analytics - 1/3 width */}
          <div className="space-y-6">
            {/* Fraud Risk Distribution */}
            <InteractiveCard>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <AlertTriangle className="mr-2 h-5 w-5 text-red-500" />
                  Fraud Risk Analysis
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="w-3 h-3 rounded-full bg-red-500" />
                      <span className="font-medium">High Risk</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div className="w-24 bg-gray-200 rounded-full h-2">
                        <div className="h-2 rounded-full bg-red-500" style={{ width: '15%' }} />
                      </div>
                      <span className="text-sm font-medium w-8">3</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="w-3 h-3 rounded-full bg-yellow-500" />
                      <span className="font-medium">Medium Risk</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div className="w-24 bg-gray-200 rounded-full h-2">
                        <div className="h-2 rounded-full bg-yellow-500" style={{ width: '25%' }} />
                      </div>
                      <span className="text-sm font-medium w-8">8</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="w-3 h-3 rounded-full bg-green-500" />
                      <span className="font-medium">Low Risk</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div className="w-24 bg-gray-200 rounded-full h-2">
                        <div className="h-2 rounded-full bg-green-500" style={{ width: '60%' }} />
                      </div>
                      <span className="text-sm font-medium w-8">145</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </InteractiveCard>

            {/* Recent Actions */}
            <InteractiveCard>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Clock className="mr-2 h-5 w-5 text-blue-600" />
                  Recent Actions
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {recentActions.map((action) => (
                    <div key={action.id} className="flex items-start space-x-3 p-3 hover:bg-gray-50 rounded-lg transition-colors">
                      <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm text-gray-900 font-medium">{action.action}</p>
                        <p className="text-xs text-gray-500 mt-1">{action.time}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </InteractiveCard>
          </div>
        </div>
      </main>
    </div>
  );
};
