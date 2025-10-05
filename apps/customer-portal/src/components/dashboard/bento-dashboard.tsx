'use client';

import React from 'react';
import { InteractiveCard, CardContent, CardHeader, CardTitle } from '../ui/interactive-card';
import { ModernButton } from '../ui/modern-button';
import { 
  TrendingUp, 
  Clock, 
  FileText, 
  MessageSquare, 
  User, 
  Upload,
  Plus,
  Bell,
  Calendar,
  DollarSign
} from 'lucide-react';

interface DashboardProps {
  user: {
    first_name: string;
  };
  stats: {
    total: number;
    pending: number;
    approved: number;
    totalValue: string;
  };
  recentActivities: Array<{
    id: string;
    type: string;
    description: string;
    timestamp: string;
    status: string;
  }>;
}

const QuickActionButton = ({ 
  icon, 
  label, 
  onClick 
}: { 
  icon: React.ReactNode; 
  label: string; 
  onClick: () => void;
}) => (
  <button
    onClick={onClick}
    className="flex items-center w-full p-3 text-left hover:bg-gray-50 rounded-lg transition-colors group"
  >
    <div className="flex items-center justify-center w-10 h-10 bg-blue-50 rounded-lg mr-3 group-hover:bg-blue-100 transition-colors">
      {icon}
    </div>
    <span className="font-medium text-gray-700 group-hover:text-gray-900">{label}</span>
  </button>
);

const ActivityItem = ({ 
  activity 
}: { 
  activity: DashboardProps['recentActivities'][0] 
}) => {
  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'approved': return 'text-green-600 bg-green-50';
      case 'pending': return 'text-yellow-600 bg-yellow-50';
      case 'rejected': return 'text-red-600 bg-red-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  return (
    <div className="flex items-start space-x-3 p-3 hover:bg-gray-50 rounded-lg transition-colors">
      <div className="flex-shrink-0 w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
      <div className="flex-1 min-w-0">
        <p className="text-sm text-gray-900 font-medium">{activity.description}</p>
        <div className="flex items-center mt-1 space-x-2">
          <span className="text-xs text-gray-500">{activity.timestamp}</span>
          <span className={`text-xs px-2 py-1 rounded-full font-medium ${getStatusColor(activity.status)}`}>
            {activity.status}
          </span>
        </div>
      </div>
    </div>
  );
};

export const BentoDashboard: React.FC<DashboardProps> = ({
  user,
  stats,
  recentActivities
}) => {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-40 backdrop-blur-sm bg-white/95">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                Good morning, {user.first_name} 👋
              </h1>
              <p className="text-gray-600">Here's what's happening with your claims</p>
            </div>
            <ModernButton 
              variant="primary"
              icon={<Plus className="w-4 h-4" />}
            >
              New Claim
            </ModernButton>
          </div>
        </div>
      </header>

      {/* Bento Grid Layout */}
      <main className="p-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          
          {/* Quick Stats - Spans 2 columns */}
          <InteractiveCard className="md:col-span-2 bg-gradient-to-br from-blue-500 to-purple-600 text-white border-0 hover:shadow-2xl">
            <CardContent className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold">Claims Overview</h3>
                <TrendingUp className="h-5 w-5 opacity-80" />
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <div className="text-3xl font-bold">{stats.total}</div>
                  <div className="text-sm opacity-80">Total Claims</div>
                </div>
                <div>
                  <div className="text-3xl font-bold text-yellow-200">{stats.pending}</div>
                  <div className="text-sm opacity-80">In Progress</div>
                </div>
                <div>
                  <div className="text-3xl font-bold text-green-200">{stats.approved}</div>
                  <div className="text-sm opacity-80">Completed</div>
                </div>
                <div>
                  <div className="text-3xl font-bold text-blue-200">{stats.totalValue}</div>
                  <div className="text-sm opacity-80">Total Value</div>
                </div>
              </div>
            </CardContent>
          </InteractiveCard>

          {/* Recent Activity */}
          <InteractiveCard className="lg:col-span-2">
            <CardHeader>
              <CardTitle className="flex items-center">
                <Clock className="mr-2 h-5 w-5 text-blue-600" />
                Recent Activity
              </CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              <div className="max-h-80 overflow-y-auto">
                {recentActivities.map((activity) => (
                  <ActivityItem key={activity.id} activity={activity} />
                ))}
              </div>
            </CardContent>
          </InteractiveCard>

          {/* Quick Actions */}
          <InteractiveCard>
            <CardHeader>
              <CardTitle className="flex items-center">
                <FileText className="mr-2 h-5 w-5 text-purple-600" />
                Quick Actions
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <QuickActionButton
                icon={<Upload className="w-5 h-5 text-blue-600" />}
                label="Upload Documents"
                onClick={() => {/* TODO: Implement document upload */}}
              />
              <QuickActionButton
                icon={<MessageSquare className="w-5 h-5 text-green-600" />}
                label="Contact Support"
                onClick={() => {/* TODO: Implement support contact */}}
              />
              <QuickActionButton
                icon={<User className="w-5 h-5 text-purple-600" />}
                label="Update Profile"
                onClick={() => {/* TODO: Implement profile update */}}
              />
              <QuickActionButton
                icon={<Calendar className="w-5 h-5 text-orange-600" />}
                label="Schedule Call"
                onClick={() => {/* TODO: Implement call scheduling */}}
              />
            </CardContent>
          </InteractiveCard>

          {/* Notifications */}
          <InteractiveCard className="md:col-span-2 lg:col-span-1">
            <CardHeader>
              <CardTitle className="flex items-center">
                <Bell className="mr-2 h-5 w-5 text-yellow-600" />
                Notifications
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex items-start space-x-3 p-3 bg-blue-50 rounded-lg">
                  <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                  <div>
                    <p className="text-sm font-medium text-gray-900">Claim #12345 approved</p>
                    <p className="text-xs text-gray-500">2 hours ago</p>
                  </div>
                </div>
                <div className="flex items-start space-x-3 p-3 bg-yellow-50 rounded-lg">
                  <div className="w-2 h-2 bg-yellow-500 rounded-full mt-2 flex-shrink-0"></div>
                  <div>
                    <p className="text-sm font-medium text-gray-900">Document required for claim #12346</p>
                    <p className="text-xs text-gray-500">1 day ago</p>
                  </div>
                </div>
                <div className="flex items-start space-x-3 p-3 bg-green-50 rounded-lg">
                  <div className="w-2 h-2 bg-green-500 rounded-full mt-2 flex-shrink-0"></div>
                  <div>
                    <p className="text-sm font-medium text-gray-900">Payment processed for claim #12344</p>
                    <p className="text-xs text-gray-500">3 days ago</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </InteractiveCard>

          {/* Performance Metrics */}
          <InteractiveCard className="lg:col-span-2">
            <CardHeader>
              <CardTitle className="flex items-center">
                <DollarSign className="mr-2 h-5 w-5 text-green-600" />
                Your Claims Performance
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center p-4 bg-green-50 rounded-lg">
                  <div className="text-2xl font-bold text-green-600">18.5hrs</div>
                  <div className="text-sm text-gray-600">Avg Processing Time</div>
                </div>
                <div className="text-center p-4 bg-blue-50 rounded-lg">
                  <div className="text-2xl font-bold text-blue-600">97.2%</div>
                  <div className="text-sm text-gray-600">Approval Rate</div>
                </div>
              </div>
            </CardContent>
          </InteractiveCard>
        </div>
      </main>
    </div>
  );
};
