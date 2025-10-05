import React from 'react';
import { InteractiveCard, CardContent, CardHeader, CardTitle } from '../components/ui/interactive-card';
import { ModernButton } from '../components/ui/modern-button';
import { 
  DollarSign, 
  TrendingUp, 
  Shield, 
  Star, 
  Download,
  Server,
  Users,
  AlertTriangle,
  Plus,
  Settings,
  BarChart3,
  Activity,
  Clock,
  CheckCircle,
  User,
  Database,
  Zap,
  Globe
} from 'lucide-react';

interface ExecutiveKPICardProps {
  title: string;
  value: string;
  change: string;
  period: string;
  trend: 'up' | 'down';
  icon: React.ReactNode;
  color: 'green' | 'blue' | 'purple' | 'yellow';
}

const ExecutiveKPICard: React.FC<ExecutiveKPICardProps> = ({
  title,
  value,
  change,
  period,
  trend,
  icon,
  color
}) => {
  const colorClasses = {
    green: 'from-green-500 to-emerald-600',
    blue: 'from-blue-500 to-blue-600',
    purple: 'from-purple-500 to-pink-600',
    yellow: 'from-yellow-500 to-orange-500'
  };

  const trendColor = trend === 'up' ? 'text-green-600' : 'text-red-600';

  return (
    <InteractiveCard className="group">
      <CardContent className="p-6">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-gray-600 mb-1">{title}</p>
            <p className="text-3xl font-bold text-gray-900">{value}</p>
            <div className="flex items-center mt-2">
              <TrendingUp className={`h-3 w-3 mr-1 ${trendColor}`} />
              <span className={`text-sm font-medium ${trendColor}`}>{change}</span>
              <span className="text-sm text-gray-500 ml-1">{period}</span>
            </div>
          </div>
          <div className={`w-16 h-16 bg-gradient-to-br ${colorClasses[color]} rounded-2xl flex items-center justify-center group-hover:scale-110 transition-transform duration-200 shadow-lg`}>
            <div className="text-white">
              {icon}
            </div>
          </div>
        </div>
      </CardContent>
    </InteractiveCard>
  );
};

const StatusIndicator: React.FC<{ status: 'healthy' | 'warning' | 'error' }> = ({ status }) => {
  const statusConfig = {
    healthy: { color: 'bg-green-500', text: 'Healthy' },
    warning: { color: 'bg-yellow-500', text: 'Warning' },
    error: { color: 'bg-red-500', text: 'Error' }
  };

  const config = statusConfig[status];

  return (
    <div className="flex items-center">
      <div className={`w-2 h-2 rounded-full ${config.color} mr-2`}></div>
      <span className="text-sm font-medium text-gray-700">{config.text}</span>
    </div>
  );
};

const SystemHealthIndicators: React.FC = () => {
  const services = [
    { name: 'API Gateway', status: 'healthy' as const, uptime: '99.97%', responseTime: '245ms' },
    { name: 'Claims Service', status: 'healthy' as const, uptime: '99.95%', responseTime: '180ms' },
    { name: 'AI Service', status: 'warning' as const, uptime: '99.89%', responseTime: '320ms' },
    { name: 'Auth Service', status: 'healthy' as const, uptime: '99.99%', responseTime: '95ms' },
  ];

  return (
    <div className="space-y-4">
      {services.map((service) => (
        <div key={service.name} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
          <div className="flex items-center space-x-3">
            <StatusIndicator status={service.status} />
            <div>
              <div className="font-medium text-gray-900">{service.name}</div>
              <div className="text-sm text-gray-500">Uptime: {service.uptime}</div>
            </div>
          </div>
          <div className="text-right text-sm">
            <div className="font-medium text-gray-900">{service.responseTime}</div>
            <div className="text-gray-500">avg response</div>
          </div>
        </div>
      ))}
    </div>
  );
};

const UserActivityChart: React.FC = () => {
  const activities = [
    { period: 'Last 24h', users: 1247, color: 'bg-blue-500' },
    { period: 'Last 7d', users: 8934, color: 'bg-green-500' },
    { period: 'Last 30d', users: 34567, color: 'bg-purple-500' }
  ];

  return (
    <div className="space-y-4">
      {activities.map((activity, index) => (
        <div key={index} className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className={`w-3 h-3 rounded-full ${activity.color}`}></div>
            <span className="font-medium text-gray-700">{activity.period}</span>
          </div>
          <span className="text-lg font-bold text-gray-900">{activity.users.toLocaleString()}</span>
        </div>
      ))}
    </div>
  );
};

const RecentAlertsList: React.FC = () => {
  const alerts = [
    { id: 1, message: 'High CPU usage on server-03', time: '2 min ago', severity: 'warning' },
    { id: 2, message: 'Fraud detection model updated', time: '1 hour ago', severity: 'info' },
    { id: 3, message: 'Database backup completed', time: '3 hours ago', severity: 'success' }
  ];

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'warning': return 'text-yellow-600 bg-yellow-50';
      case 'error': return 'text-red-600 bg-red-50';
      case 'success': return 'text-green-600 bg-green-50';
      default: return 'text-blue-600 bg-blue-50';
    }
  };

  return (
    <div className="space-y-3">
      {alerts.map((alert) => (
        <div key={alert.id} className={`p-3 rounded-lg ${getSeverityColor(alert.severity)}`}>
          <p className="text-sm font-medium">{alert.message}</p>
          <p className="text-xs opacity-75 mt-1">{alert.time}</p>
        </div>
      ))}
    </div>
  );
};

export default function AdminDashboard() {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b backdrop-blur-sm bg-white/95 sticky top-0 z-40">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Executive Dashboard</h1>
              <p className="text-gray-600">System performance and business insights</p>
            </div>
            <div className="flex items-center space-x-4">
              <select className="px-4 py-2 border border-gray-200 rounded-lg focus:border-blue-500 focus:ring-4 focus:ring-blue-500/20 transition-all duration-200">
                <option>Last 30 days</option>
                <option>Last 90 days</option>
                <option>Last year</option>
              </select>
              <ModernButton variant="secondary" icon={<Download className="h-4 w-4" />}>
                Export Report
              </ModernButton>
            </div>
          </div>
        </div>
      </header>

      {/* Dashboard Content */}
      <div className="p-6">
        {/* Executive KPIs */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <ExecutiveKPICard
            title="Total Claims Value"
            value="$2.4M"
            change="+18.2%"
            period="vs last month"
            trend="up"
            icon={<DollarSign className="h-8 w-8" />}
            color="green"
          />
          <ExecutiveKPICard
            title="Processing Efficiency"
            value="94.7%"
            change="+2.1%"
            period="vs last month"
            trend="up"
            icon={<TrendingUp className="h-8 w-8" />}
            color="blue"
          />
          <ExecutiveKPICard
            title="Fraud Prevention Savings"
            value="$950K"
            change="+12.8%"
            period="this quarter"
            trend="up"
            icon={<Shield className="h-8 w-8" />}
            color="purple"
          />
          <ExecutiveKPICard
            title="Customer Satisfaction"
            value="97.2%"
            change="+0.8%"
            period="CSAT Score"
            trend="up"
            icon={<Star className="h-8 w-8" />}
            color="yellow"
          />
        </div>

        {/* Charts Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Claims Volume Trend */}
          <InteractiveCard>
            <CardHeader>
              <CardTitle className="flex items-center">
                <BarChart3 className="mr-2 h-5 w-5 text-blue-600" />
                Claims Volume & Resolution Trends
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64 flex items-center justify-center bg-gradient-to-br from-blue-50 to-purple-50 rounded-lg">
                <div className="text-center">
                  <Activity className="h-12 w-12 text-blue-500 mx-auto mb-2" />
                  <p className="text-gray-600">Interactive Chart Placeholder</p>
                  <p className="text-sm text-gray-500">Claims processing trends over time</p>
                </div>
              </div>
            </CardContent>
          </InteractiveCard>

          {/* ML Model Performance */}
          <InteractiveCard>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Zap className="mr-2 h-5 w-5 text-purple-600" />
                ML Model Performance
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64 flex items-center justify-center bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg">
                <div className="text-center">
                  <Globe className="h-12 w-12 text-purple-500 mx-auto mb-2" />
                  <p className="text-gray-600">AI Performance Metrics</p>
                  <p className="text-sm text-gray-500">Fraud detection accuracy: 97.8%</p>
                </div>
              </div>
            </CardContent>
          </InteractiveCard>
        </div>

        {/* System Health Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* System Status */}
          <InteractiveCard>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Server className="mr-2 h-5 w-5 text-green-600" />
                System Health
              </CardTitle>
            </CardHeader>
            <CardContent>
              <SystemHealthIndicators />
            </CardContent>
          </InteractiveCard>

          {/* Active Users */}
          <InteractiveCard>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Users className="mr-2 h-5 w-5 text-blue-600" />
                User Activity
              </CardTitle>
            </CardHeader>
            <CardContent>
              <UserActivityChart />
            </CardContent>
          </InteractiveCard>

          {/* Recent Alerts */}
          <InteractiveCard>
            <CardHeader>
              <CardTitle className="flex items-center">
                <AlertTriangle className="mr-2 h-5 w-5 text-yellow-600" />
                System Alerts
              </CardTitle>
            </CardHeader>
            <CardContent>
              <RecentAlertsList />
            </CardContent>
          </InteractiveCard>
        </div>

        {/* Management Tables */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          {/* User Management */}
          <InteractiveCard>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center">
                  <User className="mr-2 h-5 w-5 text-purple-600" />
                  User Management
                </CardTitle>
                <ModernButton size="sm" icon={<Plus className="h-4 w-4" />}>
                  Add User
                </ModernButton>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                      <User className="h-4 w-4 text-white" />
                    </div>
                    <div>
                      <div className="font-medium text-gray-900">Sarah Wilson</div>
                      <div className="text-sm text-gray-500">Administrator</div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    <span className="text-sm text-green-600">Active</span>
                  </div>
                </div>
                <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-gradient-to-br from-green-500 to-emerald-600 rounded-full flex items-center justify-center">
                      <User className="h-4 w-4 text-white" />
                    </div>
                    <div>
                      <div className="font-medium text-gray-900">Mike Chen</div>
                      <div className="text-sm text-gray-500">Adjuster</div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Clock className="h-4 w-4 text-yellow-500" />
                    <span className="text-sm text-yellow-600">Away</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </InteractiveCard>

          {/* Configuration Management */}
          <InteractiveCard>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center">
                  <Database className="mr-2 h-5 w-5 text-green-600" />
                  System Configuration
                </CardTitle>
                <ModernButton size="sm" variant="secondary" icon={<Settings className="h-4 w-4" />}>
                  Settings
                </ModernButton>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div>
                    <div className="font-medium text-gray-900">Database Connections</div>
                    <div className="text-sm text-gray-500">Active: 45/100</div>
                  </div>
                  <div className="w-16 bg-gray-200 rounded-full h-2">
                    <div className="bg-green-500 h-2 rounded-full" style={{ width: '45%' }}></div>
                  </div>
                </div>
                <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div>
                    <div className="font-medium text-gray-900">API Rate Limits</div>
                    <div className="text-sm text-gray-500">Usage: 2.1K/10K req/min</div>
                  </div>
                  <div className="w-16 bg-gray-200 rounded-full h-2">
                    <div className="bg-blue-500 h-2 rounded-full" style={{ width: '21%' }}></div>
                  </div>
                </div>
                <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div>
                    <div className="font-medium text-gray-900">Storage Usage</div>
                    <div className="text-sm text-gray-500">Used: 1.2TB/5TB</div>
                  </div>
                  <div className="w-16 bg-gray-200 rounded-full h-2">
                    <div className="bg-yellow-500 h-2 rounded-full" style={{ width: '24%' }}></div>
                  </div>
                </div>
              </div>
            </CardContent>
          </InteractiveCard>
        </div>
      </div>
    </div>
  );
}
