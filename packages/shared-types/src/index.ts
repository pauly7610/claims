import { z } from 'zod';

// User and Authentication Types
export const UserRole = z.enum(['customer', 'adjuster', 'admin', 'manager']);
export type UserRole = z.infer<typeof UserRole>;

export const User = z.object({
  id: z.string().uuid(),
  email: z.string().email(),
  firstName: z.string(),
  lastName: z.string(),
  role: UserRole,
  phone: z.string().optional(),
  isActive: z.boolean(),
  emailVerified: z.boolean(),
  createdAt: z.string().datetime(),
  updatedAt: z.string().datetime(),
});
export type User = z.infer<typeof User>;

// Claims Types
export const ClaimStatus = z.enum([
  'submitted',
  'under_review',
  'requires_documents',
  'ai_processing',
  'adjuster_review',
  'approved',
  'denied',
  'paid',
  'closed'
]);
export type ClaimStatus = z.infer<typeof ClaimStatus>;

export const ClaimType = z.enum([
  'auto',
  'home',
  'health',
  'life',
  'travel',
  'business'
]);
export type ClaimType = z.infer<typeof ClaimType>;

export const ClaimPriority = z.enum(['low', 'medium', 'high', 'urgent']);
export type ClaimPriority = z.infer<typeof ClaimPriority>;

export const Claim = z.object({
  id: z.string().uuid(),
  claimNumber: z.string(),
  policyId: z.string().uuid(),
  customerId: z.string().uuid(),
  incidentDate: z.string().date(),
  reportedDate: z.string().datetime(),
  claimType: ClaimType,
  description: z.string(),
  estimatedAmount: z.number().optional(),
  approvedAmount: z.number().optional(),
  status: ClaimStatus,
  priority: ClaimPriority,
  assignedAdjusterId: z.string().uuid().optional(),
  fraudScore: z.number().min(0).max(1),
  aiConfidence: z.number().min(0).max(1),
  createdAt: z.string().datetime(),
  updatedAt: z.string().datetime(),
});
export type Claim = z.infer<typeof Claim>;

export const ClaimDocument = z.object({
  id: z.string().uuid(),
  claimId: z.string().uuid(),
  fileName: z.string(),
  filePath: z.string(),
  fileType: z.string(),
  fileSize: z.number(),
  documentType: z.string(),
  aiExtractedData: z.record(z.any()).optional(),
  createdAt: z.string().datetime(),
});
export type ClaimDocument = z.infer<typeof ClaimDocument>;

export const ClaimHistory = z.object({
  id: z.string().uuid(),
  claimId: z.string().uuid(),
  statusFrom: ClaimStatus.optional(),
  statusTo: ClaimStatus,
  changedBy: z.string().uuid(),
  changeReason: z.string().optional(),
  aiDecision: z.boolean(),
  metadata: z.record(z.any()).optional(),
  createdAt: z.string().datetime(),
});
export type ClaimHistory = z.infer<typeof ClaimHistory>;

// Policy Types
export const PolicyStatus = z.enum(['active', 'inactive', 'expired', 'cancelled']);
export type PolicyStatus = z.infer<typeof PolicyStatus>;

export const Policy = z.object({
  id: z.string().uuid(),
  policyNumber: z.string(),
  customerId: z.string().uuid(),
  policyType: z.string(),
  coverageAmount: z.number(),
  deductible: z.number(),
  premium: z.number(),
  startDate: z.string().date(),
  endDate: z.string().date(),
  status: PolicyStatus,
  createdAt: z.string().datetime(),
  updatedAt: z.string().datetime(),
});
export type Policy = z.infer<typeof Policy>;

// Payment Types
export const PaymentStatus = z.enum(['pending', 'processing', 'completed', 'failed', 'cancelled']);
export type PaymentStatus = z.infer<typeof PaymentStatus>;

export const Payment = z.object({
  id: z.string().uuid(),
  claimId: z.string().uuid(),
  amount: z.number(),
  currency: z.string().default('USD'),
  paymentMethod: z.string(),
  paymentStatus: PaymentStatus,
  externalPaymentId: z.string().optional(),
  processedAt: z.string().datetime().optional(),
  createdAt: z.string().datetime(),
  updatedAt: z.string().datetime(),
});
export type Payment = z.infer<typeof Payment>;

// Notification Types
export const NotificationType = z.enum([
  'claim_submitted',
  'claim_approved',
  'claim_denied',
  'documents_required',
  'payment_processed',
  'fraud_detected',
  'system_alert'
]);
export type NotificationType = z.infer<typeof NotificationType>;

export const NotificationChannel = z.enum(['email', 'sms', 'push', 'in_app']);
export type NotificationChannel = z.infer<typeof NotificationChannel>;

export const NotificationStatus = z.enum(['pending', 'sent', 'delivered', 'failed', 'read']);
export type NotificationStatus = z.infer<typeof NotificationStatus>;

export const Notification = z.object({
  id: z.string().uuid(),
  userId: z.string().uuid(),
  claimId: z.string().uuid().optional(),
  type: NotificationType,
  title: z.string(),
  message: z.string(),
  channel: NotificationChannel,
  status: NotificationStatus,
  sentAt: z.string().datetime().optional(),
  readAt: z.string().datetime().optional(),
  createdAt: z.string().datetime(),
});
export type Notification = z.infer<typeof Notification>;

// AI/ML Types
export const AIModelType = z.enum([
  'fraud_detection',
  'document_extraction',
  'damage_assessment',
  'settlement_prediction'
]);
export type AIModelType = z.infer<typeof AIModelType>;

export const AIDecision = z.object({
  modelType: AIModelType,
  confidence: z.number().min(0).max(1),
  prediction: z.any(),
  features: z.record(z.any()),
  modelVersion: z.string(),
  timestamp: z.string().datetime(),
});
export type AIDecision = z.infer<typeof AIDecision>;

// API Response Types
export const ApiResponse = <T extends z.ZodType>(dataSchema: T) =>
  z.object({
    success: z.boolean(),
    data: dataSchema.optional(),
    error: z.string().optional(),
    message: z.string().optional(),
    timestamp: z.string().datetime(),
  });

export const PaginatedResponse = <T extends z.ZodType>(itemSchema: T) =>
  z.object({
    items: z.array(itemSchema),
    total: z.number(),
    page: z.number(),
    pageSize: z.number(),
    totalPages: z.number(),
  });

// Form Types
export const ClaimSubmissionForm = z.object({
  policyNumber: z.string().min(1, 'Policy number is required'),
  incidentDate: z.string().date(),
  claimType: ClaimType,
  description: z.string().min(10, 'Description must be at least 10 characters'),
  estimatedAmount: z.number().positive().optional(),
  documents: z.array(z.instanceof(File)).optional(),
});
export type ClaimSubmissionForm = z.infer<typeof ClaimSubmissionForm>;

export const LoginForm = z.object({
  email: z.string().email('Invalid email address'),
  password: z.string().min(8, 'Password must be at least 8 characters'),
});
export type LoginForm = z.infer<typeof LoginForm>;

// Dashboard/Analytics Types
export const ClaimMetrics = z.object({
  totalClaims: z.number(),
  pendingClaims: z.number(),
  approvedClaims: z.number(),
  deniedClaims: z.number(),
  averageResolutionTime: z.number(), // in hours
  fraudDetectionRate: z.number(),
  customerSatisfactionScore: z.number(),
  totalPayouts: z.number(),
});
export type ClaimMetrics = z.infer<typeof ClaimMetrics>;

export const TimeSeriesData = z.object({
  timestamp: z.string().datetime(),
  value: z.number(),
  label: z.string().optional(),
});
export type TimeSeriesData = z.infer<typeof TimeSeriesData>;

// Export all schemas for validation
export const schemas = {
  User,
  Claim,
  ClaimDocument,
  ClaimHistory,
  Policy,
  Payment,
  Notification,
  AIDecision,
  ClaimSubmissionForm,
  LoginForm,
  ClaimMetrics,
  TimeSeriesData,
}; 