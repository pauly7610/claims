// Simple API client for the customer portal
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface ApiResponse<T> {
  data: T
  success: boolean
  message?: string
}

interface Claim {
  id: string
  claim_number: string
  claim_type: string
  status: string
  estimated_amount: number
  created_at: string
  description: string
  last_updated: string
  incident_date?: string
  location?: string
  policy_number?: string
}

class ApiClient {
  private baseUrl: string

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        ...options,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      return data
    } catch (error) {
      console.error('API request failed:', error)
      throw error
    }
  }

  // Claims API methods
  async getClaims(): Promise<ApiResponse<{ claims: Claim[] }>> {
    return this.request<{ claims: Claim[] }>('/api/claims')
  }

  async getClaim(id: string): Promise<ApiResponse<Claim>> {
    return this.request<Claim>(`/api/claims/${id}`)
  }

  async createClaim(claimData: Partial<Claim>): Promise<ApiResponse<{ claim: Claim }>> {
    return this.request<{ claim: Claim }>('/api/claims', {
      method: 'POST',
      body: JSON.stringify(claimData),
    })
  }

  async updateClaim(id: string, claimData: Partial<Claim>): Promise<ApiResponse<Claim>> {
    return this.request<Claim>(`/api/claims/${id}`, {
      method: 'PUT',
      body: JSON.stringify(claimData),
    })
  }

  // File upload method
  async uploadFile(file: File, claimId?: string): Promise<ApiResponse<any>> {
    const formData = new FormData()
    formData.append('file', file)
    if (claimId) {
      formData.append('claim_id', claimId)
    }

    try {
      const response = await fetch(`${this.baseUrl}/api/files/upload`, {
        method: 'POST',
        body: formData,
      })
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      return await response.json()
    } catch (error) {
      console.error('File upload failed:', error)
      throw error
    }
  }

  // Auth API methods
  async login(email: string, password: string): Promise<ApiResponse<{ token: string; user: any }>> {
    return this.request<{ token: string; user: any }>('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    })
  }

  async register(userData: any): Promise<ApiResponse<{ token: string; user: any }>> {
    return this.request<{ token: string; user: any }>('/api/auth/register', {
      method: 'POST',
      body: JSON.stringify(userData),
    })
  }

  async getProfile(): Promise<ApiResponse<any>> {
    return this.request<any>('/api/auth/profile')
  }
}

// Export singleton instance
export const claimsApi = new ApiClient()
export default claimsApi 