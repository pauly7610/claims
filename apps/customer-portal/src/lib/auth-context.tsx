'use client'

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'

interface User {
  id: string
  email: string
  first_name: string
  last_name: string
  role: string
}

interface AuthContextType {
  user: User | null
  isLoading: boolean
  isAuthenticated: boolean
  login: (email: string, password: string) => Promise<void>
  register: (userData: any) => Promise<void>
  logout: () => void
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

interface AuthProviderProps {
  children: ReactNode
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  // Mock user for development
  useEffect(() => {
    // Simulate loading and setting a mock user
    const timer = setTimeout(() => {
      setUser({
        id: '1',
        email: 'john.doe@example.com',
        first_name: 'John',
        last_name: 'Doe',
        role: 'customer'
      })
      setIsLoading(false)
    }, 1000)

    return () => clearTimeout(timer)
  }, [])

  const login = async (email: string, password: string) => {
    setIsLoading(true)
    try {
      // Mock login - in real app, this would call the API
      await new Promise(resolve => setTimeout(resolve, 1000)) // Simulate API call
      
      setUser({
        id: '1',
        email,
        first_name: 'John',
        last_name: 'Doe',
        role: 'customer'
      })
    } catch (error) {
      console.error('Login failed:', error)
      throw error
    } finally {
      setIsLoading(false)
    }
  }

  const register = async (userData: any) => {
    setIsLoading(true)
    try {
      // Mock registration - in real app, this would call the API
      await new Promise(resolve => setTimeout(resolve, 1000)) // Simulate API call
      
      setUser({
        id: '1',
        email: userData.email,
        first_name: userData.first_name,
        last_name: userData.last_name,
        role: 'customer'
      })
    } catch (error) {
      console.error('Registration failed:', error)
      throw error
    } finally {
      setIsLoading(false)
    }
  }

  const logout = () => {
    setUser(null)
    // In real app, would also clear tokens, etc.
  }

  const value: AuthContextType = {
    user,
    isLoading,
    isAuthenticated: !!user,
    login,
    register,
    logout
  }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth(): AuthContextType {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

export default AuthContext 