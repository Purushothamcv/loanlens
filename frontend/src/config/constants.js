// API Configuration
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

// App Configuration
export const APP_NAME = import.meta.env.VITE_APP_NAME || 'LoanLens'
export const APP_DESCRIPTION = import.meta.env.VITE_APP_DESCRIPTION || 'AI-powered loan prediction system'

// Environment
export const ENVIRONMENT = import.meta.env.VITE_ENVIRONMENT || 'development'
export const IS_DEVELOPMENT = ENVIRONMENT === 'development'
export const IS_PRODUCTION = ENVIRONMENT === 'production'

// API Endpoints
export const API_ENDPOINTS = {
  predict: '/predict',
  health: '/health',
  root: '/'
}

// App Constants
export const FORM_STEPS = {
  PERSONAL: 1,
  FINANCIAL: 2,
  LOAN_DETAILS: 3
}

export const LOAN_STATUS = {
  APPROVED: 1,
  REJECTED: 0
}
