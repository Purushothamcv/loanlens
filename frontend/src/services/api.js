import axios from 'axios'

// Get the base URL from environment or fallback to localhost
const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

// Create axios instance with base configuration
const api = axios.create({
  baseURL: BASE_URL,
  timeout: 15000, // Increased timeout
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
  withCredentials: false, // Important: set to false to avoid CORS issues
})

// Request interceptor
api.interceptors.request.use(
  (config) => {
    console.log('🚀 API Request:', config.method?.toUpperCase(), `${config.baseURL}${config.url}`)
    if (config.data) {
      console.log('📤 Request data:', config.data)
    }
    return config
  },
  (error) => {
    console.error('❌ Request interceptor error:', error)
    return Promise.reject(error)
  }
)

// Response interceptor
api.interceptors.response.use(
  (response) => {
    console.log('✅ API Response:', response.status, response.config.url)
    console.log('📥 Response data:', response.data)
    return response
  },
  (error) => {
    console.error('❌ API Error:', error)
    
    if (error.response) {
      // Server responded with error status
      console.error('Status:', error.response.status)
      console.error('Data:', error.response.data)
      console.error('Headers:', error.response.headers)
    } else if (error.request) {
      // Request was made but no response received
      console.error('No response received:', error.request)
    } else {
      // Something else happened
      console.error('Error message:', error.message)
    }
    
    if (error.code === 'ERR_NETWORK') {
      console.error('🌐 Network error - Backend might be down or wrong URL')
      console.error(`Trying to reach: ${BASE_URL}`)
    }
    
    return Promise.reject(error)
  }
)

export const loanApi = {
  // Test backend connection
  testConnection: async () => {
    try {
      const response = await api.get('/health')
      return { success: true, data: response.data }
    } catch (error) {
      return { success: false, error: error.message }
    }
  },

  // Predict loan status
  predictLoan: async (loanData) => {
    try {
      console.log('🔮 Sending prediction request with data:', loanData)
      const response = await api.post('/predict', loanData)
      return response.data
    } catch (error) {
      console.error('🚨 Prediction failed:', error)
      throw error
    }
  },

  // Health check
  healthCheck: async () => {
    const response = await api.get('/health')
    return response.data
  },

  // Get API info
  getApiInfo: async () => {
    const response = await api.get('/')
    return response.data
  },

  // Get model info (debug endpoint)
  getModelInfo: async () => {
    const response = await api.get('/model-info')
    return response.data
  }
}

export default api
