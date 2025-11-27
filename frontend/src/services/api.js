import axios from 'axios'

// Create axios instance with base configuration
const api = axios.create({
  baseURL: 'http://localhost:8000',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: false, // Set to false to avoid CORS issues
})

// Request interceptor
api.interceptors.request.use(
  (config) => {
    console.log('API Request:', config.method?.toUpperCase(), config.url)
    console.log('Request data:', config.data)
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor
api.interceptors.response.use(
  (response) => {
    console.log('API Response:', response.status, response.config.url)
    return response
  },
  (error) => {
    console.error('API Error:', error)
    if (error.code === 'ERR_NETWORK') {
      console.error('Network error - check if backend is running on http://localhost:8000')
    }
    return Promise.reject(error)
  }
)

export const loanApi = {
  // Predict loan status
  predictLoan: async (loanData) => {
    const response = await api.post('/predict', loanData)
    return response.data
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
  }
}

export default api
