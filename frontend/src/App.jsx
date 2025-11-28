import React, { useEffect, useState } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import Navbar from './components/Navbar'
import HomePage from './pages/HomePage'
import PredictionPage from './pages/PredictionPage'
import Footer from './components/Footer'
import { loanApi } from './services/api'

function App() {
  const [backendStatus, setBackendStatus] = useState('checking')

  useEffect(() => {
    // Test backend connection on app load
    const testBackend = async () => {
      try {
        const result = await loanApi.testConnection()
        if (result.success) {
          setBackendStatus('connected')
          console.log('✅ Backend connected successfully')
        } else {
          setBackendStatus('error')
          console.error('❌ Backend connection failed:', result.error)
        }
      } catch (error) {
        setBackendStatus('error')
        console.error('❌ Backend test failed:', error)
      }
    }

    testBackend()
  }, [])

  return (
    <Router future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <div className="min-h-screen flex flex-col bg-gray-50">
        <Navbar />
        
        {/* Backend Status Banner */}
        {backendStatus === 'error' && (
          <div className="bg-red-500 text-white px-4 py-2 text-center">
            ⚠️ Backend connection failed. Please ensure the API server is running on http://localhost:8000
          </div>
        )}
        
        {backendStatus === 'checking' && (
          <div className="bg-yellow-500 text-white px-4 py-2 text-center">
            🔄 Checking backend connection...
          </div>
        )}

        <main className="flex-grow">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/predict" element={<PredictionPage backendStatus={backendStatus} />} />
          </Routes>
        </main>
        
        <Footer />
        
        <Toaster 
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: '#363636',
              color: '#fff',
            },
          }}
        />
      </div>
    </Router>
  )
}

export default App
