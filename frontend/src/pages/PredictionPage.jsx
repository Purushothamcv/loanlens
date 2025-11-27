import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  User, 
  DollarSign, 
  FileText, 
  CheckCircle, 
  XCircle, 
  AlertCircle,
  TrendingUp,
  Calculator
} from 'lucide-react'
import toast from 'react-hot-toast'
import LoadingSpinner from '../components/LoadingSpinner'
import { loanApi } from '../services/api'
import { FORM_STEPS, LOAN_STATUS } from '../config/constants'

const PredictionPage = () => {
  const [formData, setFormData] = useState({
    Gender: '',
    Married: '',
    Dependents: '',
    Education: '',
    Self_Employed: '',
    ApplicantIncome: '',
    CoapplicantIncome: '',
    LoanAmount: '',
    Loan_Amount_Term: '',
    Credit_History: '',
    Property_Area: ''
  })

  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [currentStep, setCurrentStep] = useState(FORM_STEPS.PERSONAL)

  const handleInputChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value
    }))
  }

  const validateStep = (step) => {
    switch (step) {
      case 1:
        return formData.Gender && formData.Married && formData.Dependents !== '' && formData.Education
      case 2:
        return formData.Self_Employed && formData.ApplicantIncome && formData.CoapplicantIncome !== ''
      case 3:
        return formData.LoanAmount && formData.Loan_Amount_Term && formData.Credit_History && formData.Property_Area
      default:
        return false
    }
  }

  const nextStep = () => {
    if (validateStep(currentStep)) {
      setCurrentStep(prev => Math.min(prev + 1, 3))
    } else {
      toast.error('Please fill in all required fields')
    }
  }

  const prevStep = () => {
    setCurrentStep(prev => Math.max(prev - 1, 1))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    if (!validateStep(3)) {
      toast.error('Please fill in all required fields')
      return
    }

    setLoading(true)
    
    try {
      // Convert string values to appropriate types
      const submissionData = {
        ...formData,
        Dependents: parseInt(formData.Dependents),
        ApplicantIncome: parseFloat(formData.ApplicantIncome),
        CoapplicantIncome: parseFloat(formData.CoapplicantIncome),
        LoanAmount: parseFloat(formData.LoanAmount),
        Loan_Amount_Term: parseFloat(formData.Loan_Amount_Term)
      }

      const response = await loanApi.predictLoan(submissionData)
      setPrediction(response)
      toast.success('Prediction completed successfully!')
    } catch (error) {
      console.error('Error:', error)
      toast.error(error.response?.data?.detail || 'Failed to get prediction')
    } finally {
      setLoading(false)
    }
  }

  const resetForm = () => {
    setFormData({
      Gender: '',
      Married: '',
      Dependents: '',
      Education: '',
      Self_Employed: '',
      ApplicantIncome: '',
      CoapplicantIncome: '',
      LoanAmount: '',
      Loan_Amount_Term: '',
      Credit_History: '',
      Property_Area: ''
    })
    setPrediction(null)
    setCurrentStep(1)
  }

  const renderStepContent = () => {
    switch (currentStep) {
      case 1:
        return (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="space-y-6"
          >
            <div className="flex items-center space-x-3 mb-6">
              <User className="h-6 w-6 text-blue-600" />
              <h3 className="text-xl font-semibold text-gray-900">Personal Information</h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Gender *
                </label>
                <select
                  name="Gender"
                  value={formData.Gender}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200 bg-white appearance-none"
                  required
                >
                  <option value="">Select Gender</option>
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Marital Status *
                </label>
                <select
                  name="Married"
                  value={formData.Married}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200 bg-white appearance-none"
                  required
                >
                  <option value="">Select Status</option>
                  <option value="Yes">Married</option>
                  <option value="No">Single</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Number of Dependents *
                </label>
                <select
                  name="Dependents"
                  value={formData.Dependents}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200 bg-white appearance-none"
                  required
                >
                  <option value="">Select Dependents</option>
                  <option value="0">0</option>
                  <option value="1">1</option>
                  <option value="2">2</option>
                  <option value="3">3+</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Education *
                </label>
                <select
                  name="Education"
                  value={formData.Education}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200 bg-white appearance-none"
                  required
                >
                  <option value="">Select Education</option>
                  <option value="Graduate">Graduate</option>
                  <option value="Not Graduate">Not Graduate</option>
                </select>
              </div>
            </div>
          </motion.div>
        )

      case 2:
        return (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="space-y-6"
          >
            <div className="flex items-center space-x-3 mb-6">
              <DollarSign className="h-6 w-6 text-blue-600" />
              <h3 className="text-xl font-semibold text-gray-900">Financial Information</h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Self Employed *
                </label>
                <select
                  name="Self_Employed"
                  value={formData.Self_Employed}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200 bg-white appearance-none"
                  required
                >
                  <option value="">Select Employment</option>
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Applicant Income ($) *
                </label>
                <input
                  type="number"
                  name="ApplicantIncome"
                  value={formData.ApplicantIncome}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200 bg-white"
                  placeholder="Enter monthly income"
                  min="0"
                  required
                />
              </div>

              <div className="md:col-span-2">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Co-applicant Income ($)
                </label>
                <input
                  type="number"
                  name="CoapplicantIncome"
                  value={formData.CoapplicantIncome}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200 bg-white"
                  placeholder="Enter co-applicant income (0 if none)"
                  min="0"
                />
              </div>
            </div>
          </motion.div>
        )

      case 3:
        return (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="space-y-6"
          >
            <div className="flex items-center space-x-3 mb-6">
              <FileText className="h-6 w-6 text-blue-600" />
              <h3 className="text-xl font-semibold text-gray-900">Loan Details</h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Loan Amount ($) *
                </label>
                <input
                  type="number"
                  name="LoanAmount"
                  value={formData.LoanAmount}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200 bg-white"
                  placeholder="Enter loan amount"
                  min="0"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Loan Term (days) *
                </label>
                <input
                  type="number"
                  name="Loan_Amount_Term"
                  value={formData.Loan_Amount_Term}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200 bg-white"
                  placeholder="Enter loan term in days"
                  min="0"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Credit History *
                </label>
                <select
                  name="Credit_History"
                  value={formData.Credit_History}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200 bg-white appearance-none"
                  required
                >
                  <option value="">Select Credit History</option>
                  <option value="Yes">Good Credit History</option>
                  <option value="No">Poor Credit History</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Property Area *
                </label>
                <select
                  name="Property_Area"
                  value={formData.Property_Area}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200 bg-white appearance-none"
                  required
                >
                  <option value="">Select Area</option>
                  <option value="Urban">Urban</option>
                  <option value="Semiurban">Semi-urban</option>
                  <option value="Rural">Rural</option>
                </select>
              </div>
            </div>
          </motion.div>
        )

      default:
        return null
    }
  }

  if (prediction) {
    return (
      <div className="min-h-screen bg-gray-50 py-12">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-white rounded-2xl shadow-xl p-8"
          >
            <div className="text-center mb-8">
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
                className={`w-20 h-20 mx-auto mb-4 rounded-full flex items-center justify-center ${
                  prediction.loan_status === LOAN_STATUS.APPROVED
                    ? 'bg-green-100 text-green-600' 
                    : 'bg-red-100 text-red-600'
                }`}
              >
                {prediction.loan_status === LOAN_STATUS.APPROVED ? (
                  <CheckCircle className="h-10 w-10" />
                ) : (
                  <XCircle className="h-10 w-10" />
                )}
              </motion.div>
              
              <h2 className="text-3xl font-bold text-gray-900 mb-2">
                Loan Prediction Result
              </h2>
              
              <p className={`text-xl font-semibold ${
                prediction.loan_status === LOAN_STATUS.APPROVED ? 'text-green-600' : 'text-red-600'
              }`}>
                {prediction.loan_status_text}
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
              <div className="bg-gray-50 rounded-lg p-6 text-center">
                <div className="text-2xl font-bold text-gray-900 mb-2">
                  {prediction.loan_status === LOAN_STATUS.APPROVED ? 'Approved' : 'Rejected'}
                </div>
                <div className="text-gray-600">Status</div>
              </div>
              
              <div className="bg-gray-50 rounded-lg p-6 text-center">
                <div className="text-2xl font-bold text-blue-600 mb-2">
                  {(prediction.probability * 100).toFixed(1)}%
                </div>
                <div className="text-gray-600">Approval Probability</div>
              </div>
              
              <div className="bg-gray-50 rounded-lg p-6 text-center">
                <div className="text-2xl font-bold text-gray-900 mb-2">
                  AI
                </div>
                <div className="text-gray-600">Powered Analysis</div>
              </div>
            </div>

            <div className="bg-blue-50 border border-blue-200 rounded-lg p-6 mb-8">
              <div className="flex items-start space-x-3">
                <AlertCircle className="h-6 w-6 text-blue-600 mt-0.5" />
                <div>
                  <h3 className="font-semibold text-blue-900 mb-2">Important Note</h3>
                  <p className="text-blue-800 text-sm">
                    This prediction is based on machine learning algorithms and historical data. 
                    The actual loan approval decision may vary based on additional factors not 
                    included in this model. Please consult with financial institutions for 
                    official loan applications.
                  </p>
                </div>
              </div>
            </div>

            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button
                onClick={resetForm}
                className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-6 rounded-lg transition-all duration-200 transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
              >
                <Calculator className="mr-2 h-5 w-5" />
                New Prediction
              </button>
              <button className="bg-white border border-gray-300 hover:bg-gray-50 text-gray-700 font-semibold py-3 px-6 rounded-lg transition-all duration-200 transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2">
                <TrendingUp className="mr-2 h-5 w-5" />
                Improve Score
              </button>
            </div>
          </motion.div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-8">
          <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
            Loan Prediction Form
          </h1>
          <p className="text-lg text-gray-600">
            Fill in your details to get an instant loan approval prediction
          </p>
        </div>

        {/* Progress Bar */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            {[1, 2, 3].map((step) => (
              <div
                key={step}
                className={`flex items-center justify-center w-10 h-10 rounded-full border-2 transition-all duration-300 ${
                  step <= currentStep
                    ? 'bg-blue-600 border-blue-600 text-white'
                    : 'bg-white border-gray-300 text-gray-400'
                }`}
              >
                {step < currentStep ? (
                  <CheckCircle className="h-6 w-6" />
                ) : (
                  <span className="font-semibold">{step}</span>
                )}
              </div>
            ))}
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <motion.div
              className="bg-blue-600 h-2 rounded-full"
              initial={{ width: '33%' }}
              animate={{ width: `${(currentStep / 3) * 100}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
        </div>

        <form onSubmit={handleSubmit} className="bg-white rounded-xl shadow-lg p-8">
          <AnimatePresence mode="wait">
            {renderStepContent()}
          </AnimatePresence>

          <div className="flex justify-between mt-8">
            <button
              type="button"
              onClick={prevStep}
              disabled={currentStep === 1}
              className={`bg-white border border-gray-300 hover:bg-gray-50 text-gray-700 font-semibold py-3 px-6 rounded-lg transition-all duration-200 transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 ${
                currentStep === 1 ? 'opacity-50 cursor-not-allowed' : ''
              }`}
            >
              Previous
            </button>

            {currentStep < 3 ? (
              <button
                type="button"
                onClick={nextStep}
                className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-6 rounded-lg transition-all duration-200 transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
              >
                Next
              </button>
            ) : (
              <button
                type="submit"
                disabled={loading}
                className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-6 rounded-lg transition-all duration-200 transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? (
                  <LoadingSpinner size="sm" text="" />
                ) : (
                  'Get Prediction'
                )}
              </button>
            )}
          </div>
        </form>
      </div>
    </div>
  )
}

export default PredictionPage
