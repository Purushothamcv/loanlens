from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import joblib
import numpy as np
from typing import Optional
from enum import Enum
import os
import socket
from config import settings

# Helper function to check port availability
def is_port_available(host: str, port: int) -> bool:
    """Check if a port is available for binding"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            return True
    except OSError:
        return False

def find_available_port(host: str, start_port: int, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        if is_port_available(host, port):
            return port
    raise RuntimeError(f"No available ports found in range {start_port}-{start_port + max_attempts}")

# Initialize FastAPI app with environment config
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None
)

# Add CORS middleware with more permissive settings for development
cors_origins = ["*"] if settings.ENVIRONMENT == "development" else settings.cors_origins_list

print(f"🔒 CORS Configuration:")
print(f"   Environment: {settings.ENVIRONMENT}")
print(f"   Allowed Origins: {cors_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Load the trained model
try:
    model_path = os.path.join(os.path.dirname(__file__), settings.MODEL_PATH)
    model = joblib.load(model_path)
    print(f"✅ Model loaded successfully from {model_path}")
except FileNotFoundError:
    print(f"❌ Model file '{settings.MODEL_PATH}' not found. Please ensure the model is trained and saved.")
    model = None
except Exception as e:
    print(f"⚠️ Warning: Model version mismatch or error loading model: {e}")
    print("🔄 Model loaded but may have compatibility issues.")
    model = joblib.load(model_path)  # Load anyway

# Define enums for categorical fields
class GenderEnum(str, Enum):
    FEMALE = "Female"
    MALE = "Male"

class MaritalStatusEnum(str, Enum):
    NO = "No"
    YES = "Yes"

class EducationEnum(str, Enum):
    NOT_GRADUATE = "Not Graduate"
    GRADUATE = "Graduate"

class SelfEmployedEnum(str, Enum):
    NO = "No"
    YES = "Yes"

class PropertyAreaEnum(str, Enum):
    RURAL = "Rural"
    SEMIURBAN = "Semiurban"
    URBAN = "Urban"

class CreditHistoryEnum(str, Enum):
    NO = "No"
    YES = "Yes"

# Define the input data model
class LoanApplication(BaseModel):
    Gender: GenderEnum = Field(..., description="Gender")
    Married: MaritalStatusEnum = Field(..., description="Marital Status")
    Dependents: int = Field(..., description="Number of dependents (0, 1, 2, or 3+ as 4)", ge=0, le=4)
    Education: EducationEnum = Field(..., description="Education level")
    Self_Employed: SelfEmployedEnum = Field(..., description="Self Employment status")
    ApplicantIncome: float = Field(..., description="Applicant Income", gt=0)
    CoapplicantIncome: float = Field(0.0, description="Coapplicant Income", ge=0)
    LoanAmount: float = Field(..., description="Loan Amount", gt=0)
    Loan_Amount_Term: float = Field(..., description="Loan Amount Term in days", gt=0)
    Credit_History: CreditHistoryEnum = Field(..., description="Credit History")
    Property_Area: PropertyAreaEnum = Field(..., description="Property Area")

# Define the response model
class LoanPredictionResponse(BaseModel):
    loan_status: int = Field(..., description="Loan Status (0: Not Approved, 1: Approved)")
    loan_status_text: str = Field(..., description="Loan Status in text format")
    probability: float = Field(..., description="Probability of loan approval")

# Serve the HTML file at root for testing
@app.get("/")
async def root():
    """Serve the HTML test interface"""
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    else:
        return {
            "message": "Loan Prediction API",
            "description": "Use /predict endpoint to predict loan approval status",
            "version": settings.API_VERSION,
            "environment": settings.ENVIRONMENT,
            "docs": "/docs" if settings.DEBUG else "Documentation disabled in production",
            "test_interface": "index.html not found - using React frontend instead"
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "environment": settings.ENVIRONMENT,
        "version": settings.API_VERSION
    }

@app.post("/predict", response_model=LoanPredictionResponse)
async def predict_loan_status(loan_data: LoanApplication):
    """
    Predict loan approval status based on applicant information
    
    Returns:
    - loan_status: 0 (Not Approved) or 1 (Approved)
    - loan_status_text: Human readable loan status
    - probability: Probability of loan approval
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please ensure the model file exists and is properly trained.")
    
    try:
        # Convert input data to numeric format
        numeric_data = convert_categorical_to_numeric(loan_data)
        
        # Create input array in the correct order
        feature_names = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
                        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                        'Loan_Amount_Term', 'Credit_History', 'Property_Area']
        
        input_data = np.array([[numeric_data[feature] for feature in feature_names]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]  # Probability of class 1 (approved)
        
        # Convert prediction to text
        loan_status_text = "Approved" if prediction == 1 else "Not Approved"
        
        return LoanPredictionResponse(
            loan_status=int(prediction),
            loan_status_text=loan_status_text,
            probability=float(probability)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.options("/predict")
async def predict_options():
    """Handle OPTIONS request for CORS preflight"""
    return {"message": "OK"}

# Helper function for categorical conversion
def convert_categorical_to_numeric(loan_data: LoanApplication) -> dict:
    """Convert categorical fields to numeric values as expected by the model"""
    
    # Gender conversion
    gender_numeric = 1 if loan_data.Gender == GenderEnum.MALE else 0
    
    # Marital Status conversion
    married_numeric = 1 if loan_data.Married == MaritalStatusEnum.YES else 0
    
    # Education conversion
    education_numeric = 1 if loan_data.Education == EducationEnum.GRADUATE else 0
    
    # Self Employed conversion
    self_employed_numeric = 1 if loan_data.Self_Employed == SelfEmployedEnum.YES else 0
    
    # Credit History conversion
    credit_history_numeric = 1 if loan_data.Credit_History == CreditHistoryEnum.YES else 0
    
    # Property Area conversion
    property_area_mapping = {
        PropertyAreaEnum.RURAL: 0,
        PropertyAreaEnum.SEMIURBAN: 1,
        PropertyAreaEnum.URBAN: 2
    }
    property_area_numeric = property_area_mapping[loan_data.Property_Area]
    
    return {
        'Gender': gender_numeric,
        'Married': married_numeric,
        'Dependents': loan_data.Dependents,
        'Education': education_numeric,
        'Self_Employed': self_employed_numeric,
        'ApplicantIncome': loan_data.ApplicantIncome,
        'CoapplicantIncome': loan_data.CoapplicantIncome,
        'LoanAmount': loan_data.LoanAmount,
        'Loan_Amount_Term': loan_data.Loan_Amount_Term,
        'Credit_History': credit_history_numeric,
        'Property_Area': property_area_numeric
    }

if __name__ == "__main__":
    import uvicorn
    
    # Determine the port to use
    host = settings.HOST
    port = settings.PORT
    
    # Check if default port is available
    if not is_port_available(host, port):
        print(f"⚠️ Port {port} is already in use!")
        try:
            port = find_available_port(host, port + 1)
            print(f"🔄 Using alternative port: {port}")
        except RuntimeError as e:
            print(f"❌ Error: {e}")
            print("💡 Try stopping other services or use a different port")
            print("💡 Or kill the existing process with: taskkill /F /PID <PID>")
            
            # Try to find what's using the port
            import subprocess
            try:
                result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
                lines = result.stdout.split('\n')
                for line in lines:
                    if f':{settings.PORT}' in line and 'LISTENING' in line:
                        print(f"🔍 Port {settings.PORT} is being used by: {line.strip()}")
                        break
            except:
                pass
            
            exit(1)
    
    print(f"🚀 Starting server on {host}:{port}")
    print(f"🌐 CORS origins: {settings.cors_origins_list}")
    print(f"📚 API docs available at: http://localhost:{port}/docs")
    print(f"🏥 Health check at: http://localhost:{port}/health")
    print(f"🔮 Predict endpoint: http://localhost:{port}/predict")
    
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")