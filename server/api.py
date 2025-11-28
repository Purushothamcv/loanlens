from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import joblib
import numpy as np
from typing import Optional
from enum import Enum
import os
import sys
import socket
from pathlib import Path

# CRITICAL: Ensure we're looking in the right directory
if getattr(sys, 'frozen', False):
    # Running as compiled executable
    current_dir = Path(sys.executable).parent.absolute()
else:
    # Running as script
    current_dir = Path(__file__).parent.absolute()

print(f"🔍 API starting from: {current_dir}")
print(f"📂 Python path: {sys.path}")

# Change working directory to where api.py is located
os.chdir(current_dir)
print(f"✅ Working directory set to: {Path.cwd()}")

# Add to Python path
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

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

# Enhanced model loading with ABSOLUTE paths
model = None
feature_names = None
MODEL_FEATURES = []

def load_model_with_fallbacks():
    """Try multiple methods to load the model with absolute paths"""
    global model, feature_names, MODEL_FEATURES
    
    # Use ABSOLUTE paths only
    model_paths = [
        current_dir / settings.MODEL_PATH,  # server/loan_model.pkl
        current_dir / "loan_model.pkl",  # server/loan_model.pkl (explicit)
        Path(settings.MODEL_PATH),  # relative to current
    ]
    
    print("\n" + "="*60)
    print("🔍 SEARCHING FOR MODEL FILE")
    print("="*60)
    print(f"Current directory: {current_dir}")
    print(f"Working directory: {Path.cwd()}")
    print(f"Model filename from settings: {settings.MODEL_PATH}")
    
    # List what's actually in the directory
    print(f"\n📁 Files in {current_dir}:")
    try:
        files = list(current_dir.glob("*"))
        pkl_files = [f for f in files if f.suffix == '.pkl']
        
        print(f"   Total files: {len(files)}")
        print(f"   PKL files: {len(pkl_files)}")
        
        if pkl_files:
            for pkl in pkl_files:
                print(f"   ✅ Found: {pkl.name} ({pkl.stat().st_size} bytes)")
        else:
            print(f"   ❌ No PKL files found!")
            print(f"   💡 Please run the training notebook to create loan_model.pkl")
    except Exception as e:
        print(f"   ❌ Error listing files: {e}")
    
    print(f"\n🔎 Trying to load model from these locations:")
    for i, model_path in enumerate(model_paths, 1):
        print(f"{i}. {model_path}")
    
    # Try each path
    for idx, model_path in enumerate(model_paths, 1):
        print(f"\n📍 Attempt {idx}: {model_path}")
        print(f"   Exists: {model_path.exists()}")
        print(f"   Is file: {model_path.is_file() if model_path.exists() else 'N/A'}")
        
        if model_path.exists() and model_path.is_file():
            try:
                print(f"   🔄 Loading...")
                loaded_data = joblib.load(str(model_path))
                
                # Handle different model formats
                if hasattr(loaded_data, 'predict'):
                    model = loaded_data
                    feature_names = None
                    print(f"   ✅ SUCCESS: Standard sklearn model")
                    print(f"   📊 Model type: {type(model).__name__}")
                    
                elif isinstance(loaded_data, dict) and 'model' in loaded_data:
                    model = loaded_data['model']
                    feature_names = loaded_data.get('feature_names', None)
                    print(f"   ✅ SUCCESS: Dictionary with metadata")
                    print(f"   📊 Features: {feature_names}")
                    if 'accuracy' in loaded_data:
                        print(f"   📈 Accuracy: {loaded_data['accuracy']:.4f}")
                else:
                    print(f"   ⚠️ Unknown format: {type(loaded_data)}")
                    continue
                
                # Set feature names
                EXPECTED_FEATURES = [
                    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                    'Loan_Amount_Term', 'Credit_History', 'Property_Area'
                ]
                
                MODEL_FEATURES = feature_names if feature_names else EXPECTED_FEATURES
                
                # Test the model
                test_input = np.array([[1, 1, 1, 1, 0, 5000, 1000, 150, 360, 1, 1]])
                test_pred = model.predict(test_input)
                print(f"   ✅ Test prediction: {test_pred[0]}")
                
                return True
                
            except Exception as e:
                print(f"   ❌ Loading failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        else:
            print(f"   ⏭️  Skipping (file not found)")
    
    # If we get here, no model was loaded
    print("\n" + "="*60)
    print("❌ CRITICAL: MODEL FILE NOT FOUND!")
    print("="*60)
    print(f"Searched in:")
    for path in model_paths:
        print(f"  - {path}")
    
    print(f"\n💡 TO FIX THIS:")
    print(f"1. Make sure you've run the training notebook (main.ipynb)")
    print(f"2. Verify loan_model.pkl exists in: {current_dir}")
    print(f"3. Check the file was created by running Cell 3 in the notebook")
    print(f"4. The file should be ~2KB in size")
    
    return False

# Try to load the model on startup
print("\n" + "="*60)
print("🚀 INITIALIZING API - LOADING MODEL")
print("="*60)

model_loaded = load_model_with_fallbacks()

if not model_loaded:
    print("="*60)
    print("⚠️ WARNING: API STARTING WITHOUT MODEL!")
    print("⚠️ ALL PREDICTIONS WILL FAIL!")
    print("="*60)
    print("\n🔧 TO CREATE THE MODEL:")
    print("1. Open: server/main.ipynb")
    print("2. Run Cell 1 (install packages)")
    print("3. Run Cell 2 (import libraries)")  
    print("4. Run Cell 3 (train and save model)")
    print("5. Restart this API server")
    print("="*60 + "\n")
else:
    print("="*60)
    print("✅ MODEL LOADED SUCCESSFULLY")
    print(f"📋 Features: {MODEL_FEATURES}")
    print("="*60 + "\n")

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

# Define the expected feature order (must match training data)
# This should match the feature order from your training pipeline
EXPECTED_FEATURES = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
    'Loan_Amount_Term', 'Credit_History', 'Property_Area'
]

# Use model's feature names if available, otherwise use expected order
if feature_names and len(feature_names) > 0:
    MODEL_FEATURES = feature_names
    print(f"📋 Using model's feature order: {MODEL_FEATURES}")
else:
    MODEL_FEATURES = EXPECTED_FEATURES
    print(f"📋 Using default feature order: {MODEL_FEATURES}")

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
            "model_loaded": model is not None
        }

@app.get("/health")
async def health_check():
    """Health check endpoint with detailed model status"""
    model_status = {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else None,
        "environment": settings.ENVIRONMENT,
        "version": settings.API_VERSION,
        "current_dir": str(current_dir),
        "working_dir": str(Path.cwd()),
        "model_path": settings.MODEL_PATH,
        "model_file_exists": (current_dir / settings.MODEL_PATH).exists(),
        "features_count": len(MODEL_FEATURES) if MODEL_FEATURES else 0
    }
    
    if not model:
        model_status["error"] = "Model not loaded - please train model using main.ipynb"
        model_status["fix_instructions"] = [
            "1. Open server/main.ipynb",
            "2. Run cells 1, 2, 3 in order",
            "3. Verify loan_model.pkl is created",
            "4. Restart this API server"
        ]
    
    return model_status

@app.get("/model-info")
async def get_model_info():
    """Get detailed information about the loaded model"""
    if model is None:
        return {
            "status": "No model loaded",
            "error": "Model file not found or failed to load",
            "search_paths": [
                str(Path(settings.MODEL_PATH)),
                str(Path(current_dir) / settings.MODEL_PATH),
                str(Path(current_dir) / "loan_model.pkl"),
            ],
            "current_dir": str(current_dir),
            "files_in_dir": [f.name for f in Path(current_dir).glob("*.pkl")]
        }
    
    try:
        return {
            "model_type": str(type(model).__name__),
            "model_loaded": True,
            "feature_names": MODEL_FEATURES,
            "feature_count": len(MODEL_FEATURES),
            "has_predict": hasattr(model, 'predict'),
            "has_predict_proba": hasattr(model, 'predict_proba'),
            "model_file": settings.MODEL_PATH,
            "model_location": str(current_dir / settings.MODEL_PATH)
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict", response_model=LoanPredictionResponse)
async def predict_loan_status(loan_data: LoanApplication):
    """Predict loan approval status"""
    
    # Check if model is loaded with detailed error
    if model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Model not loaded",
                "message": "The ML model file (loan_model.pkl) is not loaded",
                "current_dir": str(current_dir),
                "expected_location": str(current_dir / "loan_model.pkl"),
                "file_exists": (current_dir / "loan_model.pkl").exists(),
                "fix": "Run the training notebook (main.ipynb) cells 1-3 to create the model",
                "then_restart": "Restart this API server after model is created"
            }
        )
    
    try:
        # Convert input data to numeric format
        numeric_data = convert_categorical_to_numeric(loan_data)
        
        # Create input array in the correct order
        input_data = np.array([[numeric_data[feature] for feature in MODEL_FEATURES]])
        
        print(f"🔍 Prediction request:")
        print(f"   Input shape: {input_data.shape}")
        print(f"   Input values: {input_data[0]}")
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Get probability
        try:
            probability = model.predict_proba(input_data)[0][1]
        except Exception as prob_error:
            print(f"⚠️ Probability calculation failed: {prob_error}")
            # Fallback probability
            probability = 0.5 if prediction == 0 else 0.8
        
        loan_status_text = "Approved" if prediction == 1 else "Not Approved"
        
        print(f"✅ Prediction: {loan_status_text} ({probability:.4f})")
        
        return LoanPredictionResponse(
            loan_status=int(prediction),
            loan_status_text=loan_status_text,
            probability=float(probability)
        )
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.options("/predict")
async def predict_options():
    """Handle OPTIONS request for CORS preflight"""
    return {"message": "OK"}

# Helper function for categorical conversion
def convert_categorical_to_numeric(loan_data: LoanApplication) -> dict:
    """Convert categorical fields to numeric values as expected by the model"""
    result = {
        'Gender': 1 if loan_data.Gender == GenderEnum.MALE else 0,
        'Married': 1 if loan_data.Married == MaritalStatusEnum.YES else 0,
        'Dependents': loan_data.Dependents,
        'Education': 1 if loan_data.Education == EducationEnum.GRADUATE else 0,
        'Self_Employed': 1 if loan_data.Self_Employed == SelfEmployedEnum.YES else 0,
        'ApplicantIncome': float(loan_data.ApplicantIncome),
        'CoapplicantIncome': float(loan_data.CoapplicantIncome),
        'LoanAmount': float(loan_data.LoanAmount),
        'Loan_Amount_Term': float(loan_data.Loan_Amount_Term),
        'Credit_History': 1 if loan_data.Credit_History == CreditHistoryEnum.YES else 0,
        'Property_Area': {
            PropertyAreaEnum.RURAL: 0,
            PropertyAreaEnum.SEMIURBAN: 1,
            PropertyAreaEnum.URBAN: 2
        }[loan_data.Property_Area]
    }
    return result

# Add a debug endpoint to check model details
@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        return {"status": "No model loaded"}
    
    try:
        return {
            "model_type": str(type(model).__name__),
            "model_loaded": True,
            "feature_names": MODEL_FEATURES,
            "feature_count": len(MODEL_FEATURES),
            "has_predict_proba": hasattr(model, 'predict_proba'),
            "model_file": settings.MODEL_PATH
        }
    except Exception as e:
        return {"error": str(e)}

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
            exit(1)
    
    print(f"🚀 Starting server on {host}:{port}")
    print(f"📚 API docs available at: http://{host}:{port}/docs")
    print(f"🏥 Health check at: http://{host}:{port}/health")
    print(f"🔮 Predict endpoint: http://{host}:{port}/predict")
    
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")