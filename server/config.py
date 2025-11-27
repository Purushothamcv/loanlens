import os
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    def __init__(self):
        # Server Configuration
        self.PORT = int(os.getenv("PORT", 8000))
        self.HOST = os.getenv("HOST", "127.0.0.1")  # Changed to localhost for development
        
        # API Configuration
        self.API_TITLE = os.getenv("API_TITLE", "Loan Prediction API")
        self.API_DESCRIPTION = os.getenv("API_DESCRIPTION", "API for predicting loan approval status using machine learning")
        self.API_VERSION = os.getenv("API_VERSION", "1.0.0")
        
        # CORS Configuration - Allow all localhost origins in development
        self.CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001,http://localhost:5173,http://127.0.0.1:5173,http://localhost:8080,http://127.0.0.1:8080")
        
        # Model Configuration
        self.MODEL_PATH = os.getenv("MODEL_PATH", "loan_model.pkl")
        
        # Environment
        self.ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
        self.DEBUG = os.getenv("DEBUG", "true").lower() == "true"
    
    @property
    def cors_origins_list(self) -> List[str]:
        # In development, allow all origins for easier testing
        if self.ENVIRONMENT == "development":
            return ["*"]
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

settings = Settings()
