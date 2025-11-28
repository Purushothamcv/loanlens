import os
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    def __init__(self):
        # Server Configuration
        self.PORT = int(os.getenv("PORT", 8000))
        self.HOST = os.getenv("HOST", "0.0.0.0")  # Allow all connections in dev
        
        # API Configuration
        self.API_TITLE = os.getenv("API_TITLE", "Loan Prediction API")
        self.API_DESCRIPTION = os.getenv("API_DESCRIPTION", "API for predicting loan approval status using machine learning")
        self.API_VERSION = os.getenv("API_VERSION", "1.0.0")
        
        # CORS Configuration - Be more permissive in development
        default_cors = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173,http://localhost:8080,http://127.0.0.1:8080"
        self.CORS_ORIGINS = os.getenv("CORS_ORIGINS", default_cors)
        
        # Model Configuration
        self.MODEL_PATH = os.getenv("MODEL_PATH", "loan_model.pkl")
        
        # Environment
        self.ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
        self.DEBUG = os.getenv("DEBUG", "true").lower() == "true"
    
    @property
    def cors_origins_list(self) -> List[str]:
        # In development, allow all origins for easier testing
        if self.ENVIRONMENT == "development":
            origins = [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
            # Add localhost variations
            origins.extend([
                "http://localhost:3000",
                "http://127.0.0.1:3000",
                "http://localhost:5173", 
                "http://127.0.0.1:5173",
                "*"  # Allow all in development
            ])
            return list(set(origins))  # Remove duplicates
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

settings = Settings()
