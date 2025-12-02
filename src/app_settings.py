from pydantic_settings import BaseSettings

"""Automatically loads environment variables from a .env file"""
class Settings(BaseSettings):
    
    # API Keys
    frontend_api_key: str
    llm_service_api_key: str
    
    # LLM Service Configuration
    llm_service_url: str
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
