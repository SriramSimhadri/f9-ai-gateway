from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    """
    Gateway application configuration manager.
    Loads values from environment variables or a local .env file.
    """
    # 1. Server settings
    HOST: str = Field(default="0.0.0.0", description="IP address to bind the FastAPI server to")
    PORT: int = Field(default=8000, description="Port to expose the gateway server on")
    
    # 2. Valkey Database Connection Settings
    VALKEY_HOST: str = Field(default="localhost", description="Valkey/Redis server hostname")
    VALKEY_PORT: int = Field(default=6379, description="Valkey port")
    VALKEY_INDEX_NAME: str = Field(default="llm_semantic_cache", description="Index name for vector lookup")
    
    # 3. Cache & Model Tuning
    CACHE_TTL: int = Field(default=86400, description="Time-To-Live for cache entries in seconds (24 hours)")
    SIMILARITY_THRESHOLD: float = Field(default=0.92, description="Cosine similarity match threshold (0.0 to 1.0)")
    
    # 4. Upstream LLM Provider Endpoint (for local testing default is Ollama)
    UPSTREAM_LLM_URL: str = Field(
        default="http://localhost:11434/v1", 
        description="The base URL of the target LLM provider (e.g., Ollama, OpenAI)"
    )
    UPSTREAM_LLM_MODEL: str = Field(
        default="llama3", 
        description="Default model used when proxying requests upstream"
    )

    # Tell Pydantic to read environment variables and look for an optional .env file
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore" # Ignore extra env variables not declared here
    )

# Instantiate a global settings object to import across our codebase
settings = Settings()