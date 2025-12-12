"""Configuration manager with YAML loading and environment variable override support."""
import yaml
import os
from pathlib import Path
from typing import Optional
from .config_models import AppConfig
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """Centralized configuration manager with environment variable override support."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to config.yaml. If None, uses default location.
        """
        if config_path is None:
            # Default to config.yaml in the same directory
            config_path = Path(__file__).parent / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config: Optional[AppConfig] = None
        self.load_config()
    
    def load_config(self):
        """Load configuration from YAML and apply environment variable overrides."""
        try:
            with open(self.config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Apply environment variable overrides
            config_dict = self._apply_env_overrides(config_dict)
            
            # Validate with Pydantic
            self._config = AppConfig(**config_dict)
            logger.info(f"Configuration loaded successfully from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _apply_env_overrides(self, config_dict: dict) -> dict:
        """
        Apply environment variable overrides to config.
        
        Environment variables follow the pattern: SECTION_SUBSECTION_KEY
        Example: RAG_ENABLED=true overrides rag.enabled
        """
        
        # RAG overrides
        if os.getenv("RAG_ENABLED") is not None:
            config_dict["rag"]["enabled"] = os.getenv("RAG_ENABLED").lower() == "true"
        if os.getenv("RAG_N_RESULTS"):
            config_dict["rag"]["n_results"] = int(os.getenv("RAG_N_RESULTS"))
        if os.getenv("RAG_RELEVANCE_THRESHOLD"):
            config_dict["rag"]["relevance_threshold"] = float(os.getenv("RAG_RELEVANCE_THRESHOLD"))
        if os.getenv("RAG_COLLECTION_NAME"):
            config_dict["rag"]["collection_name"] = os.getenv("RAG_COLLECTION_NAME")
        
        # LLM overrides
        if os.getenv("DEFAULT_MODEL"):
            config_dict["llm"]["default_model"] = os.getenv("DEFAULT_MODEL")
        
        # Academic search overrides
        if os.getenv("ACADEMIC_SEARCH_ENABLED") is not None:
            config_dict["academic_search"]["enabled"] = os.getenv("ACADEMIC_SEARCH_ENABLED").lower() == "true"
        if os.getenv("ACADEMIC_SEARCH_MAX_RESULTS"):
            config_dict["academic_search"]["max_results"] = int(os.getenv("ACADEMIC_SEARCH_MAX_RESULTS"))
        
        # Document processing overrides
        if os.getenv("CHUNK_SIZE"):
            config_dict["document_processing"]["chunking"]["max_chars"] = int(os.getenv("CHUNK_SIZE"))
        if os.getenv("CHUNK_OVERLAP"):
            config_dict["document_processing"]["chunking"]["overlap"] = int(os.getenv("CHUNK_OVERLAP"))
        if os.getenv("BATCH_SIZE"):
            config_dict["document_processing"]["batch_size"] = int(os.getenv("BATCH_SIZE"))
        
        # Service URL overrides
        if os.getenv("LLM_SERVICE_URL"):
            config_dict["services"]["llm_url"] = os.getenv("LLM_SERVICE_URL")
        if os.getenv("VECTORDB_URL"):
            config_dict["services"]["vectordb_url"] = os.getenv("VECTORDB_URL")
        
        return config_dict
    
    def reload_config(self):
        """Reload configuration from file."""
        logger.info("Reloading configuration...")
        self.load_config()
    
    @property
    def config(self) -> AppConfig:
        """Get current configuration."""
        if self._config is None:
            self.load_config()
        return self._config
    
    # Convenience methods for accessing specific config sections
    
    def get_rag_config(self):
        """Get RAG configuration."""
        return self.config.rag
    
    def get_llm_config(self):
        """Get LLM configuration."""
        return self.config.llm
    
    def get_academic_search_config(self):
        """Get academic search configuration."""
        return self.config.academic_search
    
    def get_document_processing_config(self):
        """Get document processing configuration."""
        return self.config.document_processing
    
    def get_vectordb_config(self):
        """Get vector database configuration."""
        return self.config.vectordb
    
    def get_prompts_config(self):
        """Get prompts configuration."""
        return self.config.prompts
    
    def get_services_config(self):
        """Get services configuration."""
        return self.config.services
    
    def get_logging_config(self):
        """Get logging configuration."""
        return self.config.logging


# Global singleton instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get or create the global configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> AppConfig:
    """Get current application configuration (convenience function)."""
    return get_config_manager().config
