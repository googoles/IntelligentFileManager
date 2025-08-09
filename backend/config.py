"""
Configuration settings for the Research File Manager backend.

This module handles environment variables, database settings, and other
configuration options for the application.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import logging


class Config:
    """
    Configuration class for the Research File Manager backend.
    
    Handles environment variables and provides default values for
    database connections, file processing, and other settings.
    """
    
    # Database settings
    DATABASE_URL: Optional[str] = os.getenv('DATABASE_URL')
    
    # If no DATABASE_URL is provided, use SQLite in the data directory
    if not DATABASE_URL:
        DATA_DIR = Path(os.getenv('DATA_DIR', 'data'))
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        DB_DIR = DATA_DIR / 'db'
        DB_DIR.mkdir(parents=True, exist_ok=True)
        
        DATABASE_URL = f"sqlite:///{DB_DIR}/research.db"
    
    # Logging configuration
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE: Optional[str] = os.getenv('LOG_FILE')
    
    # File processing settings
    MAX_FILE_SIZE: int = int(os.getenv('MAX_FILE_SIZE', 50 * 1024 * 1024))  # 50MB
    SUPPORTED_TEXT_EXTENSIONS: list = [
        '.txt', '.md', '.py', '.js', '.java', '.cpp', '.c', '.h',
        '.html', '.css', '.xml', '.json', '.yaml', '.yml', '.ini',
        '.cfg', '.conf', '.log', '.sql', '.sh', '.bat', '.ps1'
    ]
    
    SUPPORTED_DOCUMENT_EXTENSIONS: list = [
        '.pdf', '.doc', '.docx', '.odt', '.rtf'
    ]
    
    SUPPORTED_DATA_EXTENSIONS: list = [
        '.csv', '.xlsx', '.xls', '.json', '.xml', '.parquet', '.feather'
    ]
    
    SUPPORTED_IMAGE_EXTENSIONS: list = [
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.svg', '.webp'
    ]
    
    # Text processing settings
    MAX_CONTENT_LENGTH: int = int(os.getenv('MAX_CONTENT_LENGTH', 10000))  # Characters
    CHUNK_SIZE: int = int(os.getenv('CHUNK_SIZE', 500))  # For text chunking
    
    # Vector embedding settings
    EMBEDDING_MODEL: str = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    EMBEDDING_DIMENSION: int = int(os.getenv('EMBEDDING_DIMENSION', 384))
    
    # ChromaDB settings
    CHROMA_DB_PATH: str = os.getenv('CHROMA_DB_PATH', str(DATA_DIR / 'db' / 'chroma'))
    
    # LLM (Ollama) settings
    LLM_ENABLED: bool = os.getenv('LLM_ENABLED', 'True').lower() in ('true', '1', 'yes')
    LLM_MODEL_NAME: str = os.getenv('LLM_MODEL_NAME', 'llama3.2:3b')
    LLM_TIMEOUT: int = int(os.getenv('LLM_TIMEOUT', 60))  # seconds
    LLM_MAX_CONTEXT_LENGTH: int = int(os.getenv('LLM_MAX_CONTEXT_LENGTH', 4000))  # characters
    LLM_TEMPERATURE: float = float(os.getenv('LLM_TEMPERATURE', 0.3))
    LLM_FALLBACK_ENABLED: bool = os.getenv('LLM_FALLBACK_ENABLED', 'True').lower() in ('true', '1', 'yes')
    LLM_MAX_RETRIES: int = int(os.getenv('LLM_MAX_RETRIES', 3))
    LLM_CHUNK_SIZE: int = int(os.getenv('LLM_CHUNK_SIZE', 2000))
    
    # Project template settings
    DEFAULT_PROJECT_TEMPLATE: str = os.getenv('DEFAULT_PROJECT_TEMPLATE', 'research')
    
    PROJECT_TEMPLATES: Dict[str, list] = {
        'research': [
            'literature',
            'data/raw',
            'data/processed',
            'code',
            'results/figures',
            'results/tables',
            'drafts',
            'notes'
        ],
        'minimal': [
            'input',
            'output',
            'workspace'
        ],
        'analysis': [
            'data',
            'scripts',
            'outputs',
            'reports',
            'figures'
        ]
    }
    
    # File organization rules
    FILE_ORGANIZATION_RULES: Dict[str, list] = {
        'documents': SUPPORTED_DOCUMENT_EXTENSIONS + ['.txt', '.md'],
        'data': SUPPORTED_DATA_EXTENSIONS,
        'code': ['.py', '.js', '.r', '.ipynb', '.java', '.cpp', '.c'],
        'images': SUPPORTED_IMAGE_EXTENSIONS,
        'results': []  # Will be populated by keyword matching
    }
    
    # Keywords for result file detection
    RESULT_KEYWORDS: list = [
        'result', 'output', 'figure', 'plot', 'graph', 'chart',
        'analysis', 'report', 'summary', 'conclusion'
    ]
    
    @classmethod
    def get_all_supported_extensions(cls) -> list:
        """Get all supported file extensions."""
        all_extensions = []
        all_extensions.extend(cls.SUPPORTED_TEXT_EXTENSIONS)
        all_extensions.extend(cls.SUPPORTED_DOCUMENT_EXTENSIONS)
        all_extensions.extend(cls.SUPPORTED_DATA_EXTENSIONS)
        all_extensions.extend(cls.SUPPORTED_IMAGE_EXTENSIONS)
        return list(set(all_extensions))  # Remove duplicates
    
    @classmethod
    def is_supported_file(cls, filename: str) -> bool:
        """Check if a file is supported based on its extension."""
        file_ext = Path(filename).suffix.lower()
        return file_ext in cls.get_all_supported_extensions()
    
    @classmethod
    def get_file_category(cls, filename: str) -> str:
        """Get the category of a file based on its extension."""
        file_ext = Path(filename).suffix.lower()
        filename_lower = filename.lower()
        
        # Check for result keywords first
        if any(keyword in filename_lower for keyword in cls.RESULT_KEYWORDS):
            return 'results'
        
        # Check by extension
        for category, extensions in cls.FILE_ORGANIZATION_RULES.items():
            if file_ext in extensions:
                return category
        
        return 'unsorted'
    
    @classmethod
    def setup_logging(cls) -> None:
        """Setup logging configuration."""
        log_level = getattr(logging, cls.LOG_LEVEL.upper(), logging.INFO)
        
        # Configure logging format
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Setup handlers
        handlers = [logging.StreamHandler()]
        
        if cls.LOG_FILE:
            # Ensure log directory exists
            log_path = Path(cls.LOG_FILE)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(cls.LOG_FILE))
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=handlers,
            force=True  # Override existing configuration
        )
        
        # Set specific logger levels
        logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
        logging.getLogger('chromadb').setLevel(logging.WARNING)


class DevelopmentConfig(Config):
    """Development configuration with debug settings."""
    
    LOG_LEVEL: str = 'DEBUG'
    
    # Enable SQL query logging in development
    DATABASE_ECHO: bool = True
    
    # Smaller limits for testing
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    MAX_CONTENT_LENGTH: int = 5000


class ProductionConfig(Config):
    """Production configuration with optimized settings."""
    
    LOG_LEVEL: str = 'INFO'
    DATABASE_ECHO: bool = False
    
    # Higher limits for production
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    MAX_CONTENT_LENGTH: int = 50000


class TestingConfig(Config):
    """Testing configuration with test database."""
    
    LOG_LEVEL: str = 'WARNING'
    
    # Use in-memory SQLite for tests
    DATABASE_URL: str = 'sqlite:///:memory:'
    
    # Small limits for fast tests
    MAX_FILE_SIZE: int = 1024 * 1024  # 1MB
    MAX_CONTENT_LENGTH: int = 1000
    CHUNK_SIZE: int = 100


def get_config() -> Config:
    """
    Get configuration based on environment.
    
    Returns:
        Configuration instance based on ENVIRONMENT variable
    """
    env = os.getenv('ENVIRONMENT', 'development').lower()
    
    config_map = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'testing': TestingConfig
    }
    
    config_class = config_map.get(env, DevelopmentConfig)
    config = config_class()
    
    # Setup logging with the selected configuration
    config.setup_logging()
    
    return config


# Global config instance
config = get_config()


if __name__ == "__main__":
    # Test configuration
    print("Configuration Test")
    print(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    print(f"Database URL: {config.DATABASE_URL}")
    print(f"Log Level: {config.LOG_LEVEL}")
    print(f"Max File Size: {config.MAX_FILE_SIZE / (1024*1024):.1f}MB")
    print(f"Supported Extensions: {len(config.get_all_supported_extensions())}")
    print(f"Example file category for 'results.csv': {config.get_file_category('results.csv')}")
    print(f"Example file category for 'analysis_results.pdf': {config.get_file_category('analysis_results.pdf')}")