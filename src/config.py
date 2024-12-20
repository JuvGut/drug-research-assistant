# src/config.py

import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class Config:
    # Model configurations
    MODEL_ID: str = "facebook/opt-125m" #"TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Performance settings
    BATCH_SIZE: int = 32
    NUM_WORKERS: int = 4
    MAX_CONCURRENT_REQUESTS: int = 10
    REQUEST_TIMEOUT: int = 30
    
    # API configurations
    PUBMED_BASE_URL: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    PUBMED_API_KEY: str = field(default_factory=lambda: os.getenv("PUBMED_API_KEY"))
    API_RETRY_ATTEMPTS: int = 3
    API_RETRY_DELAY: int = 1

    # Analysis parameters
    MAX_RESULTS: int = 10
    DEFAULT_SIMILARITY_K: int = 3

    # Query types
    ANALYSIS_TYPES: List[str] = field(default_factory=lambda: [
        "mechanism of action",
        "clinical trials summary",
        "adverse effects",
        "drug interactions",
        "therapeutic applications",
        "pharmacokinetics",
        "dosage recommendations",
        "contraindications",
        "patient outcomes",
        "cost effectiveness"
    ])
    
    # Cache settings
    CACHE_DIR: str = ".cache"
    CACHE_EXPIRY: int = 3600
    MAX_CACHE_SIZE: int = 1000
    MEMORY_CACHE_SIZE: int = 100
    
    # Chemical structure analysis settings (using RDKit)
    ENABLE_STRUCTURE_ANALYSIS: bool = True
    SIMILARITY_THRESHOLD: float = 0.7