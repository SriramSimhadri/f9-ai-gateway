import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

from gateway.config import settings

class EmbeddingEngine:
    """
    Singleton Semantic Embedding Engine.
    Handles the initialization of our local all-MiniLM-L6-v2 transformer model
    and safely generates text embeddings without blocking the async event loop.
    """
    _instance = None
    _model = None
    _executor = None

    def __new__(cls):
        # Implementation of the Singleton pattern
        if cls._instance is None:
            cls._instance = super(EmbeddingEngine, cls).__new__(cls)
            # 1. Load the model once into memory
            # It targets the model downloaded during our Docker build step
            cls._model = SentenceTransformer("all-MiniLM-L6-v2")
            
            # 2. Allocate a dedicated worker pool for heavy CPU math tasks
            cls._executor = ThreadPoolExecutor(max_workers=4)
        return cls._instance

    def _sync_encode(self, text: str) -> List[float]:
        """
        Synchronous wrapper that performs the actual vector mathematics.
        """
        # Convert text into a NumPy array of 384 numbers
        embedding: np.ndarray = self._model.encode(text, normalize_embeddings=True)
        
        # Valkey requires a standard Python list of floats or binary bytes
        return embedding.tolist()

    async def get_embedding(self, text: str) -> List[float]:
        """
        Asynchronously generates a 384-dimensional vector embedding for a given text.
        Safely offloads CPU-bound calculations to avoid blocking incoming proxy connections.
        """
        # Clean the input text to prevent whitespace mutations affecting coordinates
        cleaned_text = text.strip()
        
        # Grab the running FastAPI event loop
        loop = asyncio.get_running_loop()
        
        # Offload the math: run_in_executor sends _sync_encode to our ThreadPoolExecutor.
        # This keeps the main FastAPI event loop free to handle other users.
        embedding = await loop.run_in_executor(
            self._executor, 
            self._sync_encode, 
            cleaned_text
        )
        return embedding

# Instantiate a single global instance for the gateway package
embedding_engine = EmbeddingEngine()