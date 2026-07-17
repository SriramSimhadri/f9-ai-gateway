import logging
import numpy as np
import valkey
from valkey.commands.search.field import VectorField, TextField
from valkey.commands.search.indexDefinition import IndexDefinition, IndexType
from valkey.commands.search.query import Query

from gateway.config import settings

logger = logging.getLogger(__name__)

class ValkeyCacheManager:
    """
    Manages connections to Valkey, index initialization, 
    and execution of semantic vector similarity queries.
    """
    def __init__(self):
        # Establish an asynchronous TCP connection pool to Valkey
        self.client = valkey.Valkey(
            host=settings.VALKEY_HOST, 
            port=settings.VALKEY_PORT, 
            decode_responses=False  # Keep raw bytes for vector precision matching
        )
        self.index_name = settings.VALKEY_INDEX_NAME
        self._initialize_vector_index()

    def _initialize_vector_index(self):
        """
        Creates the specialized HNSW Vector Index inside Valkey if it doesn't exist.
        """
        try:
            # Check if index exists by querying its metadata info
            self.client.ft(self.index_name).info()
            logger.info(f"Valkey vector index '{self.index_name}' already exists. Skipping creation.")
        except valkey.exceptions.ResponseError:
            logger.info(f"Creating new Valkey vector index: {self.index_name}")
            
            # Define fields: text field for human reference, vector field for math search
            fields = [
                TextField("prompt", weight=1.0),
                VectorField(
                    "prompt_vector",
                    "HNSW",  # Highly efficient vector search graph algorithm
                    {
                        "TYPE": "FLOAT32",
                        "DIM": 384,          # Matches all-MiniLM-L6-v2 dimensions
                        "DISTANCE_METRIC": "COSINE"  # Search strategy based on vector angles
                    }
                )
            ]
            
            # Direct Valkey to watch keys prefixing with 'cache:'
            definition = IndexDefinition(prefix=["cache:"], index_type=IndexType.HASH)
            
            # Build the physical index graph inside Valkey's memory space
            self.client.ft(self.index_name).create_index(fields=fields, definition=definition)

    def _float_list_to_bytes(self, vector: list[float]) -> bytes:
        """
        Converts a standard Python list of floats into dense binary bytes for Valkey.
        """
        return np.array(vector, dtype=np.float32).tobytes()

    async def get_semantic_match(self, vector: list[float]) -> dict | None:
        """
        Queries Valkey to find an existing cache match that is within our similarity threshold.
        """
        # Convert our query vector into dense binary bytes
        vector_bytes = self._float_list_to_bytes(vector)
        
        # Construct a Valkey vector KNN search query
        # This asks Valkey: find the nearest item ($vector), call its distance score 'score'
        query_string = f"*=>[KNN 1 @prompt_vector $vector AS score]"
        
        query = (
            Query(query_string)
            .return_fields("prompt", "response", "score")
            .dialect(2)  # Dialect 2 enables advanced vector scoring syntax
        )
        
        try:
            # Execute search
            results = self.client.ft(self.index_name).search(query, query_params={"vector": vector_bytes})
            
            if results.docs:
                doc = results.docs[0]
                similarity_score = 1.0 - float(doc.score)
                
                # ADD THIS LINE TO VISUALLY TRACK PERFORMANCE:
                print(f"--- [DEBUG] Raw Valkey Semantic Similarity Score calculated: {similarity_score:.4f} ---")
                
                if similarity_score >= settings.SIMILARITY_THRESHOLD:
                    # Helper function to decode field only if it is a byte string
                    def safe_decode(field) -> str:
                        return field.decode("utf-8") if isinstance(field, bytes) else str(field)

                    return {
                        "prompt": safe_decode(doc.prompt),
                        "response": safe_decode(doc.response),
                        "similarity": similarity_score
                    }
            return None
        except Exception as e:
            logger.error(f"Valkey lookup exception encountered: {e}")
            return None

    async def set_cache_entry(self, prompt: str, response: str, vector: list[float]):
        """
        Saves a newly generated LLM answer and its prompt embedding into Valkey with an expiration window.
        """
        # Formulate a unique hash key prefixing with 'cache:' so the index tracks it
        cache_key = f"cache:{hash(prompt)}"
        vector_bytes = self._float_list_to_bytes(vector)
        
        # Build payload mapping
        payload = {
            "prompt": prompt.encode("utf-8"),
            "response": response.encode("utf-8"),
            "prompt_vector": vector_bytes
        }
        
        try:
            # Write key-value dictionary to Valkey using a pipeline to group commands
            pipe = self.client.pipeline()
            pipe.hset(cache_key, mapping=payload)
            pipe.expire(cache_key, settings.CACHE_TTL)
            pipe.execute()
        except Exception as e:
            logger.error(f"Failed writing payload entry to Valkey: {e}")

# Global cache instance for our gateway proxy orchestration layer
valkey_cache = ValkeyCacheManager()
