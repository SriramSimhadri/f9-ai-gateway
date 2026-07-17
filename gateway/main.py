import json
import re
from fastapi import FastAPI, Request, Header
from fastapi.responses import StreamingResponse
import httpx

from gateway.config import settings
from gateway.embedding import embedding_engine
from gateway.cache import valkey_cache

app = FastAPI(title="AI Gateway Semantic Cache Proxy")

# Compiled regex to quickly scan for temporal/real-time keywords
TEMPORAL_KEYWORDS_REGEX = re.compile(
    r"\b(today|current|now|weather|stock|latest|news|at the moment|ticker)\b", 
    re.IGNORECASE
)

def _is_cache_bypass_required(prompt: str, x_bypass_cache: str | None) -> bool:
    """
    Evaluates whether the request should skip the vector cache.
    """
    if x_bypass_cache and x_bypass_cache.lower() == "true":
        return True
    return bool(TEMPORAL_KEYWORDS_REGEX.search(prompt))

async def _simulate_stream_reply(cached_text: str):
    """
    Chops cached text blocks into simulated SSE chunk streams.
    This provides an identical experience to client SDKs expecting real-time typing.
    """
    # Chop text into small pieces of word bundles
    chunks = [cached_text[i:i+8] for i in range(0, len(cached_text), 8)]
    for chunk in chunks:
        # Wrap chunk inside standard OpenAI chunk completion specifications
        chunk_payload = {
            "choices": [{"delta": {"content": chunk}, "finish_reason": None}]
        }
        yield f"data: {json.dumps(chunk_payload)}\n\n"
    
    # Send the mandatory termination chunk signal to close the client's listener stream
    final_payload = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
    yield f"data: {json.dumps(final_payload)}\n\n"
    yield "data: [DONE]\n\n"

async def _proxy_and_cache_stream(request_body: dict, prompt_text: str, prompt_vector: list[float] | None, skip_cache: bool = False):
    """
    Proxies request to the live LLM, streams chunks to the client instantly, 
    and conditionally copies data to Valkey depending on skip_cache flag.
    """
    request_body["stream"] = True
    if "model" not in request_body:
        request_body["model"] = settings.UPSTREAM_LLM_MODEL

    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST", 
            f"{settings.UPSTREAM_LLM_URL}/chat/completions", 
            json=request_body,
            timeout=60.0
        ) as response:
            
            full_response_text = ""
            async for raw_line in response.aiter_lines():
                if not raw_line.strip():
                    continue
                
                yield f"{raw_line}\n\n"
                
                # Only compile the string buffer if we actually intend to save it
                if not skip_cache and raw_line.startswith("data: ") and not raw_line.endswith("[DONE]"):
                    try:
                        data_json = json.loads(raw_line[6:])
                        content = data_json["choices"][0]["delta"].get("content", "")
                        full_response_text += content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
            
            # Asynchronously commit raw string + vector ONLY if bypass was not requested
            if not skip_cache and full_response_text and prompt_vector:
                await valkey_cache.set_cache_entry(prompt_text, full_response_text, prompt_vector)


@app.post("/v1/chat/completions")
async def handle_chat_completion(
    request: Request,
    x_bypass_cache: str | None = Header(default=None, alias="X-Bypass-Cache")
):
    """
    Main OpenAI-compatible Reverse Proxy routing hook.
    """
    body = await request.json()
    messages = body.get("messages", [])
    prompt_text = messages[-1].get("content", "") if messages else ""
    
    # 1. Evaluate Guardrail Bypass Strategies
    if _is_cache_bypass_required(prompt_text, x_bypass_cache):
        print("\n[GUARDRAIL BYPASS] Bypass condition met. Routing stream directly to LLM without cache sync...")
        return StreamingResponse(
            _proxy_and_cache_stream(body, prompt_text, prompt_vector=None, skip_cache=True),
            media_type="text/event-stream"
        )

    # 2. Compute Vector Embeddings using our background worker ThreadPool
    prompt_vector = await embedding_engine.get_embedding(prompt_text)
    
    # 3. Query Valkey Vector Graph
    cache_match = await valkey_cache.get_semantic_match(prompt_vector)
    
    if cache_match:
        print(f"\n[CACHE HIT] Found match with Similarity Score: {cache_match['similarity']:.4f}")
        return StreamingResponse(
            _simulate_stream_reply(cache_match["response"]), 
            media_type="text/event-stream"
        )
    
    print("\n[CACHE MISS] No match found in Valkey. Routing request directly to upstream LLM...")
    return StreamingResponse(
        _proxy_and_cache_stream(body, prompt_text, prompt_vector, skip_cache=False),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("gateway.main:app", host=settings.HOST, port=settings.PORT, reload=True)
