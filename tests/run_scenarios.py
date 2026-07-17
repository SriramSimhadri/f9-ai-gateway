import asyncio
import json
import time
import httpx

GATEWAY_URL = "http://localhost:8000/v1/chat/completions"

async def fire_request(scenario_title: str, prompt: str, headers: dict = None):
    """
    Helper function to send streaming requests to the gateway and print metrics.
    """
    print(f"\n⚡ RUNNING: {scenario_title}")
    print(f"   Prompt: \"{prompt}\"")
    if headers:
        print(f"   Headers: {headers}")

    payload = {
        "messages": [{"role": "user", "content": prompt}]
    }
    
    start_time = time.perf_counter()
    chunk_count = 0
    full_text = ""
    is_ollama = False

    async with httpx.AsyncClient() as client:
        try:
            async with client.stream("POST", GATEWAY_URL, json=payload, headers=headers, timeout=30.0) as response:
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    
                    if line.startswith("data: ") and not line.endswith("[DONE]"):
                        chunk_count += 1
                        try:
                            data_json = json.loads(line[6:])
                            # Identify source using OpenAI's fingerprint formatting pattern
                            if data_json.get("system_fingerprint") == "fp_ollama":
                                is_ollama = True
                            
                            content = data_json["choices"][0]["delta"].get("content", "")
                            full_text += content
                        except Exception:
                            continue
        except Exception as e:
            print(f"❌ Connection Failure: {e}")
            return

    elapsed_time = (time.perf_counter() - start_time) * 1000
    
    # Visual Output Reporting Block
    print(f"⏱️  Finished in: {elapsed_time:.2f}ms | Received chunks: {chunk_count}")
    print(f"📥 Response Source: {'🟢 LIVE UPSTREAM (Ollama)' if is_ollama else '🚀 SEMANTIC CACHE (Valkey)'}")
    print(f"📄 Text Snippet: {full_text[:80].strip()}...")
    print("-" * 60)

async def main():
    print("=" * 60)
    print("      LAUNCHING SEMANTIC AI GATEWAY INTEGRATION SCENARIOS     ")
    print("=" * 60)

    # Scenario 1: Cold Start Miss
    await fire_request(
        "Scenario 1: Cache Miss (Cold Start)", 
        "Explain what a semantic cache gateway does inside a system architecture."
    )
    
    # Tiny pause to let background async task write to Valkey completely
    await asyncio.sleep(1)

    # Scenario 2: Perfect Hot Cache Hit
    await fire_request(
        "Scenario 2: Exact Cache Hit (Hot Read)", 
        "Explain what a semantic cache gateway does inside a system architecture."
    )

    # Scenario 3: Semantic Match (Phrasing Variation)
    await fire_request(
        "Scenario 3: Semantic Cache Hit (Fuzzy Match)", 
        "Can you describe how a semantic caching proxy works for LLMs?"
    )

    # Scenario 4: Real-time keyword filter bypass execution
    await fire_request(
        "Scenario 4: Cache Bypass Guardrail (Real-time keyword)", 
        "What is the latest stock price of Apple today?"
    )

    # Scenario 5: Manual Header Interception Override
    await fire_request(
        "Scenario 5: Explicit Header Bypass Override", 
        "Explain what a semantic cache gateway does inside a system architecture.",
        headers={"X-Bypass-Cache": "true"}
    )

    # Scenario 6: Manual Header Interception Override
    await fire_request(
        "Scenario 5: Very random scenario to test the system", 
        "Explain how motion pictures work in 2 lines"
    )

        # Scenario 6: Manual Header Interception Override
    await fire_request(
        "Scenario 5: Very random scenario to test the system - 2", 
        "Explain motion pictures"
    )

if __name__ == "__main__":
    asyncio.run(main())
