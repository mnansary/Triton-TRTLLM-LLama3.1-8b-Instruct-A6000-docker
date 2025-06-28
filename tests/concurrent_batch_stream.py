import httpx
import asyncio
from datetime import datetime

# --- Configuration ---
# The base URL of your running llm_service.
BASE_URL = "http://localhost:24434"
STREAM_URL = f"{BASE_URL}/generate_stream"

# --- Helper function to make a single streaming request and print its output ---
async def make_and_consume_stream(
    client: httpx.AsyncClient, 
    request_id: str, 
    payload: dict
):
    """
    Makes a single streaming request and prints the output chunks as they arrive.
    The request_id is used for clear logging.
    """
    print(f"[{datetime.now().isoformat()}] STARTING request: '{request_id}'")
    
    try:
        async with client.stream("POST", STREAM_URL, json=payload, timeout=120.0) as response:
            # Check for HTTP errors (like 422 Validation Error)
            response.raise_for_status()
            
            # The 'async for' loop will process each chunk of text as it's received
            async for chunk in response.aiter_text():
                # Print the raw text chunk immediately. The `flush=True` is crucial
                # to ensure the output appears in real-time without buffering.
                print(chunk, end="", flush=True)

    except httpx.HTTPStatusError as e:
        print(f"\n\n[ERROR for '{request_id}'] HTTP Error: {e.response.status_code} - {e.response.text}\n")
    except Exception as e:
        print(f"\n\n[ERROR for '{request_id}'] An unexpected error occurred: {e}\n")

    print(f"\n[{datetime.now().isoformat()}] FINISHED request: '{request_id}'\n")


# --- Main function to orchestrate the concurrent tests ---
async def main():
    """
    Simulates multiple concurrent users by creating and running several
    streaming requests in parallel using asyncio.gather.
    """
    print("--- Starting Concurrent Streaming Test ---")
    print(f"Targeting API at: {STREAM_URL}\n")
    
    # --- Define 4 different request payloads to simulate 4 concurrent users ---
    # This matches the `max_batch_size=4` of the compiled model.
    request_payloads = [
        {
            "request_id": "Poem-About-Mars",
            "payload": {
                "prompt": "Write a short, four-line poem about the planet Mars.",
                "max_tokens": 60, "temperature": 0.7
            }
        },
        {
            "request_id": "Largest-Cities",
            "payload": {
                "prompt": "List the 5 largest cities in the world by population, starting with the largest.",
                "max_tokens": 100, "temperature": 0.2
            }
        },
        {
            "request_id": "Quantum-Entanglement",
            "payload": {
                "prompt": "Explain the concept of quantum entanglement in three simple sentences.",
                "max_tokens": 150, "temperature": 0.5
            }
        },
        {
            "request_id": "Spicy-Cookie-Recipe",
            "payload": {
                "prompt": "Create a simple recipe for a spicy chocolate cookie.",
                "max_tokens": 120, "temperature": 0.9
            }
        }
    ]

    # Use a single httpx.AsyncClient for all requests for efficiency
    async with httpx.AsyncClient() as client:
        # --- Create a list of asyncio Tasks ---
        # This is the key to concurrency. We are preparing all our requests
        # before running any of them.
        tasks = []
        for req in request_payloads:
            task = make_and_consume_stream(
                client=client, 
                request_id=req["request_id"], 
                payload=req["payload"]
            )
            tasks.append(task)
        
        # --- Run all tasks concurrently ---
        # asyncio.gather runs all the prepared tasks in parallel and waits for
        # them all to complete. This is what sends the flood of requests to
        # your server, triggering the in-flight batching.
        print(f"Dispatching {len(tasks)} requests concurrently...\n" + "="*40)
        await asyncio.gather(*tasks)

    print("="*40 + "\n--- Concurrent Streaming Test Finished ---")


# --- Standard entry point ---
if __name__ == "__main__":
    asyncio.run(main())