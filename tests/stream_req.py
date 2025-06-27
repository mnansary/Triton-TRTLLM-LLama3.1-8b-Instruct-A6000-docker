import httpx
import asyncio

# The endpoint for streaming generation
url = "http://localhost:24434/generate_stream"

async def stream_response():
    """
    Connects to the streaming endpoint and correctly processes the raw text stream.
    """
    payload = {
        "prompt": "What are the three most important features of the NVIDIA A6000 GPU for AI?",
        "temperature": 0.2,
        "max_tokens": 150
    }

    print("--- Sending streaming request ---")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                print("--- Receiving stream ---", flush=True)

                # Use aiter_text() to get raw chunks of text as they arrive.
                # This is the correct method since the server sends raw strings.
                async for text_chunk in response.aiter_text():
                    # Print the chunk directly to the console without a newline.
                    # flush=True ensures it appears immediately.
                    print(text_chunk, end="", flush=True)

    except httpx.RequestError as e:
        print(f"\n[Error] An error occurred while requesting {e.request.url!r}: {e}")
    except httpx.HTTPStatusError as e:
        print(f"\n[Error] Received status code {e.response.status_code} for {e.request.url!r}.")
        print(f"Response body: {e.response.text}")

    print("\n--- Stream finished ---")

if __name__ == "__main__":
    asyncio.run(stream_response())