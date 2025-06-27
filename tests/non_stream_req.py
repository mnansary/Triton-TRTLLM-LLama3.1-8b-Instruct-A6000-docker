import httpx
import asyncio
import json

# The endpoint for streaming generation
url = "http://localhost:24434/generate_stream"

async def stream_response():
    payload = {
        "prompt": "What are the three most important features of the NVIDIA A6000 GPU for AI?",
        "temperature": 0.2,
        "max_tokens": 150
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        print("--- Sending streaming request ---")
        try:
            async with client.stream("POST", url, json=payload) as response:
                # Ensure the server responded with a success status code
                response.raise_for_status()
                
                print("--- Receiving stream ---")
                async for line in response.aiter_lines():
                    # Check if the line has content after stripping whitespace
                    if not line.strip():
                        continue  # Skip empty lines

                    try:
                        # Attempt to parse the line as JSON
                        token_data = json.loads(line)
                        # Safely get the 'text' key, defaulting to an empty string
                        print(token_data.get("text", ""), end="", flush=True)
                    
                    except json.JSONDecodeError:
                        # This will catch any line that is not valid JSON.
                        # For debugging, we can print a warning.
                        # In production, you might just want to 'continue' silently.
                        print(f"\n[Warning] Received a non-JSON line from stream: '{line}'")
                        continue

        except httpx.RequestError as e:
            print(f"\nAn error occurred while requesting {e.request.url!r}.")
            return
        except httpx.HTTPStatusError as e:
            print(f"\nError response {e.response.status_code} while requesting {e.request.url!r}.")
            # You can print the response body for more details
            print(f"Response body: {e.response.text}")
            return


    print("\n--- Stream finished ---")

if __name__ == "__main__":
    asyncio.run(stream_response())