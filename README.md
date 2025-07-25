# High-Performance Llama 3.1 8B API with 96k Context on Triton

This repository provides a complete, step-by-step guide to deploying Meta's Llama 3.1 8B Instruct model with a 96,000 token context length. The deployment uses NVIDIA TensorRT-LLM for high-performance inference, served via the Triton Inference Server, and is fronted by a production-grade FastAPI application for easy integration.

The final result is a robust, streaming-capable API service that can handle multiple concurrent users efficiently on a single NVIDIA A6000 GPU.

## Table of Contents

- [**Introduction**](#introduction)
    - [**From Prototype to Production**](#from-prototype-to-production)
    - [**System Architecture**](#system-architecture)
    - [**A Deep Dive into Quantization**](#a-deep-dive-into-quantization)
    - [**GPU Memory & Performance Demystified**](#gpu-memory--performance-demystified)
    - [**Concurrency, Batching, and a Tale of Two Elevators**](#concurrency-batching-and-a-tale-of-two-elevators)
- [**Installation**](#installation)
    -   [**Step 1: System Prerequisites**](#step-1-system-prerequisites)
        -   [1.1 Install Docker](#11-install-docker)
        -   [1.2 Install NVIDIA Container Toolkit](#12-install-nvidia-container-toolkit)
    -   [**Step 2: NVIDIA NGC Authentication**](#step-2-nvidia-ngc-authentication)
        -   [2.1 Get Your NGC API Key](#21-get-your-ngc-api-key)
        -   [2.2 Log in via Docker](#22-log-in-via-docker)
    -   [**Step 3: Download Triton Server Image**](#step-3-download-triton-server-image)
    -   [**Step 4: Download the Llama 3.1 Model**](#step-4-download-the-llama-31-model)
    -   [**Step 5: Build the TensorRT-LLM Engine**](#step-5-build-the-tensorrt-llm-engine)  
        -   [5.1 Convert Hugging Face Model to TensorRT-LLM Checkpoint](#51-convert-hugging-face-model-to-tensorrt-llm-checkpoint)
        -   [5.2 Build the Optimized Engine from the Checkpoint](#52-build-the-optimized-engine-from-the-checkpoint)
    -   [**Step 6: Prepare and Launch Triton Server**](#step-6-prepare-and-launch-triton-server)
        -   [6.1 Prepare the Triton Model Repository](#61-prepare-the-triton-model-repository)
        -   [6.2 Correct Host File Permissions](#62-correct-host-file-permissions)
        -   [6.3 Launch the Triton Server Container](#63-launch-the-triton-server-container)

- [**Inference**](#inference)
    -   [**Step 0: The "Why": FastAPI Service vs. Direct Triton Access**](#step-0-the-why-fastapi-service-vs-direct-triton-access)
    -   [**Step 1: Run the Production API Service**](#step-1-run-the-production-api-service)
        -   [1.1 Prerequisites](#11-prerequisites)
        -   [1.2 Create Conda Environment](#12-create-conda-environment)
        -   [1.3 Install Dependencies](#13-install-dependencies)
        -   [1.4 Configure the Service](#14-configure-the-service)
        -   [1.5 Run the API Server](#15-run-the-api-server)
        -   [1.6 Test the API Endpoints](#16-test-the-api-endpoints)
    - [**Step 2:Turning it into a managed, reliable background service.**](#step-2turning-it-into-a-managed-reliable-background-service)
        -   [Step 2.1: Create the `systemd` Service File](#step-21-create-the-systemd-service-file)
        -   [Step 2.2: Install and Manage the Service](#step-22-install-and-manage-the-service)
        -   [Step 2.3: Controlling Your Service (The Commands You Wanted)](#step-23-controlling-your-service-the-commands-you-wanted)
        -   [Step 2.4: Viewing Logs](#step-24-viewing-logs)
    - [**Step 3: Isolated Inference: Consuming the API from Any Application**](#step-3-isolated-inference-consuming-the-api-from-any-application)
        - [3.1. Python: Using `requests` and `httpx`](#31-python-using-requests-and-httpx)
        - [3.2. JavaScript (Browser / Node.js): Using the `fetch` API (NOT TESTED:  BUT YOU WILL GET THE IDEA RIGHT!!)](#32-javascript-browser--nodejs-using-the-fetch-api-not-tested--but-you-will-get-the-idea-right)
        - [3.3 Using All Available Parameters](#33-using-all-available-parameters)


- [**Insights**](#insights) 
    - [Unit Testing the API: Ensuring Reliability and Understanding Behavior](#unit-testing-the-api-ensuring-reliability-and-understanding-behavior)



> Note: A6000 and L40s, step 5.1,5.2 and 6.1 are different but you can follow along . You will find links for redirection and getting back.

---

# Introduction
## From Prototype to Production

For anyone coming from a research or prototyping background, the Hugging Face `transformers` library is the gold standard. It is an incredible tool for its ease of use, extensive model hub, and vibrant community. A simple call can get a powerful model running in minutes:

```python
from transformers import pipeline

# This is familiar, easy, and powerful for single-user scenarios.
generator = pipeline("text-generation", model="meta-llama/Meta-Llama-3.1-8B-Instruct")
result = generator("The future of AI is...")
```

So, why not just wrap this `pipeline` in a FastAPI server and call it a day?

Because a **prototype** and a **production service** solve fundamentally different problems. A prototype needs to *work*. A production service needs to *perform*—reliably, efficiently, and for many users at once.

---

### The Production Challenge: The Limits of a Simple Pipeline

Imagine your service gets 10 user requests at the same time. A basic Hugging Face implementation, by default, will handle them **sequentially**. This creates two major problems: high latency and wasted GPU resources.

We can visualize this blocking behavior:

```mermaid
sequenceDiagram
    participant User 1
    participant User 2
    participant Server

    Note over User 1, User 2: Both send requests at the same time.

    User 1->>+Server: /generate (Prompt A)
    Note right of Server: GPU is now busy with Prompt A.
    User 2->>+Server: /generate (Prompt B)
    Note right of Server: Request B is queued. User 2 must wait.

    Server-->>-User 1: Response for A
    Note right of Server: GPU is now free.
    Note right of Server: Server starts processing the next item in the queue.
    Note right of Server: GPU is now busy with Prompt B.

    Server-->>-User 2: Response for B
```
This leads to:
1.  **High Latency:** The last user in the queue waits for everyone else to finish. Their experience is terrible.
2.  **Wasted GPU Potential:** The GPU, a massively parallel processor, is often underutilized, only working on one sequence at a time.

To solve this, we introduce a specialized stack: **TensorRT-LLM** and the **Triton Inference Server**.

---

### The Solution, Component by Component

Instead of one tool that does everything reasonably well, we use two specialized tools that do their individual jobs with maximum performance.

### 1. What is TensorRT-LLM? (The Engine Optimizer)

**Analogy:** Think of the default PyTorch model from Hugging Face as a high-quality, road-legal engine in a luxury car. It's powerful and versatile. **TensorRT-LLM is a Formula 1 engineering team that rebuilds that engine specifically for a race track (your NVIDIA A6000 GPU).**

The F1 team does several things:
*   **Strips Out Unnecessary Parts:** Removes parts of the model code only used for training, not inference.
*   **Fuses Operations (Kernel Fusion):** Instead of telling the GPU "do task A, then task B, then task C," it fuses them into a single, highly optimized instruction: "do task ABC." This dramatically reduces overhead.
*   **Uses Lighter Materials (Quantization):** It converts model weights from heavy `FP16` numbers into lighter `INT8` numbers. This means the model takes up less VRAM and calculations are much faster on the GPU's specialized INT8 Tensor Cores. 
*   **Optimizes for the Track (Hardware-Specific Compilation):** The final engine is compiled *specifically* for the architecture of your A6000 GPU, taking advantage of its unique features. It's not a general-purpose engine anymore; it's a bespoke racing engine.

The output of `trtllm-build` is a `.engine` file. This is no longer a flexible PyTorch model; it's a rigid, black-box, speed-optimized inference engine.

### 2. What is Triton Inference Server? (The Airport Traffic Controller)

**Analogy:** If TensorRT-LLM builds the lightning-fast race car, **Triton is the entire airport operations team managing hundreds of planes (user requests) landing on and taking off from the runways (your GPU).**

A single car on a track is fast, but an airport needs to handle massive, concurrent traffic safely and efficiently. Triton's job is to:

*   **Load Models (`.engine` files):** It loads our TensorRT-LLM engine into GPU memory, ready to receive requests.
*   **Manage Concurrent Requests:** It's built from the ground up to accept thousands of simultaneous connections without breaking a sweat.
*   **Dynamic Batching:** This is its most critical feature. Instead of waiting for a "batch" of 4 requests to arrive before starting (static batching), Triton can dynamically group requests that are in-flight, maximizing GPU utilization. 
*   **Orchestrate Pipelines (Ensembles):** A raw request is just text. The engine needs token IDs. Triton manages the entire pipeline, as defined in our model repository.

Here is what Triton's internal workflow (our "ensemble" model) looks like:

```mermaid
graph TD
    A["Incoming Request <br> (raw text, temp, etc.)"] --> B(preprocessing);
    B --> C{"tensorrt_llm_bls <br> (Business Logic)"};
    C --> D["tensorrt_llm <br> GPU Inference"];
    D --> E(postprocessing);
    E --> F["Final Response <br> (generated text stream)"];
```
Triton manages sending the data between these steps automatically. This lets us use simple Python for pre/post-processing while using the hyper-optimized engine for the heavy lifting on the GPU.

---

### The Production Solution in Action

With our full stack, the workflow for handling 2 concurrent users looks completely different and far more efficient.

```mermaid
sequenceDiagram
    participant User 1
    participant User 2
    participant FastAPI
    participant Triton Server
    participant GPU

    Note over User 1, User 2: Both send requests at the same time.

    User 1->>+FastAPI: /generate (Prompt A)
    FastAPI->>+Triton Server: gRPC Request A
    
    User 2->>+FastAPI: /generate (Prompt B)
    FastAPI->>+Triton Server: gRPC Request B

    Note right of Triton Server: Triton receives both requests and uses<br>In-Flight Batching to schedule them.

    loop Generation Steps
        Triton Server->>+GPU: Process fused batch (part of A + part of B)
        GPU-->>-Triton Server: Return generated tokens for A and B
        
        Triton Server-->>-FastAPI: Stream token for A
        FastAPI-->>-User 1: Stream token for A
        
        Triton Server-->>-FastAPI: Stream token for B
        FastAPI-->>-User 2: Stream token for B
    end

```

### Final Comparison

| Feature                 | Hugging Face `pipeline` (Default) | Triton + TensorRT-LLM Stack                                  |
| :---------------------- | :-------------------------------- | :----------------------------------------------------------- |
| **Primary Goal**        | Ease of use, rapid prototyping.   | High performance, high concurrency, production stability.    |
| **Key Technology**      | PyTorch / TensorFlow              | **TensorRT-LLM:** Kernel Fusion, Quantization<br>**Triton:** In-Flight Batching, C++ Server |
| **Performance**         | Baseline                          | **State-of-the-Art.** Up to 8x faster than baseline on NVIDIA GPUs. |
| **Concurrency**         | **Sequential.** Processes one request at a time. | **Truly Concurrent.** Manages thousands of simultaneous requests. |
| **Memory Usage**        | High (FP16/FP32 weights)          | **Optimized.** INT8 weights + Paged KV Cache reduce VRAM pressure significantly. |
| **Deployment Model**    | Simple Python script              | Decoupled client-server architecture.                        |

**Conclusion:** While Hugging Face is the perfect tool to start your journey, it's not designed for the demands of a high-traffic, low-latency production environment. By graduating to a specialized stack like **TensorRT-LLM** for optimization and **Triton** for serving, you are moving from building a working car to engineering a complete, high-performance transportation system.


## System Architecture

Understanding how a user's request travels through the system is key to debugging issues and appreciating the role of each component. Our service is not a single program, but a pipeline of specialized applications working together. This document traces that path from start to finish.

### The Journey of a Request

At a high level, the system is a client-server architecture. Our FastAPI application acts as a client to the true powerhouse: the Triton Inference Server.

Here is the sequence of events for a single API call to our service:

```mermaid
sequenceDiagram
    participant UserApp as User Application
    participant FastAPI as FastAPI Service
    participant Triton as Triton Server
    participant GPU as NVIDIA A6000

    UserApp->>+FastAPI: 1. POST /generate_stream (HTTP Request)
    Note over FastAPI: Validates request, creates gRPC client call

    FastAPI->>+Triton: 2. StreamInfer (gRPC Request)
    Note over Triton: Receives request, begins Ensemble pipeline

    Triton->>Triton: 3. Preprocessing (Text -> Token IDs)
    Note over Triton: This step runs on the CPU.

    loop For each new token
        Triton->>+GPU: 4. Inference Step (Fused Batch)
        Note over GPU: Runs TensorRT-LLM engine
        GPU-->>-Triton: Generated Token ID

        Triton->>Triton: 5. Postprocessing (Token ID -> Text)
        Note over Triton: This step runs on the CPU.

        Triton-->>FastAPI: 6. Streamed Result (gRPC)
    end
    
    FastAPI-->>-UserApp: 7. Streamed Result (HTTP)
```

### Component Roles & Responsibilities

Let's break down the job of each piece of the puzzle.

### 1. The FastAPI Service (The Front Door)

-   **File:** `llm_service/main.py`
-   **What it is:** A user-friendly web server acting as the public-facing interface for our entire system. It runs using Gunicorn and Uvicorn workers for production-grade HTTP handling.
-   **Its Responsibilities:**
    -   **Expose Clean Endpoints:** Provides simple, intuitive HTTP endpoints like `/generate` and `/generate_stream`.
    -   **Validate User Input:** Uses Pydantic models (`llm_service/models.py`) to ensure incoming requests have the correct data types (e.g., `prompt` is a string, `temperature` is a float). This prevents bad data from ever reaching the core model.
    -   **Act as a Triton Client:** This is its most important role. It translates the validated HTTP request into a highly efficient gRPC call that the Triton server understands. It uses the logic in `llm_service/triton_client.py` to do this.
    -   **Handle Streaming:** For the `/generate_stream` endpoint, it receives the gRPC stream from Triton and relays it to the user as a streaming HTTP response.

### 2. Triton Inference Server (The Airport Traffic Controller)

-   **What it is:** A high-performance, general-purpose inference server developed by NVIDIA, written in C++.
-   **Its Responsibilities:**
    -   **Load & Manage Models:** When Triton starts, it scans the `/models` directory and loads all the model configurations (`config.pbtxt`) and the actual engine files into GPU memory.
    -   **Listen for Requests:** It opens ports (e.g., 8001 for gRPC) and listens for incoming inference requests from clients like our FastAPI service.
    -   **Schedule Work for the GPU:** It holds the queue of incoming requests and uses its chosen scheduling strategy (in our case, `inflight_fused_batching`) to pack work together efficiently, maximizing GPU utilization.
    -   **Execute Model Pipelines:** When a request arrives for the `ensemble` model, Triton takes on the role of a conductor, executing the defined pipeline step-by-step.

### 3. The Triton "Ensemble" Model (The Pipeline Conductor)

-   **File:** `~/trtllm-triton-repo/ensemble/config.pbtxt`
-   **What it is:** A virtual model inside Triton. It doesn't perform any inference itself but instead defines a **Directed Acyclic Graph (DAG)** that dictates the flow of data between other models.
-   **Its Responsibilities:**
    1.  Acts as the single entry point for a request.
    2.  Receives the initial data (`prompt`, `max_tokens`, etc.) from the client.
    3.  Directs the `prompt` string to the `preprocessing` model.
    4.  Takes the output (token IDs) from `preprocessing` and sends them to the `tensorrt_llm_bls` model (the logic manager).
    5.  The `tensorrt_llm_bls` interacts with the core `tensorrt_llm` engine to generate new token IDs.
    6.  Directs the generated token IDs to the `postprocessing` model to be converted back into text.
    7.  Streams the final text from `postprocessing` back to the client that called the ensemble.

### 4. The Pipeline Sub-Models (The Specialists)

These are the individual workers that the Ensemble model conducts.

-   **`preprocessing` (CPU-bound):**
    -   A simple Python model.
    -   **Job:** Takes a raw text string and uses the Hugging Face tokenizer to convert it into a sequence of integer token IDs.

-   **`tensorrt_llm_bls` (CPU-bound):**
    -   The Business Logic Scripting (BLS) model.
    -   **Job:** A crucial "manager" script that contains the stateful logic needed to interface with the in-flight batcher. It keeps track of request states, stop conditions, and streaming flags. It's the glue between the simple pre/post processing steps and the complex core engine.

-   **`tensorrt_llm` (GPU-bound):**
    -   **This is the core engine.** It's the `rank0.engine` file we built with `trtllm-build`.
    -   **Job:** Takes a batch of token IDs and executes the hyper-optimized forward pass on the GPU's Tensor Cores to predict the next token ID. It is incredibly fast but only understands numbers, not text.

-   **`postprocessing` (CPU-bound):**
    -   Another simple Python model.
    -   **Job:** Takes the integer token IDs generated by the engine and uses the Hugging Face tokenizer to convert them back into human-readable text.

## A Deep Dive into Quantization

Quantization is one of the most critical techniques for deploying Large Language Models in production. It is the process of reducing the numerical precision of a model's numbers (its weights and, in some cases, its activations).

The primary goals of quantization are to:
1.  **Reduce Memory Footprint:** Decrease the amount of GPU VRAM the model requires.
2.  **Increase Inference Speed:** Accelerate computation by using faster, lower-precision hardware capabilities.

Think of it like saving a professional photograph. You could save it as a massive, uncompressed `RAW` file with perfect fidelity, or as a high-quality `JPEG` which is a fraction of the size and visually almost identical. Quantization is a form of intelligent, lossy compression for AI models.

### The Building Blocks: Data Types in AI

The precision of a number in a computer is determined by how many bits are used to store it. In deep learning, these are the most common data types:

| Data Type | Bits | Technical Name | Analogy | Pros & Cons |
| :--- | :--- | :--- | :--- | :--- |
| **FP32** | 32 | Single-Precision Floating-Point | **Uncompressed `RAW` Photo** | **Pro:** Highest precision; the standard for training models to ensure accuracy and stable convergence. <br> **Con:** Very large memory footprint and computationally intensive. |
| **FP16** | 16 | Half-Precision Floating-Point | **High-Quality `PNG` File** | **Pro:** **Half the size of FP32** and significantly faster on modern GPUs with Tensor Cores. The standard for high-performance inference. <br> **Con:** Minor, often negligible, loss of precision compared to FP32. |
| **INT8** | 8 | 8-bit Integer | **High-Quality `JPEG` File** | **Pro:** **75% smaller than FP32** and **extremely fast** when using a GPU's specialized INT8 Tensor Cores. <br> **Con:** Can noticeably degrade model accuracy if not applied carefully. |

### Our Strategy: W8A16 Weight-Only Quantization

Simply converting the entire model to INT8 would be fast but could harm the model's predictive quality. Instead, we use a more sophisticated and effective strategy called **Weight-Only Quantization**, specifically the **W8A16** scheme.

This means we treat two different parts of the model—the static **Weights** and the dynamic **Activations**—differently.

#### 1. The Weights (`W8`): The Model's Brain

-   **What they are:** The billions of parameters that the model "learned" during its extensive training. They represent the crystallized knowledge of the AI. For Llama 3.1 8B, this is roughly 8 billion numbers.
-   **Our Action:** We convert these weights from their original FP16 format into **INT8**.
-   **The Impact:**
    -   **Massive Memory Savings:** This is the single biggest win for reducing the model's on-disk and in-memory size.
      - *FP16 Weights:* `8 Billion parameters * 2 bytes/FP16 ≈ 16 GB`
      - *INT8 Weights:* `8 Billion parameters * 1 byte/INT8 ≈ 8 GB`
      - We instantly save **~8 GB of VRAM** just by quantizing the weights.
    -   **Extreme Speed Boost:** The NVIDIA A6000 GPU has specialized hardware units called **INT8 Tensor Cores**. When the GPU performs matrix multiplications (the core operation of an LLM), loading these INT8 weights into those cores is dramatically faster than using FP16.

#### 2. The Activations (`A16`): The Flow of Thought

-   **What they are:** The numbers that flow *through* the model's layers during a single inference request. They represent the intermediate calculations for the specific prompt you provided.
-   **Our Action:** We keep these transient numbers at the high-precision **FP16** format.
-   **The Impact:**
    -   **Preserves Accuracy:** The intermediate calculations in an LLM are highly sensitive to numerical precision. Keeping them in FP16 ensures that the model's reasoning and generation quality remain high, avoiding the potential degradation that could come from using INT8 for these values.

The **W8A16** approach gives us the best of both worlds: the immense storage and speed benefits of INT8 for the static model weights, combined with the accuracy and stability of FP16 for the dynamic activation values.

### The Critical Exception: The KV Cache

There is a third, crucial component to consider: the **KV Cache**.

-   **What it is:** The model's short-term memory or "conversation scratchpad." After processing each token in a prompt, the model calculates and stores a "Key" (K) and "Value" (V) pair. When generating the next token, the model attends to everything in this cache to understand the context, preventing it from having to re-read the entire prompt from scratch every single time.
-   **The VRAM Problem:** The size of the KV Cache is directly proportional to the context length. For long-context models, it is the **single largest consumer of VRAM during inference**.
    -   *Rough calculation for one 96k request:* `96,000 tokens * 32 layers * 4096 hidden_dim * 2 (K and V) * 2 bytes/FP16 ≈ 50.3 GB`
-   **Our Deliberate Choice: Keeping the KV Cache in FP16**
    -   **The Reason:** The technology that enables efficient long-context inference, **Paged Attention (`use_paged_context_fmha`)**, has a critical requirement in our version of TensorRT-LLM: it is only compatible with an FP16 KV Cache.
    -   **The Engineering Trade-off:** We could have attempted to quantize the KV Cache to INT8 to save even more memory. However, this would have forced us to disable Paged Attention, making it impossible to handle a 96k context length. Therefore, we made a conscious engineering decision: **Long-context capability is more important than the additional memory savings from KV Cache quantization.**

### Summary of Our Quantization Strategy

This table summarizes the choices made for each component of the model:

| Component | Our Chosen Precision | Justification |
| :--- | :--- | :--- |
| **Model Weights** | **INT8** | **Performance & Memory.** Halves the model's storage size and leverages the A6000's ultra-fast INT8 Tensor Cores for maximum throughput. |
| **Activations** | **FP16** | **Accuracy & Stability.** Preserves the numerical precision of in-flight calculations, ensuring the model's output quality is not compromised. |
| **KV Cache** | **FP16** | **Compatibility & Functionality.** A mandatory requirement to enable `use_paged_context_fmha`, the core technology needed to serve a 96k+ context length. |

By understanding these deliberate trade-offs, you can see that our setup is not just a conversion, but a carefully engineered configuration to achieve a specific, high-performance goal: serving a very long context model to multiple users with low latency.


## GPU Memory & Performance Demystified

An NVIDIA A6000 GPU comes with a generous 48 GB of VRAM. For most deep learning tasks, this is an enormous amount. However, for Large Language Models—especially those running with a long context—every single gigabyte is precious. Understanding where this memory goes and how to measure performance is essential for running a stable and efficient service.

### A Deep Dive into VRAM Usage

When Triton loads and runs our model, the GPU's memory is allocated to several distinct components. We can broadly categorize them into three buckets: static, highly dynamic, and transient.

### 1. Model Weights (Static Cost)

-   **What it is:** This is the physical size of the model's brain—the ~8 billion parameters of Llama 3.1 that we quantized to INT8.
-   **Characteristic:** This is a **fixed, one-time cost**. As soon as the Triton server loads the model, this memory is allocated and will not be freed until the model is unloaded. It is independent of how many users are connected or what prompts they send.
-   **VRAM Calculation:**
    -   Original Model Size (FP16): `~8,000,000,000 parameters * 2 bytes/FP16 ≈ 16 GB`
    -   Our Quantized Model Size (INT8): `~8,000,000,000 parameters * 1 byte/INT8 ≈ 8 GB`
    -   **Result:** This is our baseline memory usage. Roughly **8 GB** of our 48 GB is consumed just by having the model loaded.

### 2. The KV Cache (The Dominant, Dynamic Cost)

-   **What it is:** The model's "short-term memory" or context-keeping scratchpad. For every token of input *and* output, the model stores a corresponding Key/Value (K,V) pair. The size of this cache is the **single largest driver of VRAM consumption** during inference.
-   **Characteristic:** This cost is **highly dynamic**. It grows with every active user and with the length of their context (prompt + generated response).
-   **VRAM Calculation (The Shocking Math):**
    -   The formula is roughly: `sequence_length * num_layers * hidden_dimension * 2 (for K and V) * bytes_per_element`
    -   Let's plug in the numbers for a **single, worst-case user request** with a 96k token prompt:
        -   `96,000 tokens * 32 layers * 4096 hidden_dim * 2 * 2 bytes/FP16 ≈ 50.3 GB`
-   **How Is This Possible? The Magic of Paged Attention**
    -   The raw calculation shows that a single max-length request would require more VRAM than our GPU has! This is why the `use_paged_context_fmha` build flag is not just an optimization—it is **absolutely essential**.
    -   **Analogy:** Think of how your computer's operating system uses virtual memory (RAM + a page file on your SSD). It intelligently moves "pages" of memory around.
    -   **Paged KV Cache** works similarly. Instead of allocating one enormous, contiguous block of VRAM for the cache, it allocates memory in smaller, fixed-size "pages" or "blocks." This has two huge benefits:
        1.  **Eliminates Internal Fragmentation:** It prevents wasted memory between requests of different sizes.
        2.  **Enables Much Larger Context:** It allows the total logical size of the KV cache to exceed what could be stored in a single memory block.
-   **Controlling the Cache:** Our `tensorrt_llm/config.pbtxt` has a critical parameter: `kv_cache_free_gpu_mem_fraction: 0.85`. This tells Triton: "After you've loaded the model weights (~8 GB), take 85% of the *remaining free VRAM* and reserve it as a shared memory pool for the Paged KV Cache." This dynamic pool is then used to serve all concurrent users.

### 3. Activation Buffers & Workspace (Transient Cost)

-   **What it is:** This is the temporary memory the GPU needs for its internal calculations—the "scratchpad" for performing matrix multiplications and other operations during the forward pass.
-   **Characteristic:** This memory is **transient and fluctuating**. It is allocated and freed very rapidly for each step of the generation process. Its peak size depends on the `max_batch_size` we configured.
-   **Result:** This is a smaller but non-zero consumer of VRAM that is handled automatically by the TensorRT engine.

### Measuring Performance: The Key Performance Indicators (KPIs)

"Performance" is not a single number. For a user-facing LLM service, we care about two primary dimensions: latency (speed for one user) and throughput (capacity of the system).

### Latency: The User's Perception of Speed

| Metric | What It Measures | Why It Matters | Dominated By |
| :--- | :--- | :--- | :--- |
| **Time to First Token (TTFT)** | The time from when a user sends a request to when the very first word of the response appears. | This is the most important metric for **perceived responsiveness**. A low TTFT makes the service feel snappy and alive, even if the full response takes time. | **Prompt Processing.** The initial phase where the model "ingests" the entire user prompt into its KV Cache. |
| **Time per Output Token (TPOT)** | The time it takes to generate each subsequent token after the first one. This is often measured in milliseconds/token. | This determines the "typing speed" of the AI. A low TPOT results in a smooth, continuous stream of text. | **Decoding Steps.** A series of smaller, faster forward passes, each generating one new token. |

### Throughput: The System's Overall Capacity

| Metric | What It Measures | Why It Matters |
| :--- | :--- | :--- |
| **Requests per Second** | The number of independent user requests the service can complete in one second. | A classic server metric, but can be misleading for LLMs where one request might be 10 tokens and another might be 10,000. |
| **Output Tokens per Second** | The **total number of tokens** generated by the server across all concurrent users in one second. | **This is the best metric for LLM throughput.** It measures the actual "work output" of the GPU and normalizes for requests of different lengths. Maximizing this KPI means you are getting the most value out of your hardware. |

### The Fundamental Trade-off

In any serving system, there is a core trade-off: **increasing batching typically increases throughput but also increases latency**. Our system uses **In-Flight Fused Batching** to find a "best of both worlds" scenario, dynamically creating batches to keep the GPU busy (high throughput) without making new users wait in a queue (low latency).

### How to Monitor This in Practice

The primary command-line tool for observing your GPU in real-time is `nvidia-smi`.

```bash
# Run this command in your terminal for a live, auto-updating view
watch -n 1 nvidia-smi
```

When you run this and put your service under load, pay attention to these columns:
-   **`Memory-Usage`**: This will show you the total VRAM consumption. You'll see it jump to ~8 GB when the server starts, and then increase further as users connect and the KV Cache pool is utilized.
-   **`GPU-Util`**: This shows the percentage of time the GPU's processing cores were active. When serving requests, you want this number to be as high as possible. A consistently high `GPU-Util` means your server is efficiently feeding the GPU with work, maximizing your throughput.

## Concurrency, Batching, and a Tale of Two Elevators

Concurrency is the ability of a service to handle multiple user requests at the same time. For a web server handling simple database queries, this is a solved problem. For a Large Language Model service, where a single request can tie up a multi-thousand-dollar GPU for seconds or even minutes, it is the single most important challenge to solve for production.

The strategy used to achieve concurrency determines the throughput, latency, and ultimate efficiency of the entire system.

### The Naive Approach: The Single-Lane Road (Sequential Processing)

The simplest possible implementation would process requests one by one, in the order they arrive.

`[User 1] -> [GPU Process] -> [Response 1]`
`                  (User 2 waits in line)  `
`[User 2] -> [GPU Process] -> [Response 2]`

-   **Problem:** This is incredibly inefficient. The GPU, a massively parallel processor, is often idle between requests. More importantly, User 2 has to wait for User 1's entire request to finish, leading to terrible latency. This model does not scale beyond a single user.

---

### The First Evolution: The Tour Bus (Static Batching)

A simple improvement is to group requests together into a "static batch." The server waits until it has collected a certain number of requests (e.g., a batch size of 4) before sending them all to the GPU at once.

**Analogy:** This is like a tour bus. The bus driver (the server) waits at the station until the bus is full before starting the tour (processing).

```mermaid
graph TD
    subgraph "Static Batching (The Tour Bus)"
        direction LR
        U1[User 1] --> B{Batch Queue};
        U2[User 2] --> B;
        U3[User 3] --> B;
        U4[User 4] --> B;
        B -- "Waits until full (batch_size=4)" --> P([GPU Process]);
        P --> R1[Resp 1];
        P --> R2[Resp 2];
        P --> R3[Resp 3];
        P --> R4[Resp 4];
    end
```

This approach has two critical flaws:

1.  **Head-of-Line Blocking:** If User 1 arrives, they have to wait for Users 2, 3, and 4 to show up before their request even starts. This creates high initial latency.
2.  **Wasted Computation (Padding):** LLM generation is iterative. If User 1's prompt is 100 tokens and User 2's is 1000 tokens, the server must "pad" User 1's input with 900 empty tokens to make the requests the same size for the batch. The GPU wastes cycles processing these empty pads.

---

### The Production Solution: The Modern Elevator (In-Flight Fused Batching)

This is the advanced, state-of-the-art technique used by TensorRT-LLM and enabled by our `batching_strategy: inflight_fused_batching` configuration.

**Analogy:** Forget the tour bus. Think of a modern, smart elevator in a skyscraper.
-   The static batching "tour bus" waits until it's full before moving at all.
-   The in-flight batching "elevator" is **constantly in motion**. It can be going up, pick someone up on Floor 5 (a new request), let someone off on Floor 8 (a finished request), and continue on to Floor 12 (a long-running request), all without stopping and waiting to be full.

**How it works for LLMs:**

In-Flight Batching deconstructs the generation process into individual token-by-token steps. The Triton scheduler can then dynamically "fuse" requests that are at completely different stages of their lifecycle into a single computational batch for the GPU.

At any given microsecond, the batch being sent to the GPU might contain a mix of operations:

| Request ID | Current Task |
| :--- | :--- |
| **Request A** | Processing token #50,000 of a very long initial prompt. |
| **Request B** | Generating the 5th output token (decoding step). |
| **Request C** | Generating the 120th output token (decoding step). |
| **Request D** | Processing token #10 of a brand new, short prompt that just arrived. |

The TensorRT-LLM engine executes this "fused" batch, and the Triton server intelligently routes the results back to the correct users. User B gets their 6th token, User C gets their 121st, and the processing of Prompts A and D continues.

```mermaid
graph TD
    subgraph "In-Flight Fused Batching (The Smart Elevator)"
        direction LR
        U1["User 1<br>(Long Prompt)"] --> S(Scheduler);
        U2["User 2<br>(Short Prompt)"] --> S;
        U3["User 3<br>(Just Arrived)"] --> S;
        
        S -- "Dynamically fuses active steps" --> P([GPU Process]);

        subgraph "Continuous GPU Batches"
            direction TB
            P --> B1["Batch 1:<br>Prompt A (step n)<br>Prompt B (step m)"];
            B1 --> B2["Batch 2:<br>Prompt A (step n+1)<br>Prompt B (step m+1)<br>Prompt C (step 1)"];
            B2 --> B3["Batch 3:<br>Prompt A (finished)<br>Prompt B (step m+2)<br>Prompt C (step 2)"];
        end

        B1 -->|Token for A| R1[User 1];
        B1 -->|Token for B| R2[User 2];
        B2 -->|Token for A| R1;
        B2 -->|Token for B| R2;
        B2 -->|Token for C| R3[User 3];

    end
```

### The Overwhelming Benefits

This modern approach is what makes high-performance LLM serving possible:

1.  **Maximum GPU Utilization:** The GPU is never idle as long as there is at least one active request in the system. There is no waiting for batches to fill up.
2.  **Extremely Low "Time to First Token":** New requests can be added to the batch and begin processing on the very next iteration, meaning users get their first token back almost immediately.
3.  **High Overall Throughput:** By keeping the GPU constantly busy with useful, non-padded work, the total number of tokens generated per second across all users is maximized.
4.  **Fairness:** The scheduler ensures that long-running requests don't starve new, short requests.

By choosing this strategy, we move from a simple, blocking system to a dynamic, fluid, and highly efficient service capable of handling a demanding, real-world user load.

---

# Installation

---

## Step 1: System Prerequisites

This section covers the base software required on your host machine to interact with NVIDIA GPUs inside Docker containers.

### 1.1 Install Docker

```bash
sudo apt-get update
sudo apt-get install -y docker.io

# Add your user to the docker group to run docker without sudo
# NOTE: You must log out and log back in for this change to take effect.
sudo usermod -aG docker ${USER}

echo "----------------------------------------------------"
echo "-> You will need to log out and log back in for docker group changes to apply."
echo "-> Or, you can start a new shell with: newgrp docker"
echo "----------------------------------------------------"
```

### 1.2 Install NVIDIA Container Toolkit

This toolkit allows Docker containers to access your NVIDIA GPU.

```bash
# Add the NVIDIA repository and key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install the toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use the NVIDIA runtime and restart the Docker daemon
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## Step 2: NVIDIA NGC Authentication

To download the pre-built Triton container, you need to authenticate with the NVIDIA NGC catalog.

### 2.1 Get Your NGC API Key

1.  Go to the NVIDIA NGC website: **[https://ngc.nvidia.com](https://ngc.nvidia.com)**
2.  Sign in or create a free account.
3.  In the top-right corner, click your user name and select **"Setup"**.
4.  On the Setup page, click **"Get API Key"** and then **"Generate API Key"**.
5.  **IMPORTANT:** A long alphanumeric string will be displayed. This is your API key. Copy this key immediately and save it somewhere safe. You will not be able to see it again.

### 2.2 Log in via Docker

Use the API key to log in to the NVIDIA Container Registry (`nvcr.io`).

```bash
docker login nvcr.io
```

The command will prompt you for a `Username` and a `Password`:

-   **Username:** Enter the literal string `$oauthtoken`
-   **Password:** Paste the **NGC API Key** you just generated. (Note: The password will be invisible as you type/paste it).

You should see a `Login Succeeded` message.

---

## Step 3: Download Triton Server Image

Pull the specific Triton Server image that includes TensorRT-LLM support.

```bash
docker pull nvcr.io/nvidia/tritonserver:25.05-trtllm-python-py3
```

---

## Step 4: Download the Llama 3.1 Model

We will download the model weights from Hugging Face and create directories to store model data and the final Triton repository.

```bash
# Create directories on the host machine
mkdir ~/trtllm-triton-repo
mkdir ~/trtllm-data

# Install the Hugging Face command-line tool
pip install huggingface_hub

# Log in with your Hugging Face token (it will prompt you to paste it)
huggingface-cli login

# Download the model files into the data directory
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct --local-dir ~/trtllm-data/llama-3.1-8b-hf
```

---

## Step 5: Build the TensorRT-LLM Engine

**Option-1**: First, launch an interactive container session with the required directories mounted.

```bash
# Start an interactive container
docker run --gpus all -it --rm \
  -v ~/trtllm-data:/data \
  -v ~/trtllm-triton-repo:/triton_repo \
  nvcr.io/nvidia/tritonserver:25.05-trtllm-python-py3
```

> You are now inside the container.
> - The HF model is available at `/data/llama-3.1-8b-hf`
> - The output repository will be at `/triton_repo`

Now, run the following commands *inside the container*:

```bash
# Inside container: Create a workspace and clone TensorRT-LLM
mkdir -p /workspace && cd /workspace
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git checkout v0.19.0
```

**Option-2** : Alternatively what you can do is git clone TensorRT-LLM repo (Faster) in your host machine 

```bash
# install git lfs
sudo apt-get install git-lfs
# initialize
git lfs install

mkdir ~/trtllm-repo
cd trtllm-repo/
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git checkout v0.19.0
```

and now you can mount this dir as well 

```bash
docker run --gpus all -it --rm   -v ~/trtllm-data:/data   -v ~/trtllm-triton-repo:/triton_repo   -v ~/trtllm-repo:/workspace  nvcr.io/nvidia/tritonserver:25.05-trtllm-python-py3

```


The next steps are same in both cases from here:

```bash
# Inside container: Install dependencies
pip install -r requirements.txt
# Inside container: Create a directory for the converted checkpoint
mkdir -p /data/llama-3.1-8b-tllm-ckpt
```

**Important**: here you will divert if you have L40s instead of A6000. You can follow step 5.1,5.2 and 6.1 from [**L40s Installation**](#l40s-installation). If you have A6000 continue.


#### **This is the most critical step, where we convert the Hugging Face model into a highly optimized engine for inference. These commands are run inside the Triton container.**

### 5.1 Convert Hugging Face Model to TensorRT-LLM Checkpoint
### FOR A6000 - FP8 is not Available
```bash
# Inside container: Run the conversion script (INT8 weight-only quantization)
python examples/llama/convert_checkpoint.py \
    --model_dir /data/llama-3.1-8b-hf \
    --output_dir /data/llama-3.1-8b-tllm-ckpt \
    --dtype float16 \
    --use_weight_only \
    --weight_only_precision int8
```
#### `convert_checkpoint.py` Arguments

| Category | Argument | Your Value | Justification |
|:---|:---|:---|:---|
| **I/O** | `--model_dir` | `/data/llama-3.1-8b-hf`| **Required.** Points to the source Hugging Face model. |
| | `--output_dir` | `/data/llama-3.1-8b-tllm-ckpt`| **Required.** Destination for the converted checkpoint. |
| **Quantization** | `--dtype` | `float16` | **Best Choice.** Sets the precision for activations and the KV Cache. `float16` KV Cache is **required** for compatibility with `use_paged_context_fmha` (long context). |
| | `--use_weight_only`| `(Set)` | **Essential.** The master switch that enables INT8 weight-only quantization for the model's linear layers. |
| | `--weight_only_precision`| `int8` | **Best Choice.** Instructs the converter to quantize weights to INT8, providing the best performance on the A6000's INT8 Tensor Cores. |
| **KV Cache** | `--int8_kv_cache` | `(Not Set)` | **Correctly Omitted.** Using `int8_kv_cache` is **incompatible** with `use_paged_context_fmha`. Achieving the 96k context goal requires paged attention, so this must be omitted. |

### 5.2 Build the Optimized Engine from the Checkpoint

This command uses the checkpoint to build the final, runnable engine, configured for our specific needs (96k input length, paged attention).

```bash
# Inside container: Build the engine
trtllm-build \
    --checkpoint_dir /data/llama-3.1-8b-tllm-ckpt \
    --output_dir /data/llama-3.1-8b-96k-engine/ \
    --gpt_attention_plugin auto \
    --max_batch_size 4 \
    --max_input_len 96000 \
    --max_seq_len 131072 \
    --max_num_tokens 131072\
    --use_paged_context_fmha enable \
    --multiple_profiles enable
```

#### `trtllm-build` Arguments

**Keep the max_num_tokens=131072 for using the model with full capacity this is crucial**

| Category | Argument | Your Value | Justification |
|:---|:---|:---|:---|
| **I/O** | `--checkpoint_dir` | `/data/llama-3.1-8b-tllm-ckpt`| **Required.** Path to the converted checkpoint. |
| | `--output_dir` | `/data/llama-3.1-8b-96k-engine/`| **Required.** Where to save the final, compiled engine files. |
| **Engine Capacity**| `--max_batch_size` | `4` | Defines the maximum number of requests to process in a single batch. Must be consistent across all configs. |
| | `--max_input_len` | `96000` | **Your Goal.** Sets the maximum number of tokens allowed in the input prompt. |
| | `--max_seq_len` | `131072` | **Your Goal.** Sets the absolute maximum sequence length (input + output). This must be >= `max_input_len`. |
| **Plugins** | `--gpt_attention_plugin`| `auto` | Automatically selects the most optimized attention kernel available. |
| | `--use_paged_context_fmha`| `enable` | **Critical.** Enables Paged and Fused Multi-Head Attention, the key technology for handling very long context lengths efficiently. |
| | `--multiple_profiles` | `enable` | Creates multiple optimization profiles, which can improve performance for inputs of varying lengths. |


Once the build is complete, you can leave the interactive container: The reason to leave the container is to keep it clean from the heavy TensorRT-LLM repo. 
```bash
# Inside container:
exit
```

---

## Step 6: Prepare and Launch Triton Server

Now we configure the Triton model repository and launch the server as a background service.

### 6.1 Prepare the Triton Model Repository

These steps are also performed *inside a container*.

```bash
# Start a new interactive container
docker run --gpus all -it --rm \
  -v ~/trtllm-data:/data \
  -v ~/trtllm-triton-repo:/triton_repo \
  nvcr.io/nvidia/tritonserver:25.05-trtllm-python-py3
```

Inside the new container, run these commands:

```bash
# Inside container: Clone the backend repository
cd /workspace
git clone -b r25.05 https://github.com/triton-inference-server/tensorrtllm_backend.git

# Inside container: Copy the model template files
cp -r /workspace/tensorrtllm_backend/all_models/inflight_batcher_llm/* /triton_repo/
```

#### ensambe
```bash
# Inside container: Use the fill_template.py script to configure the models
# Note the engine_dir path now correctly points to our built engine.

# ensemble/config.pbtxt
python /workspace/tensorrtllm_backend/tools/fill_template.py -i /triton_repo/ensemble/config.pbtxt \
    triton_max_batch_size:4,logits_datatype:TYPE_FP32
```
#### Configure `ensemble/config.pbtxt`

This is the top-level configuration that defines the entire inference pipeline, chaining the preprocessing, business logic, core model, and postprocessing steps together.

| Parameter | Your Value | Grounded Logic & Reasoning |
| :--- | :--- | :--- |
| **`triton_max_batch_size`** | `4` | **Why:** This value **must match** the `--max_batch_size 4` you used when you ran `trtllm-build`. This top-level ensemble config tells Triton the maximum number of requests it can group together before sending them down the pipeline. Consistency is critical for the server to function correctly. |
| **`logits_datatype`** | `TYPE_FP32` | **Why:** This controls the precision of the model's final output scores before sampling. `TYPE_FP32` is the safest and most standard choice for preserving numerical accuracy and avoiding potential issues during the final token selection (e.g., in temperature or top-p sampling). |


#### preprocessing
```bash
# preprocessing/config.pbtxt
python /workspace/tensorrtllm_backend/tools/fill_template.py -i /triton_repo/preprocessing/config.pbtxt \
    triton_max_batch_size:4,tokenizer_dir:/data/llama-3.1-8b-hf,preprocessing_instance_count:1
```

#### Configure `preprocessing/config.pbtxt`

This model is responsible for taking the raw input string from the user and converting it into token IDs that the TensorRT-LLM engine can understand.

| Parameter | Your Value | Grounded Logic & Reasoning |
| :--- | :--- | :--- |
| **`triton_max_batch_size`** | `4` | **Why:** This must be consistent across all configs. It matches the `max_batch_size` used for the ensemble config and the `trtllm-build` command. |
| **`tokenizer_dir`** | `/data/llama-3.1-8b-hf` | **Why:** This is the absolute path *inside the container* to the directory where the Hugging Face tokenizer files (`tokenizer.model`, `tokenizer.json`, etc.) are located. The Python backend for this step needs this exact path to load the vocabulary and correctly convert input text into token IDs. |
| **`preprocessing_instance_count`** | `1` | **Why:** This sets how many parallel CPU processes to use for tokenization. `1` is a safe, standard default. You could increase this later if performance monitoring reveals that text tokenization is a CPU bottleneck for your service under heavy load. |


#### postprocessing

```bash
# postprocessing/config.pbtxt
python /workspace/tensorrtllm_backend/tools/fill_template.py -i /triton_repo/postprocessing/config.pbtxt \
    triton_max_batch_size:4,tokenizer_dir:/data/llama-3.1-8b-hf,postprocessing_instance_count:1
```

#### Configure `postprocessing/config.pbtxt`

This model does the reverse of preprocessing: it takes the output token IDs generated by the engine and converts them back into human-readable text.

| Parameter | Your Value | Grounded Logic & Reasoning |
| :--- | :--- | :--- |
| **`triton_max_batch_size`** | `4` | **Why:** This must be consistent across all configs and match the batch size you built the engine with (`4`). |
| **`tokenizer_dir`** | `/data/llama-3.1-8b-hf` | **Why:** This is the absolute path to the tokenizer files. The Python backend for this step needs this path to correctly map the output token IDs back into words. |
| **`postprocessing_instance_count`** | `1` | **Why:** This sets how many parallel CPU processes to use for detokenization. `1` is a safe, standard default. It can be increased later if postprocessing becomes a bottleneck. |

#### tensorrt_llm

```bash
# tensorrt_llm/config.pbtxt (Corrected)
python /workspace/tensorrtllm_backend/tools/fill_template.py -i /triton_repo/tensorrt_llm/config.pbtxt \
    triton_backend:tensorrtllm,triton_max_batch_size:4,decoupled_mode:True,max_beam_width:1,engine_dir:/data/llama-3.1-8b-96k-engine,batching_strategy:inflight_fused_batching,max_tokens_in_paged_kv_cache:131072,kv_cache_free_gpu_mem_fraction:0.85,exclude_input_in_output:True,logits_datatype:TYPE_FP32,encoder_input_features_data_type:TYPE_FP16
```
#### configure `tensorrt_llm/config.pbtxt` Arguments : THE MOST IMPORTANT ONE

This table explains the parameters for the final `tensorrt_llm/config.pbtxt` configuration.

| Parameter | Your Value | Grounded Logic & Reasoning |
|:---|:---|:---|
| **`batching_strategy`** | `inflight_fused_batching` | **Logic:** This enables the core feature for high throughput: in-flight batching. It dynamically manages requests, maximizing GPU utilization, which is essential for serving multiple users. |
| **`decoupled_mode`** | `True` | **Logic:** This enables streaming output (tokens are sent back as they are generated). This is a requirement for any interactive application. |
| **`max_tokens_in_paged_kv_cache`** | `131072` | **Logic:** This is a **critical safeguard** for your long context. It defines the absolute maximum number of tokens that the paged KV cache manager can hold across all active requests. It **must** be at least as large as your `max_seq_len` (131072) to handle even a single max-length request. |
| **`kv_cache_free_gpu_mem_fraction`** | `0.85` | **Logic:** This is your primary memory control knob. It tells the backend: "After loading the model weights, take 85% of the *remaining free VRAM* and dedicate it to the KV cache." For a high-VRAM card like the A6000, this is the best approach to dynamically allocate a large pool for the KV cache. |
| **`engine_dir`** | `/data/llama-3.1-8b-96k-engine` | **Logic:** **This is the correct path.** It is the absolute path inside the container to the directory containing your compiled `rank0.engine`. It must match the `--output_dir` from the `trtllm-build` step. |
| **`exclude_input_in_output`**| `True` | **Logic:** A standard setting for chat/instruction models. It prevents the server from echoing the user's (potentially very long) prompt, saving bandwidth. |
| **`encoder_input_features_data_type`**| `TYPE_FP16` | **Logic:** This specifies the data type for certain inputs. Since our model's `--dtype` was `float16`, this ensures compatibility. |
| **`logits_datatype`** | `TYPE_FP32` | **Logic:** Ensures numerical stability and consistency. It matches the data type for logits specified in the other configuration files, ensuring data flows correctly through the pipeline. |
| **Others** | `triton_backend:tensorrtllm`, `triton_max_batch_size:4`, etc. | **Logic:** These ensure consistency with the other configuration files and the engine build itself. They are mandatory and must match. |


#### tensorrt_llm_bls (bussiness logic) 

```bash
# tensorrt_llm_bls/config.pbtxt
python /workspace/tensorrtllm_backend/tools/fill_template.py -i /triton_repo/tensorrt_llm_bls/config.pbtxt \
    triton_max_batch_size:4,decoupled_mode:True,bls_instance_count:1,accumulate_tokens:False,logits_datatype:TYPE_FP32
```

#### Configure `tensorrt_llm_bls/config.pbtxt`

This model implements the Business Logic Scripting (BLS) for the pipeline. It acts as the "manager" that orchestrates the flow of data to and from the core `tensorrt_llm` model, especially for complex patterns like in-flight batching and streaming.

| Parameter | Your Value | Grounded Logic & Detailed Reasoning |
|:---|:---|:---|
| **`triton_max_batch_size`** | `4` | **Logic:** Ensures consistency across the entire pipeline. It must match the value used in `trtllm-build` and all other `config.pbtxt` files. |
| **`decoupled_mode`** | `True` | **Logic:** **Crucial for streaming.** This must match the `decoupled_mode:True` setting from the `tensorrt_llm/config.pbtxt`. It tells this "manager" component to operate in an asynchronous, non-blocking way, which is required to handle the streaming of tokens from the backend engine. |
| **`bls_instance_count`** | `1` | **Logic:** This defines how many parallel CPU processes run this business logic script. `1` is a safe and standard default. This can be tuned later if this specific step is identified as a performance bottleneck. |
| **`accumulate_tokens`** | `False` | **Logic:** **Also crucial for streaming.** When set to `False`, this component does *not* wait for the entire sequence to be generated. It passes tokens through to the `postprocessing` step as soon as they arrive from the engine. If this were `True`, it would buffer the entire response and destroy the streaming effect. |
| **`logits_datatype`** | `TYPE_FP32` | **Logic:** Ensures numerical stability and consistency. It matches the data type for logits specified in the other configuration files, ensuring data flows correctly through the pipeline. |



Once done, exit the container: `exit`.

### 6.2 Correct Host File Permissions

The files created inside the container are owned by the root user. Change ownership to your current user.

```bash
# On your host machine
sudo chown -R $USER:$USER ~/trtllm-triton-repo
```

### 6.3 Launch the Triton Server Container

Run the Triton server in detached mode, exposing the necessary ports and mounting the data and model repository directories.

```bash
docker run --gpus all -d --name triton_server \
  --shm-size=1g \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v ~/trtllm-data:/data \
  -v ~/trtllm-triton-repo:/models \
  nvcr.io/nvidia/tritonserver:25.05-trtllm-python-py3 \
  tritonserver --model-repository=/models 
```
You can add the option of verbose logging for debug purposes: 

```bash
docker run --gpus all -d --name triton_server \
  --shm-size=1g \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v ~/trtllm-data:/data \
  -v ~/trtllm-triton-repo:/models \
  nvcr.io/nvidia/tritonserver:25.05-trtllm-python-py3 \
  tritonserver --model-repository=/models --log-verbose=1
```

You can monitor its startup logs to ensure the models load correctly:
```bash
docker logs -f triton_server
```
Look for a `READY` status for the `ensemble` model. 

o remove the server later, run `docker rm -f triton_server`.

---

# Inference

## **Step 0: The "Why": FastAPI Service vs. Direct Triton Access**

The Triton server is running, its ports are exposed, and it's ready to accept inference requests. A logical question arises: **Why add another layer with a FastAPI service? Why not just have our applications communicate directly with Triton's endpoints?**

The answer lies in a fundamental principle of production system design: **separation of concerns**. While Triton is an unparalleled engine for high-performance inference, it is not designed to be a user-friendly, public-facing application server. Our FastAPI service acts as a crucial **API Gateway** or **façade**, translating the raw power of Triton into a secure, robust, and easy-to-use product.

Think of it this way: Triton is a Formula 1 engine. You wouldn't bolt it directly to a go-kart and give it to a regular driver. You build a complete car around it—with a familiar steering wheel, a protective chassis, and a dashboard that hides the complexity. Our FastAPI service is that car.

Here is a direct comparison of the two approaches:

| Aspect | Direct Triton Endpoint Access | FastAPI Service Layer (Our Approach) |
| :--- | :--- | :--- |
| **Communication Protocol** | **gRPC (or raw HTTP)**. Highly efficient but requires specialized client libraries and complex, model-specific request construction. | **Standard RESTful HTTP/JSON**. The universal language of the web. Can be called from any language or tool (`curl`, Python `requests`, JavaScript `fetch`) with no special setup. |
| **Ease of Integration** | **Difficult.** Developers need to understand Triton's API, data types (e.g., `TYPE_FP32`), and tensor shapes. Every application needs its own Triton client. | **Effortless.** Developers just call a simple endpoint like `/generate` with a JSON body. The complexity of Triton is completely hidden. |
| **Data Validation** | **None.** If you send a request with a missing parameter or the wrong data type, Triton will reject it with a low-level, often cryptic error. The burden is on the client. | **Robust.** Pydantic models automatically validate all incoming requests. If a user sends `"temperature": "hot"`, they get a clean `422 Unprocessable Entity` error explaining the mistake. |
| **Security & Access Control** | **Minimal.** You are exposing a raw port. Implementing API keys, rate limiting, or user authentication is complex and not Triton's core function. | **Built-in.** It is trivial to add FastAPI middleware for API key authentication, IP whitelisting, rate limiting, CORS headers, and other essential security features. |
| **Flexibility & Abstraction**| **Rigid.** Your application is tightly coupled to this specific Triton model (`ensemble`) and its input/output schema. Changing the backend model would require rewriting all client applications. | **Decoupled.** The FastAPI service provides a stable API "contract." You could replace the entire backend (e.g., switch to vLLM or a different model) and your users' applications would **never know the difference**. |
| **Logging & Observability**| **System-Level.** You see Triton's server logs, but it's hard to trace a specific user's request or log application-level context. | **Application-Level.** You can log precisely what you need: which user called the API, with what prompt, what parameters they used, and how long it took. This is invaluable for analytics and debugging. |

#### In Summary: From a Raw Engine to a Managed Product

By not exposing Triton directly, we gain:

1.  **Universal Accessibility:** Any developer, on any platform, can instantly use your LLM service. They don't need to learn gRPC or install NVIDIA's client libraries. They just need to make an HTTP request.
2.  **Robustness:** Your service won't crash or return confusing errors due to malformed user input. It provides clear, actionable feedback.
3.  **Future-Proofing:** The FastAPI layer is an **abstraction barrier**. It protects your users from backend changes and protects your backend from your users. You can evolve your inference stack (upgrade TensorRT-LLM, change the model, add a new one) without breaking a single client integration.
4.  **Security:** You have a single, controlled entry point where you can enforce all your security and access policies.

In conclusion, the FastAPI service isn't just an optional add-on; it is the component that elevates our high-performance inference engine into a production-ready, reliable, and secure API product.

## Step 1: Run the Production API Service

With the Triton server running, we can now launch our user-facing FastAPI service which provides a clean, easy-to-use interface.

### 1.1 Prerequisites

This service uses Python. We recommend using `conda` to manage environments.

-   [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) or [Anaconda](https://www.anaconda.com/download) must be installed.

Git clone this repository

```
git clone https://github.com/mnansary/Triton-TRTLLM-LLama3.1-8b-Instruct-A6000-docker.git
```


### 1.2 Create Conda Environment

Create a dedicated environment for the service to avoid dependency conflicts.

```bash
conda create -n llm_service python=3.10 -y
conda activate llm_service
```

### 1.3 Install Dependencies

Install all required Python packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 1.4 Configure the Service

The API service is configured via the `.env` file in the project root. Create it -Open it and ensure the settings match your setup. The defaults should work if you followed this guide exactly.

-   `TRITON_URL`: Should be `localhost:8001`.
-   `MODEL_NAME`: Should be `ensemble`.
-   `MAX_TOKENS`: The default maximum output length. Can be overridden per-request.

the ```.env``` should look something like this

```bash
# --- LLM Service Environment Variables ---
# Use this file to override default settings from config.py

# --- Triton and Model Settings ---
TRITON_URL="localhost:8001"
MODEL_NAME="ensemble"

# --- Server Settings ---
HOST="0.0.0.0"
PORT=24434
# For a machine with 16 CPU cores, a value of 8 might provide better performance.
WORKERS=4

# --- Logging Settings ---
LOG_LEVEL="INFO" # Options: DEBUG, INFO, WARNING, ERROR

# ===================================================================
# --- System-Wide Overrides for Generation Defaults ---
# Any value set here will override the default from config.py.
# This is useful for setting a different "house style" for your API.
# ===================================================================
DEFAULT_MAX_TOKENS=4096      # Setting a higher default for this environment.
DEFAULT_TEMPERATURE=0.7      # Making the model slightly more creative by default here.
DEFAULT_TOP_P=0.95           # A slightly larger sampling pool.
DEFAULT_REPETITION_PENALTY=1.15
```

### 1.5 Run the API Server

Use the provided shell script to launch the application with Gunicorn, a production-ready web server.

```bash
chmod +x run.sh
./run.sh
```

You should see output indicating the server has started on `http://0.0.0.0:24434`.

### 1.6 Test the API Endpoints

You can now send requests from any of your applications or test with `curl`.

#### Non-Streaming Request

This waits for the full response before returning.

```bash
curl -X POST http://localhost:24434/generate \
-H "Content-Type: application/json" \
-d '{
  "prompt": "An NVIDIA A6000 GPU is a powerful tool for",
  "temperature": 0.1
}'
```

#### Streaming Request

This returns tokens as soon as they are generated, ideal for interactive UIs.

```bash
curl -N -X POST http://localhost:24434/generate_stream \
-H "Content-Type: application/json" \
-d '{
  "prompt": "The future of artificial intelligence will likely be",
  "temperature": 0.5,
  "max_tokens": 100
}'
```
---

### 1.7 Available Generation Parameters for Your Triton Service

This table summarizes the key parameters you can expose through your service. These are defined in the `input` section of your `ensemble/config.pbtxt` and can be passed with each inference request to control the model's output.

#### Core Generation Control

| Parameter | Data Type | Description | Default/Example |
| :--- | :--- | :--- | :--- |
| `prompt` | String | The initial text prompt to start the generation. (Mapped to `text_input`). | **Required.** |
| `max_tokens` | Integer | The maximum number of new tokens to generate in the response. | **Required.** e.g., `1024` |
| `min_tokens` | Integer | The minimum number of new tokens to generate. The model will not stop before this count. | e.g., `32` |

#### Sampling & Creativity Control

| Parameter | Data Type | Description | Default/Example |
| :--- | :--- | :--- | :--- |
| `temperature` | Float | Controls randomness. Lower values (e.g., `0.1`) make the output more deterministic and focused. Higher values (e.g., `0.9`) increase creativity and diversity. A value of `0.0` is equivalent to greedy decoding. | e.g., `0.7` |
| `top_k` | Integer | Restricts the sampling pool to the `k` most likely next tokens. A value of `1` is greedy decoding. A value of `0` disables Top-K sampling. | e.g., `50` |
| `top_p` | Float | Restricts the sampling pool to a cumulative probability mass. For `top_p=0.9`, the model considers only the most likely tokens whose probabilities add up to 95%. A value of `1.0` disables Top-P sampling. | e.g., `0.95` |
| `seed` | Unsigned 64-bit Int | A seed for the random number generator to ensure reproducible outputs. If you use the same prompt and seed, you will get the exact same result every time. | e.g., `42` |

**Note:** The model typically uses either Top-K or Top-P sampling, whichever is more restrictive. It's common to set one to its disabled state (like `top_k=0`) while tuning the other.

#### Content & Penalty Control

| Parameter | Data Type | Description | Default/Example |
| :--- | :--- | :--- | :--- |
| `repetition_penalty` | Float | Penalizes tokens that have already appeared in the text (prompt + generation). A value > 1.0 discourages repetition; a value < 1.0 encourages it. `1.0` means no penalty. | e.g., `1.2` |
| `presence_penalty` | Float | Similar to repetition penalty, but applies a flat penalty to any token that has appeared at least once, regardless of frequency. Positive values discourage repeating tokens. | e.g., `0.0` |
| `frequency_penalty` | Float | Applies a penalty to tokens that increases based on how many times they have already appeared. Positive values discourage repeating tokens more strongly the more they are repeated. | e.g., `0.0` |
| `length_penalty` | Float | Adjusts the model's preference for longer or shorter sequences, primarily used with beam search. A value > 1.0 encourages longer sequences; a value < 1.0 encourages shorter ones. | e.g., `1.0` |

#### Stopping & Filtering

| Parameter | Data Type | Description | Default/Example |
| :--- | :--- | :--- | :--- |
| `stop_words` | List of Strings | A list of words or phrases that will immediately stop the generation process if produced. | `["\nUser:", "<|eot_id|>"]` |
| `bad_words` | List of Strings | A list of words that are forbidden from being generated. The model will never output these tokens. | e.g., `["nsfw_word"]` |
| `end_id` | Integer | The token ID that signals the end of a sequence. For Llama 3, this is `128009` (`<|eot_id|>`). This is usually handled automatically but can be overridden. | e.g., `128009` |

#### Advanced Control

| Parameter | Data Type | Description | Default/Example |
| :--- | :--- | :--- | :--- |
| `beam_width` | Integer | The number of beams to use for beam search. A value of `1` (the default) effectively uses standard sampling. A value > 1 enables beam search, which can produce higher-quality results at the cost of performance. | e.g., `1` |
| `num_return_sequences`| Integer | The number of alternative sequences to generate. Requires `beam_width` to be at least this large. | e.g., `1` |
| `return_log_probs` | Boolean | If `true`, returns the log probabilities of the generated tokens. Useful for analysis but adds computational overhead. | `false` |
---




## Step 2:Turning it into a managed, reliable background service.

The best and most standard way to do this on a modern Linux system (like Ubuntu, which you appear to be using) is with `systemd`. This is the system and service manager that controls almost everything on your machine.

Creating a `systemd` service will give you:
-   **Automatic Startup:** The service will start automatically when your machine boots.
-   **Automatic Restart:** If your application crashes for any reason, `systemd` will automatically restart it.
-   **Centralized Logging:** All output from your script is automatically captured and managed by the system's logger (`journalctl`).
-   **Simple Commands:** Easy, standardized commands like `start`, `stop`, `restart`, and `status`.

Here is the complete, step-by-step guide.

---

### Step 2.1: Create the `systemd` Service File

This file is a simple text file that tells `systemd` everything it needs to know about your service.

First, you need to create this file in the system-wide services directory. You will need `sudo` privileges.

```bash
# Create and open the service file with the nano text editor
sudo nano /etc/systemd/system/llm_service.service
```

Now, **copy the following content and paste it into the `nano` editor**.

**Crucially, you must replace the placeholders `<your_username>` and `<full_path_to_your_project>` with your actual values.**

```ini
[Unit]
Description=LLM Inference API Service
After=network.target docker.service

[Service]
# --- IMPORTANT: Replace with your actual user and group ---
User=<your_username>
Group=<your_username>

# --- IMPORTANT: Replace with the full path to your project directory ---
WorkingDirectory=<full_path_to_your_project>

# --- IMPORTANT: Replace with your actual user ---
# This sets the PATH so systemd can find the correct python/gunicorn
Environment="PATH=/home/<your_username>/<anaconda3 or miniconda3>/envs/llm_service/bin:/usr/bin:/bin"

# This command uses the full path to the script for reliability
ExecStart=/bin/bash <full_path_to_your_project>/run.sh

# Restart the service if it fails
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

#### How to Find Your Placeholders:

1.  **`<your_username>`**: Your username is `ansary`.
2.  **`<full_path_to_your_project>`**:
    -   Navigate to your project directory in your terminal: `cd ~/services/Triton-TRTLLM-LLama3.1-8b-Instruct-A6000`
    -   Run the command `pwd`. It will print the full path, for example: `/home/ansary/services/Triton-TRTLLM-LLama3.1-8b-Instruct-A6000`.
    -   Use this full path in the service file.
3.  **anaconda or miniconda**: whichever you installed and using

#### Your Final `llm_service.service` File Should Look Like This:

```ini
[Unit]
Description=LLM Inference API Service
# Wait for the network and docker to be up before starting
After=network.target docker.service

[Service]
# Run the service as your user to handle permissions correctly
User=ansary
Group=ansary

# Start the service from your project's root directory
WorkingDirectory=/home/ansary/services/Triton-TRTLLM-LLama3.1-8b-Instruct-A6000-docker

# Set the PATH environment variable so systemd can find the conda environment's gunicorn
Environment="PATH=/home/ansary/anaconda3/envs/llm_service/bin:/usr/bin:/bin"

# The command to execute, using absolute paths
ExecStart=/bin/bash /home/ansary/services/Triton-TRTLLM-LLama3.1-8b-Instruct-A6000-docker/run.sh

# Automatically restart the service if it crashes
Restart=on-failure
RestartSec=5

[Install]
# Enable the service to start on boot
WantedBy=multi-user.target
```

Once you have pasted and edited the content, save the file and exit `nano` by pressing `Ctrl+X`, then `Y`, then `Enter`.

---

### Step 2.2: Install and Manage the Service

Now you will use the `systemctl` command to control your new service.

**1. Reload the `systemd` daemon:**
This tells `systemd` to read your new service file.
```bash
sudo systemctl daemon-reload
```

**2. Enable the service:**
This registers your service to start automatically on system boot.
```bash
sudo systemctl enable llm_service
```

**3. Start the service:**
This starts the service right now. You will no longer see the output in your terminal because it's running in the background.
```bash
sudo systemctl start llm_service
```

### Step 2.3: Controlling Your Service (The Commands You Wanted)

You now have a powerful set of commands to manage your application.

#### Check the Status
This is the most important command. It will tell you if the service is `active (running)` or if it has failed.
```bash
sudo systemctl status llm_service
```
A healthy output will look like this:
```
● llm_service.service - LLM Inference API Service
     Loaded: loaded (/etc/systemd/system/llm_service.service; enabled; vendor preset: enabled)
     Active: active (running) since Thu 2025-06-27 06:30:00 +06; 1min 10s ago
   Main PID: 1450123 (bash)
      Tasks: 6 (limit: 9389)
     Memory: 500.0M
        CPU: 10.5s
     CGroup: /system.slice/llm_service.service
             ...
```

#### Stop the Service (The "Kill" Command)
This will gracefully stop your application.
```bash
sudo systemctl stop llm_service
```

#### Restart the Service
Use this after you make any changes to your Python code. It's a convenient one-step stop and start.
```bash
sudo systemctl restart llm_service
```

### Step 2.4: Viewing Logs

Since the service is running in the background, you no longer see its logs in your terminal. `systemd` automatically captures them for you via `journalctl`.

#### View All Logs for Your Service
```bash
sudo journalctl -u llm_service.service
```

#### View Live, Tailing Logs (Most Useful!)
This is the equivalent of `tail -f`. It will show you a live stream of logs as they happen, which is perfect for debugging.
```bash
sudo journalctl -u llm_service.service -f
```

Press `Ctrl+C` to exit the live view.

You have now successfully converted your script into a robust, manageable, production-style system service.
---


Of course. Here is the new section for your documentation, designed to follow "Step 0" and fit perfectly within the "Inference" chapter.

---

Of course. I have revised the section according to your instructions, replacing the concurrent request example with clear, focused examples of streaming in both Python and JavaScript.

Here is the updated section.

---

## **Step 3: Isolated Inference: Consuming the API from Any Application**

The true power of wrapping our Triton server with a FastAPI service is that we've created a **standard, universal interface**. Our high-performance LLM engine now speaks the common language of the web: HTTP and JSON.

This means you can consume the service from virtually any programming language, application, or environment with **extremely low dependency requirements**. You are no longer tied to specific Python libraries or a complex gRPC setup. Your application is now **isolated** from the inference backend; it only needs to know how to make a web request.

Let's demonstrate how easy this is with practical examples in both Python and JavaScript.

### 3.1. Python: Using `requests` and `httpx`

For Python applications, you only need a standard HTTP client library.

-   **Dependency:** The de-facto standard `requests` library for simple calls, or `httpx` for modern asynchronous streaming.
    ```bash
    # Install if you don't have them
    pip install requests httpx
    ```

**Example 1: Simple Synchronous Request**

This is perfect for scripts or backend tasks where you wait for the full response before proceeding.

```python
import requests
import json

# The endpoint for non-streaming generation
url = "http://localhost:24434/generate"

# Define the prompt and control parameters
payload = {
    "prompt": "Write a short, dramatic story about a lonely lighthouse keeper who discovers a message in a bottle.",
    "temperature": 0.8,
    "max_tokens": 256,
    "repetition_penalty": 1.15
}

print("Sending request...")
try:
    # Make the POST request
    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

    # Print the clean JSON response
    response_data = response.json()
    print("\n--- Full Response Received ---")
    print(json.dumps(response_data, indent=2))

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
```

**Example 2: Streaming the Response Asynchronously**

This pattern is ideal for interactive applications (like chatbots) where you want to display the response to the user as it's being generated. We use `httpx` for its excellent async and streaming support.

```python
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
```

You can find this codes that displays : 
1. sync non stream: ```tests/non_stream_req.py```
2. async stream: ```test/stream_req.py```
3. concurrent_batch_stream: ```tests/concurrent_batch_stream.py```
> no 3- It will show jumbled data because all the requests are being processed with streaming concurrently 



###  3.2. JavaScript (Browser / Node.js): Using the `fetch` API (NOT TESTED:  BUT YOU WILL GET THE IDEA RIGHT!!)

For any web front-end or Node.js backend, you can use the built-in `fetch` API.

-   **Dependency:** **None.** `fetch` is native to all modern browsers and Node.js (v18+).

**Example: Calling the Streaming Endpoint from JavaScript**

This code could run directly in a browser's developer console or be part of a web application's logic to display a response to a user in real-time.

```javascript
// The endpoint for streaming generation
const url = 'http://localhost:24434/generate_stream';

const payload = {
    prompt: "Explain the concept of In-Flight Batching for LLMs like you're explaining it to a 5th grader.",
    temperature: 0.5,
    max_tokens: 300,
};

async function streamLlamaResponse() {
    console.log("Sending streaming request to the LLM service...");
    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        // Process the stream
        while (true) {
            const { done, value } = await reader.read();
            if (done) {
                console.log("\n--- Stream finished ---");
                break;
            }
            
            // Decode the chunk of data into a raw string.
            const text_chunk = decoder.decode(value);
            
            // No JSON parsing needed! Just write the raw text to the console.
            // This assumes a Node.js environment. In a browser, you'd
            // append this to an HTML element's innerText.
            process.stdout.write(text_chunk); 
        }
    } catch (error) {
        console.error("\nFailed to fetch LLM response:", error);
    }
}

// Run the function
streamLlamaResponse();
```



---

### 3.3 Using All Available Parameters

The FastAPI service exposes all the powerful generation parameters supported by the TensorRT-LLM backend. You can fine-tune the model's behavior on a per-request basis by including these in your JSON payload.

Below are examples in Python and JavaScript showing a request that uses a comprehensive set of these parameters.

#### Python Example: Full Parameter Control

```python

# A payload demonstrating many available parameters
full_payload = {
    "prompt": "Invent a name and a short, one-sentence description for a fantasy novel about a cursed map.",
    
    # --- Core Generation Control ---
    "max_tokens": 128,
    "min_tokens": 15,
    
    # --- Sampling & Creativity Control ---
    "temperature": 0.85,
    "top_k": 40,
    "top_p": 0.9,
    "seed": 42,
    
    # --- Content & Penalty Control ---
    "repetition_penalty": 1.2,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "length_penalty": 1.0,

    # --- Stopping & Filtering ---
    "stop_words": ["The End.", "\n\n"],
    "bad_words": ["cliche", "unoriginal"],
    
    # --- Advanced Control (Defaults are usually fine) ---
    "beam_width": 1, 
    "return_log_probs": False
}


```

#### JavaScript Example: Full Parameter Control

This example can be run in a Node.js environment or adapted for the browser.

```javascript
// A payload demonstrating many available parameters
const fullPayload = {
    prompt: "Invent a name and a short, one-sentence description for a fantasy novel about a cursed map.",

    // --- Core Generation Control ---
    max_tokens: 128,
    min_tokens: 15,

    // --- Sampling & Creativity Control ---
    temperature: 0.85,
    top_k: 40,
    top_p: 0.9,
    seed: 42,

    // --- Content & Penalty Control ---
    repetition_penalty: 1.2,
    presence_penalty: 0.0,
    frequency_penalty: 0.0,
    length_penalty: 1.0,

    // --- Stopping & Filtering ---
    stop_words: ["The End.", "\n\n"],
    bad_words: ["cliche", "unoriginal"],

    // --- Advanced Control (Defaults are usually fine) ---
    beam_width: 1,
    return_log_probs: false,
};

```

# Insights 


## **Unit Testing the API: Ensuring Reliability and Understanding Behavior**

A robust API is a tested API. This repository includes a comprehensive test suite using `pytest` to verify the functionality, reliability, and specific behaviors of the FastAPI service and the underlying Triton backend.

These tests serve three primary purposes:
1.  **Correctness:** To ensure that all endpoints work as expected and that parameters are correctly passed to the model.
2.  **Validation:** To confirm that the API correctly rejects invalid or malformed requests, protecting the backend from bad data.
3.  **Behavioral Analysis:** To gain deep insights into how the TensorRT-LLM backend interprets parameters, which is crucial for predictable and controllable generation.

The tests are located in the `/tests` directory and can be run with `pytest`. They provide a living specification of how the service is intended to work.

* the pytests can be run with: ```pytest -v tests/```
* It purely tests the api's behaviour under various conditions and it should display the following 

```bash
==================================================================== test session starts ====================================================================
platform linux -- Python 3.10.18, pytest-8.4.1, pluggy-1.6.0 -- /home/ansary/anaconda3/envs/llm_service/bin/python3.10
cachedir: .pytest_cache
rootdir: /home/ansary/services/Triton-TRTLLM-LLama3.1-8b-Instruct-A6000-docker
plugins: anyio-4.9.0, asyncio-1.0.0
asyncio: mode=strict, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collected 21 items                                                                                                                                          

tests/test_api.py::test_health_check PASSED                                                                                                           [  4%]
tests/test_api.py::test_generate_basic_success PASSED                                                                                                 [  9%]
tests/test_api.py::test_generate_with_zero_temperature_is_deterministic PASSED                                                                        [ 14%]
tests/test_api.py::test_generate_with_high_temperature_is_different PASSED                                                                            [ 19%]
tests/test_api.py::test_generate_with_seed_has_an_effect PASSED                                                                                       [ 23%]
tests/test_api.py::test_generate_with_zero_temp_is_truly_deterministic PASSED                                                                         [ 28%]
tests/test_api.py::test_generate_with_max_tokens PASSED                                                                                               [ 33%]
tests/test_api.py::test_generate_with_bad_words PASSED                                                                                                [ 38%]
tests/test_api.py::test_generate_with_stop_words PASSED                                                                                               [ 42%]
tests/test_api.py::test_generate_with_various_sampling_params[top_k-5] PASSED                                                                         [ 47%]
tests/test_api.py::test_generate_with_various_sampling_params[top_p-0.5] PASSED                                                                       [ 52%]
tests/test_api.py::test_generate_with_various_sampling_params[repetition_penalty-1.5] PASSED                                                          [ 57%]
tests/test_api.py::test_generate_with_various_sampling_params[presence_penalty-0.5] PASSED                                                            [ 61%]
tests/test_api.py::test_generate_with_various_sampling_params[frequency_penalty-0.5] PASSED                                                           [ 66%]
tests/test_api.py::test_generate_custom_request_id PASSED                                                                                             [ 71%]
tests/test_api.py::test_generate_missing_prompt_validation PASSED                                                                                     [ 76%]
tests/test_api.py::test_generate_invalid_param_validation PASSED                                                                                      [ 80%]
tests/test_api.py::test_stream_basic_success PASSED                                                                                                   [ 85%]
tests/test_api.py::test_stream_produces_multiple_chunks PASSED                                                                                        [ 90%]
tests/test_api.py::test_stream_with_stop_words PASSED                                                                                                 [ 95%]
tests/test_api.py::test_stream_missing_prompt_validation PASSED                                                                                       [100%]

=============================================================== 21 passed in 61.87s (0:01:01) ===============================================================

```

### Test Suite Summary

The following table provides a detailed breakdown of each test case, its purpose, and the specific behavior it validates.

#### General and Health Tests

| Test Function | Purpose |
| :--- | :--- |
| `test_health_check` | Verifies that the service is running and the `/health` endpoint returns a `200 OK` status with the expected JSON payload. |

#### Non-Streaming Endpoint (`/generate`)

| Test Function | Purpose |
| :--- | :--- |
| `test_generate_basic_success` | **Primary Happy Path:** Confirms that a standard, valid request receives a `200 OK` response with generated text. |
| `test_generate_with_zero_temperature_is_deterministic` | **Reproducibility Test:** Verifies that with `temperature=0.0`, two identical requests produce the exact same output. |
| `test_generate_with_high_temperature_is_different` | **Creativity Test:** Confirms that with `temperature > 0.0`, two requests with different `seed` values produce different, non-deterministic outputs. |
| `test_generate_with_seed_has_an_effect` | **Seed Functionality Test:** Proves that the `seed` parameter is respected by showing that different seeds lead to different creative outputs. |
| `test_generate_with_zero_temp_is_truly_deterministic` | **Greedy Decoding Test:** A stricter version of the determinism test, proving that `temperature=0.0` ignores the seed and always produces the same result. |
| `test_generate_with_max_tokens` | **Output Length Control:** Ensures the `max_tokens` parameter effectively limits the length of the generated response. |
| `test_generate_with_bad_words` | **Content Filtering (Nuanced):** Tests that the `bad_words` filter can successfully prevent a forbidden word from being generated in a non-greedy context. |
| `test_generate_with_stop_words` | **Generation Halting:** Verifies the precise behavior of `stop_words`: the model generates the stop word and then immediately ceases generation. |
| `test_generate_with_various_sampling_params` | **Parameter Compatibility:** A parameterized test that ensures various sampling parameters (`top_k`, `top_p`, penalties) are accepted and processed without error. |
| `test_generate_custom_request_id` | **Metadata Passthrough:** Confirms that a custom `request_id` sent in the payload is correctly returned in the response. |
| `test_generate_missing_prompt_validation` | **Required Field Validation:** Checks that the API correctly returns a `422 Unprocessable Entity` error if the mandatory `prompt` field is missing. |
| `test_generate_invalid_param_validation` | **Parameter Boundary Validation:** Checks that the API correctly returns a `422` error if a parameter is given a value outside its valid range (e.g., `temperature > 2.0`). |

#### Streaming Endpoint (`/generate_stream`)

| Test Function | Purpose |
| :--- | :--- |
| `test_stream_basic_success` | **Primary Streaming Happy Path:** Confirms that a valid streaming request receives a `200 OK` response and successfully streams back text content. |
| `test_stream_produces_multiple_chunks` | **Stream Viability:** Ensures that for a longer generation, the response is broken into multiple, distinct chunks, proving the streaming mechanism is working correctly. |
| `test_stream_with_stop_words` | **Streaming Content Control:** Verifies that the `stop_words` parameter functions correctly in streaming mode, halting the stream after the stop word is sent. |
| `test_stream_missing_prompt_validation` | **Streaming Endpoint Validation:** Confirms that the FastAPI validation layer also protects the streaming endpoint, returning a `422` error for invalid requests. |

#### Key Insights from the Test Suite

The test suite reveals several important characteristics of our production-ready LLM service.

##### 1. Core Functionality and Robust Validation

-   **Health & Availability:** The `test_health_check` confirms the service is running and responsive.
-   **Endpoint Success:** `test_generate_basic_success` and `test_stream_basic_success` are the "happy path" tests, ensuring both non-streaming and streaming endpoints can successfully generate text without errors.
-   **Input Validation:** The tests for missing or invalid parameters (e.g., `test_generate_missing_prompt_validation`) prove that the Pydantic models in our FastAPI service are working correctly, returning a `422 Unprocessable Entity` status code before a request ever reaches Triton. This is a critical feature for a production API.

##### 2. Controlling Creativity and Determinism

This is where the tests provide the most valuable "scientific" insights into the model's behavior.

-   **Perfect Determinism (`temperature=0.0`):** The `test_generate_with_zero_temp_is_truly_deterministic` test proves a critical concept: when `temperature` is `0.0`, the model enters a **greedy decoding** mode. In this mode, it will *always* produce the exact same output for a given prompt, and the `seed` parameter is correctly ignored. This is essential for tasks requiring absolute reproducibility.

-   **Controlled Stochasticity (`temperature > 0.0`):**
    -   The `test_generate_with_high_temperature_is_different` test confirms that with a higher temperature, using **different seeds** results in **different outputs**. This proves the model is behaving stochastically as intended.
    -   The `test_generate_with_seed_has_an_effect` test, when combined with the previous one, confirms that providing the **same seed** with the same high temperature will produce the **exact same "creative" output**.

**Conclusion:** The tests give us confidence that we have full control over the reproducibility of our generations. We can either lock the model into a deterministic mode or perfectly replicate any creative output on demand.

##### 3. Content Control and Filtering

These tests verify our ability to guide and constrain the model's output.

-   **`stop_words` Behavior:** The `test_stream_with_stop_words` test reveals a precise and important behavior: the backend generates the stop word itself and **then immediately halts the process**. The final output will contain the stop word. This is different from systems that might stop *before* producing the word, and understanding this is key to using the feature effectively.

-   **A Deep Dive on the `bad_words` Limitation:** The `test_generate_with_bad_words` test passes, but it uncovers a known, nuanced limitation in some high-performance backends.
    -   **The Insight:** If a forbidden word is the overwhelmingly most probable *first token* for the model to generate (e.g., answering "The sky is" with "blue"), the `bad_words` filter may not be able to override this deterministic, high-probability choice. It's like trying to switch a train's tracks when it's already past the junction.
    -   **The Engineering Trade-off:** This is often an intentional design choice in performance-oriented backends. Implementing a filter powerful enough to override a greedy choice can introduce latency into the critical generation path.
    -   **The Test's Proof:** The test is specifically designed to show that the feature **works correctly in more common scenarios**—where the forbidden word is a likely, but not guaranteed, choice. By prompting with "My favorite color is not red, it is", we confirm the filter can successfully prevent "blue" from being chosen and guide the model to an alternative.

By understanding these tested behaviors, developers can use the API with confidence, knowing exactly how to achieve the results they need, from perfectly reproducible outputs to creatively guided text.


---

---

# L40s Installation 

-   [5.1 Convert Hugging Face Model to TensorRT-LLM Checkpoint](#51-convert-hugging-face-model-to-tensorrt-llm-checkpoint)
-   [5.2 Build the Optimized Engine from the Checkpoint](#52-build-the-optimized-engine-from-the-checkpoint)
-   [6.1 Prepare the Triton Model Repository](#61-prepare-the-triton-model-repository)


---

### 5.1 (L40s) Convert Hugging Face Model to TensorRT-LLM Checkpoint 
```bash
# Inside container: Run the conversion script
python examples/llama/convert_checkpoint.py \
    --model_dir /data/llama-3.1-8b-hf \
    --output_dir /data/llama-3.1-8b-fp8-tllm-ckpt \
    --dtype bfloat16 \
    --use_fp8_rowwise \
    --fp8_kv_cache \
    --use_meta_fp8_rowwise_recipe \
    --calib_dataset cnn_dailymail \
    --calib_size 32 \
    --calib_max_seq_length 512
```
#### `convert_checkpoint.py` Arguments
| Category | Argument | Your Value | Justification |
|:---|:---|:---|:---|
| **I/O** | `--model_dir` | `/data/llama-3.1-8b-hf` | **Required.** Points to the source Hugging Face model directory. |
| | `--output_dir` | `/data/llama-3.1-8b-fp8-tllm-ckpt` | **Required.** Specifies the destination for the converted FP8 checkpoint. |
| **Quantization** | `--dtype` | `bfloat16` | **Best Choice.** Sets the base precision for non-quantized layers and calculations. `bfloat16` is highly performant and numerically stable on the Ada (L40S) architecture. |
| | `--use_fp8_rowwise`| `(Set)` | **Critical Fix.** Enables the per-token, per-channel FP8 quantization mode. This is the essential flag that correctly initializes the FP8 quantization pathway, which the meta recipe relies upon. |
| | `--use_meta_fp8_rowwise_recipe`| `(Set)`| **Expert Choice.** Applies a validated, high-accuracy FP8 quantization strategy from Meta. It modifies the base `fp8_rowwise` behavior by intelligently skipping quantization on sensitive layers to preserve model quality. |
| **KV Cache** | `--fp8_kv_cache` | `(Set)` | **Essential for Long Context.** Quantizes the Key-Value cache to FP8, which is critical for reducing the massive VRAM footprint of a 96k context window and improving memory bandwidth. |
| **Calibration**| `--calib_dataset` | `cnn_dailymail` | **Required.** Provides a dataset to generate the FP8 scaling factors (`amax` values) on-the-fly, as they don't exist in the source model. This resolves the `TypeError`. |
| | `--calib_size` | `32` | **Required.** Specifies using 32 samples from the dataset for calibration, which is sufficient to accurately estimate the scaling factors without a lengthy process. |
| | `--calib_max_seq_length` | `512` | **Required.** Manages memory during the calibration step by setting a maximum sequence length. This does not limit the final model's context length. |

### 5.2(L40s) Build the Optimized Engine from the Checkpoint

This command uses the checkpoint to build the final, runnable engine, configured for our specific needs (96k input length, paged attention).

```bash
# Inside container: Build the engine
trtllm-build \
    --checkpoint_dir /data/llama-3.1-8b-fp8-tllm-ckpt \
    --output_dir /data/llama-3.1-8b-96k-fp8-engine/ \
    --max_batch_size 4 \
    --max_input_len 96000 \
    --max_seq_len 131072 \
    --max_num_tokens 131072\
    --kv_cache_type paged \
    --fp8_rowwise_gemm_plugin auto \
    --gpt_attention_plugin auto \
    --use_paged_context_fmha enable \
    --use_fp8_context_fmha enable \
    --multiple_profiles enable \
    --use_fused_mlp enable
```

Of course. Here is the justification table for the final, optimized command.

#### `trtllm-build` Arguments

**Keep the max_num_tokens=131072 for using the model with full capacity this is crucial**

| Category | Argument | Your Value | Justification |
|:---|:---|:---|:---|
| **I/O** | `--checkpoint_dir` | `/data/llama-3.1-8b-fp8-tllm-ckpt` | **Required.** Points to the FP8-quantized checkpoint from the conversion step. |
| | `--output_dir` | `/data/llama-3.1-8b-96k-fp8-engine/` | **Required.** Destination for the final, compiled FP8 engine files. |
| **Engine Capacity**| `--max_batch_size` | `4` | **User Defined.** Sets the maximum number of concurrent requests the engine can schedule. |
| | `--max_input_len`| `96000` | **Your Goal.** Allocates resources to handle input prompts up to 96,000 tokens long. |
| | `--max_seq_len` | `131072` | **Your Goal.** Sets the total memory for prompt + generated tokens, providing a robust buffer for output. |
| **Plugins (FP8)** | `--fp8_rowwise_gemm_plugin`| `auto` | **Critical.** Enables the specific GEMM plugin that corresponds to the `use_fp8_rowwise` conversion, ensuring the correct high-performance FP8 kernels are used. |
| | `--use_fp8_context_fmha`| `enable` | **Critical.** Accelerates prompt processing by running the Fused Multi-Head Attention kernel in FP8, leveraging the L40S's hardware. |
| **Plugins (Attention & Memory)** | `--kv_cache_type` | `paged` | **Essential.** Enables the modern, efficient paged KV cache memory manager, which is necessary for handling extremely long context lengths like 96k. |
| | `--gpt_attention_plugin`| `auto` | **Best Practice.** Automatically selects the most optimized Fused Multi-Head Attention kernels for the generation phase. |
| | `--use_paged_context_fmha`| `enable` | **Essential.** The companion to `kv_cache_type paged`, this enables the specific attention kernel designed to operate on the paged memory layout. |
| **Plugins (Performance)** | `--use_fused_mlp` | `enable` | **Performance.** Fuses the GEMM and activation functions in the MLP block into a single kernel, reducing overhead and significantly improving throughput. |
| | `--multiple_profiles`| `enable` | **Performance.** Instructs TensorRT to build and test more kernel variations, improving the engine's performance across different sequence lengths at the cost of longer build time. |


Once the build is complete, you can leave the interactive container: The reason to leave the container is to keep it clean from the heavy TensorRT-LLM repo. 
```bash
# Inside container:
exit
```

---


### 6.1(L40s) Prepare the Triton Model Repository

These steps are also performed *inside a container*.

```bash
# Start a new interactive container
docker run --gpus all -it --rm \
  -v ~/trtllm-data:/data \
  -v ~/trtllm-triton-repo:/triton_repo \
  nvcr.io/nvidia/tritonserver:25.05-trtllm-python-py3
```

Inside the new container, run these commands:

```bash
# Inside container: Clone the backend repository
cd /workspace
git clone -b r25.05 https://github.com/triton-inference-server/tensorrtllm_backend.git

# Inside container: Copy the model template files
cp -r /workspace/tensorrtllm_backend/all_models/inflight_batcher_llm/* /triton_repo/
```

#### ensambe
```bash
# Inside container: Use the fill_template.py script to configure the models
# Note the engine_dir path now correctly points to our built engine.

# ensemble/config.pbtxt
python /workspace/tensorrtllm_backend/tools/fill_template.py -i /triton_repo/ensemble/config.pbtxt \
    triton_max_batch_size:4,logits_datatype:TYPE_FP32
```
#### Configure `ensemble/config.pbtxt`

This is the top-level configuration that defines the entire inference pipeline, chaining the preprocessing, business logic, core model, and postprocessing steps together.

| Parameter | Your Value | Grounded Logic & Reasoning |
| :--- | :--- | :--- |
| **`triton_max_batch_size`** | `4` | **Why:** This value **must match** the `--max_batch_size 4` you used when you ran `trtllm-build`. This top-level ensemble config tells Triton the maximum number of requests it can group together before sending them down the pipeline. Consistency is critical for the server to function correctly. |
| **`logits_datatype`** | `TYPE_FP32` | **Why:** This controls the precision of the model's final output scores before sampling. `TYPE_FP32` is the safest and most standard choice for preserving numerical accuracy and avoiding potential issues during the final token selection (e.g., in temperature or top-p sampling). |


#### preprocessing
```bash
# preprocessing/config.pbtxt
python /workspace/tensorrtllm_backend/tools/fill_template.py -i /triton_repo/preprocessing/config.pbtxt \
    triton_max_batch_size:4,tokenizer_dir:/data/llama-3.1-8b-hf,preprocessing_instance_count:1
```

#### Configure `preprocessing/config.pbtxt`

This model is responsible for taking the raw input string from the user and converting it into token IDs that the TensorRT-LLM engine can understand.

| Parameter | Your Value | Grounded Logic & Reasoning |
| :--- | :--- | :--- |
| **`triton_max_batch_size`** | `4` | **Why:** This must be consistent across all configs. It matches the `max_batch_size` used for the ensemble config and the `trtllm-build` command. |
| **`tokenizer_dir`** | `/data/llama-3.1-8b-hf` | **Why:** This is the absolute path *inside the container* to the directory where the Hugging Face tokenizer files (`tokenizer.model`, `tokenizer.json`, etc.) are located. The Python backend for this step needs this exact path to load the vocabulary and correctly convert input text into token IDs. |
| **`preprocessing_instance_count`** | `1` | **Why:** This sets how many parallel CPU processes to use for tokenization. `1` is a safe, standard default. You could increase this later if performance monitoring reveals that text tokenization is a CPU bottleneck for your service under heavy load. |


#### postprocessing

```bash
# postprocessing/config.pbtxt
python /workspace/tensorrtllm_backend/tools/fill_template.py -i /triton_repo/postprocessing/config.pbtxt \
    triton_max_batch_size:4,tokenizer_dir:/data/llama-3.1-8b-hf,postprocessing_instance_count:1
```

#### Configure `postprocessing/config.pbtxt`

This model does the reverse of preprocessing: it takes the output token IDs generated by the engine and converts them back into human-readable text.

| Parameter | Your Value | Grounded Logic & Reasoning |
| :--- | :--- | :--- |
| **`triton_max_batch_size`** | `4` | **Why:** This must be consistent across all configs and match the batch size you built the engine with (`4`). |
| **`tokenizer_dir`** | `/data/llama-3.1-8b-hf` | **Why:** This is the absolute path to the tokenizer files. The Python backend for this step needs this path to correctly map the output token IDs back into words. |
| **`postprocessing_instance_count`** | `1` | **Why:** This sets how many parallel CPU processes to use for detokenization. `1` is a safe, standard default. It can be increased later if postprocessing becomes a bottleneck. |

#### tensorrt_llm

```bash
# tensorrt_llm/config.pbtxt (Corrected)
python /workspace/tensorrtllm_backend/tools/fill_template.py -i /triton_repo/tensorrt_llm/config.pbtxt \
    triton_backend:tensorrtllm,triton_max_batch_size:4,decoupled_mode:True,max_beam_width:1,engine_dir:/data/llama-3.1-8b-96k-fp8-engine,batching_strategy:inflight_fused_batching,max_tokens_in_paged_kv_cache:131072,enable_chunked_context:True,exclude_input_in_output:True,logits_datatype:TYPE_FP32,encoder_input_features_data_type:TYPE_FP16
```
#### configure `tensorrt_llm/config.pbtxt` Arguments : THE MOST IMPORTANT ONE

This table explains the parameters for the final `tensorrt_llm/config.pbtxt` configuration.

| Parameter | Your Value | Grounded Logic & Reasoning |
|:---|:---|:---|
| **`engine_dir`** | `/data/llama-3.1-8b-96k-fp8-engine` | **Logic:** This is the absolute path inside the container to your compiled FP8 engine. It **must** match the `--output_dir` from the `trtllm-build` step. |
| **`batching_strategy`** | `inflight_fused_batching` | **Logic:** Enables the core feature for high throughput: in-flight batching. It dynamically manages and batches requests to maximize GPU utilization, which is essential for serving multiple users. |
| **`decoupled_mode`** | `True` | **Logic:** Enables streaming output, where tokens are sent back as they are generated. This is a requirement for any interactive chat or real-time application. |
| **`max_tokens_in_paged_kv_cache`**| `131072` | **Logic:** **Critical for stability.** This defines the total KV cache pool size in tokens (`max_batch_size` * `max_seq_len`). This value is large enough to handle a single full batch of worst-case, max-length requests, preventing memory exhaustion. |
| **`enable_chunked_context`** | `True` | **Logic:** **Essential for long context.** This processes the huge 96k input prompt in smaller chunks, which prevents a single massive memory allocation that could cause out-of-memory errors and improves stability. |
| **`exclude_input_in_output`**| `True` | **Logic:** A standard setting for chat models. It prevents the server from echoing the user's (potentially very long) prompt, saving bandwidth and compute. |
| **`encoder_input_features_data_type`**| `TYPE_FP16` | **Logic:** This specifies the data type for certain inputs. Since our model's `--dtype` was `float16`, this ensures compatibility. |
| **`logits_datatype`** | `TYPE_FP32` | **Logic:** Ensures numerical stability and consistency. It matches the data type for logits specified in the other configuration files, ensuring data flows correctly through the pipeline. |
| **Others** | `triton_backend:tensorrtllm`, `triton_max_batch_size:4`, etc. | **Logic:** These ensure consistency with the other configuration files and the engine build itself. They are mandatory and must match across the entire pipeline. |

#### tensorrt_llm_bls (bussiness logic) 

```bash
# tensorrt_llm_bls/config.pbtxt
python /workspace/tensorrtllm_backend/tools/fill_template.py -i /triton_repo/tensorrt_llm_bls/config.pbtxt \
    triton_max_batch_size:4,decoupled_mode:True,bls_instance_count:1,accumulate_tokens:False,logits_datatype:TYPE_FP32
```

#### Configure `tensorrt_llm_bls/config.pbtxt`

This model implements the Business Logic Scripting (BLS) for the pipeline. It acts as the "manager" that orchestrates the flow of data to and from the core `tensorrt_llm` model, especially for complex patterns like in-flight batching and streaming.

| Parameter | Your Value | Grounded Logic & Detailed Reasoning |
|:---|:---|:---|
| **`triton_max_batch_size`** | `4` | **Logic:** Ensures consistency across the entire pipeline. It must match the value used in `trtllm-build` and all other `config.pbtxt` files. |
| **`decoupled_mode`** | `True` | **Logic:** **Crucial for streaming.** This must match the `decoupled_mode:True` setting from the `tensorrt_llm/config.pbtxt`. It tells this "manager" component to operate in an asynchronous, non-blocking way, which is required to handle the streaming of tokens from the backend engine. |
| **`bls_instance_count`** | `1` | **Logic:** This defines how many parallel CPU processes run this business logic script. `1` is a safe and standard default. This can be tuned later if this specific step is identified as a performance bottleneck. |
| **`accumulate_tokens`** | `False` | **Logic:** **Also crucial for streaming.** When set to `False`, this component does *not* wait for the entire sequence to be generated. It passes tokens through to the `postprocessing` step as soon as they arrive from the engine. If this were `True`, it would buffer the entire response and destroy the streaming effect. |
| **`logits_datatype`** | `TYPE_FP32` | **Logic:** Ensures numerical stability and consistency. It matches the data type for logits specified in the other configuration files, ensuring data flows correctly through the pipeline. |



Once done, exit the container: `exit`.

now got back to [**step 6.2**](#62-correct-host-file-permissions)
---



# TODO 
* Add adapters like: 
    * standalone 
    * lanchain 
    * openai 
* unittest the adapters 
* benchmark concurrency and load testing

# Customization References

If you want to customize your own service the following links might help. 

- https://github.com/triton-inference-server/tensorrtllm_backend
- https://github.com/NVIDIA/TensorRT-LLM
- https://nvidia.github.io/TensorRT-LLM/installation/linux.html 