# Multi-Agent Agentic RAG (Local)

## Prerequisites

Ensure you have the following installed on your machine:
- Docker Desktop (required for Magic-PDF and vLLM)
- Conda (for creating and managing environments)

## Project Setup

Follow these steps to set up the development environment for this project.

Note: magic-pdf explicitly required transformers==4.42.4 to work and the project required an updated transformers version, hence, creating different containers with different environments.

### 1. *Build and start the Docker container using `docker-compose`*
 ```bash
 docker-compose up --build -d
 ```

### 2. *Pull vLLM image from dockerhub*
```bash
docker pull vllm/vllm-openai
```

### 3. *Create a .env file containing:*
1. HF_TOKEN_WRITE
2. HF_TOKEN_READ
3. TAVILY_API_KEY
4. VLLM_API_KEY

### 4. *Download model from huggingface*
```bash
python download_huggingface_model.py
```

### 5. *Build container using vllm-openai image*
```bash
docker run --gpus all `
   --name vllm_qwen2.5_14b_instruct_bnb_4bit `
   -v "*path to downloaded model directory that contains config.json (e.g, ...\snapshots\f010a5cd44911b4fff441fcfa67200643ed811c4)*:/model" `
   -e VLLM_ATTENTION_BACKEND=FLASHINFER `
   --ipc=host `
   -p 8080:8080 `
   vllm/vllm-openai:latest `
   --model "/model" `
   --max-num-seqs 8 `
   --seed 42 `
   --max-model-len 6144  `
   --port 8080 `
   --quantization bitsandbytes `
   --load-format bitsandbytes `
   --api-key vllm `
   --dtype bfloat16 `
   --kv-cache-dtype fp8_e4m3
```

### 4. *Create conda environment*
```bash
conda create -n agentic_rag python=3.10
conda activate agentic_rag
```

### 5. *Install PyTorch with CUDA Compatibility*
```bash
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
```

### 6. *Install requirements*
```bash
pip install -r requirements.txt
```

### 7. *Clear cache after everything installed successfully*
```bash
pip cache purge
conda clean --all -y
```