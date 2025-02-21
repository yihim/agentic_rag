# Multi-Agent Agentic RAG (Local)

## Prerequisites

Ensure you have the following installed on your machine:
- Docker Desktop (required for Magic-PDF and vLLM)
- Conda (for creating and managing environments)

## Project Setup

Follow these steps to set up the development environment for this project.

Note: magic-pdf explicitly required transformers==4.42.4 to work and the project required an updated transformers version, hence, creating different containers with different environments.

### 1. *Create a .env file containing:*
1. HF_TOKEN_WRITE - Create write access huggingface token at [here](https://huggingface.co/security-checkup?next=%2Fsettings%2Ftokens)
2. HF_TOKEN_READ - Create read access huggingface token at [here](https://huggingface.co/security-checkup?next=%2Fsettings%2Ftokens)
3. TAVILY_API_KEY - Get your api key at [here](https://tavily.com/)
4. VLLM_API_KEY - Define when building the container at *Step 5*

### 2. *Download model from huggingface*
```bash
python download_huggingface_model.py
```

### 3. *Configure docker-compose.yml file --volumes parameter*
- Change the path to the downloaded model directory that contains config.json

### 4. *Build and start the Docker container using `docker-compose`*
 ```bash
 docker-compose up --build -d
 ```

### 5. *Create conda environment*
```bash
conda create -n agentic_rag python=3.10
conda activate agentic_rag
```

### 6. *Install PyTorch with CUDA Compatibility*
```bash
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
```

### 7. *Install requirements*
```bash
pip install -r requirements.txt
```

### 8. *Clear cache after everything installed successfully*
```bash
pip cache purge
conda clean --all -y
```