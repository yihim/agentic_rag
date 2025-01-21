# Agentic RAG

## Prerequisites

Ensure you have the following installed on your machine:
- Docker Desktop (required for Magic-PDF)
- Conda (for creating and managing environments)

## Project Setup

Follow these steps to set up the development environment for this project.


### 1. *Data Extraction (Magic-PDF)*

1. Navigate to the `data_extraction` directory:
    ```bash
    cd ./data_extraction
    ```

2. Build and start the Docker container using `docker-compose`:
    ```bash
    docker-compose up --build -d mineru
    ```

3. Once the container is running, use `requests` to post with a json object containing `file_path` as the key. The extracted data will be saved in the `./data_extraction/tmp/md` folder.

Note: magic-pdf explicitly required transformers==4.42.4 to work and the project required an updated transformers version to work.

### 2. *Create conda environment*
```bash
conda create -n agentic_rag python=3.10
conda activate agentic_rag
```

### 3. *Install PyTorch with CUDA Compatibility*
```bash
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
```

### 4. *Install requirements*
```bash
pip install -r requirements.txt
```