# Use a base image with Conda pre-installed for simplicity
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y jq --no-install-recommends \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Create and activate conda environment
RUN conda create -n mineru python=3.10 -y && \
    echo "conda activate mineru" >> ~/.bashrc
SHELL ["bash", "-c"]

# Install Python packages
RUN conda run -n mineru pip install --no-cache-dir --default-timeout=2000 -U magic-pdf[full] --extra-index-url https://wheels.myhloli.com
RUN conda run -n mineru pip install --no-cache-dir --default-timeout=2000 paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# Copy and install application requirements
COPY requirements.txt .
RUN conda run -n mineru pip install -r requirements.txt

# Download models
RUN wget https://github.com/opendatalab/MinerU/raw/master/scripts/download_models_hf.py -O download_models_hf.py && \
    conda run -n mineru python download_models_hf.py

# Copy application code
COPY . .

RUN jq '.["device-mode"] = "cuda"' /root/magic-pdf.json > /root/magic-pdf.json.tmp && mv /root/magic-pdf.json.tmp /root/magic-pdf.json

# Default command to run the app
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "mineru"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
