# Build stage
FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3.10-distutils \
        python3-pip \
        python3-venv \
        gcc \
        g++ && \
    rm -rf /var/lib/apt/lists/*

# Set up virtual environment
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-distutils \
        curl \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set up Python alternatives
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --set python3 /usr/bin/python3.10 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Copy only the specified files: requirements.txt, app.py and constants.py
COPY --chown=appuser:appuser requirements.txt app.py constants.py /app/

RUN mkdir -p /app/pdf_data && chown appuser:appuser /app/pdf_data && chmod 755 /app/pdf_data

# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Expose API port
EXPOSE 8060

# Container metadata
LABEL maintainer="Yihim <yihim_1999@hotmail.com>" \
      description="Multi-Agent Agentic RAG App" \
      version="1.0"
