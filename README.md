# agentic_rag

# Project Setup

Follow these instructions to set up the development environment for this project.

## 1. Create Conda Environment

```bash
conda create --name myenv python=3.10
conda activate myenv
```

## 2. Install PyTorch
```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```

## 3. Install Magic-PDF
```bash
pip install magic-pdf[full] --extra-index-url https://wheels.myhloli.com
```

## 4. Verify Magic-PDF Version
```bash
magic-pdf --version
```

## 5. Install PaddlePaddle GPU Version
```bash
python -m pip install paddlepaddle-gpu==2.6.1.post120 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
```

## 6. Install Other Requirements
```bash
pip install -r requirements.txt
```