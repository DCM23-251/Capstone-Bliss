FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && python -m pip install \
        --index-url https://pypi.org/simple \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1+cpu \
        torchvision==0.20.1+cpu \
        torchaudio==2.5.1+cpu \
    && python -m pip install -r requirements.txt \
    && python -m pip install --force-reinstall --no-deps opencv-contrib-python-headless

COPY . .

RUN mkdir -p models users_data

RUN python -c "from huggingface_hub import hf_hub_download; import shutil; \
shutil.copyfile(hf_hub_download('SalahALHaismawi/yolov26-fire-detection', 'best.pt', revision='5397b43994c39c2ed4ef81a6996c6334e2b7a5fa'), 'models/best.pt'); \
shutil.copyfile(hf_hub_download('Ultralytics/YOLO26', 'yolo26m.pt', revision='070ac3c51435984d992ba92eb3f234daf4e4500d'), 'models/yolo26m.pt')"

EXPOSE 7860

CMD ["python", "main.py"]
