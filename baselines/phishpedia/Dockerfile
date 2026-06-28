FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    python3-venv \
    wget \
    curl \
    unzip \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libnspr4 \
    libnss3 \
    libgconf-2-4 \
    && rm -rf /var/lib/apt/lists/*

# Install Google Chrome (chromium-browser is broken in Docker on Ubuntu 22.04)
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" \
       >> /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && apt-get clean

# Install pixi
RUN curl -fsSL https://pixi.sh/install.sh | sh \
    && echo 'export PATH="/root/.pixi/bin:$PATH"' >> /etc/profile.d/pixi.sh

ENV PATH="/root/.pixi/bin:$PATH"

# Clone Phishpedia
RUN git clone --depth 1 https://github.com/lindsey98/Phishpedia.git /app/phishpedia

WORKDIR /app/phishpedia

# Install dependencies via pixi (may fail silently on some platforms)
RUN pixi install || true

# Install all runtime dependencies via pip (pixi install may fail in Docker)
# CUDA 12.1 PyTorch — host CUDA Version: 12.2
RUN pip3 install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --no-cache-dir \
    gdown "Pillow==8.4.0" requests tldextract "selenium>=4.10" \
    opencv-python-headless pyyaml \
    numpy==1.23.0 scikit-learn spacy beautifulsoup4 matplotlib pandas \
    nltk tqdm unidecode scipy fvcore lxml psutil \
    flask flask-cors pycocotools \
    && pip3 install --no-build-isolation --no-cache-dir \
    "git+https://github.com/facebookresearch/detectron2.git"

# Download models via setup.sh (may need gdown)
RUN chmod +x setup.sh \
    && (./setup.sh || echo "Setup script had issues, models may need manual download")

# Patch logo_matching.py:
# 1. Convert logo_feat_list from list to numpy array
RUN sed -i 's/return logo_feat_list, file_name_list/import numpy as np; return np.array(logo_feat_list), file_name_list/' \
    /app/phishpedia/logo_matching.py || true
# 2. Fix KeyError when brand not in domain_map
RUN sed -i 's/top3_domainlist = \[domain_map\[x\] for x in top3_brandlist\]/top3_domainlist = [domain_map[x] for x in top3_brandlist if x in domain_map]/' \
    /app/phishpedia/logo_matching.py || true

# Copy wrapper and training scripts
COPY run.py /app/run.py
COPY screenshot.py /app/screenshot.py
COPY train.py /app/phishpedia/train.py

ENTRYPOINT ["python3", "/app/run.py"]
