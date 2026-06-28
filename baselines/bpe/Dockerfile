FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    dnsutils \
    && rm -rf /var/lib/apt/lists/*

# Clone the original BPE repo (CCS'22)
RUN git clone --depth 1 https://github.com/taerikkk/BPE.git /app/bpe

# Install Python dependencies matching the paper's environment
RUN pip install --no-cache-dir \
    numpy==1.23.5 \
    "scipy<1.12" \
    scikit-learn==1.3.1 \
    networkx==3.2.1 \
    gensim==4.3.2 \
    tldextract==5.1.1 \
    tqdm==4.64.1 \
    requests==2.31.0

# Copy wrapper scripts
COPY run.py /app/run.py

WORKDIR /app

ENTRYPOINT ["python", "/app/run.py"]
