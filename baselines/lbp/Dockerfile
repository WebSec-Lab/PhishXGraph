FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    dnsutils \
    && rm -rf /var/lib/apt/lists/*

# Clone the LBP repo (ICC'25)
RUN git clone --depth 1 https://github.com/wenyeguo/LBP.git /app/lbp

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy==1.23.5 \
    "scipy<1.12" \
    scikit-learn==1.3.1 \
    networkx==3.2.1 \
    gensim==4.3.2 \
    tldextract==5.1.1 \
    tqdm==4.64.1 \
    requests==2.31.0

COPY run.py /app/run.py

WORKDIR /app

ENTRYPOINT ["python", "/app/run.py"]
