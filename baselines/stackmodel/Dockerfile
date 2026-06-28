FROM python:3.8-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone the PhishingBaseline repo (StackModel portion)
RUN git clone --depth 1 https://github.com/lindsey98/PhishingBaseline.git /app/repo

# Install Python dependencies (upgraded for Python 3.8 compatibility)
RUN pip install --no-cache-dir \
    numpy==1.21.6 \
    scikit-learn==1.0.2 \
    xgboost==1.6.2 \
    pandas==1.3.5 \
    joblib==1.1.1 \
    tldextract==3.4.4 \
    beautifulsoup4==4.11.2 \
    requests==2.28.2 \
    lxml==4.9.3 \
    gdown==4.7.1 \
    matplotlib==3.5.3

# Download pretrained model from Google Drive
RUN gdown --folder https://drive.google.com/drive/folders/15WPsuPJDCzhp2SvYZLRj8mAlT3zmoAMW \
       -O /app/repo/StackModel/ || echo "Model download failed, will need manual download"

# Copy wrapper script
COPY run.py /app/run.py

WORKDIR /app/repo/StackModel

ENTRYPOINT ["python", "/app/run.py"]
