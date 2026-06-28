FROM python:3.6-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone the PhishingBaseline repo (URLNet portion)
RUN git clone --depth 1 https://github.com/lindsey98/PhishingBaseline.git /app/repo

# Create symlinks for import compatibility
RUN mkdir -p /app/repo/experiment/URLnet && \
    ln -s /app/repo/URLNet/TextCNN.py /app/repo/experiment/URLnet/TextCNN.py && \
    ln -s /app/repo/URLNet/utils.py /app/repo/experiment/URLnet/utils.py && \
    touch /app/repo/experiment/__init__.py && \
    touch /app/repo/experiment/URLnet/__init__.py

# Install Python dependencies
RUN pip install --no-cache-dir \
    tensorflow==1.13.1 \
    tflearn==0.3.2 \
    numpy==1.16.6 \
    scikit-learn==0.20.4 \
    tqdm

# Copy wrapper script
COPY run.py /app/run.py

WORKDIR /app/repo/URLNet

ENV PYTHONPATH=/app/repo

# Default entrypoint
ENTRYPOINT ["python", "/app/run.py"]
