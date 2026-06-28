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
    dos2unix \
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

# Install pixi and Chrome
RUN curl -fsSL https://pixi.sh/install.sh | sh \
    && echo 'export PATH="/root/.pixi/bin:$PATH"' >> /etc/profile.d/pixi.sh \
    && wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" \
       >> /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && apt-get clean

ENV PATH="/opt/google/chrome:/usr/local/bin:/root/.pixi/bin:$PATH"
ENV CHROME_BIN="/opt/google/chrome/chrome"

# Clone PhishIntention
RUN git clone --depth 1 https://github.com/lindsey98/PhishIntention.git /app/phishintention

WORKDIR /app/phishintention

# Install dependencies + Selenium 4 for screenshots
# CUDA 12.1 PyTorch — host CUDA Version: 12.2
RUN pixi install || true \
    && pip3 install --no-cache-dir \
       torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --no-cache-dir \
       gdown "Pillow==8.4.0" requests tldextract "selenium>=4.10" \
       opencv-python-headless pyyaml \
       numpy==1.23.0 scikit-learn spacy beautifulsoup4 matplotlib pandas \
       nltk tqdm unidecode scipy fvcore lxml psutil \
       editdistance selenium-wire helium \
    && pip3 install --no-build-isolation --no-cache-dir \
       "git+https://github.com/facebookresearch/detectron2.git"

# Patch logo_matching.py: fix KeyError when brand not in domain_map
RUN sed -i 's/top3_domainlist = \[domain_map\[x\] for x in top3_brandlist\]/top3_domainlist = [domain_map[x] for x in top3_brandlist if x in domain_map]/' \
    /app/phishintention/modules/logo_matching.py || true

# Patch logo_matching.py: convert pred_bbox tensor to list for PIL crop compatibility
RUN sed -i "s|cropped = img.crop((pred_bbox\[0\], pred_bbox\[1\], pred_bbox\[2\], pred_bbox\[3\]))|cropped = img.crop((int(pred_bbox[0]), int(pred_bbox[1]), int(pred_bbox[2]), int(pred_bbox[3])))|" \
    /app/phishintention/modules/logo_matching.py || true

# Patch web_utils.py: use Selenium 4 auto chromedriver instead of hardcoded path
RUN python3 -c "\
import re; p='/app/phishintention/utils/web_utils.py'; t=open(p).read(); \
t=t.replace('service = ChromeService(executable_path=\"./chromedriver/chromedriver\")', '# auto chromedriver'); \
t=t.replace('service = ChromeService(executable_path=\"./chromedriver/chromedriver.exe\")', '# auto chromedriver'); \
t=t.replace('driver = webdriver.Chrome(service=service, options=options)', 'driver = webdriver.Chrome(options=options)'); \
open(p,'w').write(t)"

# Patch phishintention.py: use cached CRP screenshot instead of live browser navigation
RUN python3 -c "\
p='/app/phishintention/phishintention.py'; t=open(p).read(); \
old='def _step4_dynamic_analysis(self, url, screenshot_path, pred_boxes, pred_classes):'; \
new='''def _step4_dynamic_analysis(self, url, screenshot_path, pred_boxes, pred_classes):\n\
        # Check for cached CRP screenshot from feature collection\n\
        import json as _json\n\
        site_dir = os.path.dirname(screenshot_path)\n\
        cached_crp_shot = os.path.join(site_dir, \"crp_shot.png\")\n\
        cached_crp_result = os.path.join(site_dir, \"crp_result.json\")\n\
        if os.path.isfile(cached_crp_result):\n\
            try:\n\
                with open(cached_crp_result) as _f:\n\
                    _crp = _json.load(_f)\n\
                if _crp.get(\"crp_found\") and os.path.isfile(cached_crp_shot):\n\
                    logger.info(\"Using cached CRP screenshot\")\n\
                    return _crp.get(\"redirected_url\", url), cached_crp_shot, True, 0.0\n\
                else:\n\
                    logger.info(\"Cached CRP: no CRP page found\")\n\
                    return url, screenshot_path, False, 0.0\n\
            except Exception:\n\
                pass'''; \
t=t.replace(old, new, 1); \
open(p,'w').write(t)"

RUN dos2unix setup.sh \
    && chmod +x setup.sh \
    && (./setup.sh || echo "Setup script had issues, models may need manual download")

# Copy wrapper and training scripts
COPY run.py /app/run.py
COPY screenshot.py /app/screenshot.py
COPY train.py /app/phishintention/train.py

ENTRYPOINT ["python3", "/app/run.py"]
