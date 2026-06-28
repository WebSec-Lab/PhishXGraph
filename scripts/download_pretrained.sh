#!/bin/bash
set -e

# =============================================================================
# download_pretrained.sh — Download pre-trained models for Phishpedia & PhishIntention
#
# These models are required for inference. Docker build runs setup.sh but
# gdown often fails due to Google Drive rate limits. This script downloads
# models to the host and mounts them into containers.
#
# Usage:
#   ./scripts/download_pretrained.sh [phishpedia|phishintention|all]
#
# Requirements: pip install gdown
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

command -v gdown >/dev/null 2>&1 || {
    echo "gdown not found. Install with: pip install gdown"
    exit 1
}

download_gdrive() {
    local id="$1"
    local dest="$2"
    if [[ -f "$dest" ]]; then
        echo "  [skip] $(basename "$dest") already exists"
        return
    fi
    echo "  [download] $(basename "$dest") ..."
    gdown "$id" -O "$dest" --quiet || {
        echo "  [FAIL] $(basename "$dest") — try manual download from https://drive.google.com/uc?id=$id"
        return 1
    }
}

# ═══════════════════════════════════════════
# Phishpedia models
# ═══════════════════════════════════════════
download_phishpedia() {
    local MODEL_DIR="${PROJECT_DIR}/models/phishpedia/pretrained"
    mkdir -p "$MODEL_DIR"
    echo "=== Downloading Phishpedia pre-trained models ==="
    echo "  Destination: $MODEL_DIR"

    download_gdrive "1tE2Mu5WC8uqCxei3XqAd7AWaP5JTmVWH" "$MODEL_DIR/rcnn_bet365.pth"
    download_gdrive "1Q6lqjpl4exW7q_dPbComcj0udBMDl8CW" "$MODEL_DIR/faster_rcnn.yaml"
    download_gdrive "1H0Q_DbdKPLFcZee8I14K62qV7TTy7xvS" "$MODEL_DIR/resnetv2_rgb_new.pth.tar"
    download_gdrive "1fr5ZxBKyDiNZ_1B6rRAfZbAHBBoUjZ7I" "$MODEL_DIR/expand_targetlist.zip"
    download_gdrive "1qSdkSSoCYUkZMKs44Rup_1DPBxHnEKl1" "$MODEL_DIR/domain_map.pkl"

    # Unzip target list
    if [[ -f "$MODEL_DIR/expand_targetlist.zip" && ! -d "$MODEL_DIR/expand_targetlist" ]]; then
        echo "  [unzip] expand_targetlist.zip ..."
        unzip -qo "$MODEL_DIR/expand_targetlist.zip" -d "$MODEL_DIR/"
    fi

    echo "  Done. $(ls "$MODEL_DIR"/*.pth "$MODEL_DIR"/*.pth.tar "$MODEL_DIR"/*.pkl 2>/dev/null | wc -l) model files ready."
    echo ""
}

# ═══════════════════════════════════════════
# PhishIntention models
# ═══════════════════════════════════════════
download_phishintention() {
    local MODEL_DIR="${PROJECT_DIR}/models/phishintention/pretrained"
    mkdir -p "$MODEL_DIR"
    echo "=== Downloading PhishIntention pre-trained models ==="
    echo "  Destination: $MODEL_DIR"

    download_gdrive "1HWjE5Fv-c3nCDzLCBc7I3vClP1IeuP_I" "$MODEL_DIR/layout_detector.pth"
    download_gdrive "1igEMRz0vFBonxAILeYMRWTyd7A9sRirO" "$MODEL_DIR/crp_classifier.pth.tar"
    download_gdrive "1_O5SALqaJqvWoZDrdIVpsZyCnmSkzQcm" "$MODEL_DIR/crp_locator.pth"
    download_gdrive "15pfVWnZR-at46gqxd50cWhrXemP8oaxp" "$MODEL_DIR/ocr_pretrained.pth.tar"
    download_gdrive "1BxJf5lAcNEnnC0In55flWZ89xwlYkzPk" "$MODEL_DIR/ocr_siamese.pth.tar"
    download_gdrive "1fr5ZxBKyDiNZ_1B6rRAfZbAHBBoUjZ7I" "$MODEL_DIR/expand_targetlist.zip"
    download_gdrive "1qSdkSSoCYUkZMKs44Rup_1DPBxHnEKl1" "$MODEL_DIR/domain_map.pkl"

    # Unzip target list
    if [[ -f "$MODEL_DIR/expand_targetlist.zip" && ! -d "$MODEL_DIR/expand_targetlist" ]]; then
        echo "  [unzip] expand_targetlist.zip ..."
        unzip -qo "$MODEL_DIR/expand_targetlist.zip" -d "$MODEL_DIR/"
    fi

    echo "  Done. $(ls "$MODEL_DIR"/*.pth "$MODEL_DIR"/*.pth.tar "$MODEL_DIR"/*.pkl 2>/dev/null | wc -l) model files ready."
    echo ""
}

# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════
TARGET="${1:-all}"

case "$TARGET" in
    phishpedia)     download_phishpedia ;;
    phishintention) download_phishintention ;;
    all)
        download_phishpedia
        download_phishintention
        echo "=== All pre-trained models downloaded ==="
        echo ""
        echo "Next steps:"
        echo "  1. Models are in models/{phishpedia,phishintention}/pretrained/"
        echo "  2. Rebuild containers: ./scripts/run_pipeline.sh build"
        echo "     (or mount the pretrained dir into the containers)"
        echo "  3. See docker-compose.yml volume mounts for model paths"
        ;;
    *)
        echo "Usage: $0 [phishpedia|phishintention|all]"
        exit 1
        ;;
esac
