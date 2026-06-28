#!/usr/bin/env python3
"""
Phishpedia Training Script

Implements training for Phishpedia's two main components:
1. Logo Detector (Detectron2 Faster R-CNN) - fine-tuned with pseudo-labels
2. Siamese Network (ResNet50) - contrastive learning for logo-brand matching
Also builds the reference logo database from benign training samples.

Paper: "Phishpedia: A Hybrid Deep Learning Based Approach to Visually Identify
        Phishing Webpages" (USENIX Security 2021)

Usage:
    python3 train.py --folder <train_sites_dir> --output <model_output_dir>

Input folder structure:
    train_sites/
      site_0/
        info.txt    (URL)
        label.txt   (phish or benign)
        shot.png    (screenshot)
      site_1/
        ...
"""

import argparse
import json
import os
import sys
import logging
import random
from collections import defaultdict
from urllib.parse import urlparse

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

logging.basicConfig(level=logging.INFO, format="[Phishpedia-Train] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PHISHPEDIA_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_INPUT_SIZE = (128, 128)
SIAMESE_EMB_DIM = 128
DETECTOR_SCORE_THRESH = 0.3
SIAMESE_LR = 1e-4
SIAMESE_EPOCHS = 10
SIAMESE_BATCH_SIZE = 32
DETECTOR_LR = 1e-3
DETECTOR_EPOCHS = 5
DETECTOR_BATCH_SIZE = 4


# =========================================================================
# 1. Data Loading
# =========================================================================

def load_training_sites(folder):
    """Load all training sites from the folder structure."""
    sites = []
    if not os.path.isdir(folder):
        logger.error("Training folder not found: %s", folder)
        return sites

    for entry in sorted(os.listdir(folder)):
        site_dir = os.path.join(folder, entry)
        if not os.path.isdir(site_dir):
            continue

        info_path = os.path.join(site_dir, "info.txt")
        label_path = os.path.join(site_dir, "label.txt")
        shot_path = os.path.join(site_dir, "shot.png")

        url = ""
        if os.path.isfile(info_path):
            with open(info_path) as f:
                url = f.read().strip()

        label = "unknown"
        if os.path.isfile(label_path):
            with open(label_path) as f:
                label = f.read().strip()

        if not os.path.isfile(shot_path):
            continue

        domain = ""
        if url:
            try:
                domain = urlparse(url).netloc.lower()
                # Remove www. prefix and port
                if domain.startswith("www."):
                    domain = domain[4:]
                if ":" in domain:
                    domain = domain.split(":")[0]
            except Exception:
                pass

        sites.append({
            "dir": site_dir,
            "url": url,
            "label": label,
            "shot_path": shot_path,
            "domain": domain,
            "name": entry,
        })

    logger.info("Loaded %d training sites (%d phish, %d benign)",
                len(sites),
                sum(1 for s in sites if s["label"] == "phish"),
                sum(1 for s in sites if s["label"] == "benign"))
    return sites


# =========================================================================
# 2. Logo Detection (Detectron2)
# =========================================================================

def _load_detectron2_detector():
    """Load the pretrained Phishpedia logo detector."""
    try:
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from detectron2 import model_zoo
    except ImportError:
        logger.warning("detectron2 not available, skipping detector training")
        return None, None

    cfg = get_cfg()

    # Try to find Phishpedia's detector config
    config_candidates = [
        os.path.join(PHISHPEDIA_DIR, "configs", "faster_rcnn.yaml"),
        os.path.join(PHISHPEDIA_DIR, "configs", "faster_rcnn_web.yaml"),
        os.path.join(PHISHPEDIA_DIR, "src", "detectron2_pedia", "configs",
                     "faster_rcnn.yaml"),
    ]

    config_found = False
    for config_path in config_candidates:
        if os.path.isfile(config_path):
            cfg.merge_from_file(config_path)
            config_found = True
            logger.info("Using detector config: %s", config_path)
            break

    if not config_found:
        # Use default Faster R-CNN config
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # logo only
        logger.info("Using default Faster R-CNN config")

    # Try to load pretrained weights
    weight_candidates = [
        os.path.join(PHISHPEDIA_DIR, "models", "rcnn_bet365.pth"),
        os.path.join(PHISHPEDIA_DIR, "models", "rcnn_logo.pth"),
        os.path.join(PHISHPEDIA_DIR, "models", "faster_rcnn.pth"),
        os.path.join(PHISHPEDIA_DIR, "src", "detectron2_pedia", "output",
                     "rcnn_bet365", "model_final.pth"),
    ]

    for weight_path in weight_candidates:
        if os.path.isfile(weight_path):
            cfg.MODEL.WEIGHTS = weight_path
            logger.info("Using detector weights: %s", weight_path)
            break

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = DETECTOR_SCORE_THRESH
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        predictor = DefaultPredictor(cfg)
        return predictor, cfg
    except Exception as e:
        logger.warning("Failed to load detector: %s", e)
        return None, None


def detect_logos(predictor, screenshot_path):
    """Detect logos in a screenshot and return bounding boxes + crops."""
    import cv2

    img = cv2.imread(screenshot_path)
    if img is None:
        return []

    try:
        outputs = predictor(img)
        instances = outputs["instances"]

        results = []
        for i in range(len(instances)):
            bbox = instances.pred_boxes[i].tensor.cpu().numpy()[0]
            score = instances.scores[i].item()
            x1, y1, x2, y2 = map(int, bbox)

            # Ensure valid crop
            h, w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue

            crop = img[y1:y2, x1:x2]
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

            results.append({
                "bbox": [x1, y1, x2, y2],
                "score": score,
                "crop": crop_pil,
            })

        return results
    except Exception as e:
        logger.debug("Detection failed for %s: %s", screenshot_path, e)
        return []


def detect_logos_all_sites(predictor, sites):
    """Run logo detection on all training sites."""
    all_detections = {}
    for i, site in enumerate(sites):
        detections = detect_logos(predictor, site["shot_path"])
        all_detections[site["name"]] = detections
        if (i + 1) % 50 == 0:
            logger.info("Detected logos: %d/%d sites", i + 1, len(sites))

    total_logos = sum(len(d) for d in all_detections.values())
    logger.info("Total logos detected: %d from %d sites", total_logos, len(sites))
    return all_detections


# =========================================================================
# 3. Reference Logo Database (from benign sites)
# =========================================================================

def build_reference_list(sites, detections, output_dir):
    """Build reference logo database from benign training sites.

    For each benign site, the detected logos are saved as reference logos
    for that domain/brand. This reference list is used during inference
    to match detected logos against known brands.
    """
    ref_dir = os.path.join(output_dir, "reference_list")
    os.makedirs(ref_dir, exist_ok=True)

    domain_logos = defaultdict(list)

    for site in sites:
        if site["label"] != "benign":
            continue
        if not site["domain"]:
            continue

        site_detections = detections.get(site["name"], [])
        for det in site_detections:
            domain_logos[site["domain"]].append(det["crop"])

    saved_count = 0
    domain_map = {}
    for domain, logos in domain_logos.items():
        brand_dir = os.path.join(ref_dir, domain)
        os.makedirs(brand_dir, exist_ok=True)

        for j, logo in enumerate(logos):
            logo_path = os.path.join(brand_dir, "logo_{}.png".format(j))
            logo_resized = logo.resize(LOGO_INPUT_SIZE, Image.BILINEAR)
            logo_resized.save(logo_path)
            saved_count += 1

        domain_map[domain] = domain

    # Save domain map
    map_path = os.path.join(ref_dir, "domain_map.json")
    with open(map_path, "w") as f:
        json.dump(domain_map, f, indent=2)

    logger.info("Reference list: %d logos from %d brands saved to %s",
                saved_count, len(domain_logos), ref_dir)
    return ref_dir


# =========================================================================
# 4. Siamese Network Training
# =========================================================================

class SiameseEmbedding(nn.Module):
    """ResNet50-based embedding network for logo matching.

    Reproduces the Siamese architecture from the Phishpedia paper:
    - ResNet50 backbone (pretrained on ImageNet)
    - Global average pooling
    - FC layer to embedding space
    """

    def __init__(self, emb_dim=SIAMESE_EMB_DIM, pretrained_path=None):
        super().__init__()
        import torchvision.models as models

        resnet = models.resnet50(pretrained=True)
        # Remove the final FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(2048, emb_dim)

        if pretrained_path and os.path.isfile(pretrained_path):
            try:
                state = torch.load(pretrained_path, map_location="cpu")
                self.load_state_dict(state, strict=False)
                logger.info("Loaded pretrained Siamese weights: %s",
                            pretrained_path)
            except Exception as e:
                logger.warning("Could not load Siamese weights: %s", e)

    def forward(self, x):
        feat = self.backbone(x)
        feat = feat.view(feat.size(0), -1)
        emb = self.fc(feat)
        return nn.functional.normalize(emb, p=2, dim=1)


class ContrastiveLoss(nn.Module):
    """Contrastive loss for Siamese network training.

    L = (1-Y) * 0.5 * D^2 + Y * 0.5 * max(0, margin - D)^2

    Y=0 for same-brand pairs, Y=1 for different-brand pairs.
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        dist = torch.nn.functional.pairwise_distance(emb1, emb2)
        loss = (1 - label) * 0.5 * dist.pow(2) + \
               label * 0.5 * torch.clamp(self.margin - dist, min=0).pow(2)
        return loss.mean()


class LogoPairDataset(Dataset):
    """Dataset of logo pairs for contrastive learning.

    Generates:
    - Positive pairs: two logos from the same brand/domain
    - Negative pairs: two logos from different brands/domains
    """

    def __init__(self, brand_logos, transform=None, pairs_per_epoch=2000):
        self.transform = transform or transforms.Compose([
            transforms.Resize(LOGO_INPUT_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # brand_logos: {brand_name: [PIL.Image, ...]}
        self.brands = [b for b, logos in brand_logos.items() if len(logos) >= 1]
        self.brand_logos = {b: brand_logos[b] for b in self.brands}
        self.pairs_per_epoch = pairs_per_epoch

        # Pre-generate pairs
        self.pairs = self._generate_pairs()

    def _generate_pairs(self):
        pairs = []
        brands_with_multiple = [b for b in self.brands
                                if len(self.brand_logos[b]) >= 2]

        n_pos = self.pairs_per_epoch // 2
        n_neg = self.pairs_per_epoch - n_pos

        # Positive pairs (same brand)
        for _ in range(n_pos):
            if brands_with_multiple:
                brand = random.choice(brands_with_multiple)
                logos = self.brand_logos[brand]
                i1, i2 = random.sample(range(len(logos)), 2)
                pairs.append((logos[i1], logos[i2], 0))  # 0 = same
            elif self.brands:
                # Augmented positive: same logo with different augmentation
                brand = random.choice(self.brands)
                logo = self.brand_logos[brand][0]
                pairs.append((logo, logo, 0))

        # Negative pairs (different brands)
        for _ in range(n_neg):
            if len(self.brands) >= 2:
                b1, b2 = random.sample(self.brands, 2)
                l1 = random.choice(self.brand_logos[b1])
                l2 = random.choice(self.brand_logos[b2])
                pairs.append((l1, l2, 1))  # 1 = different

        random.shuffle(pairs)
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1, img2, label = self.pairs[idx]

        if img1.mode != "RGB":
            img1 = img1.convert("RGB")
        if img2.mode != "RGB":
            img2 = img2.convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)


def _find_pretrained_siamese():
    """Find pretrained Siamese model weights."""
    candidates = [
        os.path.join(PHISHPEDIA_DIR, "models", "resnetv2_rgb_new.pth.tar"),
        os.path.join(PHISHPEDIA_DIR, "models", "siamese_pedia.pth"),
        os.path.join(PHISHPEDIA_DIR, "models", "resnetv2_rgb_new.pth"),
        os.path.join(PHISHPEDIA_DIR, "models",
                     "expand_targetlist", "resnetv2_rgb_new.pth"),
        os.path.join(PHISHPEDIA_DIR, "src", "siamese_pedia",
                     "siamese_retrain", "resnetv2_rgb_new.pth"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def train_siamese_network(sites, detections, output_dir):
    """Train the Siamese network for logo-brand matching.

    Training procedure (following the paper):
    1. Collect logo crops grouped by brand (domain for benign sites)
    2. Create positive pairs (same brand) and negative pairs (different brands)
    3. Train with contrastive loss
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training Siamese network on %s", device)

    # Collect logos by brand
    brand_logos = defaultdict(list)

    for site in sites:
        if not site["domain"]:
            continue
        site_dets = detections.get(site["name"], [])
        brand = site["domain"]
        for det in site_dets:
            brand_logos[brand].append(det["crop"])

    # Need at least 2 brands for contrastive learning
    brands_with_logos = {b: logos for b, logos in brand_logos.items()
                         if len(logos) >= 1}
    if len(brands_with_logos) < 2:
        logger.warning("Not enough brands with detected logos (%d). "
                        "Need at least 2 for Siamese training. Skipping.",
                        len(brands_with_logos))
        return None

    logger.info("Siamese training: %d brands, %d total logos",
                len(brands_with_logos),
                sum(len(v) for v in brands_with_logos.values()))

    # Load pretrained Siamese model
    pretrained_path = _find_pretrained_siamese()
    model = SiameseEmbedding(
        emb_dim=SIAMESE_EMB_DIM,
        pretrained_path=pretrained_path,
    ).to(device)

    # Create dataset and dataloader
    dataset = LogoPairDataset(
        brands_with_logos,
        pairs_per_epoch=min(2000, len(brands_with_logos) * 50),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=SIAMESE_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    # Training
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=SIAMESE_LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    model.train()
    for epoch in range(SIAMESE_EPOCHS):
        total_loss = 0.0
        n_batches = 0

        for img1, img2, labels in dataloader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)

            emb1 = model(img1)
            emb2 = model(img2)
            loss = criterion(emb1, emb2, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        logger.info("Siamese epoch %d/%d - loss: %.4f",
                     epoch + 1, SIAMESE_EPOCHS, avg_loss)

        # Re-generate pairs for next epoch
        dataset.pairs = dataset._generate_pairs()

    # Save trained model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "siamese_pedia.pth")
    torch.save(model.state_dict(), model_path)
    logger.info("Siamese model saved to %s", model_path)

    # Compute and save reference embeddings
    _save_reference_embeddings(model, brands_with_logos, output_dir, device)

    return model_path


def _save_reference_embeddings(model, brand_logos, output_dir, device):
    """Pre-compute and save embeddings for the reference logo database."""
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(LOGO_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    embeddings = {}
    with torch.no_grad():
        for brand, logos in brand_logos.items():
            brand_embs = []
            for logo in logos:
                if logo.mode != "RGB":
                    logo = logo.convert("RGB")
                img_t = transform(logo).unsqueeze(0).to(device)
                emb = model(img_t).cpu().numpy()[0]
                brand_embs.append(emb.tolist())
            embeddings[brand] = brand_embs

    emb_path = os.path.join(output_dir, "reference_embeddings.json")
    with open(emb_path, "w") as f:
        json.dump(embeddings, f)
    logger.info("Reference embeddings saved: %d brands", len(embeddings))


# =========================================================================
# 5. Logo Detector Fine-tuning (Detectron2)
# =========================================================================

def _prepare_detector_dataset(sites, detections):
    """Prepare dataset for Detectron2 fine-tuning using pseudo-labels.

    Uses the pretrained detector's predictions as pseudo ground-truth
    for fine-tuning on the new domain.
    """
    dataset_dicts = []

    for site in sites:
        site_dets = detections.get(site["name"], [])
        if not site_dets:
            continue

        shot_path = site["shot_path"]
        try:
            img = Image.open(shot_path)
            w, h = img.size
        except Exception:
            continue

        annos = []
        for det in site_dets:
            x1, y1, x2, y2 = det["bbox"]
            annos.append({
                "bbox": [x1, y1, x2, y2],
                "bbox_mode": 0,  # BoxMode.XYXY_ABS
                "category_id": 0,  # logo class
            })

        dataset_dicts.append({
            "file_name": shot_path,
            "height": h,
            "width": w,
            "annotations": annos,
        })

    return dataset_dicts


def train_detector(predictor, cfg, sites, detections, output_dir):
    """Fine-tune the Detectron2 logo detector.

    Uses pseudo-labels from the pretrained detector to fine-tune
    on the training screenshots, adapting to the new domain.
    """
    try:
        from detectron2.data import DatasetCatalog, MetadataCatalog
        from detectron2.engine import DefaultTrainer
        from detectron2.config import get_cfg as _get_cfg
    except ImportError:
        logger.warning("detectron2 not available, skipping detector fine-tuning")
        return None

    dataset_dicts = _prepare_detector_dataset(sites, detections)
    if len(dataset_dicts) < 10:
        logger.warning("Too few samples with detections (%d) for detector "
                        "fine-tuning. Skipping.", len(dataset_dicts))
        return None

    logger.info("Fine-tuning detector with %d annotated images", len(dataset_dicts))

    # Register dataset
    dataset_name = "phishpedia_train_logos"
    if dataset_name in DatasetCatalog:
        DatasetCatalog.remove(dataset_name)
    DatasetCatalog.register(dataset_name, lambda: dataset_dicts)
    MetadataCatalog.get(dataset_name).set(thing_classes=["logo"])

    # Configure training
    train_cfg = cfg.clone()
    train_cfg.DATASETS.TRAIN = (dataset_name,)
    train_cfg.DATASETS.TEST = ()
    train_cfg.DATALOADER.NUM_WORKERS = 0
    train_cfg.SOLVER.BASE_LR = DETECTOR_LR
    train_cfg.SOLVER.MAX_ITER = len(dataset_dicts) * DETECTOR_EPOCHS // DETECTOR_BATCH_SIZE
    train_cfg.SOLVER.MAX_ITER = max(train_cfg.SOLVER.MAX_ITER, 100)
    train_cfg.SOLVER.IMS_PER_BATCH = min(DETECTOR_BATCH_SIZE, len(dataset_dicts))
    train_cfg.SOLVER.CHECKPOINT_PERIOD = train_cfg.SOLVER.MAX_ITER  # save at end
    train_cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    train_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    det_output = os.path.join(output_dir, "detector")
    os.makedirs(det_output, exist_ok=True)
    train_cfg.OUTPUT_DIR = det_output

    # Train
    trainer = DefaultTrainer(train_cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Save final model
    final_model = os.path.join(det_output, "model_final.pth")
    if os.path.isfile(final_model):
        out_path = os.path.join(output_dir, "rcnn_logo.pth")
        import shutil
        shutil.copy2(final_model, out_path)
        logger.info("Detector model saved to %s", out_path)
        return out_path

    return None


# =========================================================================
# 6. Main Training Pipeline
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Phishpedia Training")
    parser.add_argument("--folder", required=True,
                        help="Training sites directory")
    parser.add_argument("--output", required=True,
                        help="Output directory for trained models")
    parser.add_argument("--skip-detector", action="store_true",
                        help="Skip logo detector fine-tuning")
    parser.add_argument("--skip-siamese", action="store_true",
                        help="Skip Siamese network training")
    parser.add_argument("--siamese-epochs", type=int, default=SIAMESE_EPOCHS,
                        help="Number of Siamese training epochs")
    parser.add_argument("--detector-epochs", type=int, default=DETECTOR_EPOCHS,
                        help="Number of detector fine-tuning epochs")
    args = parser.parse_args()

    siamese_epochs = args.siamese_epochs
    detector_epochs = args.detector_epochs

    os.makedirs(args.output, exist_ok=True)

    # Step 1: Load training data
    logger.info("=" * 60)
    logger.info("Step 1: Loading training data")
    logger.info("=" * 60)
    sites = load_training_sites(args.folder)
    if not sites:
        logger.error("No training sites found. Exiting.")
        sys.exit(1)

    # Step 2: Load pretrained logo detector
    logger.info("=" * 60)
    logger.info("Step 2: Loading pretrained logo detector")
    logger.info("=" * 60)
    predictor, cfg = _load_detectron2_detector()

    # Step 3: Detect logos in all training screenshots
    if predictor is not None:
        logger.info("=" * 60)
        logger.info("Step 3: Detecting logos in training screenshots")
        logger.info("=" * 60)
        detections = detect_logos_all_sites(predictor, sites)
    else:
        logger.warning("No detector available. Using fallback logo extraction.")
        detections = _fallback_logo_extraction(sites)

    # Step 4: Build reference list from benign sites
    logger.info("=" * 60)
    logger.info("Step 4: Building reference logo database")
    logger.info("=" * 60)
    build_reference_list(sites, detections, args.output)

    # Step 5: Train Siamese network
    if not args.skip_siamese:
        logger.info("=" * 60)
        logger.info("Step 5: Training Siamese network")
        logger.info("=" * 60)
        train_siamese_network(sites, detections, args.output)
    else:
        logger.info("Skipping Siamese network training (--skip-siamese)")

    # Step 6: Fine-tune logo detector
    if not args.skip_detector and predictor is not None and cfg is not None:
        logger.info("=" * 60)
        logger.info("Step 6: Fine-tuning logo detector")
        logger.info("=" * 60)
        train_detector(predictor, cfg, sites, detections, args.output)
    else:
        logger.info("Skipping detector fine-tuning")

    # Save training metadata
    meta = {
        "n_sites": len(sites),
        "n_phish": sum(1 for s in sites if s["label"] == "phish"),
        "n_benign": sum(1 for s in sites if s["label"] == "benign"),
        "n_logos_detected": sum(len(d) for d in detections.values()),
        "siamese_epochs": siamese_epochs,
        "detector_epochs": detector_epochs,
        "siamese_emb_dim": SIAMESE_EMB_DIM,
    }
    with open(os.path.join(args.output, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("=" * 60)
    logger.info("Training complete. Models saved to %s", args.output)
    logger.info("=" * 60)


def _fallback_logo_extraction(sites):
    """Fallback: extract logo-like regions from screenshots when detector
    is not available.

    Uses a simple heuristic: crop the top-left region of the screenshot
    where logos are typically located.
    """
    detections = {}
    for site in sites:
        try:
            img = Image.open(site["shot_path"])
            w, h = img.size

            # Logo heuristic regions:
            # 1. Top-left corner (most common logo position)
            # 2. Top-center
            crops = []

            # Top-left: 20% width, 15% height
            x2 = min(int(w * 0.25), w)
            y2 = min(int(h * 0.12), h)
            if x2 > 30 and y2 > 30:
                crop = img.crop((0, 0, x2, y2))
                crops.append({
                    "bbox": [0, 0, x2, y2],
                    "score": 0.5,
                    "crop": crop,
                })

            detections[site["name"]] = crops
        except Exception:
            detections[site["name"]] = []

    logger.info("Fallback extraction: %d sites processed",
                len(detections))
    return detections


if __name__ == "__main__":
    main()
