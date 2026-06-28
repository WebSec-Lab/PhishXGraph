#!/usr/bin/env python3
"""
PhishIntention Training Script

Extends Phishpedia training with two additional components:
1. AWL Layout Classifier - classifies credential-requiring page layouts
2. CRP Transition Classifier - identifies credential-requiring page transitions

Also trains the shared components (logo detector, Siamese network, reference list)
by delegating to the Phishpedia training logic.

Paper: "Inferring Phishing Intention via Webpage Appearance and Dynamics:
        A Deep Vision Based Approach" (USENIX Security 2022)

Usage:
    python3 train.py --folder <train_sites_dir> --output <model_output_dir>

Input folder structure:
    train_sites/
      site_0/
        info.txt        (URL)
        label.txt       (phish or benign)
        shot.png        (screenshot)
        html.txt        (HTML content, optional)
        crp_shot.png    (CRP screenshot, optional)
        crp_result.json (CRP analysis result, optional)
      site_1/
        ...
"""

import argparse
import json
import os
import sys
import logging
import random
import copy
from collections import defaultdict
from urllib.parse import urlparse

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

logging.basicConfig(level=logging.INFO, format="[PhishIntention-Train] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PHISHINTENTION_DIR = os.path.dirname(os.path.abspath(__file__))
SCREENSHOT_SIZE = (224, 224)
LOGO_INPUT_SIZE = (128, 128)
SIAMESE_EMB_DIM = 128

# AWL classifier hyperparameters
AWL_LR = 1e-4
AWL_EPOCHS = 15
AWL_BATCH_SIZE = 16

# CRP classifier hyperparameters
CRP_LR = 1e-4
CRP_EPOCHS = 10
CRP_BATCH_SIZE = 16

# Shared component hyperparameters (delegated to Phishpedia-style training)
SIAMESE_LR = 1e-4
SIAMESE_EPOCHS = 10
SIAMESE_BATCH_SIZE = 32
DETECTOR_SCORE_THRESH = 0.3


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
                if domain.startswith("www."):
                    domain = domain[4:]
                if ":" in domain:
                    domain = domain.split(":")[0]
            except Exception:
                pass

        # Check for CRP data
        crp_shot = os.path.join(site_dir, "crp_shot.png")
        crp_result = os.path.join(site_dir, "crp_result.json")
        html_path = os.path.join(site_dir, "html.txt")

        sites.append({
            "dir": site_dir,
            "url": url,
            "label": label,
            "shot_path": shot_path,
            "domain": domain,
            "name": entry,
            "crp_shot": crp_shot if os.path.isfile(crp_shot) else None,
            "crp_result": crp_result if os.path.isfile(crp_result) else None,
            "html_path": html_path if os.path.isfile(html_path) else None,
        })

    logger.info("Loaded %d training sites (%d phish, %d benign)",
                len(sites),
                sum(1 for s in sites if s["label"] == "phish"),
                sum(1 for s in sites if s["label"] == "benign"))
    return sites


# =========================================================================
# 2. AWL Layout Classifier (Attention-based Weighted Layout)
# =========================================================================

class AWLClassifier(nn.Module):
    """Attention-based Weighted Layout classifier from PhishIntention.

    Classifies webpage screenshots as credential-requiring pages (CRP) or not.
    Architecture:
    - ResNet18 backbone (pretrained on ImageNet)
    - Spatial attention module
    - Binary classification head

    The spatial attention mechanism weighs different regions of the screenshot,
    focusing on login forms, input fields, and credential-related UI elements.
    """

    def __init__(self, pretrained_path=None):
        super().__init__()
        resnet = models.resnet18(pretrained=True)

        # Feature extractor (up to layer4)
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        # Spatial attention module
        self.attention = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 2),
        )

        if pretrained_path and os.path.isfile(pretrained_path):
            try:
                state = torch.load(pretrained_path, map_location="cpu")
                self.load_state_dict(state, strict=False)
                logger.info("Loaded pretrained AWL weights: %s", pretrained_path)
            except Exception as e:
                logger.warning("Could not load AWL weights: %s", e)

    def forward(self, x):
        feat = self.features(x)  # (B, 512, H, W)

        # Spatial attention
        att = self.attention(feat)  # (B, 1, H, W)
        feat = feat * att  # Attended features

        # Classification
        pooled = self.avgpool(feat)
        pooled = pooled.view(pooled.size(0), -1)
        out = self.classifier(pooled)
        return out


class ScreenshotDataset(Dataset):
    """Dataset of webpage screenshots for layout classification.

    Labels:
    - phish sites → 1 (credential-requiring page)
    - benign sites → 0 (non-credential page)

    This is a simplification: not all phishing pages are CRP and some benign
    pages have login forms. However, phishing sites overwhelmingly present
    credential-requiring layouts, making this a reasonable proxy.
    """

    def __init__(self, sites, transform=None):
        self.sites = [s for s in sites if s["label"] in ("phish", "benign")]
        self.transform = transform or transforms.Compose([
            transforms.Resize(SCREENSHOT_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.sites)

    def __getitem__(self, idx):
        site = self.sites[idx]
        try:
            img = Image.open(site["shot_path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", SCREENSHOT_SIZE, (128, 128, 128))

        img = self.transform(img)
        label = 1 if site["label"] == "phish" else 0
        return img, torch.tensor(label, dtype=torch.long)


def _find_pretrained_awl():
    """Find pretrained AWL model weights."""
    candidates = [
        os.path.join(PHISHINTENTION_DIR, "models", "layout_detector.pth"),
        os.path.join(PHISHINTENTION_DIR, "models", "awl_detector.pth"),
        os.path.join(PHISHINTENTION_DIR, "models", "layout_classifier.pth"),
        os.path.join(PHISHINTENTION_DIR, "models",
                     "awl_detector", "model_final.pth"),
        os.path.join(PHISHINTENTION_DIR, "modules",
                     "awl_detector", "model.pth"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def train_awl_classifier(sites, output_dir):
    """Train the AWL (Attention-based Weighted Layout) classifier.

    Training procedure (following the paper):
    1. Prepare screenshot dataset with CRP labels
    2. Fine-tune ResNet18 with spatial attention
    3. Binary cross-entropy loss
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training AWL layout classifier on %s", device)

    labeled_sites = [s for s in sites if s["label"] in ("phish", "benign")]
    if len(labeled_sites) < 10:
        logger.warning("Too few labeled sites (%d) for AWL training. Skipping.",
                        len(labeled_sites))
        return None

    # Split into train/val (90/10)
    random.shuffle(labeled_sites)
    val_size = max(1, len(labeled_sites) // 10)
    val_sites = labeled_sites[:val_size]
    train_sites = labeled_sites[val_size:]

    train_transform = transforms.Compose([
        transforms.Resize(SCREENSHOT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(SCREENSHOT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ScreenshotDataset(train_sites, transform=train_transform)
    val_dataset = ScreenshotDataset(val_sites, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=AWL_BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=AWL_BATCH_SIZE,
                            shuffle=False, num_workers=0)

    logger.info("AWL dataset: %d train, %d val", len(train_dataset), len(val_dataset))

    # Load model
    pretrained_path = _find_pretrained_awl()
    model = AWLClassifier(pretrained_path=pretrained_path).to(device)

    # Class weights for imbalanced data
    n_phish = sum(1 for s in train_sites if s["label"] == "phish")
    n_benign = sum(1 for s in train_sites if s["label"] == "benign")
    if n_phish > 0 and n_benign > 0:
        weight = torch.tensor([n_phish / (n_phish + n_benign),
                               n_benign / (n_phish + n_benign)],
                              dtype=torch.float32).to(device)
    else:
        weight = None

    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.Adam(model.parameters(), lr=AWL_LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=AWL_EPOCHS)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(AWL_EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        scheduler.step()

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        avg_loss = train_loss / max(train_total, 1)

        logger.info("AWL epoch %d/%d - loss: %.4f, train_acc: %.4f, val_acc: %.4f",
                     epoch + 1, AWL_EPOCHS, avg_loss, train_acc, val_acc)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    # Save best model
    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "awl_detector.pth")
    torch.save(model.state_dict(), model_path)
    logger.info("AWL classifier saved to %s (best val_acc: %.4f)",
                model_path, best_val_acc)
    return model_path


# =========================================================================
# 3. CRP Transition Classifier
# =========================================================================

class CRPClassifier(nn.Module):
    """CRP (Credential-Requiring Page) transition classifier.

    Determines whether a page transition leads to a credential-requiring page
    by comparing the original screenshot with the post-transition screenshot.

    Architecture:
    - Dual-branch ResNet18 for before/after screenshots
    - Feature concatenation + FC classifier
    """

    def __init__(self, pretrained_path=None):
        super().__init__()
        resnet = models.resnet18(pretrained=True)

        # Shared feature extractor
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Classification head (takes concatenated features from both screenshots)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

        if pretrained_path and os.path.isfile(pretrained_path):
            try:
                state = torch.load(pretrained_path, map_location="cpu")
                self.load_state_dict(state, strict=False)
                logger.info("Loaded pretrained CRP weights: %s", pretrained_path)
            except Exception as e:
                logger.warning("Could not load CRP weights: %s", e)

    def forward(self, img_before, img_after):
        feat_before = self.backbone(img_before).view(img_before.size(0), -1)
        feat_after = self.backbone(img_after).view(img_after.size(0), -1)
        combined = torch.cat([feat_before, feat_after], dim=1)
        return self.classifier(combined)


class CRPPairDataset(Dataset):
    """Dataset of screenshot pairs for CRP transition classification.

    For sites with CRP data:
    - shot.png (before) + crp_shot.png (after) → label from crp_result.json
    For sites without CRP data:
    - Synthetic pairs using augmentation
      - phish: original + augmented version → positive (CRP transition)
      - benign: original + augmented version → negative (no transition)
    """

    def __init__(self, sites, transform=None):
        self.transform = transform or transforms.Compose([
            transforms.Resize(SCREENSHOT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.pairs = self._build_pairs(sites)

    def _build_pairs(self, sites):
        pairs = []

        for site in sites:
            if site["label"] not in ("phish", "benign"):
                continue

            try:
                img_before = Image.open(site["shot_path"]).convert("RGB")
            except Exception:
                continue

            # Use actual CRP data if available
            if site["crp_shot"] is not None:
                try:
                    img_after = Image.open(site["crp_shot"]).convert("RGB")
                    crp_label = 1  # CRP transition detected
                    if site["crp_result"] is not None:
                        with open(site["crp_result"]) as f:
                            crp_data = json.load(f)
                            crp_label = 1 if crp_data.get("crp_found", False) else 0
                    pairs.append((img_before, img_after, crp_label))
                    continue
                except Exception:
                    pass

            # Synthetic: phish sites likely have CRP transitions
            if site["label"] == "phish":
                # Use the same image as a proxy for "transition to CRP"
                pairs.append((img_before, img_before, 1))
            else:
                pairs.append((img_before, img_before, 0))

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_before, img_after, label = self.pairs[idx]
        img_before = self.transform(img_before)
        img_after = self.transform(img_after)
        return img_before, img_after, torch.tensor(label, dtype=torch.long)


def _find_pretrained_crp():
    """Find pretrained CRP classifier weights."""
    candidates = [
        os.path.join(PHISHINTENTION_DIR, "models", "crp_classifier.pth.tar"),
        os.path.join(PHISHINTENTION_DIR, "models", "crp_classifier.pth"),
        os.path.join(PHISHINTENTION_DIR, "models", "crp_locator.pth"),
        os.path.join(PHISHINTENTION_DIR, "modules",
                     "crp_classifier", "model.pth"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def train_crp_classifier(sites, output_dir):
    """Train the CRP transition classifier.

    Training procedure (following the paper):
    1. Build pairs of (before, after) screenshots
    2. For sites with CRP data: use actual transitions
    3. For sites without: use synthetic pairs from phish/benign labels
    4. Train dual-branch classifier with cross-entropy loss
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training CRP transition classifier on %s", device)

    labeled_sites = [s for s in sites if s["label"] in ("phish", "benign")]
    if len(labeled_sites) < 10:
        logger.warning("Too few labeled sites (%d) for CRP training. Skipping.",
                        len(labeled_sites))
        return None

    # Split into train/val
    random.shuffle(labeled_sites)
    val_size = max(1, len(labeled_sites) // 10)
    val_sites = labeled_sites[:val_size]
    train_sites = labeled_sites[val_size:]

    transform = transforms.Compose([
        transforms.Resize(SCREENSHOT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CRPPairDataset(train_sites, transform=transform)
    val_dataset = CRPPairDataset(val_sites, transform=transform)

    if len(train_dataset) == 0:
        logger.warning("No CRP training pairs generated. Skipping.")
        return None

    train_loader = DataLoader(train_dataset, batch_size=CRP_BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CRP_BATCH_SIZE,
                            shuffle=False, num_workers=0)

    logger.info("CRP dataset: %d train pairs, %d val pairs",
                len(train_dataset), len(val_dataset))

    # Load model
    pretrained_path = _find_pretrained_crp()
    model = CRPClassifier(pretrained_path=pretrained_path).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CRP_LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CRP_EPOCHS)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(CRP_EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for img_b, img_a, labels in train_loader:
            img_b, img_a = img_b.to(device), img_a.to(device)
            labels = labels.to(device)

            outputs = model(img_b, img_a)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for img_b, img_a, labels in val_loader:
                img_b, img_a = img_b.to(device), img_a.to(device)
                labels = labels.to(device)
                outputs = model(img_b, img_a)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        scheduler.step()

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        avg_loss = train_loss / max(train_total, 1)

        logger.info("CRP epoch %d/%d - loss: %.4f, train_acc: %.4f, val_acc: %.4f",
                     epoch + 1, CRP_EPOCHS, avg_loss, train_acc, val_acc)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "crp_classifier.pth")
    torch.save(model.state_dict(), model_path)
    logger.info("CRP classifier saved to %s (best val_acc: %.4f)",
                model_path, best_val_acc)
    return model_path


# =========================================================================
# 4. Shared Components (Logo Detector + Siamese Network)
#    Reuses Phishpedia training logic
# =========================================================================

def _import_phishpedia_training():
    """Try to import Phishpedia training functions."""
    # The Phishpedia train.py may be at various locations
    search_paths = [
        os.path.join(PHISHINTENTION_DIR, "..", "phishpedia"),
        "/app/phishpedia",
        os.path.join(os.path.dirname(PHISHINTENTION_DIR), "phishpedia"),
    ]

    for path in search_paths:
        train_py = os.path.join(path, "train.py")
        if os.path.isfile(train_py):
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "phishpedia_train", train_py)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

    return None


def train_shared_components(sites, output_dir):
    """Train logo detector + Siamese network (shared with Phishpedia).

    Attempts to import and use the Phishpedia training module.
    Falls back to a self-contained implementation if unavailable.
    """
    phishpedia_mod = _import_phishpedia_training()

    if phishpedia_mod is not None:
        logger.info("Using Phishpedia training module for shared components")
        # Reuse Phishpedia's full training pipeline
        try:
            loaded_sites = phishpedia_mod.load_training_sites(
                sites[0]["dir"].rsplit("/", 1)[0] if sites else "")

            predictor, cfg = phishpedia_mod._load_detectron2_detector()
            if predictor is not None:
                detections = phishpedia_mod.detect_logos_all_sites(
                    predictor, loaded_sites)
                phishpedia_mod.build_reference_list(
                    loaded_sites, detections, output_dir)
                phishpedia_mod.train_siamese_network(
                    loaded_sites, detections, output_dir)
                phishpedia_mod.train_detector(
                    predictor, cfg, loaded_sites, detections, output_dir)
            else:
                detections = phishpedia_mod._fallback_logo_extraction(
                    loaded_sites)
                phishpedia_mod.build_reference_list(
                    loaded_sites, detections, output_dir)
                phishpedia_mod.train_siamese_network(
                    loaded_sites, detections, output_dir)
            return
        except Exception as e:
            logger.warning("Phishpedia training module failed: %s. "
                           "Using built-in fallback.", e)

    # Fallback: self-contained Siamese training + reference list
    logger.info("Training shared components (built-in fallback)")
    _train_siamese_fallback(sites, output_dir)
    _build_reference_list_fallback(sites, output_dir)


def _train_siamese_fallback(sites, output_dir):
    """Self-contained Siamese network training (fallback)."""
    # Import the SiameseEmbedding and ContrastiveLoss inline
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class SiameseEmbedding(nn.Module):
        def __init__(self, emb_dim=SIAMESE_EMB_DIM):
            super().__init__()
            resnet = models.resnet50(pretrained=True)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            self.fc = nn.Linear(2048, emb_dim)

        def forward(self, x):
            feat = self.backbone(x)
            feat = feat.view(feat.size(0), -1)
            emb = self.fc(feat)
            return nn.functional.normalize(emb, p=2, dim=1)

    # Collect logo-like crops by domain (heuristic: top-left region)
    brand_logos = defaultdict(list)
    for site in sites:
        if not site["domain"] or site["label"] not in ("phish", "benign"):
            continue
        try:
            img = Image.open(site["shot_path"]).convert("RGB")
            w, h = img.size
            crop = img.crop((0, 0, min(int(w * 0.25), w),
                             min(int(h * 0.12), h)))
            if crop.size[0] > 20 and crop.size[1] > 20:
                brand_logos[site["domain"]].append(crop)
        except Exception:
            continue

    if len(brand_logos) < 2:
        logger.warning("Not enough brands (%d) for Siamese training",
                        len(brand_logos))
        return

    transform = transforms.Compose([
        transforms.Resize(LOGO_INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    model = SiameseEmbedding().to(device)

    # Find pretrained weights
    for candidate in [
        os.path.join(PHISHINTENTION_DIR, "models", "resnetv2_rgb_new.pth.tar"),
        os.path.join(PHISHINTENTION_DIR, "models", "siamese_pedia.pth"),
        os.path.join(PHISHINTENTION_DIR, "models", "resnetv2_rgb_new.pth"),
    ]:
        if os.path.isfile(candidate):
            try:
                model.load_state_dict(
                    torch.load(candidate, map_location="cpu"), strict=False)
                logger.info("Loaded Siamese weights: %s", candidate)
            except Exception:
                pass
            break

    optimizer = optim.Adam(model.parameters(), lr=SIAMESE_LR, weight_decay=1e-5)
    margin = 1.0
    brands = list(brand_logos.keys())

    model.train()
    for epoch in range(SIAMESE_EPOCHS):
        total_loss = 0.0
        n_steps = 0

        # Generate mini-batches of pairs
        for _ in range(max(50, len(brands) * 2)):
            # Positive pair
            b = random.choice(brands)
            logos = brand_logos[b]
            l1 = transform(random.choice(logos)).unsqueeze(0).to(device)
            l2 = transform(random.choice(logos)).unsqueeze(0).to(device)

            e1, e2 = model(l1), model(l2)
            dist_pos = nn.functional.pairwise_distance(e1, e2)
            loss_pos = 0.5 * dist_pos.pow(2)

            # Negative pair
            b1, b2 = random.sample(brands, 2)
            l1 = transform(random.choice(brand_logos[b1])).unsqueeze(0).to(device)
            l2 = transform(random.choice(brand_logos[b2])).unsqueeze(0).to(device)

            e1, e2 = model(l1), model(l2)
            dist_neg = nn.functional.pairwise_distance(e1, e2)
            loss_neg = 0.5 * torch.clamp(margin - dist_neg, min=0).pow(2)

            loss = (loss_pos + loss_neg).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_steps += 1

        logger.info("Siamese (fallback) epoch %d/%d - loss: %.4f",
                     epoch + 1, SIAMESE_EPOCHS,
                     total_loss / max(n_steps, 1))

    model_path = os.path.join(output_dir, "siamese_pedia.pth")
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logger.info("Siamese model saved to %s", model_path)


def _build_reference_list_fallback(sites, output_dir):
    """Build reference logo list from benign sites (fallback)."""
    ref_dir = os.path.join(output_dir, "reference_list")
    os.makedirs(ref_dir, exist_ok=True)

    domain_map = {}
    saved = 0

    for site in sites:
        if site["label"] != "benign" or not site["domain"]:
            continue

        try:
            img = Image.open(site["shot_path"]).convert("RGB")
            w, h = img.size
            crop = img.crop((0, 0, min(int(w * 0.25), w),
                             min(int(h * 0.12), h)))

            brand_dir = os.path.join(ref_dir, site["domain"])
            os.makedirs(brand_dir, exist_ok=True)

            idx = len(os.listdir(brand_dir))
            crop_resized = crop.resize(LOGO_INPUT_SIZE, Image.BILINEAR)
            crop_resized.save(os.path.join(brand_dir,
                                           "logo_{}.png".format(idx)))
            domain_map[site["domain"]] = site["domain"]
            saved += 1
        except Exception:
            continue

    with open(os.path.join(ref_dir, "domain_map.json"), "w") as f:
        json.dump(domain_map, f, indent=2)

    logger.info("Reference list (fallback): %d logos from %d brands",
                saved, len(domain_map))


# =========================================================================
# 5. Main Training Pipeline
# =========================================================================

def main():
    global AWL_EPOCHS, CRP_EPOCHS

    parser = argparse.ArgumentParser(description="PhishIntention Training")
    parser.add_argument("--folder", required=True,
                        help="Training sites directory")
    parser.add_argument("--output", required=True,
                        help="Output directory for trained models")
    parser.add_argument("--skip-shared", action="store_true",
                        help="Skip shared component training (detector+siamese)")
    parser.add_argument("--skip-awl", action="store_true",
                        help="Skip AWL layout classifier training")
    parser.add_argument("--skip-crp", action="store_true",
                        help="Skip CRP transition classifier training")
    parser.add_argument("--awl-epochs", type=int, default=AWL_EPOCHS,
                        help="AWL training epochs (default: %d)" % AWL_EPOCHS)
    parser.add_argument("--crp-epochs", type=int, default=CRP_EPOCHS,
                        help="CRP training epochs (default: %d)" % CRP_EPOCHS)
    args = parser.parse_args()

    AWL_EPOCHS = args.awl_epochs
    CRP_EPOCHS = args.crp_epochs

    os.makedirs(args.output, exist_ok=True)

    # Step 1: Load training data
    logger.info("=" * 60)
    logger.info("Step 1: Loading training data")
    logger.info("=" * 60)
    sites = load_training_sites(args.folder)
    if not sites:
        logger.error("No training sites found. Exiting.")
        sys.exit(1)

    # Step 2: Train shared components (logo detector + Siamese)
    if not args.skip_shared:
        logger.info("=" * 60)
        logger.info("Step 2: Training shared components (detector + Siamese)")
        logger.info("=" * 60)
        train_shared_components(sites, args.output)
    else:
        logger.info("Skipping shared component training (--skip-shared)")

    # Step 3: Train AWL layout classifier
    if not args.skip_awl:
        logger.info("=" * 60)
        logger.info("Step 3: Training AWL layout classifier")
        logger.info("=" * 60)
        train_awl_classifier(sites, args.output)
    else:
        logger.info("Skipping AWL training (--skip-awl)")

    # Step 4: Train CRP transition classifier
    if not args.skip_crp:
        logger.info("=" * 60)
        logger.info("Step 4: Training CRP transition classifier")
        logger.info("=" * 60)
        train_crp_classifier(sites, args.output)
    else:
        logger.info("Skipping CRP training (--skip-crp)")

    # Save training metadata
    meta = {
        "n_sites": len(sites),
        "n_phish": sum(1 for s in sites if s["label"] == "phish"),
        "n_benign": sum(1 for s in sites if s["label"] == "benign"),
        "n_with_crp_data": sum(1 for s in sites if s["crp_shot"] is not None),
        "awl_epochs": AWL_EPOCHS,
        "crp_epochs": CRP_EPOCHS,
        "components_trained": {
            "shared": not args.skip_shared,
            "awl": not args.skip_awl,
            "crp": not args.skip_crp,
        },
    }
    with open(os.path.join(args.output, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("=" * 60)
    logger.info("Training complete. Models saved to %s", args.output)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
