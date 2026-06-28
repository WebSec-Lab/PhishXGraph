#!/usr/bin/env python
"""
URLNet wrapper: reads phishinglist.csv, runs URLNet train or test.
Supports MODE=train (train from labeled data) and MODE=test (inference).
"""
import csv
import os
import sys
import subprocess
import tempfile
import time

INPUT_CSV = os.environ.get("INPUT_CSV", "/data/input/phishinglist.csv")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/data/results/urlnet")
MODEL_DIR = os.environ.get("MODEL_DIR", "/data/models/urlnet")
FEATURES_DIR = os.environ.get("FEATURES_DIR", "/data/features")
MODE = os.environ.get("MODE", "test")
URLNET_DIR = "/app/repo/URLNet"

# Training hyperparameters - defaults match original URLNet train.py defaults
URLNET_DEV_PCT = os.environ.get("URLNET_DEV_PCT", "0.05")
URLNET_EPOCHS = os.environ.get("URLNET_EPOCHS", "5")
URLNET_BATCH_SIZE = os.environ.get("URLNET_BATCH_SIZE", "128")
URLNET_LR = os.environ.get("URLNET_LR", "0.001")
URLNET_EMB_MODE = os.environ.get("URLNET_EMB_MODE", "5")
URLNET_EMB_DIM = os.environ.get("URLNET_EMB_DIM", "32")
URLNET_FILTER_SIZES = os.environ.get("URLNET_FILTER_SIZES", "3,4,5,6")
URLNET_CHECKPOINT_EVERY = os.environ.get("URLNET_CHECKPOINT_EVERY", "500")
URLNET_EVAL_EVERY = os.environ.get("URLNET_EVAL_EVERY", "500")
URLNET_PRINT_EVERY = os.environ.get("URLNET_PRINT_EVERY", "50")


def read_urls(csv_path):
    """Read URLs and labels from phishinglist.csv"""
    urls, labels = [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            urls.append(row["url"])
            labels.append(row.get("label", "unknown"))
    return urls, labels


def prepare_input(urls, labels, tmp_path, for_train=False):
    """Write URLs in URLNet format: <label>\t<url>"""
    with open(tmp_path, "w") as f:
        for url, label in zip(urls, labels):
            if for_train:
                urlnet_label = "+1" if label == "phish" else "-1"
            else:
                urlnet_label = "-1"  # dummy label for inference
            f.write("{}\t{}\n".format(urlnet_label, url))


def train_urlnet(input_path, output_dir):
    """Run URLNet train.py"""
    os.makedirs(output_dir, exist_ok=True)
    # URLNet expects output_dir to end with /
    output_dir_with_slash = output_dir if output_dir.endswith("/") else output_dir + "/"
    cmd = [
        sys.executable, os.path.join(URLNET_DIR, "train.py"),
        "--data.data_dir", input_path,
        "--data.dev_pct", URLNET_DEV_PCT,
        "--data.delimit_mode", "1",
        "--data.min_word_freq", "1",
        "--model.emb_mode", URLNET_EMB_MODE,
        "--model.emb_dim", URLNET_EMB_DIM,
        "--model.filter_sizes", URLNET_FILTER_SIZES,
        "--train.nb_epochs", URLNET_EPOCHS,
        "--train.batch_size", URLNET_BATCH_SIZE,
        "--train.lr", URLNET_LR,
        "--log.output_dir", output_dir_with_slash,
        "--log.checkpoint_every", URLNET_CHECKPOINT_EVERY,
        "--log.eval_every", URLNET_EVAL_EVERY,
        "--log.print_every", URLNET_PRINT_EVERY,
    ]
    print("[URLNet] Training: {}".format(" ".join(cmd)))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=URLNET_DIR)
    print(result.stdout.decode('utf-8', errors='replace'))
    if result.returncode != 0:
        print("[URLNet] STDERR: {}".format(result.stderr.decode('utf-8', errors='replace')), file=sys.stderr)
    return result.returncode


def run_urlnet(input_path, output_path):
    """Run URLNet test.py"""
    # Find the latest checkpoint file
    checkpoint_dir = os.path.join(MODEL_DIR, "checkpoints")
    checkpoint_file = None
    if os.path.exists(checkpoint_dir):
        # Read checkpoint file to get latest model path
        ckpt_state_file = os.path.join(checkpoint_dir, "checkpoint")
        if os.path.exists(ckpt_state_file):
            with open(ckpt_state_file, "r") as f:
                first_line = f.readline().strip()
                # Format: model_checkpoint_path: "model-75"
                if "model_checkpoint_path" in first_line:
                    model_name = first_line.split('"')[1]
                    checkpoint_file = os.path.join(checkpoint_dir, model_name)

    if not checkpoint_file or not os.path.exists(checkpoint_file + ".meta"):
        print("[URLNet] ERROR: No valid checkpoint found in {}".format(checkpoint_dir))
        return 1

    cmd = [
        sys.executable, os.path.join(URLNET_DIR, "test.py"),
        "--data.data_dir", input_path,
        "--data.delimit_mode", "1",
        "--data.word_dict_dir", os.path.join(MODEL_DIR, "words_dict.p"),
        "--data.subword_dict_dir", os.path.join(MODEL_DIR, "subwords_dict.p"),
        "--data.char_dict_dir", os.path.join(MODEL_DIR, "chars_dict.p"),
        "--log.checkpoint_dir", checkpoint_file,
        "--log.output_dir", output_path,
        "--model.emb_mode", URLNET_EMB_MODE,
        "--model.emb_dim", URLNET_EMB_DIM,
        "--test.batch_size", URLNET_BATCH_SIZE,
    ]

    print("[URLNet] Running: {}".format(" ".join(cmd)))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=URLNET_DIR)
    print(result.stdout.decode('utf-8', errors='replace'))
    if result.returncode != 0:
        print("[URLNet] STDERR: {}".format(result.stderr.decode('utf-8', errors='replace')), file=sys.stderr)
    return result.returncode


def parse_results(urls, result_file):
    """Parse URLNet output and write standardized results.csv"""
    predictions = []

    # URLNet writes results as: url\tlabel\tpredict\tscore (tab-separated with header)
    if os.path.exists(result_file):
        with open(result_file, "r") as f:
            lines = f.readlines()
            # Skip header line
            for line in lines[1:]:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    url = parts[0]
                    pred_val = int(parts[2])  # 1 = phish, -1 = benign
                    score = float(parts[3])
                    pred = "phish" if pred_val == 1 else "benign"
                    predictions.append((url, pred, score))
    else:
        # If no results, mark all as error
        for url in urls:
            predictions.append((url, "error", 0.0))

    return predictions


def write_output(predictions, output_path):
    """Write standardized results CSV"""
    os.makedirs(output_path, exist_ok=True)
    csv_path = os.path.join(output_path, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "prediction", "score"])
        for url, pred, score in predictions:
            writer.writerow([url, pred, score])
    print("[URLNet] Results written to {}".format(csv_path))


def main():
    print("[URLNet] Starting (mode={})...".format(MODE))

    if MODE == "collect":
        print("[URLNet] Collect mode: feature-collector service handles extraction. Sleeping.")
        while True:
            time.sleep(3600)

    if not os.path.exists(INPUT_CSV):
        print("[URLNet] ERROR: Input file not found: {}".format(INPUT_CSV))
        sys.exit(1)

    urls, labels = read_urls(INPUT_CSV)
    print("[URLNet] Loaded {} URLs".format(len(urls)))

    if MODE == "train":
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "train_urls.txt")
            prepare_input(urls, labels, input_path, for_train=True)
            ret = train_urlnet(input_path, MODEL_DIR)
            if ret == 0:
                print("[URLNet] Training complete. Model saved to {}".format(MODEL_DIR))
            else:
                print("[URLNet] Training failed.")
                sys.exit(1)
        return

    # Test mode
    if not os.path.exists(MODEL_DIR):
        print("[URLNet] WARNING: Model not found at {}".format(MODEL_DIR))
        print("[URLNet] Run with MODE=train first, or provide pretrained model")
        predictions = [(url, "unknown", 0.0) for url in urls]
        write_output(predictions, OUTPUT_DIR)
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "test_urls.txt")
        result_file = os.path.join(tmpdir, "results.txt")

        prepare_input(urls, labels, input_path, for_train=False)
        ret = run_urlnet(input_path, result_file)

        if ret == 0:
            predictions = parse_results(urls, result_file)
        else:
            print("[URLNet] Inference failed, writing fallback results")
            predictions = [(url, "error", 0.0) for url in urls]

    write_output(predictions, OUTPUT_DIR)
    print("[URLNet] Done.")


if __name__ == "__main__":
    main()
