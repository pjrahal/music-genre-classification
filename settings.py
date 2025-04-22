import os
import logging

# === Logger Setup ===
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("MusicClassifier")

# === File Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "../fma_small")
METADATA_DIR = os.path.join(BASE_DIR, "../fma_metadata")
FEATURES_DIR = os.path.join(BASE_DIR, "../features")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "../models/best_model_val.pth")

# === Feature Extraction Options ===
# FEATURES_USED = ["mel", "delta", "delta2", "mfcc"]
FEATURES_USED = ["mel", "delta", "delta2"]

# === Hyperparameters ===
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
PATIENCE = 5

# === Package Requirements ===
REQUIRED_PACKAGES = [
    "librosa",
    "numpy",
    "pandas",
    "torch",
    "tqdm",
    "scikit-learn"
]

# === Virtual Environment Directory ===
ENV_DIR = ".venv"