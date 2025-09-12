import torch

class Config:
    # --- General Settings ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42

    # --- Dataset & Data Loading ---
    DATASET_PATH = "data/images/"
    ANNOTATION_FILE = "data/annotations.txt"
    NUM_CLASSES = 2  # +1 for background, e.g., if you have 1 class, set this to 2
    BATCH_SIZE = 4
    NUM_WORKERS = 2

    # --- Model & Training Parameters ---
    LEARNING_RATE_G = 1e-4
    LEARNING_RATE_D = 1e-4
    LEARNING_RATE_DETECTOR = 1e-5
    NUM_EPOCHS = 20

    # --- Paths for saving models and results ---
    CHECKPOINT_DIR = "results/checkpoints"
    RESULT_DIR = "results/visualizations"
    SAVE_EVERY_N_EPOCHS = 5

    def __init__(self):
        print(f"Using device: {self.DEVICE}")

# Initialize the config
cfg = Config()
