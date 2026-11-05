"""Central configuration for Task 2 model development (Member B).

Single source of truth: every entry script (train / predict / smoke test)
reads paths, seeds and hyper-parameters from here, so a full run is
reproducible by keeping this file unchanged.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths --
ROOT = Path(__file__).resolve().parent            # .../project/task2_model
DATA_DIR = ROOT / "data"                         # put train.csv / test.csv here
ARTIFACT_DIR = ROOT / "artifacts"                # fitted vectoriser + weights
SUBMISSION_DIR = ROOT / "submissions"            # task2_submission.csv lands here

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
VECTORIZER_FILE = ARTIFACT_DIR / "tfidf.npz"
MODEL_FILE = ARTIFACT_DIR / "ridge_model.npz"
SPLIT_DIR = ROOT / "split"                     # canonical split (B -> C handoff)
SPLIT_FILE = SPLIT_DIR / "val_ids.csv"

# C's feature outputs (v2 "leak-free" delivery, 2026-11-03) -- parquet dirs
FEATURES_DIR = ROOT / "features"
TRAIN_FEATURES = FEATURES_DIR / "train_features"      # 7,684,645 x 21
TEST_FEATURES = FEATURES_DIR / "test_features"        # 10,000 x 21

# -------------------------------------------------------- reproducibility --
SEED = 5434            # proposed team-wide seed (see eval_protocol.md)
VAL_RATIO = 0.10       # stratified by rating label

# ------------------------------------- text representation (self-impl'd) --
MIN_DF = 3             # drop terms seen in fewer than 3 training documents
MAX_FEATURES = 30000   # vocabulary cap, keeps local GD tractable
NGRAM_RANGE = (1, 2)   # unigrams + bigrams

# ------------------- model (self-implemented ridge, gradient descent) ----
LEARNING_RATE = 0.5    # rows of X are L2-normalised -> Hessian <= I -> lr<=1 safe
LR_DECAY = 0.995       # multiplicative decay per epoch
EPOCHS = 300           # tuned 2026-11-03: 300 beats 150 (val 1.4403 vs 1.4594)
L2 = 1e-4              # ridge penalty
PATIENCE = 15          # early-stopping patience on validation RMSE

# --------------------------------------------------------------- utility --
SAMPLE_FRAC = 0.1      # None = full data. 7.68M rows is slow locally (pure-
                       # python tokenizer); pilot on 10%, full run goes
                       # through C's Spark pipeline later.

# ------------- neural text model (self-implemented, PyTorch as engine) ---
NEURAL_VOCAB_MAX = 100000   # unigram vocabulary cap (embedding models like
                            # larger vocabularies than TF-IDF's 30k)
NEURAL_EMB_DIM = 64
NEURAL_HIDDEN = 256
NEURAL_LR = 1e-3
NEURAL_BATCH = 1024
NEURAL_EPOCHS = 8
NEURAL_PATIENCE = 3

RATING_MIN = 1         # ratings live in [1, 5]; predictions are clipped
RATING_MAX = 5
