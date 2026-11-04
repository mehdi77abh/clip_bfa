"""
config.py — Central configuration for the hybrid bit-flip attack pipeline.
"""

import torch


class CFG:
    MODEL = "ViT-L/14"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TARGET_CLS = 3

    # ── Dataset (DataModule) ──
    DATA = "../data"
    DATASET = "caltech101"        # key: caltech101 | cifar10 | cifar100 | imagenet
    DATASET_NAME = "Caltech101"   # display name (used in OUT_DIR)
    DOWNLOAD = False

    BATCH = 8
    MINI_BS = 4
    WORKERS = 2

    N_TARGET_IMAGES = 32
    N_CALIB = 32
    N_GRAD = 16
    N_EVAL = 200

    QUANT_TOLERANCE = 2.0

    # ── Phase 1: Trigger ──
    TRIG_SZ = 12
    P1_EPOCHS = 300
    P1_LR = 0.1
    TV_WEIGHT = 0.0
    TRIGGER_TYPE = "center_add"
    TEMPERATURE = 0.07
    EPSILON = 8.0 / 255.0   # unused (additive trigger)
    OPACITY = 1.0           # unused (additive trigger)
    LAMBDA_VIS = 0.0

    # ── Phase 2: Routing efficiency ──
    TOP_K_LAYERS = 12

    # ── Geometry (auto-overwritten at load) ──
    PATCH_SIZE = 14
    IMG_SIZE = 224

    # ── Phase 3: Gradient pre-screening ──
    PRESCREEN_TOP_K = 30

    # ── Phase 4: Hybrid ranking ──
    GRADIENT_BETA = 10.0
    SENS_TOP_M = 150

    # ── Phase 5: Greedy search ──
    MAX_FLIPS = 80
    EVAL_BS = 32
    CANDS_PER_ROUND = 30
    ASR_TARGET = 90.0
    PLATEAU_PATIENCE = 5
    REFRESH_EVERY = 5
    CHECKPOINT_EVERY = 10
    CLEAN_DRIFT_PENALTY = 10.0
    MAX_CLEAN_DRIFT = 0.02
    INITIAL_MAX_BIT = 7
    EXPAND_BIT_AFTER = 5
    # ── Probe evaluation ──
    LP_EPOCHS = 10
    LP_LR = 1e-3
    FT_EPOCHS = 5
    FT_LR = 1e-5

    # ── Output ──
    SEED = 42
    RUN_ID = ""
    OUT_DIR = ""        # replaces OUT_DIR — set per run