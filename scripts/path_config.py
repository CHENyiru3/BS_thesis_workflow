from __future__ import annotations

import os
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parent
WORKFLOW_ROOT = SCRIPT_ROOT.parent
# Backward-compatible alias used by existing scripts.
PROJECT_ROOT = WORKFLOW_ROOT
# Backward-compatible alias used by existing scripts.
FIGURE_ROOT = WORKFLOW_ROOT

ARTIFACT_ROOT = Path(os.environ.get("BS_ARTIFACT_ROOT", str(WORKFLOW_ROOT / "artifacts"))).resolve()
DATA_ROOT = Path(os.environ.get("BS_DATA_ROOT", str(WORKFLOW_ROOT / "data"))).resolve()

MOUSE_TRAINING_DIR = DATA_ROOT / "mouse_training" / "all_train"
MOUSE_VERIFICATION_DIR = DATA_ROOT / "mouse_verification"
EXTERNAL_VERIFY_DIR = DATA_ROOT / "external_verify" / "GSE306935_RAW" / "h5ad_musclike"
HUMAN_TRAINING_DIR = DATA_ROOT / "human_training"
RESOURCES_DIR = DATA_ROOT / "resources"
CISTARGET_DIR = RESOURCES_DIR / "cistarget"
CELL_CYCLE_GENES_PATH = RESOURCES_DIR / "regev_lab_cell_cycle_genes.txt"
