#!/usr/bin/env python3
"""
Thin runnable driver for the mouse MuSC age-state classifier pipeline.
"""

from __future__ import annotations

from pathlib import Path
import os
import sys

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

from path_config import PROJECT_ROOT

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT / "scripts"))

import mouse_workflow_core as mouse_core  # noqa: E402


# Edit these values directly before running `python clock_system_mouse.py`.
TRAINING_PATHS = mouse_core.MouseTrainingPaths()
TRAINING_CONFIG = mouse_core.MouseTrainingConfig()


mouse_core.validate_required_paths(
    {
        **{f"mouse training input {name}": path for name, path in TRAINING_PATHS.input_files.items()},
        "mouse Ensembl mapping": TRAINING_PATHS.mapping_path,
    }
)

artifact_dir = Path(TRAINING_PATHS.artifact_dir)
artifact_dir.mkdir(parents=True, exist_ok=True)
mouse_core.step_setup_runtime()


adatas = mouse_core.step_load_mouse_datasets(TRAINING_PATHS)


adatas = mouse_core.step_harmonize_mouse_datasets(adatas, TRAINING_PATHS)


matrix_info = mouse_core.step_build_mouse_training_matrix(adatas, TRAINING_CONFIG)


atlas_info = mouse_core.step_save_mouse_training_atlas(
    matrix_info["adata_combined"],
    matrix_info["linkage_map"],
    TRAINING_PATHS,
    TRAINING_CONFIG,
)


model_info = mouse_core.step_train_mouse_model(matrix_info, TRAINING_CONFIG)


metrics = mouse_core.step_save_mouse_training_artifacts(
    TRAINING_PATHS,
    TRAINING_CONFIG,
    matrix_info,
    atlas_info,
    model_info,
)


_ = {
    "adatas": adatas,
    "adata_all": matrix_info["adata_all"],
    "adata_combined": matrix_info["adata_combined"],
    "adata_excluded": matrix_info["adata_excluded"],
    "adata_training_atlas": atlas_info["adata_training_atlas"],
    "adata_musc_combined": model_info["adata_musc_combined"],
    "adata_musc_pyscenic": model_info["adata_musc_pyscenic"],
    "metrics": metrics,
}
