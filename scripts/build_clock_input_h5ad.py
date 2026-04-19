#!/usr/bin/env python3
from __future__ import annotations

"""
Build a harmonized `.h5ad` input for clock verification from sample-level inputs.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import anndata as ad
import numpy as np
import pandas as pd
from scipy import io, sparse

from path_config import DATA_ROOT


ROOT = DATA_ROOT / "added_data"


@dataclass(frozen=True)
class SampleSpec:
    dataset_id: str
    sample_id: str
    sample_name: str
    sample_title: str
    genotype_raw: str
    genotype_group: str
    treatment_raw: str
    treatment_group: str
    injury_model: str
    injury_state: str
    dpi: int
    zeitgeber_time: Optional[int]
    include_for_clock: bool


SAMPLE_SPECS: list[SampleSpec] = [
    SampleSpec(
        dataset_id="GSE250049",
        sample_id="GSM7971945",
        sample_name="GSM7971945_C_CON",
        sample_title="Gsdmd cKO muscle",
        genotype_raw="Lyz2-Cre-Gsdmdf/f",
        genotype_group="KO",
        treatment_raw="no",
        treatment_group="uninjured",
        injury_model="none",
        injury_state="uninjured",
        dpi=0,
        zeitgeber_time=None,
        include_for_clock=False,
    ),
    SampleSpec(
        dataset_id="GSE250049",
        sample_id="GSM7971946",
        sample_name="GSM7971946_C_2",
        sample_title="Gsdmd cKO muscle 2dpi",
        genotype_raw="Lyz2-Cre-Gsdmdf/f",
        genotype_group="KO",
        treatment_raw="CTX intramuscluar injection 2dpi",
        treatment_group="CTX",
        injury_model="CTX",
        injury_state="injured",
        dpi=2,
        zeitgeber_time=None,
        include_for_clock=False,
    ),
    SampleSpec(
        dataset_id="GSE250049",
        sample_id="GSM7971947",
        sample_name="GSM7971947_C_10",
        sample_title="Gsdmd cKO muscle 10dpi",
        genotype_raw="Lyz2-Cre-Gsdmdf/f",
        genotype_group="KO",
        treatment_raw="CTX intramuscluar injection 10dpi",
        treatment_group="CTX",
        injury_model="CTX",
        injury_state="injured",
        dpi=10,
        zeitgeber_time=None,
        include_for_clock=False,
    ),
    SampleSpec(
        dataset_id="GSE250049",
        sample_id="GSM7971948",
        sample_name="GSM7971948_F_CON",
        sample_title="Gsdmd WT control muscle",
        genotype_raw="Gsdmdf/f",
        genotype_group="WT_like_control",
        treatment_raw="no",
        treatment_group="uninjured",
        injury_model="none",
        injury_state="uninjured",
        dpi=0,
        zeitgeber_time=None,
        include_for_clock=True,
    ),
    SampleSpec(
        dataset_id="GSE250049",
        sample_id="GSM7971949",
        sample_name="GSM7971949_F_2",
        sample_title="Gsdmd WT muscle 2dpi",
        genotype_raw="Gsdmdf/f",
        genotype_group="WT_like_control",
        treatment_raw="CTX intramuscluar injection 2dpi",
        treatment_group="CTX",
        injury_model="CTX",
        injury_state="injured",
        dpi=2,
        zeitgeber_time=None,
        include_for_clock=True,
    ),
    SampleSpec(
        dataset_id="GSE250049",
        sample_id="GSM7971950",
        sample_name="GSM7971950_F_10",
        sample_title="Gsdmd WT muscle 10dpi",
        genotype_raw="Gsdmdf/f",
        genotype_group="WT_like_control",
        treatment_raw="CTX intramuscluar injection 10dpi",
        treatment_group="CTX",
        injury_model="CTX",
        injury_state="injured",
        dpi=10,
        zeitgeber_time=None,
        include_for_clock=True,
    ),
    SampleSpec(
        dataset_id="GSE250049",
        sample_id="GSM7971951",
        sample_name="GSM7971951_M_3",
        sample_title="control muscle with control treatment 3dpi",
        genotype_raw="wildtype",
        genotype_group="WT",
        treatment_raw="CTX intramuscluar injection 3dpi with control treatment",
        treatment_group="control_treatment",
        injury_model="CTX",
        injury_state="injured",
        dpi=3,
        zeitgeber_time=None,
        include_for_clock=True,
    ),
    SampleSpec(
        dataset_id="GSE250049",
        sample_id="GSM7971952",
        sample_name="GSM7971952_M_10",
        sample_title="control muscle with control treatment 10dpi",
        genotype_raw="wildtype",
        genotype_group="WT",
        treatment_raw="CTX intramuscluar injection 10dpi with control treatment",
        treatment_group="control_treatment",
        injury_model="CTX",
        injury_state="injured",
        dpi=10,
        zeitgeber_time=None,
        include_for_clock=True,
    ),
    SampleSpec(
        dataset_id="GSE250049",
        sample_id="GSM7971953",
        sample_name="GSM7971953_E_3",
        sample_title="control muscle with EET treatment 3dpi",
        genotype_raw="wildtype",
        genotype_group="WT",
        treatment_raw="CTX intramuscluar injection 3dpi with EET treatment",
        treatment_group="EET",
        injury_model="CTX",
        injury_state="injured",
        dpi=3,
        zeitgeber_time=None,
        include_for_clock=False,
    ),
    SampleSpec(
        dataset_id="GSE250049",
        sample_id="GSM7971954",
        sample_name="GSM7971954_E_10",
        sample_title="control muscle with EET treatment 10dpi",
        genotype_raw="wildtype",
        genotype_group="WT",
        treatment_raw="CTX intramuscluar injection 10dpi with EET treatment",
        treatment_group="EET",
        injury_model="CTX",
        injury_state="injured",
        dpi=10,
        zeitgeber_time=None,
        include_for_clock=False,
    ),
    SampleSpec(
        dataset_id="GSE278177",
        sample_id="GSM8541256",
        sample_name="GSM8541256_Day0_Q4",
        sample_title="0dpi_ZT4",
        genotype_raw="wildtype",
        genotype_group="WT",
        treatment_raw="muscle cells collected from uninjured TA muscle at ZT4",
        treatment_group="uninjured",
        injury_model="none",
        injury_state="uninjured",
        dpi=0,
        zeitgeber_time=4,
        include_for_clock=True,
    ),
    SampleSpec(
        dataset_id="GSE278177",
        sample_id="GSM8541257",
        sample_name="GSM8541257_Day0_Q16",
        sample_title="0dpi_ZT16",
        genotype_raw="wildtype",
        genotype_group="WT",
        treatment_raw="muscle cells collected from uninjured TA muscle at ZT16",
        treatment_group="uninjured",
        injury_model="none",
        injury_state="uninjured",
        dpi=0,
        zeitgeber_time=16,
        include_for_clock=True,
    ),
    SampleSpec(
        dataset_id="GSE278177",
        sample_id="GSM8541258",
        sample_name="GSM8541258_Day1_A4",
        sample_title="1dpi_ZT4",
        genotype_raw="wildtype",
        genotype_group="WT",
        treatment_raw="muscle cells collected from cardiotoxin-injured TA muscle at ZT4 on day 1 post-injury",
        treatment_group="CTX",
        injury_model="CTX",
        injury_state="injured",
        dpi=1,
        zeitgeber_time=4,
        include_for_clock=True,
    ),
    SampleSpec(
        dataset_id="GSE278177",
        sample_id="GSM8541259",
        sample_name="GSM8541259_Day1_A16",
        sample_title="1dpi_ZT16",
        genotype_raw="wildtype",
        genotype_group="WT",
        treatment_raw="muscle cells collected from cardiotoxin-injured TA muscle at ZT16 on day 1 post-injury",
        treatment_group="CTX",
        injury_model="CTX",
        injury_state="injured",
        dpi=1,
        zeitgeber_time=16,
        include_for_clock=True,
    ),
    SampleSpec(
        dataset_id="GSE278177",
        sample_id="GSM8541260",
        sample_name="GSM8541260_Day3_A4",
        sample_title="3dpi_ZT4",
        genotype_raw="wildtype",
        genotype_group="WT",
        treatment_raw="muscle cells collected from cardiotoxin-injured TA muscle at ZT4 on day 3 post-injury",
        treatment_group="CTX",
        injury_model="CTX",
        injury_state="injured",
        dpi=3,
        zeitgeber_time=4,
        include_for_clock=True,
    ),
    SampleSpec(
        dataset_id="GSE278177",
        sample_id="GSM8541261",
        sample_name="GSM8541261_Day3_A16",
        sample_title="3dpi_ZT16",
        genotype_raw="wildtype",
        genotype_group="WT",
        treatment_raw="muscle cells collected from cardiotoxin-injured TA muscle at ZT16 on day 3 post-injury",
        treatment_group="CTX",
        injury_model="CTX",
        injury_state="injured",
        dpi=3,
        zeitgeber_time=16,
        include_for_clock=True,
    ),
]


DATASET_TITLES = {
    "GSE250049": "GSDMD-mediated metabolic crosstalk licenses a pro-regenerative niche for tissue repair",
    "GSE278177": "Immunomodulatory Role of the Stem Cell Circadian Clock in Muscle Repair",
}


def read_lines(path: Path) -> List[str]:
    import gzip

    with gzip.open(path, "rt") as handle:
        return [line.rstrip("\n") for line in handle]


def find_triplet(sample_dir: Path, stem: str) -> Dict[str, Path]:
    candidates = {
        "barcodes": [
            sample_dir / f"{stem}.barcodes.tsv.gz",
            sample_dir / f"{stem}_barcodes.tsv.gz",
        ],
        "features": [
            sample_dir / f"{stem}.features.tsv.gz",
            sample_dir / f"{stem}_features.tsv.gz",
        ],
        "matrix": [
            sample_dir / f"{stem}.matrix.mtx.gz",
            sample_dir / f"{stem}_matrix.mtx.gz",
        ],
    }
    out = {}
    for key, paths in candidates.items():
        match = next((path for path in paths if path.exists()), None)
        if match is None:
            raise FileNotFoundError(f"Missing {key} file for {stem} in {sample_dir}")
        out[key] = match
    return out


def load_sample(sample: SampleSpec, root: Path) -> ad.AnnData:
    sample_dir = root / sample.dataset_id / sample.sample_name
    paths = find_triplet(sample_dir, sample.sample_name)

    barcodes = read_lines(paths["barcodes"])
    features = [line.split("\t") for line in read_lines(paths["features"])]
    matrix = io.mmread(paths["matrix"]).tocsr().transpose().tocsr()

    if matrix.shape[0] != len(barcodes):
        raise ValueError(f"Barcode count mismatch for {sample.sample_name}")
    if matrix.shape[1] != len(features):
        raise ValueError(f"Feature count mismatch for {sample.sample_name}")

    gene_ids = [row[0] for row in features]
    gene_symbols = [row[1] if len(row) > 1 else row[0] for row in features]
    feature_types = [row[2] if len(row) > 2 else "Gene Expression" for row in features]

    obs_names = [f"{sample.sample_id}:{barcode}" for barcode in barcodes]
    obs = pd.DataFrame(index=obs_names)
    obs["dataset_id"] = sample.dataset_id
    obs["series_title"] = DATASET_TITLES[sample.dataset_id]
    obs["source"] = sample.dataset_id
    obs["sample_id"] = sample.sample_id
    obs["sample_name"] = sample.sample_name
    obs["sample_title"] = sample.sample_title
    obs["genotype_raw"] = sample.genotype_raw
    obs["genotype_group"] = sample.genotype_group
    obs["treatment_raw"] = sample.treatment_raw
    obs["treatment_group"] = sample.treatment_group
    obs["injury_model"] = sample.injury_model
    obs["injury_state"] = sample.injury_state
    obs["dpi"] = sample.dpi
    obs["zeitgeber_time"] = pd.Series([sample.zeitgeber_time] * len(obs), index=obs.index, dtype="Int64")
    obs["include_for_clock"] = sample.include_for_clock

    var = pd.DataFrame(index=pd.Index(gene_ids, name="gene_id"))
    var["gene_symbol"] = gene_symbols
    var["feature_type"] = feature_types
    var["ensembl_id"] = gene_ids

    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    adata.layers["counts"] = adata.X.copy()
    compute_qc(adata)
    adata = apply_qc(adata)
    adata.uns["dataset_id"] = sample.dataset_id
    adata.uns["sample_id"] = sample.sample_id
    return adata


def compute_qc(adata: ad.AnnData) -> None:
    x = adata.X.tocsr()
    detected = np.asarray((x > 0).sum(axis=1)).ravel().astype(np.int64)
    total_counts = np.asarray(x.sum(axis=1)).ravel().astype(np.int64)

    gene_names = adata.var["gene_symbol"].fillna(adata.var_names.to_series()).astype(str)
    mt_mask = gene_names.str.upper().str.startswith("MT-").to_numpy()
    if mt_mask.any():
        mt_counts = np.asarray(x[:, mt_mask].sum(axis=1)).ravel().astype(np.int64)
        pct_mt = np.divide(
            mt_counts * 100.0,
            np.maximum(total_counts, 1),
            out=np.zeros_like(total_counts, dtype=float),
            where=total_counts > 0,
        )
    else:
        pct_mt = np.zeros(x.shape[0], dtype=float)

    adata.obs["n_genes_by_counts"] = detected
    adata.obs["total_counts"] = total_counts
    adata.obs["pct_counts_mt"] = pct_mt


def apply_qc(adata: ad.AnnData, min_cells: int = 3, min_genes: int = 200, max_pct_mt: float = 20.0) -> ad.AnnData:
    x = adata.X.tocsr()
    gene_mask = np.asarray((x > 0).sum(axis=0)).ravel() >= min_cells
    adata = adata[:, gene_mask].copy()
    cell_mask = (
        (adata.obs["n_genes_by_counts"].to_numpy() >= min_genes)
        & (adata.obs["pct_counts_mt"].to_numpy() <= max_pct_mt)
    )
    adata = adata[cell_mask].copy()
    adata.layers["counts"] = adata.X.copy()
    return adata


def build_dataset(dataset_id: str, specs: Iterable[SampleSpec], root: Path) -> ad.AnnData:
    adatas = [load_sample(spec, root) for spec in specs if spec.include_for_clock]
    combined = ad.concat(adatas, join="inner", merge="same")
    combined.layers["counts"] = combined.X.copy()
    combined.obs["dataset_id"] = combined.obs["dataset_id"].astype("category")
    combined.obs["source"] = combined.obs["source"].astype("category")
    combined.obs["sample_id"] = combined.obs["sample_id"].astype("category")
    combined.obs["sample_name"] = combined.obs["sample_name"].astype("category")
    combined.obs["sample_title"] = combined.obs["sample_title"].astype("category")
    combined.obs["genotype_raw"] = combined.obs["genotype_raw"].astype("category")
    combined.obs["genotype_group"] = combined.obs["genotype_group"].astype("category")
    combined.obs["treatment_raw"] = combined.obs["treatment_raw"].astype("category")
    combined.obs["treatment_group"] = combined.obs["treatment_group"].astype("category")
    combined.obs["injury_model"] = combined.obs["injury_model"].astype("category")
    combined.obs["injury_state"] = combined.obs["injury_state"].astype("category")
    combined.obs["include_for_clock"] = combined.obs["include_for_clock"].astype(bool)
    combined.uns["dataset_id"] = dataset_id
    combined.uns["series_title"] = DATASET_TITLES[dataset_id]
    return combined


def summarize_specs(specs: Iterable[SampleSpec]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        rows.append(
            {
                "dataset_id": spec.dataset_id,
                "sample_id": spec.sample_id,
                "sample_name": spec.sample_name,
                "include_for_clock": spec.include_for_clock,
                "genotype_group": spec.genotype_group,
                "treatment_group": spec.treatment_group,
                "dpi": spec.dpi,
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build filtered raw-count h5ad files for clock input.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Root directory containing added_data.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT,
        help="Directory to write h5ad outputs into.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs_by_dataset: Dict[str, List[SampleSpec]] = {}
    for spec in SAMPLE_SPECS:
        specs_by_dataset.setdefault(spec.dataset_id, []).append(spec)

    summary = summarize_specs(SAMPLE_SPECS)
    print(summary.groupby(["dataset_id", "include_for_clock"]).size().rename("samples"))

    for dataset_id, specs in specs_by_dataset.items():
        dataset_adata = build_dataset(dataset_id, specs, args.root)
        output_path = args.output_dir / f"{dataset_id}_clock_input_raw.h5ad"
        dataset_adata.write_h5ad(output_path, compression="gzip")
        print(
            f"{dataset_id}: cells={dataset_adata.n_obs}, genes={dataset_adata.n_vars}, "
            f"samples={dataset_adata.obs['sample_id'].nunique()}, output={output_path}"
        )


if __name__ == "__main__":
    main()
