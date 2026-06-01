#!/usr/bin/env python3
"""Generate the committed golden for rsomics-sc-regress-out.

Builds a small 10x MTX directory and a per-cell covariate TSV (total_counts,
pct_counts_mt), then runs the scanpy oracle to capture the residual matrix.
Run once on a machine with the scanpy venv; commit tests/golden/.
"""
import gzip
import os
import shutil
import subprocess
import sys

import numpy as np
import scipy.io
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    rng = np.random.default_rng(42)
    n_genes, n_cells = 30, 50
    dens = (rng.random((n_genes, n_cells)) < 0.4)
    counts = (rng.poisson(3.0, (n_genes, n_cells)) * dens).astype(int)

    tenx = os.path.join(HERE, "golden", "tenx")
    os.makedirs(tenx, exist_ok=True)
    mm = sp.csr_matrix(counts)
    p = os.path.join(tenx, "matrix.mtx")
    scipy.io.mmwrite(p, mm, field="integer")
    with open(p, "rb") as fi, gzip.open(p + ".gz", "wb") as fo:
        shutil.copyfileobj(fi, fo)
    os.remove(p)
    with gzip.open(os.path.join(tenx, "barcodes.tsv.gz"), "wt") as f:
        for i in range(n_cells):
            f.write(f"CELL{i:04d}-1\n")
    with gzip.open(os.path.join(tenx, "features.tsv.gz"), "wt") as f:
        for i in range(n_genes):
            f.write(f"ENSG{i:05d}\tGENE{i}\tGene Expression\n")

    total = counts.sum(axis=0).astype(float)  # per cell
    mito = counts[:5].sum(axis=0).astype(float)  # first 5 genes as "mito"
    pct_mt = 100.0 * mito / np.maximum(total, 1.0)
    cov_path = os.path.join(HERE, "golden", "covariates.tsv")
    with open(cov_path, "w") as f:
        f.write("total_counts\tpct_counts_mt\n")
        for i in range(n_cells):
            f.write(f"{float(total[i])}\t{float(pct_mt[i])}\n")

    out = os.path.join(HERE, "golden", "regress_out.mtx")
    subprocess.run(
        [sys.executable, os.path.join(HERE, "scanpy_regress_oracle.py"),
         tenx, cov_path, out, "total_counts,pct_counts_mt"],
        check=True,
    )
    print(f"golden written: {n_genes} genes x {n_cells} cells")


if __name__ == "__main__":
    main()
