#!/usr/bin/env python3
"""scanpy oracle for rsomics-sc-regress-out.

Reads a 10x MTX directory and a per-cell covariate TSV, runs
sc.pp.regress_out on the named keys (or all covariate columns), and dumps the
dense residual as a genes x cells MatrixMarket `array` file in column-major
(cell-major) order — matching the tool's output layout and float32 storage.

Usage: scanpy_regress_oracle.py <mtx_dir> <covariates.tsv> <out.mtx> [keys_csv]
"""
import sys

import numpy as np
import pandas as pd
import scanpy as sc


def main() -> None:
    mtx_dir = sys.argv[1]
    cov_path = sys.argv[2]
    out_path = sys.argv[3]
    keys = sys.argv[4].split(",") if len(sys.argv) > 4 and sys.argv[4] else None

    adata = sc.read_10x_mtx(mtx_dir)
    cov = pd.read_csv(cov_path, sep="\t")
    use = keys if keys is not None else list(cov.columns)
    for col in use:
        adata.obs[col] = cov[col].to_numpy().astype(float)

    sc.pp.regress_out(adata, use)

    x = np.asarray(adata.X.todense() if hasattr(adata.X, "todense") else adata.X)
    gc = x.T  # genes x cells
    n_genes, n_cells = gc.shape
    flat = gc.reshape(n_genes, n_cells).flatten(order="F")
    with open(out_path, "w") as f:
        f.write("%%MatrixMarket matrix array real general\n")
        f.write(f"{n_genes} {n_cells}\n")
        for v in flat:
            f.write(f"{float(v)!r}\n")


if __name__ == "__main__":
    main()
