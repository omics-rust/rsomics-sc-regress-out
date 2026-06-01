# rsomics-sc-regress-out

Regress out unwanted per-cell covariates (total counts, mitochondrial
fraction, …) from a single-cell count matrix, numerically matching scanpy's
`sc.pp.regress_out`.

Each gene's expression across cells is fit by ordinary least squares against an
intercept plus the chosen covariate columns, and the gene's values are replaced
by the regression **residuals**. This is scanpy's fast `numpy_regress_out` path:
the normal equations `(RᵀR)⁻¹ Rᵀ X` are solved once for all genes, then the fit
`R · coeff` is subtracted from `X`. The regressor matrix `R` is built in f64
(the covariates come from per-cell annotations); the matrix `X` and the
residuals are f32, matching scanpy's float32 storage so the output is
bit-faithful.

Regressing out densifies the matrix — a gene's residual is nonzero even where
the count was an implicit zero — so the output is a full genes × cells dense
matrix in MatrixMarket `array` (column-major) layout.

A regressor matrix that is singular (e.g. a constant or collinear covariate)
is rejected rather than silently producing garbage.

## Usage

```bash
# regress out total counts and mitochondrial fraction (the common scanpy idiom)
rsomics-sc-regress-out filtered_feature_bc_matrix/ \
    -c obs.tsv -k total_counts,pct_counts_mt -o residuals.mtx
```

Input is a 10x MTX directory (`matrix.mtx` or `matrix.mtx.gz`, genes × cells)
plus a covariate TSV: a header of column names and one row per cell in barcode
order. `-k/--keys` selects which columns to regress on (all columns if omitted).
Output is a dense MatrixMarket `array real general` matrix in genes × cells
layout, one value per line in column-major (cell-major) order.

The single-categorical-key mode (scanpy's per-category-mean regression, which
falls back to the statsmodels GLM) is **not implemented**: continuous-covariate
regression on intercept + numeric columns is the routine usage, and the
categorical path is a distinct algorithm better served by its own tool.

## Origin

This crate is an independent Rust reimplementation of scanpy's
`sc.pp.regress_out` based on:

- The published method (Wolf, Angerer & Theis, "SCANPY: large-scale single-cell
  gene expression data analysis", *Genome Biology* 2018,
  doi:10.1186/s13059-017-1382-0), itself inspired by Seurat's `RegressOut`
  (Satija et al. 2015).
- The public MatrixMarket and 10x Genomics matrix file-format specs.
- Reading scanpy's `preprocessing/_simple.py` (`numpy_regress_out` / `get_resid`,
  BSD-3-Clause) to match the exact normal-equations path, the f32 X / f64
  regressor dtype split, the intercept-as-first-column convention, and the
  singular-gram fallthrough.
- Black-box value-level testing against the scanpy Python package (residuals
  match to f32 precision, max deviation ~5e-7 over 15M values).

License: MIT OR Apache-2.0.
Upstream credit: scanpy <https://github.com/scverse/scanpy> (BSD-3-Clause).
