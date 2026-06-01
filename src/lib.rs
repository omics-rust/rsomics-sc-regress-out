use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use flate2::read::MultiGzDecoder;
use rayon::prelude::*;
use rsomics_common::{Result, RsomicsError};

/// A single-cell count matrix in 10x MatrixMarket layout: rows are genes,
/// columns are cells, stored as coordinate triplets. Counts are held as f32
/// because scanpy reads the 10x matrix as float32 and runs the regression on
/// that storage; matching the dtype is what keeps the residuals bit-faithful.
pub struct CountMatrix {
    pub n_genes: usize,
    pub n_cells: usize,
    pub entries: Vec<Entry>,
}

#[derive(Clone, Copy)]
pub struct Entry {
    pub gene: u32,
    pub cell: u32,
    pub value: f32,
}

/// Per-cell covariate table: one row per cell in barcode order, plus the
/// selected column names. Values are f64 to mirror scanpy, where the
/// regressors come from `adata.obs` (float64) regardless of X dtype.
pub struct Covariates {
    pub names: Vec<String>,
    pub n_cells: usize,
    /// Row-major: `values[cell * names.len() + j]`.
    pub values: Vec<f64>,
}

pub fn open_mtx(dir: &Path) -> Result<Box<dyn Read>> {
    for name in ["matrix.mtx.gz", "matrix.mtx"] {
        let path = dir.join(name);
        if path.exists() {
            return open_maybe_gz(&path);
        }
    }
    Err(RsomicsError::InvalidInput(format!(
        "no matrix.mtx or matrix.mtx.gz in {}",
        dir.display()
    )))
}

fn open_maybe_gz(path: &Path) -> Result<Box<dyn Read>> {
    let file = File::open(path)
        .map_err(|e| RsomicsError::InvalidInput(format!("{}: {e}", path.display())))?;
    if path.extension().is_some_and(|e| e == "gz") {
        Ok(Box::new(MultiGzDecoder::new(file)))
    } else {
        Ok(Box::new(file))
    }
}

/// Parse a MatrixMarket coordinate file (real, integer, or pattern; general),
/// 10x layout (genes on rows, cells on columns). Values are kept as f32.
pub fn parse_mtx(reader: impl Read) -> Result<CountMatrix> {
    let mut reader = BufReader::new(reader);
    let mut line = String::new();

    reader.read_line(&mut line).map_err(RsomicsError::Io)?;
    let banner = line.trim();
    if !banner.starts_with("%%MatrixMarket") {
        return Err(RsomicsError::InvalidInput(
            "missing %%MatrixMarket banner".into(),
        ));
    }
    let pattern = banner.contains("pattern");

    let (n_genes, n_cells, nnz) = loop {
        line.clear();
        let n = reader.read_line(&mut line).map_err(RsomicsError::Io)?;
        if n == 0 {
            return Err(RsomicsError::InvalidInput("truncated MTX header".into()));
        }
        let t = line.trim();
        if t.is_empty() || t.starts_with('%') {
            continue;
        }
        let mut it = t.split_whitespace();
        let rows = parse_usize(it.next())?;
        let cols = parse_usize(it.next())?;
        let nnz = parse_usize(it.next())?;
        break (rows, cols, nnz);
    };

    let mut entries = Vec::with_capacity(nnz);
    for raw in reader.lines() {
        let raw = raw.map_err(RsomicsError::Io)?;
        let t = raw.trim();
        if t.is_empty() {
            continue;
        }
        let mut it = t.split_whitespace();
        let gene = parse_usize(it.next())?;
        let cell = parse_usize(it.next())?;
        let value: f32 = if pattern {
            1.0
        } else {
            it.next()
                .ok_or_else(|| RsomicsError::InvalidInput("MTX entry missing value".into()))?
                .parse::<f32>()?
        };
        if gene == 0 || gene > n_genes || cell == 0 || cell > n_cells {
            return Err(RsomicsError::InvalidInput(format!(
                "MTX index out of bounds: ({gene}, {cell})"
            )));
        }
        entries.push(Entry {
            gene: (gene - 1) as u32,
            cell: (cell - 1) as u32,
            value,
        });
    }
    if entries.len() != nnz {
        return Err(RsomicsError::InvalidInput(format!(
            "MTX declared {nnz} entries, found {}",
            entries.len()
        )));
    }

    Ok(CountMatrix {
        n_genes,
        n_cells,
        entries,
    })
}

/// Parse a covariate TSV: a header line of column names, then one row per cell.
/// `keys` selects which columns to keep (in the given order); empty keeps all.
pub fn parse_covariates(reader: impl Read, keys: &[String]) -> Result<Covariates> {
    let mut reader = BufReader::new(reader);
    let mut header = String::new();
    reader.read_line(&mut header).map_err(RsomicsError::Io)?;
    let header: Vec<&str> = header.trim_end().split('\t').collect();

    let cols: Vec<usize> = if keys.is_empty() {
        (0..header.len()).collect()
    } else {
        keys.iter()
            .map(|k| {
                header.iter().position(|h| h == k).ok_or_else(|| {
                    RsomicsError::InvalidInput(format!("covariate column not found: {k}"))
                })
            })
            .collect::<Result<_>>()?
    };
    let names: Vec<String> = cols.iter().map(|&c| header[c].to_string()).collect();
    let k = names.len();

    let mut values = Vec::new();
    let mut n_cells = 0;
    for raw in reader.lines() {
        let raw = raw.map_err(RsomicsError::Io)?;
        if raw.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = raw.trim_end().split('\t').collect();
        for &c in &cols {
            let v = fields
                .get(c)
                .ok_or_else(|| RsomicsError::InvalidInput("covariate row too short".into()))?
                .parse::<f64>()?;
            values.push(v);
        }
        n_cells += 1;
    }
    if k == 0 {
        return Err(RsomicsError::InvalidInput("no covariate columns".into()));
    }
    Ok(Covariates {
        names,
        n_cells,
        values,
    })
}

/// Residuals of each gene regressed on an intercept plus the covariates,
/// returned dense in `genes × cells` column-major (cell-major) order: every
/// gene of cell 0, then cell 1, …  This is scanpy's fast numpy_regress_out:
/// solve the normal equations once for all genes, then subtract the fit.
///
/// The regressor matrix R is `cells × (k+1)` in f64; the gram matrix
/// `G = Rᵀ R` is inverted; coefficients are `G⁻¹ (Rᵀ X)`. Residuals are stored
/// f32 — X carries f32, the fit accumulates in f64, and the difference is cast
/// back to f32, exactly as scanpy's in-place subtraction on a float32 X does.
pub fn regress_out_dense(m: &CountMatrix, cov: &Covariates) -> Result<Vec<f32>> {
    if cov.n_cells != m.n_cells {
        return Err(RsomicsError::InvalidInput(format!(
            "covariate rows ({}) != matrix cells ({})",
            cov.n_cells, m.n_cells
        )));
    }
    let nc = m.n_cells;
    let ng = m.n_genes;
    let kc = cov.names.len();
    let p = kc + 1;

    let regressor = |cell: usize, j: usize| -> f64 {
        if j == 0 {
            1.0
        } else {
            cov.values[cell * kc + (j - 1)]
        }
    };

    let mut gram = vec![0.0_f64; p * p];
    for cell in 0..nc {
        for a in 0..p {
            let ra = regressor(cell, a);
            for b in a..p {
                gram[a * p + b] += ra * regressor(cell, b);
            }
        }
    }
    for a in 0..p {
        for b in 0..a {
            gram[a * p + b] = gram[b * p + a];
        }
    }
    let ginv = invert(&gram, p)
        .ok_or_else(|| RsomicsError::InvalidInput("regressor gram matrix is singular".into()))?;

    let mut rtx = vec![0.0_f64; p * ng];
    for e in &m.entries {
        let cell = e.cell as usize;
        let gene = e.gene as usize;
        let v = e.value as f64;
        for j in 0..p {
            rtx[j * ng + gene] += regressor(cell, j) * v;
        }
    }

    let mut coeff = vec![0.0_f64; p * ng];
    coeff.par_chunks_mut(ng).enumerate().for_each(|(a, row)| {
        for gene in 0..ng {
            let mut acc = 0.0_f64;
            for b in 0..p {
                acc += ginv[a * p + b] * rtx[b * ng + gene];
            }
            row[gene] = acc;
        }
    });

    let mut x = vec![0.0_f32; ng * nc];
    for e in &m.entries {
        x[e.cell as usize * ng + e.gene as usize] = e.value;
    }
    x.par_chunks_mut(ng).enumerate().for_each(|(cell, col)| {
        let mut rcell = [0.0_f64; 64];
        let rvec: Vec<f64>;
        let r: &[f64] = if p <= 64 {
            for (j, slot) in rcell.iter_mut().take(p).enumerate() {
                *slot = regressor(cell, j);
            }
            &rcell[..p]
        } else {
            rvec = (0..p).map(|j| regressor(cell, j)).collect();
            &rvec
        };
        for gene in 0..ng {
            let mut fit = 0.0_f64;
            for j in 0..p {
                fit += r[j] * coeff[j * ng + gene];
            }
            col[gene] = (col[gene] as f64 - fit) as f32;
        }
    });
    Ok(x)
}

/// Gauss-Jordan inverse of a small dense `n × n` matrix; `None` if singular.
fn invert(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut m = a.to_vec();
    let mut inv = vec![0.0_f64; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    for col in 0..n {
        let mut pivot = col;
        let mut best = m[col * n + col].abs();
        for r in (col + 1)..n {
            let v = m[r * n + col].abs();
            if v > best {
                best = v;
                pivot = r;
            }
        }
        if best == 0.0 {
            return None;
        }
        if pivot != col {
            for k in 0..n {
                m.swap(col * n + k, pivot * n + k);
                inv.swap(col * n + k, pivot * n + k);
            }
        }
        let d = m[col * n + col];
        for k in 0..n {
            m[col * n + k] /= d;
            inv[col * n + k] /= d;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = m[r * n + col];
            if f == 0.0 {
                continue;
            }
            for k in 0..n {
                m[r * n + k] -= f * m[col * n + k];
                inv[r * n + k] -= f * inv[col * n + k];
            }
        }
    }
    Some(inv)
}

/// Write the dense residual matrix in MatrixMarket `array real general` layout:
/// banner, `n_genes n_cells`, then one value per line in column-major (cell-
/// major) order, matching scipy's dense MatrixMarket and the buffer order.
pub fn write_dense(n_genes: usize, n_cells: usize, dense: &[f32], out: impl Write) -> Result<()> {
    let mut w = BufWriter::with_capacity(1 << 20, out);
    w.write_all(b"%%MatrixMarket matrix array real general\n")
        .map_err(RsomicsError::Io)?;
    let mut header = format!("{n_genes} {n_cells}");
    header.push('\n');
    w.write_all(header.as_bytes()).map_err(RsomicsError::Io)?;

    let mut fmt = ryu::Buffer::new();
    let mut buf: Vec<u8> = Vec::with_capacity(1 << 16);
    for &v in dense {
        buf.extend_from_slice(fmt.format(v).as_bytes());
        buf.push(b'\n');
        if buf.len() >= 1 << 15 {
            w.write_all(&buf).map_err(RsomicsError::Io)?;
            buf.clear();
        }
    }
    w.write_all(&buf).map_err(RsomicsError::Io)?;
    w.flush().map_err(RsomicsError::Io)?;
    Ok(())
}

fn parse_usize(tok: Option<&str>) -> Result<usize> {
    tok.ok_or_else(|| RsomicsError::InvalidInput("MTX header missing a dimension".into()))?
        .parse::<usize>()
        .map_err(Into::into)
}

/// End-to-end: read the 10x matrix and covariates, regress out, write a dense
/// genes × cells residual matrix.
pub fn run(
    mtx_dir: &Path,
    cov_path: &Path,
    keys: &[String],
    out: impl Write,
) -> Result<(usize, usize)> {
    let m = parse_mtx(open_mtx(mtx_dir)?)?;
    let cov = parse_covariates(open_maybe_gz(cov_path)?, keys)?;
    let dense = regress_out_dense(&m, &cov)?;
    write_dense(m.n_genes, m.n_cells, &dense, out)?;
    Ok((m.n_genes, m.n_cells))
}

pub fn open_output(path: &str) -> Result<Box<dyn Write>> {
    if path == "-" {
        Ok(Box::new(std::io::stdout().lock()))
    } else {
        Ok(Box::new(
            File::create(PathBuf::from(path)).map_err(RsomicsError::Io)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cov(values: Vec<f64>, names: &[&str], n_cells: usize) -> Covariates {
        Covariates {
            names: names.iter().map(|s| s.to_string()).collect(),
            n_cells,
            values,
        }
    }

    #[test]
    fn constant_covariate_is_singular() {
        // A covariate column that is constant is collinear with the intercept,
        // so the gram matrix is singular and we bail rather than emit garbage.
        let m = CountMatrix {
            n_genes: 2,
            n_cells: 4,
            entries: vec![
                Entry {
                    gene: 0,
                    cell: 0,
                    value: 4.0,
                },
                Entry {
                    gene: 1,
                    cell: 2,
                    value: 8.0,
                },
            ],
        };
        let c = cov(vec![7.0, 7.0, 7.0, 7.0], &["z"], 4);
        assert!(regress_out_dense(&m, &c).is_err());
    }

    #[test]
    fn residual_matches_hand_ols() {
        // gene 0 over 3 cells: y = [3, 0, 6]; covariate x = [1, 2, 3].
        // R = [[1,1],[1,2],[1,3]]. OLS: slope = 1.5, intercept = 0.0
        // (means: x̄=2, ȳ=3; Sxy = (1-2)(3-3)+(2-2)(0-3)+(3-2)(6-3)=3; Sxx=2 -> slope 1.5;
        //  intercept = 3 - 1.5*2 = 0). Fitted = [1.5, 3.0, 4.5];
        // residuals = [1.5, -3.0, 1.5].
        let m = CountMatrix {
            n_genes: 1,
            n_cells: 3,
            entries: vec![
                Entry {
                    gene: 0,
                    cell: 0,
                    value: 3.0,
                },
                Entry {
                    gene: 0,
                    cell: 2,
                    value: 6.0,
                },
            ],
        };
        let c = cov(vec![1.0, 2.0, 3.0], &["x"], 3);
        let d = regress_out_dense(&m, &c).unwrap();
        // cell-major, 1 gene: [resid_cell0, resid_cell1, resid_cell2]
        assert!((d[0] - 1.5).abs() < 1e-5, "{}", d[0]);
        assert!((d[1] + 3.0).abs() < 1e-5, "{}", d[1]);
        assert!((d[2] - 1.5).abs() < 1e-5, "{}", d[2]);
    }

    #[test]
    fn singular_regressors_error() {
        let m = CountMatrix {
            n_genes: 1,
            n_cells: 3,
            entries: vec![Entry {
                gene: 0,
                cell: 0,
                value: 1.0,
            }],
        };
        // covariate identical to the intercept -> collinear -> singular.
        let c = cov(vec![1.0, 1.0, 1.0], &["ones2"], 3);
        assert!(regress_out_dense(&m, &c).is_err());
    }

    #[test]
    fn covariate_count_mismatch_errors() {
        let m = CountMatrix {
            n_genes: 1,
            n_cells: 3,
            entries: vec![],
        };
        let c = cov(vec![1.0, 2.0], &["x"], 2);
        assert!(regress_out_dense(&m, &c).is_err());
    }
}
