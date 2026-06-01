use std::path::{Path, PathBuf};
use std::process::Command;

/// Residuals are float32; ours and scanpy run the same f32 arithmetic but sum
/// the per-cell dot product in a different order, so allow a small relative or
/// absolute slack on top of f32 rounding.
const REL: f64 = 1e-4;
const ABS: f64 = 1e-4;

fn scanpy_python() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let shared = PathBuf::from(&home).join("oracle-venvs/scanpy/bin/python");
    if shared.exists() {
        return Some(shared);
    }
    for cand in ["python3", "python"] {
        let ok = Command::new(cand)
            .args(["-c", "import scanpy, pandas"])
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if ok {
            return Some(PathBuf::from(cand));
        }
    }
    None
}

/// Dense MatrixMarket `array`: banner, `rows cols`, then one value per line in
/// column-major order. Returns the values in that order.
fn parse_array(text: &str) -> Vec<f64> {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let banner = lines.next().unwrap();
    assert!(banner.starts_with("%%MatrixMarket"), "bad banner: {banner}");
    let mut vals = Vec::new();
    let mut dim_seen = false;
    for line in lines {
        let t = line.trim();
        if t.starts_with('%') {
            continue;
        }
        if !dim_seen {
            dim_seen = true;
            continue;
        }
        vals.push(t.parse::<f64>().unwrap());
    }
    vals
}

fn diff(a: &[f64], b: &[f64], label: &str) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "{label}: length {} vs {}",
        a.len(),
        b.len()
    );
    let mut max_dev = 0.0_f64;
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let dev = (x - y).abs();
        max_dev = max_dev.max(dev);
        let tol = ABS + REL * x.abs().max(y.abs());
        assert!(
            dev <= tol,
            "{label}: value {i} differs: {x} vs {y} (dev {dev:e})"
        );
    }
    max_dev
}

fn run_ours(mtx_dir: &Path, cov: &Path, keys: &[&str]) -> Vec<f64> {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_rsomics-sc-regress-out"));
    cmd.arg(mtx_dir)
        .arg("-c")
        .arg(cov)
        .arg("-o")
        .arg("-")
        .arg("-q");
    if !keys.is_empty() {
        cmd.arg("-k").arg(keys.join(","));
    }
    let out = cmd.output().expect("run rsomics-sc-regress-out");
    assert!(
        out.status.success(),
        "ours failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    parse_array(&String::from_utf8(out.stdout).unwrap())
}

/// Always runs (no oracle needed): ours vs the committed golden the oracle
/// produced once. This is the gate CI relies on.
#[test]
fn matches_committed_golden() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let mtx_dir = Path::new(manifest).join("tests/golden/tenx");
    let cov = Path::new(manifest).join("tests/golden/covariates.tsv");
    let golden = Path::new(manifest).join("tests/golden/regress_out.mtx");
    assert!(golden.exists(), "missing golden {golden:?}");

    let ours = run_ours(&mtx_dir, &cov, &["total_counts", "pct_counts_mt"]);
    let want = parse_array(&std::fs::read_to_string(&golden).unwrap());
    let dev = diff(&want, &ours, "golden");
    eprintln!(
        "golden compat OK: {} values, max deviation {dev:e}",
        ours.len()
    );
}

/// Live differential vs scanpy; loud-skips when the oracle is unavailable.
#[test]
fn matches_scanpy_value_level() {
    let Some(py) = scanpy_python() else {
        eprintln!("SKIP: scanpy venv not found (~/oracle-venvs/scanpy/bin/python); compat skipped");
        return;
    };

    let manifest = env!("CARGO_MANIFEST_DIR");
    let mtx_dir = Path::new(manifest).join("tests/golden/tenx");
    let cov = Path::new(manifest).join("tests/golden/covariates.tsv");
    let oracle_py = Path::new(manifest).join("tests/scanpy_regress_oracle.py");
    assert!(mtx_dir.exists(), "missing golden 10x dir {mtx_dir:?}");

    let scratch = std::env::var("RSOMICS_SCRATCH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir());

    let oracle_out = scratch.join("sc_regress_oracle.mtx");
    let status = Command::new(&py)
        .arg(&oracle_py)
        .arg(&mtx_dir)
        .arg(&cov)
        .arg(&oracle_out)
        .arg("total_counts,pct_counts_mt")
        .status()
        .expect("run scanpy oracle");
    assert!(status.success(), "oracle failed");
    let oracle = parse_array(&std::fs::read_to_string(&oracle_out).unwrap());
    let ours = run_ours(&mtx_dir, &cov, &["total_counts", "pct_counts_mt"]);
    let dev = diff(&oracle, &ours, "live");
    eprintln!("live compat OK: {} values, max dev {dev:e}", ours.len());
}
