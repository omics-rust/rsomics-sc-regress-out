use criterion::{Criterion, criterion_group, criterion_main};
use rsomics_sc_regress_out::{CountMatrix, Covariates, Entry, regress_out_dense};

fn synthetic(n_genes: usize, n_cells: usize, density: f64) -> (CountMatrix, Covariates) {
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut entries = Vec::new();
    let mut total = vec![0.0_f64; n_cells];
    for (cell, tot) in total.iter_mut().enumerate() {
        for gene in 0..n_genes {
            if (next() % 10_000) as f64 / 10_000.0 < density {
                let v = (next() % 50 + 1) as f32;
                entries.push(Entry {
                    gene: gene as u32,
                    cell: cell as u32,
                    value: v,
                });
                *tot += v as f64;
            }
        }
    }
    let mut values = Vec::with_capacity(n_cells * 2);
    for &t in total.iter().take(n_cells) {
        values.push(t);
        values.push((next() % 1000) as f64 / 1000.0);
    }
    let cov = Covariates {
        names: vec!["total_counts".into(), "pct_mt".into()],
        n_cells,
        values,
    };
    (
        CountMatrix {
            n_genes,
            n_cells,
            entries,
        },
        cov,
    )
}

fn bench_regress(c: &mut Criterion) {
    let (m, cov) = synthetic(2000, 5000, 0.05);
    c.bench_function("regress_out_2000x5000_2cov", |b| {
        b.iter(|| regress_out_dense(&m, &cov).unwrap())
    });
}

criterion_group!(benches, bench_regress);
criterion_main!(benches);
