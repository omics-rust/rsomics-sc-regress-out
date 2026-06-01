use std::path::PathBuf;

use clap::Parser;
use rsomics_common::{CommonFlags, Result, Tool, ToolMeta};
use rsomics_help::{Example, FlagSpec, HelpSpec, Origin, Section};

use rsomics_sc_regress_out::{open_output, run};

pub const META: ToolMeta = ToolMeta {
    name: env!("CARGO_PKG_NAME"),
    version: env!("CARGO_PKG_VERSION"),
};

#[derive(Parser, Debug)]
#[command(name = "rsomics-sc-regress-out", version, about, long_about = None, disable_help_flag = true)]
pub struct Cli {
    /// 10x MTX directory (matrix.mtx[.gz], genes×cells).
    pub input: PathBuf,

    /// Per-cell covariate TSV: header of column names, one row per cell in
    /// barcode order.
    #[arg(short = 'c', long)]
    covariates: PathBuf,

    /// Comma-separated covariate column names to regress on; empty = all.
    #[arg(short = 'k', long, value_delimiter = ',')]
    keys: Vec<String>,

    #[arg(short = 'o', long, default_value = "-")]
    output: String,

    #[command(flatten)]
    pub common: CommonFlags,
}

impl Tool for Cli {
    fn meta() -> ToolMeta {
        META
    }
    fn common(&self) -> &CommonFlags {
        &self.common
    }

    fn execute(self) -> Result<()> {
        self.common.install_rayon_pool()?;
        let out = open_output(&self.output)?;
        let (genes, cells) = run(&self.input, &self.covariates, &self.keys, out)?;
        if !self.common.quiet {
            let n = if self.keys.is_empty() {
                "all".to_string()
            } else {
                self.keys.len().to_string()
            };
            eprintln!("regressed out {n} covariates from {cells} cells × {genes} genes");
        }
        Ok(())
    }
}

pub static HELP: HelpSpec = HelpSpec {
    name: env!("CARGO_PKG_NAME"),
    version: env!("CARGO_PKG_VERSION"),
    tagline: "Regress out per-cell covariates from a single-cell matrix via per-gene OLS.",
    origin: Some(Origin {
        upstream: "scanpy sc.pp.regress_out",
        upstream_license: "BSD-3-Clause",
        our_license: "MIT OR Apache-2.0",
        paper_doi: Some("10.1186/s13059-017-1382-0"),
    }),
    usage_lines: &["<10x-mtx-dir> -c covariates.tsv [-k total_counts,pct_counts_mt] [-o out.mtx]"],
    sections: &[Section {
        title: "OPTIONS",
        flags: &[
            FlagSpec {
                short: Some('c'),
                long: "covariates",
                aliases: &[],
                value: Some("<path>"),
                type_hint: Some("Path"),
                required: true,
                default: None,
                description: "Per-cell covariate TSV (header + one row per cell).",
                why_default: None,
            },
            FlagSpec {
                short: Some('k'),
                long: "keys",
                aliases: &[],
                value: Some("<names>"),
                type_hint: Some("CSV"),
                required: false,
                default: None,
                description: "Covariate columns to regress on (comma-separated); all if omitted.",
                why_default: Some("Matches regressing on every obs column scanpy is handed."),
            },
            FlagSpec {
                short: Some('o'),
                long: "output",
                aliases: &[],
                value: Some("<path>"),
                type_hint: Some("String"),
                required: false,
                default: Some("-"),
                description: "Output dense MTX path (genes×cells array); '-' for stdout.",
                why_default: Some("Streams to stdout for pipeline composition."),
            },
        ],
    }],
    examples: &[Example {
        description: "regress out total counts and mito fraction (scanpy idiom)",
        command: "rsomics-sc-regress-out mtx_dir/ -c obs.tsv -k total_counts,pct_counts_mt -o resid.mtx",
    }],
    json_result_schema_doc: None,
};

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_debug_assert() {
        Cli::command().debug_assert();
    }
}
