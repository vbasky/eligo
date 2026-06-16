//! Command-line interface for lodestar.
//!
//! Runs the best-of-N selection loop and reports the chosen candidate. Until
//! the `candle` backend lands, it uses the deterministic mock backend/scorer so
//! the loop is exercisable end-to-end; `--out` writes the winning image as a
//! binary PPM (no image-encoding dependency yet).

use std::io::Write as _;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use lodestar::mock::{MockBackend, MockScorer};
use lodestar::{GenerateConfig, Image, RerollPolicy, best_of_n};

/// Best-of-N image generation that selects the best candidate by measurable reward.
#[derive(Debug, Parser)]
#[command(name = "lodestar", version, about)]
struct Cli {
    /// Text prompt to generate from.
    prompt: String,

    /// Number of candidates to generate, then choose the best of.
    #[arg(short = 'n', long, default_value_t = 4)]
    candidates: u32,

    /// Base RNG seed (candidate i uses seed + i).
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Regenerate the single worst candidate once and keep the better result.
    #[arg(long)]
    reroll_worst: bool,

    /// Write the winning image to this path as a binary PPM.
    #[arg(long)]
    out: Option<PathBuf>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    let cfg = GenerateConfig::new(&cli.prompt)
        .with_candidates(cli.candidates)
        .with_seed(cli.seed)
        .with_reroll(if cli.reroll_worst {
            RerollPolicy::RerollWorstOnce
        } else {
            RerollPolicy::None
        });

    let backend = MockBackend::default();
    let scorer = MockScorer;
    let selection = best_of_n(&backend, &scorer, &cfg).context("selection loop failed")?;

    println!("scored {} candidate(s):", selection.all.len());
    for (i, c) in selection.all.iter().enumerate() {
        let marker = if i == selection.best_index { "★" } else { " " };
        println!("  {marker} #{i}  seed={:<6} score={:.4}", c.seed, c.score);
    }
    let best = selection.best();
    println!("chosen: seed={} score={:.4}", best.seed, best.score);

    if let Some(path) = cli.out {
        write_ppm(&best.image, &path).with_context(|| format!("writing {}", path.display()))?;
        println!("wrote {}", path.display());
    }

    Ok(())
}

/// Write an [`Image`] as a binary (P6) PPM — keeps the CLI dependency-free until
/// a real image encoder is wired in.
fn write_ppm(image: &Image, path: &std::path::Path) -> Result<()> {
    let mut f = std::fs::File::create(path)?;
    write!(f, "P6\n{} {}\n255\n", image.width, image.height)?;
    f.write_all(&image.rgb)?;
    Ok(())
}
