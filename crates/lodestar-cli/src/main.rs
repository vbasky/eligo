//! Command-line interface for lodestar.
//!
//! Runs the best-of-N selection loop and reports the chosen candidate. The
//! default scorer is the deterministic mock; build with `--features clip` and
//! pass `--clip-model` + `--clip-tokenizer` to select with a real CLIP reward.
//! `--out` writes the winning image as a binary PPM (no image-encoding
//! dependency yet).

use std::io::Write as _;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use lodestar::mock::MockBackend;
use lodestar::{GenerateConfig, Image, RerollPolicy, Scorer, best_of_n};

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

    /// Path to a CLIP `model.onnx` (requires the `clip` feature). With
    /// `--clip-tokenizer`, scores candidates with the real CLIP reward.
    #[cfg(feature = "clip")]
    #[arg(long, requires = "clip_tokenizer")]
    clip_model: Option<PathBuf>,

    /// Path to the CLIP `tokenizer.json` (requires the `clip` feature).
    #[cfg(feature = "clip")]
    #[arg(long, requires = "clip_model")]
    clip_tokenizer: Option<PathBuf>,
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
    let (scorer, scorer_name) = select_scorer(&cli)?;
    eprintln!("scorer: {scorer_name}");

    let selection = best_of_n(&backend, scorer.as_ref(), &cfg).context("selection loop failed")?;

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

/// Pick the scorer: the real CLIP reward when built with `--features clip` and
/// given model paths, otherwise the deterministic mock.
fn select_scorer(cli: &Cli) -> Result<(Box<dyn Scorer>, &'static str)> {
    #[cfg(feature = "clip")]
    if let (Some(model), Some(tokenizer)) = (&cli.clip_model, &cli.clip_tokenizer) {
        let scorer =
            lodestar::ClipScorer::from_files(model, tokenizer).context("loading CLIP scorer")?;
        return Ok((Box::new(scorer), "CLIP (ONNX Runtime)"));
    }

    let _ = cli;
    Ok((Box::new(lodestar::mock::MockScorer), "mock (deterministic)"))
}

/// Write an [`Image`] as a binary (P6) PPM — keeps the CLI dependency-free until
/// a real image encoder is wired in.
fn write_ppm(image: &Image, path: &std::path::Path) -> Result<()> {
    let mut f = std::fs::File::create(path)?;
    write!(f, "P6\n{} {}\n255\n", image.width, image.height)?;
    f.write_all(&image.rgb)?;
    Ok(())
}
