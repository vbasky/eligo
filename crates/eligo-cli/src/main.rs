//! Command-line interface for eligo.
//!
//! Two subcommands:
//!
//! - `generate` — best-of-N: make N images and keep the one the judge rates
//!   best. Backend (the "artist") and scorer (the "judge") are the mock by
//!   default; build with `--features sd` / `--features clip` and pass model
//!   paths for real Stable Diffusion and CLIP.
//! - `similar` — rank images in a directory by visual similarity to a query
//!   image, using CLIP embeddings (requires `--features clip`).

use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use eligo::mock::MockBackend;
use eligo::{Backend, GenerateConfig, Image, RerollPolicy, Scorer, best_of_n};

/// Best-of-N image generation that selects the best candidate by measurable reward.
#[derive(Debug, Parser)]
#[command(name = "eligo", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Generate N candidates and keep the best (best-of-N selection).
    Generate(GenerateArgs),
    /// Rank images in a directory by visual similarity to a query image.
    #[cfg(feature = "clip")]
    Similar(SimilarArgs),
}

#[derive(Debug, Args)]
struct GenerateArgs {
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

    /// Write the winning image here (`.png` with the clip/sd feature, else PPM).
    #[arg(long)]
    out: Option<PathBuf>,

    /// With --out, also save every candidate (suffixed `_0`, `_1`, … and `_best`).
    #[arg(long)]
    save_all: bool,

    /// Blend a no-reference quality score (sharpness/contrast) into the reward,
    /// in `[0,1]`. 0 = prompt-match only; higher also favours crisp images.
    #[arg(long, default_value_t = 0.0)]
    quality_weight: f32,

    /// Stable Diffusion ONNX export directory (requires the `sd` feature).
    #[cfg(feature = "sd")]
    #[arg(long, requires = "sd_tokenizer")]
    sd_model_dir: Option<PathBuf>,

    /// CLIP `tokenizer.json` for the SD backend (requires the `sd` feature).
    #[cfg(feature = "sd")]
    #[arg(long, requires = "sd_model_dir")]
    sd_tokenizer: Option<PathBuf>,

    /// Denoising steps for the SD backend (more = slower, better).
    #[cfg(feature = "sd")]
    #[arg(long, default_value_t = 20)]
    steps: usize,

    /// Classifier-free guidance scale for the SD backend.
    #[cfg(feature = "sd")]
    #[arg(long, default_value_t = 7.5)]
    guidance: f32,

    /// CLIP `model.onnx` for the real scorer (requires the `clip` feature).
    #[cfg(feature = "clip")]
    #[arg(long, requires = "clip_tokenizer")]
    clip_model: Option<PathBuf>,

    /// CLIP `tokenizer.json` for the real scorer (requires the `clip` feature).
    #[cfg(feature = "clip")]
    #[arg(long, requires = "clip_model")]
    clip_tokenizer: Option<PathBuf>,
}

/// Arguments for the `similar` subcommand.
#[cfg(feature = "clip")]
#[derive(Debug, Args)]
struct SimilarArgs {
    /// Query image: results are ranked by similarity to this.
    query: PathBuf,

    /// Directory of candidate images to search.
    dir: PathBuf,

    /// How many results to show.
    #[arg(short = 'k', long, default_value_t = 5)]
    top: usize,

    /// CLIP `model.onnx`.
    #[arg(long)]
    clip_model: PathBuf,

    /// CLIP `tokenizer.json`.
    #[arg(long)]
    clip_tokenizer: PathBuf,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    match Cli::parse().command {
        Command::Generate(args) => run_generate(args),
        #[cfg(feature = "clip")]
        Command::Similar(args) => run_similar(args),
    }
}

fn run_generate(args: GenerateArgs) -> Result<()> {
    let cfg = GenerateConfig::new(&args.prompt)
        .with_candidates(args.candidates)
        .with_seed(args.seed)
        .with_reroll(if args.reroll_worst {
            RerollPolicy::RerollWorstOnce
        } else {
            RerollPolicy::None
        });

    let (backend, backend_name) = select_backend(&args)?;
    let (mut scorer, mut scorer_name) = select_scorer(&args)?;
    if args.quality_weight > 0.0 {
        scorer = Box::new(eligo::QualityWeighted::new(scorer, args.quality_weight));
        scorer_name = "blended with quality";
    }
    eprintln!("backend: {backend_name}  |  scorer: {scorer_name}");

    let selection =
        best_of_n(backend.as_ref(), scorer.as_ref(), &cfg).context("selection loop failed")?;

    println!("scored {} candidate(s):", selection.all.len());
    for (i, c) in selection.all.iter().enumerate() {
        let marker = if i == selection.best_index { "★" } else { " " };
        println!("  {marker} #{i}  seed={:<6} score={:.4}", c.seed, c.score);
    }
    let best = selection.best();
    println!("chosen: seed={} score={:.4}", best.seed, best.score);

    if let Some(path) = args.out {
        if args.save_all {
            for (i, c) in selection.all.iter().enumerate() {
                let p = numbered(&path, i, i == selection.best_index);
                write_image(&c.image, &p).with_context(|| format!("writing {}", p.display()))?;
                println!("wrote {}", p.display());
            }
        } else {
            write_image(&best.image, &path)
                .with_context(|| format!("writing {}", path.display()))?;
            println!("wrote {}", path.display());
        }
    }

    Ok(())
}

/// Rank images in a directory by CLIP similarity to the query image.
#[cfg(feature = "clip")]
fn run_similar(args: SimilarArgs) -> Result<()> {
    use eligo::{ClipEmbedder, cosine_similarity};

    let embedder = ClipEmbedder::from_files(&args.clip_model, &args.clip_tokenizer)
        .context("loading CLIP embedder")?;

    let query = Image::open(&args.query)
        .with_context(|| format!("opening query {}", args.query.display()))?;
    let query_vec = embedder.embed_image(&query).context("embedding query")?;

    let mut scored: Vec<(f32, PathBuf)> = Vec::new();
    for entry in std::fs::read_dir(&args.dir)
        .with_context(|| format!("reading directory {}", args.dir.display()))?
    {
        let path = entry?.path();
        if !is_image_file(&path) {
            continue;
        }
        let Ok(img) = Image::open(&path) else {
            eprintln!("skipping unreadable {}", path.display());
            continue;
        };
        let vec = embedder.embed_image(&img)?;
        scored.push((cosine_similarity(&query_vec, &vec), path));
    }

    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    println!("most similar to {}:", args.query.display());
    for (sim, path) in scored.iter().take(args.top) {
        println!("  {sim:.4}  {}", path.display());
    }
    if scored.is_empty() {
        println!("  (no images found in {})", args.dir.display());
    }
    Ok(())
}

/// True if the path has a raster-image extension we can decode.
#[cfg(feature = "clip")]
fn is_image_file(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).as_deref(),
        Some("png" | "jpg" | "jpeg" | "webp" | "bmp" | "gif" | "tif" | "tiff")
    )
}

/// Pick the backend: real Stable Diffusion when built with `--features sd` and
/// given model paths, otherwise the deterministic mock.
fn select_backend(args: &GenerateArgs) -> Result<(Box<dyn Backend>, &'static str)> {
    #[cfg(feature = "sd")]
    if let (Some(dir), Some(tokenizer)) = (&args.sd_model_dir, &args.sd_tokenizer) {
        let backend = eligo::SdBackend::from_dir(dir, tokenizer, args.steps, args.guidance)
            .context("loading SD backend")?;
        return Ok((Box::new(backend), "Stable Diffusion (ONNX Runtime)"));
    }

    let _ = args;
    Ok((Box::new(MockBackend::default()), "mock (deterministic)"))
}

/// Pick the scorer: the real CLIP reward when built with `--features clip` and
/// given model paths, otherwise the deterministic mock.
fn select_scorer(args: &GenerateArgs) -> Result<(Box<dyn Scorer>, &'static str)> {
    #[cfg(feature = "clip")]
    if let (Some(model), Some(tokenizer)) = (&args.clip_model, &args.clip_tokenizer) {
        let scorer =
            eligo::ClipScorer::from_files(model, tokenizer).context("loading CLIP scorer")?;
        return Ok((Box::new(scorer), "CLIP (ONNX Runtime)"));
    }

    let _ = args;
    Ok((Box::new(eligo::mock::MockScorer), "mock (deterministic)"))
}

/// Insert `_<index>` (and `_best`) before the file extension of `path`.
fn numbered(path: &Path, index: usize, is_best: bool) -> PathBuf {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("png");
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
    let tag = if is_best { "_best" } else { "" };
    path.with_file_name(format!("{stem}_{index}{tag}.{ext}"))
}

/// Write the winning image: PNG when the extension is `.png` and an image
/// encoder is compiled in (clip/sd feature), otherwise binary PPM.
fn write_image(image: &Image, path: &Path) -> Result<()> {
    #[cfg(any(feature = "clip", feature = "sd"))]
    if path.extension().and_then(|e| e.to_str()) == Some("png") {
        image.save_png(path)?;
        return Ok(());
    }
    write_ppm(image, path)
}

/// Write an [`Image`] as a binary (P6) PPM — always available, no dependencies.
fn write_ppm(image: &Image, path: &Path) -> Result<()> {
    let mut f = std::fs::File::create(path)?;
    write!(f, "P6\n{} {}\n255\n", image.width, image.height)?;
    f.write_all(&image.rgb)?;
    Ok(())
}
