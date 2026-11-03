//! CUDA chunked benchmark and strict comparison to an immutable baseline.
//! perf <record|check> <model> <reference-dir> <input.wav>...

use eyre::{ensure, Result};
use reuse_rs::{audio, GpuDevice, Offline};
use std::{path::PathBuf, time::Instant};

fn pcm16(x: f32) -> i32 {
    (x as f64 * 32768.0).round().clamp(-32768.0, 32767.0) as i32
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("fixture") {
        ensure!(
            args.len() >= 5,
            "perf fixture <out.wav> <seconds> <input.wav>..."
        );
        ensure!(
            !std::path::Path::new(&args[2]).exists(),
            "fixture already exists"
        );
        let seconds: usize = args[3].parse()?;
        ensure!((1..=3600).contains(&seconds), "invalid fixture duration");
        let mut corpus = Vec::new();
        for input in &args[4..] {
            let (samples, sr) = audio::load_wav(input)?;
            ensure!(
                sr == 44100 && !samples.is_empty(),
                "expected nonempty 44100 Hz input"
            );
            corpus.extend_from_slice(&samples);
        }
        let wanted = seconds * 44100;
        let samples: Vec<f32> = corpus.iter().copied().cycle().take(wanted).collect();
        audio::save_wav(&args[2], &samples, 44100)?;
        println!("FIXTURE samples={wanted} seconds={seconds}");
        return Ok(());
    }
    ensure!(
        args.len() >= 5,
        "perf <record|check> <model> <reference-dir> <input.wav>..."
    );
    let record = match args[1].as_str() {
        "record" => true,
        "check" => false,
        _ => return Err(eyre::eyre!("expected record or check")),
    };
    let refs = PathBuf::from(&args[3]);
    if record {
        std::fs::create_dir_all(&refs)?;
    }
    let startup = Instant::now();
    let model = Offline::load_gpu(&args[2], GpuDevice::Cuda(0))?;
    println!("load_seconds={:.6}", startup.elapsed().as_secs_f64());
    let (warm, sr) = audio::load_wav(&args[4])?;
    ensure!(sr == 44100, "benchmark requires 44100 Hz");
    let warm_start = Instant::now();
    model.enhance(&warm[..warm.len().min(sr as usize * 5)], sr)?;
    println!("warmup_seconds={:.6}", warm_start.elapsed().as_secs_f64());
    let mut elapsed = 0.0;
    let mut duration = 0.0;
    let mut all_ok = true;
    for name in &args[4..] {
        let input = PathBuf::from(name);
        let reference = refs.join(
            input
                .file_name()
                .ok_or_else(|| eyre::eyre!("invalid input"))?,
        );
        if record {
            ensure!(
                !reference.exists(),
                "reference already exists: {}",
                reference.display()
            );
        }
        let (samples, sr) = audio::load_wav(name)?;
        ensure!(
            sr == 44100 && !samples.is_empty(),
            "expected nonempty 44100 Hz input"
        );
        let seconds = samples.len() as f64 / sr as f64;
        let start = Instant::now();
        let enhanced = model.enhance_chunked(&samples, sr, 5.0, 0.5)?;
        let took = start.elapsed().as_secs_f64();
        ensure!(
            enhanced.len() == samples.len() && enhanced.iter().all(|v| v.is_finite()),
            "invalid output"
        );
        elapsed += took;
        duration += seconds;
        if record {
            audio::save_wav(reference.to_str().unwrap(), &enhanced, sr)?;
            println!(
                "RECORD file={} seconds={seconds:.6} elapsed={took:.6} rtf={:.6}",
                input.display(),
                took / seconds
            );
        } else {
            let (expected, ref_sr) = audio::load_wav(reference.to_str().unwrap())?;
            ensure!(
                ref_sr == sr && expected.len() == enhanced.len(),
                "reference length/rate mismatch"
            );
            let mut max_lsb = 0.0f64;
            let mut max_pcm = 0;
            let mut bad = 0usize;
            let mut sum_sq = 0.0f64;
            for (&a, &b) in enhanced.iter().zip(&expected) {
                ensure!(b.is_finite(), "non-finite reference");
                let lsb = (a as f64 - b as f64).abs() * 32768.0;
                let pcm = (pcm16(a) - pcm16(b)).abs();
                max_lsb = max_lsb.max(lsb);
                max_pcm = max_pcm.max(pcm);
                sum_sq += lsb * lsb;
                if lsb > 1.0 || pcm > 1 {
                    bad += 1;
                }
            }
            all_ok &= bad == 0;
            println!("CHECK file={} seconds={seconds:.6} elapsed={took:.6} rtf={:.6} max_lsb={max_lsb:.9} max_pcm={max_pcm} rms_lsb={:.9} bad_samples={bad}", input.display(), took/seconds, (sum_sq/enhanced.len() as f64).sqrt());
        }
    }
    println!("TOTAL audio_seconds={duration:.6} processing_seconds={elapsed:.6} rtf={:.6} accuracy_ok={all_ok}", elapsed/duration);
    ensure!(all_ok, "output exceeds one 16-bit LSB per sample");
    Ok(())
}
