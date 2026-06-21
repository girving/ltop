//! Spawn N child processes each burning target fractions of CPU and GPU.
//! Useful for eyeballing ltop's tree rendering, row colours, child elision,
//! and (on macOS) GPU utilisation display — without waiting for a real workload.
//!
//! The parent process runs as a worker alongside the children, so the tree
//! always has a visible root. Ctrl-C hits the whole process group at once.

use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

// ─── CLI ───────────────────────────────────────────────────────────────────

struct Config {
    /// Number of child worker processes to spawn.
    children: usize,
    /// Fraction of one CPU core to burn, 0.0..=1.0.
    cpu: f64,
    /// Fraction of GPU to burn, 0.0..=1.0.  macOS / Apple Silicon only;
    /// silently ignored on other platforms.
    gpu: f64,
    /// Internal flag: this process is a child worker, not the coordinator.
    child: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config { children: 0, cpu: 0.5, gpu: 0.0, child: false }
    }
}

fn usage() -> &'static str {
    "Usage: spin [OPTIONS]

Spawn workers that burn target fractions of CPU and GPU load.
Useful for testing ltop's tree rendering, colours, and child elision.
The parent process also runs as a worker. Ctrl-C exits the whole group.

Options:
  -n, --children <N>   Number of child workers to spawn [default: 0]
      --cpu <FRAC>     CPU busy fraction per worker, 0..1 [default: 0.5]
      --gpu <FRAC>     GPU busy fraction per worker, 0..1 [default: 0.0]
                       (macOS / Apple Silicon; silently no-op elsewhere)
  -h, --help           Show this help
"
}

fn parse_frac(flag: &str, s: &str) -> f64 {
    let v: f64 = s.parse().unwrap_or_else(|_| {
        eprintln!("spin: {flag} requires a number in [0, 1], got {s:?}");
        std::process::exit(2);
    });
    if !(0.0..=1.0).contains(&v) {
        eprintln!("spin: {flag} must be in [0, 1], got {v}");
        std::process::exit(2);
    }
    v
}

fn parse_args() -> Config {
    let mut cfg = Config::default();
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "-h" | "--help" => {
                print!("{}", usage());
                std::process::exit(0);
            }
            "--child" => cfg.child = true,
            "-n" | "--children" => {
                i += 1;
                cfg.children = raw.get(i).and_then(|s| s.parse().ok()).unwrap_or_else(|| {
                    eprintln!("spin: --children requires a non-negative integer");
                    std::process::exit(2);
                });
            }
            "--cpu" => {
                i += 1;
                let s = raw.get(i).map(String::as_str).unwrap_or("");
                cfg.cpu = parse_frac("--cpu", s);
            }
            "--gpu" => {
                i += 1;
                let s = raw.get(i).map(String::as_str).unwrap_or("");
                cfg.gpu = parse_frac("--gpu", s);
            }
            other => {
                eprintln!("spin: unknown argument {other:?}  (try --help)");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    cfg
}

// ─── CPU worker ────────────────────────────────────────────────────────────

fn cpu_worker(frac: f64) -> ! {
    if frac <= 0.0 {
        // Keep the process alive without burning CPU.
        loop {
            thread::sleep(Duration::from_secs(3600));
        }
    }
    let period = Duration::from_millis(100);
    let busy = period.mul_f64(frac.clamp(0.0, 1.0));
    let idle = period.saturating_sub(busy);
    loop {
        let start = Instant::now();
        let mut x: u64 = 0x9E3779B97F4A7C15;
        while start.elapsed() < busy {
            for _ in 0..4096 {
                x = x.wrapping_mul(0x5851F42D4C957F2D).wrapping_add(1);
            }
            std::hint::black_box(x);
        }
        if !idle.is_zero() {
            thread::sleep(idle);
        }
    }
}

// ─── GPU worker (macOS / Metal) ────────────────────────────────────────────

#[cfg(target_os = "macos")]
fn gpu_worker(frac: f64) {
    use metal::{CompileOptions, Device, MTLResourceOptions};

    if frac <= 0.0 {
        return;
    }

    let Some(device) = Device::system_default() else {
        eprintln!("spin: no Metal device available; --gpu has no effect");
        return;
    };

    // A simple compute kernel: XOR-shift loop writing to a sink so the
    // compiler cannot eliminate the work.
    let shader = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void spin_kernel(
            device uint* sink [[buffer(0)]],
            uint  id          [[thread_position_in_grid]]
        ) {
            uint x = id + 1u;
            for (uint i = 0u; i < 4096u; i++) {
                x ^= x << 13u;
                x ^= x >> 17u;
                x ^= x << 5u;
            }
            // Scatter writes prevent the loop being optimised away.
            sink[id & 255u] = x;
        }
    "#;

    let library = match device.new_library_with_source(shader, &CompileOptions::new()) {
        Ok(l) => l,
        Err(e) => { eprintln!("spin: Metal compile error: {e}"); return; }
    };
    let function = match library.get_function("spin_kernel", None) {
        Ok(f) => f,
        Err(e) => { eprintln!("spin: get_function: {e}"); return; }
    };
    let pipeline = match device.new_compute_pipeline_state_with_function(&function) {
        Ok(p) => p,
        Err(e) => { eprintln!("spin: compute pipeline: {e}"); return; }
    };
    let queue = device.new_command_queue();
    // 256-uint sink; Apple Silicon unified memory — accessible from both CPU and GPU.
    let sink = device.new_buffer(256 * 4, MTLResourceOptions::StorageModeShared);

    // Threadgroup size: 64 threads (well under the 1024 max; divides evenly
    // into most dispatch counts we'll use).
    const TG: u64 = 64;

    // Calibrate: find a threadgroup count that yields ~5 ms per dispatch,
    // giving fine-enough granularity to hit the target fraction over a 100 ms
    // period without excessive overhead.
    let mut n_tg: u64 = 512; // initial guess: 512 × 64 = 32 768 threads
    {
        let t = Instant::now();
        dispatch(&queue, &pipeline, &sink, n_tg, TG);
        let ms = t.elapsed().as_secs_f64() * 1_000.0;
        if ms > 0.01 {
            let target_ms = 5.0_f64;
            n_tg = ((n_tg as f64) * target_ms / ms).max(1.0) as u64;
        }
        // Warm-up pass with calibrated count.
        dispatch(&queue, &pipeline, &sink, n_tg, TG);
    }

    let period = Duration::from_millis(100);
    let busy = period.mul_f64(frac.clamp(0.0, 1.0));

    loop {
        let t0 = Instant::now();
        // Busy phase: keep dispatching until we've used `frac` of the period.
        while t0.elapsed() < busy {
            dispatch(&queue, &pipeline, &sink, n_tg, TG);
        }
        // Idle phase: sleep out the remainder.
        let elapsed = t0.elapsed();
        if elapsed < period {
            thread::sleep(period - elapsed);
        }
    }
}

#[cfg(target_os = "macos")]
fn dispatch(
    queue: &metal::CommandQueueRef,
    pipeline: &metal::ComputePipelineStateRef,
    sink: &metal::BufferRef,
    n_threadgroups: u64,
    threadgroup_size: u64,
) {
    use metal::MTLSize;
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(sink), 0);
    enc.dispatch_thread_groups(
        MTLSize { width: n_threadgroups, height: 1, depth: 1 },
        MTLSize { width: threadgroup_size, height: 1, depth: 1 },
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

#[cfg(not(target_os = "macos"))]
fn gpu_worker(_frac: f64) {
    // GPU spinning not supported on this platform; --gpu is silently ignored.
}

// ─── Entry point ───────────────────────────────────────────────────────────

fn main() {
    let cfg = parse_args();

    if cfg.child {
        // Internal mode: just run the workers, no spawning.
        let gpu = cfg.gpu;
        if gpu > 0.0 {
            thread::spawn(move || gpu_worker(gpu));
        }
        cpu_worker(cfg.cpu);
    }

    // Coordinator: spawn N child workers then run as a worker ourselves.
    let exe = std::env::current_exe().expect("current_exe");
    for _ in 0..cfg.children {
        Command::new(&exe)
            .args([
                "--child",
                "--cpu", &cfg.cpu.to_string(),
                "--gpu", &cfg.gpu.to_string(),
            ])
            .spawn()
            .expect("spawn child");
    }

    eprintln!(
        "spin: {} children + parent, cpu {:.0}%, gpu {:.0}%; Ctrl-C to exit",
        cfg.children,
        cfg.cpu * 100.0,
        cfg.gpu * 100.0,
    );

    let gpu = cfg.gpu;
    if gpu > 0.0 {
        thread::spawn(move || gpu_worker(gpu));
    }
    cpu_worker(cfg.cpu);
}
