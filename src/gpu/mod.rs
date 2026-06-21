//! GPU (Nvidia) info. Production uses `rm` — direct ioctls to the NVIDIA
//! kernel driver via /dev/nvidiactl + /dev/nvidia<N>, bypassing libnvidia-ml
//! and the libcuda user-mode driver it lazy-loads. NVML is kept as a reference
//! implementation in test builds only (see `#[cfg(all(test, has_nvml))]`),
//! where it's used to validate the RM backend.

#[cfg(not(target_os = "macos"))]
use core::num::NonZeroU32;

use crate::arena::{FSpan, Frame};
use crate::bytes::{f1_wide, u32d};
use crate::map::Map;
use crate::twrite::{Put, TinyWriter};

/// Element type of the cross-tick GPU prev FSpan rotated by main's run
/// loop via `init.replace2`. On macOS we need `(pid, accumulatedGPUTime_ns)`
/// pairs to diff into a per-process percentage; on Linux NVIDIA reports
/// SM% directly via a driver-owned sliding window, so the prev slot is a
/// ZST — the rotation arithmetic in `replace2` folds away at compile time.
#[cfg(target_os = "macos")]
pub type GpuPrevElem = (u32, u64);
#[cfg(not(target_os = "macos"))]
pub type GpuPrevElem = ();

#[cfg(target_os = "linux")]
pub mod rm;

#[cfg(target_os = "macos")]
pub mod agx;

#[cfg(all(test, has_nvml))]
mod nvml;

#[cfg(target_os = "linux")]
pub use rm::State;

#[cfg(target_os = "macos")]
pub use agx::State;

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub struct State<'id> { _brand: crate::arena::Brand<'id> }
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
impl<'id> State<'id> {
    pub fn has_gpu(&self) -> bool { false }
}

/// Initialise driver state — calls the platform backend's init on
/// Linux/macOS, returns an empty state elsewhere. On no-GPU hosts the
/// returned State is still valid; downstream ops no-op.
#[cfg(target_os = "linux")]
pub fn init<'a>(frame: &mut Frame<'a>) -> State<'a> { rm::init(frame) }
#[cfg(target_os = "macos")]
pub fn init<'a>(frame: &mut Frame<'a>) -> State<'a> { agx::init(frame) }
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub fn init<'a>(frame: &mut Frame<'a>) -> State<'a> {
    State { _brand: frame.brand() }
}

// ── Public types ─────────────────────────────────────────────────────────────

/// Per-process GPU usage stored on ProcInfo.
///
/// On Linux/NVIDIA the niche on `mem_mib: NonZeroU32` lets
/// `Option<GpuUsage>` be 8 bytes. On macOS the field is gone (Apple
/// Silicon's unified memory has no separate VRAM to track per-process —
/// it's already accounted for in RSS), and `Option<GpuUsage>` is 8 bytes
/// anyway from the discriminant + u32 layout.
pub struct GpuUsage {
    pub sm_pct: u32,
    #[cfg(not(target_os = "macos"))]
    pub mem_mib: NonZeroU32,
}

#[derive(Clone, Copy, Default)]
pub struct GpuProc {
    pub sm_pct: u32,
    /// Per-process GPU memory, in MiB. Linux/NVIDIA only.
    #[cfg(not(target_os = "macos"))]
    pub mem_mib: u32,
}

/// Per-tick GPU snapshot. The `procs` map is frozen over a tick-scope
/// FVec allocated inside `collect`; its lifetime matches the frame's
/// arena brand.
pub struct GpuInfo<'a> {
    pub procs: Map<'a, GpuProc>,
    pub total_util: u32,      // summed GPU utilisation % across devices
    pub total_used_mib: u64,  // summed used memory in MiB
    pub total_mib: u64,       // summed total memory in MiB
    pub n_gpus: u32,
}

impl GpuInfo<'_> {
    pub fn mem_frac(&self) -> f64 {
        if self.total_mib > 0 { self.total_used_mib as f64 / self.total_mib as f64 } else { 0.0 }
    }

    /// Per-process GPU usage, if the process has measurable GPU activity.
    ///
    /// "Measurable" is platform-specific:
    ///   - Linux/NVIDIA: non-zero GPU memory (mem_mib > 0). The driver's
    ///     SM% is sampled at irregular intervals and most processes get
    ///     0% even when they're actually using the GPU; memory is the
    ///     reliable signal.
    ///   - macOS/AGX: non-zero GPU% (sm_pct > 0). Apple Silicon's unified
    ///     memory means there's no separate VRAM to track per-process —
    ///     `accumulatedGPUTime` delta is the only signal.
    #[cfg(target_os = "macos")]
    pub fn usage(&self, pid: u32) -> Option<GpuUsage> {
        let gp = self.procs.get(&pid)?;
        (gp.sm_pct > 0).then_some(GpuUsage { sm_pct: gp.sm_pct })
    }
    #[cfg(not(target_os = "macos"))]
    pub fn usage(&self, pid: u32) -> Option<GpuUsage> {
        let gp = self.procs.get(&pid)?;
        NonZeroU32::new(gp.mem_mib).map(|mem_mib| GpuUsage { sm_pct: gp.sm_pct, mem_mib })
    }
}

impl Put for GpuInfo<'_> {
    fn put<W: TinyWriter + ?Sized>(&self, w: &mut W) {
        if self.n_gpus == 0 { w.put_byte(b'?'); return; }
        let gib = |mib: u64| mib as f64 / 1024.0;
        // Linux/NVIDIA: "<util>%  <used>G / <total>G" — fractional VRAM
        // is the load-bearing signal.
        // macOS/AGX: "<util>%  <used>G active" — `total` would be
        // `hw.memsize` (system RAM, not GPU-specific) and the fraction
        // doesn't tell you how full the GPU's memory is, so we drop it
        // and label the absolute number to make the meaning explicit.
        #[cfg(not(target_os = "macos"))]
        crate::twrite!(w,
            u32d(self.total_util / self.n_gpus), "%  ",
            f1_wide(gib(self.total_used_mib), 0), "G / ",
            f1_wide(gib(self.total_mib), 0), "G");
        #[cfg(target_os = "macos")]
        crate::twrite!(w,
            u32d(self.total_util / self.n_gpus), "%  ",
            f1_wide(gib(self.total_used_mib), 0), "G active");
    }
}

// ── Per-tick collection ─────────────────────────────────────────────────────

/// Query the driver and produce this tick's GpuInfo plus the new
/// cross-tick prev. `rm::populate` (Linux) or `agx::populate` (macOS)
/// does the heavy lifting; we wrap entries in a `Map` and attach totals.
///
/// `gpu_prev` is the previous tick's prev FSpan. macOS uses it to diff
/// cumulative `accumulatedGPUTime` counters into a per-tick percentage
/// and returns an exact-sized fresh prev for the caller to rotate via
/// `init.replace2`. Linux's prev is `FSpan<()>` — zero bytes, zero work.
///
/// `dt_ns` is the wall-clock delta since the previous tick. Linux
/// ignores it (the driver maintains its own sliding window).
///
/// On no-GPU hosts both returned FSpans are empty.
pub fn collect<'a>(
    state: &State<'_>,
    gpu_prev: &FSpan<'a, GpuPrevElem>,
    frame: &mut Frame<'a>,
    dt_ns: u64,
) -> (GpuInfo<'a>, FSpan<'a, GpuPrevElem>) {
    #[cfg(target_os = "linux")]
    let (entries, new_prev, totals) = {
        let _ = (dt_ns, gpu_prev);
        let (e, t) = rm::populate(state, frame);
        (e, frame.empty::<GpuPrevElem>(), t)
    };
    #[cfg(target_os = "macos")]
    let (entries, new_prev, totals) = agx::populate(state, gpu_prev, frame, dt_ns);
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    let (entries, new_prev, totals): (FSpan<'a, (u32, GpuProc)>, FSpan<'a, GpuPrevElem>, (u32, u32, u64, u64)) = {
        let _ = (state, dt_ns, gpu_prev);
        (frame.empty(), frame.empty(), (0, 0, 0, 0))
    };
    let (n_gpus, total_util, total_used_mib, total_mib) = totals;
    let procs = Map::new(entries.leak());
    (GpuInfo { procs, n_gpus, total_util, total_used_mib, total_mib }, new_prev)
}
