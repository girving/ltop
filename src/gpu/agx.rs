//! Apple Silicon GPU backend — talks to the AGX (Apple GPU) driver via the
//! IOKit registry. Per-process CPU% on Mac comes from `proc_pidinfo`; per-
//! process GPU% is not exposed there or in `proc_pid_rusage`. The kernel
//! does track it (xnu's `task.task_gpu_ns`) but never publishes it via the
//! BSD personality.
//!
//! What it *does* publish: each `IOUserClient` an app opens against the
//! AGX device shows up in IORegistry as an `AGXDeviceUserClient` child of
//! `AGXAccelerator`. Two properties on each are load-bearing:
//!
//!   * `IOUserClientCreator` — string `"pid <NNN>, <name>"`.
//!   * `AppUsage` — array of dicts `{API, lastSubmittedTime,
//!     accumulatedGPUTime}`. accumulatedGPUTime is in nanoseconds (the
//!     `mach_absolute_time` clock — but we don't divide by anything; the
//!     ratio (Δns / Δwall_ns) is what gives a percent regardless).
//!
//! Per-tick we walk those children, sum `accumulatedGPUTime` per PID, and
//! diff against last tick to get a percentage. Sum across PIDs ≈ system-
//! wide GPU%. Total memory comes from the AGX device's
//! `PerformanceStatistics` dict (`Alloc system memory`); active from
//! `In use system memory`.
//!
//! Cost: one `IOServiceGetMatchingServices` + per-AGX child iterator + ~30
//! `IORegistryEntryCreateCFProperties` calls per tick on a typical Mac.
//! Measured ~1 ms per tick on M2; comparable to what the Linux RM path
//! costs for its three RM_CONTROLs.

use core::ffi::c_void;
use core::ptr;

use super::GpuProc;
use crate::arena::{FSpan, FVec, Frame};

// ── Public API ───────────────────────────────────────────────────────────────

/// AGX driver state. Cross-tick: a Map of `pid -> last accumulatedGPUTime`
/// snapshot, kept in the init scope so we can compute deltas.
///
/// `total_mib` is system unified memory (Apple Silicon has no separate
/// VRAM); fetched once at init from `hw.memsize`. `n_gpus` is 1 if AGX is
/// present, 0 otherwise (no-AGX hosts get an empty State and downstream
/// ops no-op).
///
/// The cross-tick `(pid, accumulatedGPUTime_ns)` snapshot used to compute
/// per-process deltas lives outside this struct: it's an `FSpan` owned by
/// `main::run`'s init scope and rotated via `init.replace2` alongside the
/// CPU prev. `populate` takes it by reference and returns the new prev as
/// part of its result.
///
/// `_brand` carries the init scope's `'id` purely for cross-platform API
/// symmetry with `gpu::rm::State<'id>`, which holds real arena
/// allocations. State today is just two POD integers — the brand isn't
/// preventing any current bug; it's there so callers can write
/// `gpu::State<'_>` uniformly and so that a future arena-bearing field
/// would already be invariance-protected (see [`crate::arena::Brand`]).
pub struct State<'id> {
    n_gpus: u32,
    total_mib: u64,
    _brand: crate::arena::Brand<'id>,
}

impl<'id> State<'id> {
    pub fn has_gpu(&self) -> bool { self.n_gpus > 0 }
}

/// Cap on per-PID entries we read from IOKit per tick. AGXDeviceUserClient
/// children typically number ~50 on a busy desktop; 256 is comfortable
/// headroom for the per-tick scratch FVec. The cross-tick prev that
/// survives is exact-sized via `freeze_map` so this cap doesn't bound
/// memory long-term.
const MAX_TRACKED_PIDS: u32 = 256;

/// Open the AGX device (if present) and read total unified memory once.
/// On hosts without AGX (Intel Macs, headless VMs) returns a State with
/// `n_gpus=0` — every downstream call no-ops.
pub fn init<'a>(frame: &mut Frame<'a>) -> State<'a> {
    let n_gpus = count_agx_devices() as u32;
    let total_mib = if n_gpus > 0 { hw_memsize_mib() } else { 0 };
    State { n_gpus, total_mib, _brand: frame.brand() }
}

/// Match `IOServiceMatching("AGXAccelerator")` and count how many entries
/// the iterator yields (typically 1; future Macs with multiple GPUs would
/// yield more — we'd still treat them as one logical accelerator since
/// per-process accounting aggregates across them anyway).
fn count_agx_devices() -> usize {
    let master = crate::mac_sys::io_master_port();
    if master == 0 { return 0; }
    let iter = crate::mac_sys::io_service_get_matching_services_bin(
        master, &crate::mac_sys::AGX_MATCHING_BLOB);
    if iter == 0 { return 0; }
    let mut n = 0usize;
    loop {
        let svc = crate::mac_sys::io_iterator_next(iter);
        if svc == 0 { break; }
        crate::mac_sys::mach_port_deallocate(svc);
        n += 1;
    }
    crate::mac_sys::mach_port_deallocate(iter);
    n
}

/// `sysctl({CTL_HW, HW_MEMSIZE})` → total physical RAM in MiB. Apple
/// Silicon's unified memory means this is the GPU's address space too.
fn hw_memsize_mib() -> u64 {
    // CTL_HW=6, HW_MEMSIZE=24. Fixed since 10.0.
    let mut mib = [6i32, 24i32];
    let mut bytes: u64 = 0;
    let mut len: usize = core::mem::size_of::<u64>();
    let r = unsafe {
        crate::syscall::sysctl(
            mib.as_mut_ptr(), 2,
            &mut bytes as *mut _ as *mut c_void, &mut len,
            ptr::null_mut(), 0,
        )
    };
    if r != 0 { return 0; }
    bytes >> 20
}

/// Per-tick collection. Walks every AGXDeviceUserClient under each
/// AGXAccelerator, sums `accumulatedGPUTime` per PID, diffs against the
/// previous tick's snapshot to produce a percentage. Returns:
///
///   - sorted+deduped `(pid, GpuProc)` FSpan (per-process percentages,
///     transient — caller renders from this and discards).
///   - sorted `(pid, accumulatedGPUTime_ns)` FSpan to be installed as the
///     new `gpu/agx/prev` slot. Exact-sized to the deduped count, no
///     fixed cap.
///   - `(n_gpus, total_util, used_mib, total_mib)`.
///
/// `prev` is the previous tick's prev FSpan. The bytes stay alive for
/// the lifetime of this call (caller passes `&prev`); we read them via a
/// `Map` view. We do NOT modify `prev` — the caller rotates via
/// `init.replace2`, replacing it with our returned new_prev FSpan.
///
/// Build order matters: we allocate `new_prev` BEFORE `entries` so that
/// when the outer `replace2` memmoves things into place, new_prev lands
/// at gpu_prev's slot and the cpu prev (allocated even later, by main's
/// loop) lands above. Anything between them — `entries`, transient
/// scratch — gets squeezed out by replace2's two memmoves.
pub fn populate<'id>(
    state: &State<'_>,
    prev: &FSpan<'id, (u32, u64)>,
    frame: &mut Frame<'id>,
    dt_ns: u64,
) -> (FSpan<'id, (u32, GpuProc)>, FSpan<'id, (u32, u64)>, (u32, u32, u64, u64)) {
    if state.n_gpus == 0 || dt_ns == 0 {
        return (frame.empty(), frame.empty(), (0, 0, 0, state.total_mib));
    }

    // Phase 1: read IOKit, build the new prev. `compact` over the IOKit
    // walk reclaims the MAX_TRACKED_PIDS-capacity scratch (`gpu/agx/now`)
    // and shifts the deduped result down to the bottom of the compact.
    //
    // `total_used_mib` and `iokit_ok` cross the closure boundary via
    // `&mut` captures.
    let mut total_used_mib = 0u64;
    let mut iokit_ok = false;
    let new_prev: FSpan<'id, (u32, u64)> = frame.compact("gpu/agx/prev", |f| {
        let mut now: FVec<'_, (u32, u64)> =
            f.vec("gpu/agx/now", MAX_TRACKED_PIDS);

        // OSSerializeBinary checkpoint table — reused across every
        // property blob we parse this tick. 512 entries × 4 bytes =
        // 2 KB; the decoder doubles its checkpoint stride whenever a
        // blob has more objects than slots, so any blob size parses
        // (the AGXAccelerator root is ~2000+ objects, dominated by
        // IOReportLegend).
        const OSBIN_CKPTS: usize = 512;
        let mut osbin_ckpts = f.zeros::<u32>("osbinary/ckpts", OSBIN_CKPTS);

        let master = crate::mac_sys::io_master_port();
        if master == 0 { return f.empty(); }
        let iter = crate::mac_sys::io_service_get_matching_services_bin(
            master, &crate::mac_sys::AGX_MATCHING_BLOB);
        if iter == 0 { return f.empty(); }

        loop {
            let agx = crate::mac_sys::io_iterator_next(iter);
            if agx == 0 { break; }

            // Read AGXAccelerator's own properties → "PerformanceStatistics
            // ['In use system memory']". The OOL buffer drops at the end
            // of this if-let, calling vm_deallocate to release the
            // kernel pages.
            if let Some(buf) = crate::mac_sys::io_registry_entry_get_properties_bin(agx) {
                if let Some(bin) = crate::osbinary::OsBinary::parse(
                    buf.as_bytes(), &mut osbin_ckpts[..])
                {
                    if let Some(used) = read_agx_used_mib_blob(&bin) {
                        total_used_mib += used;
                    }
                }
            }

            // Walk children (AGXDeviceUserClient instances), one per
            // Metal-using process.
            let child_iter = crate::mac_sys::io_registry_entry_get_child_iterator(
                agx, b"IOService\0");
            if child_iter != 0 {
                loop {
                    let child = crate::mac_sys::io_iterator_next(child_iter);
                    if child == 0 { break; }
                    if let Some(buf) = crate::mac_sys::io_registry_entry_get_properties_bin(child) {
                        if let Some(bin) = crate::osbinary::OsBinary::parse(
                            buf.as_bytes(), &mut osbin_ckpts[..])
                        {
                            if let Some((pid, ns)) = read_user_client_blob(&bin) {
                                let _ = now.push((pid, ns));
                            }
                        }
                    }
                    crate::mac_sys::mach_port_deallocate(child);
                }
                crate::mac_sys::mach_port_deallocate(child_iter);
            }
            crate::mac_sys::mach_port_deallocate(agx);
        }
        crate::mac_sys::mach_port_deallocate(iter);
        iokit_ok = true;

        // Sort + merge per-PID (multiple userclients per process). The
        // FVec is truncated to the deduped length, then converted to an
        // FSpan; `compact` shifts it down to its entry offset, dropping
        // the trailing capacity.
        let _ = crate::map::freeze_map(&mut now, |a, b| { *a += *b; });
        now.into_fspan()
    });

    if !iokit_ok {
        return (frame.empty(), new_prev, (0, 0, 0, state.total_mib));
    }

    // Phase 2: diff old prev against new_prev to compute entries. Both
    // FSpans are sorted by pid (old prev was sorted last tick by this
    // same `freeze_map`; new_prev was just sorted), so we can iterate
    // new_prev linearly and binary-search the old.
    let prev_map = crate::map::Map::new(prev);
    let now_map = crate::map::Map::new(&new_prev);

    let entries: FSpan<'id, (u32, GpuProc)> = frame.compact("gpu/agx/procs", |f| {
        let mut buf: FVec<'_, (u32, GpuProc)> =
            f.vec("gpu/agx/procs/raw", MAX_TRACKED_PIDS);
        for &(pid, now_ns) in now_map.iter() {
            let prev_ns = prev_map.get(&pid).copied().unwrap_or(0);
            let pct = compute_pct(prev_ns, now_ns, dt_ns);
            if pct == 0 { continue; }
            let _ = buf.push((pid, GpuProc { sm_pct: pct }));
        }
        buf.into_fspan()
    });

    // total_util = sum of per-PID percentages, capped at 100. Mirrors
    // the Linux RM path which also produces 0..100 across all engines.
    let total_util: u32 = entries.iter().map(|(_, g)| g.sm_pct).sum::<u32>().min(100);
    (entries, new_prev, (state.n_gpus, total_util, total_used_mib, state.total_mib))
}

/// Per-process GPU percent from cumulative-time snapshots.
///
/// `prev_ns` and `now_ns` are `accumulatedGPUTime` snapshots one tick
/// apart (in nanoseconds, mach_absolute_time clock); `dt_ns` is the
/// wall-clock interval between them. Returns
/// `(now - prev) / dt * 100`, clamped to `[0, 100]`. Returns 0 when:
///
///   - `prev_ns == 0` (PID first observed this tick — wait one tick
///     before reporting),
///   - `now_ns <= prev_ns` (counter reset / no progress),
///   - `dt_ns == 0` (would otherwise divide by zero — caller's bug,
///     but we don't assume).
///
/// Pulled out of `populate` so the math has a unit test that doesn't
/// need a real GPU workload (see the tests module).
#[inline]
fn compute_pct(prev_ns: u64, now_ns: u64, dt_ns: u64) -> u32 {
    if prev_ns == 0 || now_ns <= prev_ns || dt_ns == 0 {
        return 0;
    }
    let delta_ns = now_ns - prev_ns;
    // delta * 100 overflows u64 past ~184 quintillion ns ≈ 5800 years.
    // saturating_mul caps at u64::MAX, then division and the clamp
    // bring us back into [0, 100].
    (delta_ns.saturating_mul(100) / dt_ns).min(100) as u32
}

/// libIOKit-free counterparts of the read_ helpers below: operate on a
/// pre-parsed OSSerializeBinary blob (kernel-emitted via
/// `io_registry_entry_get_properties_bin`) instead of going through
/// CFDictionary / CFNumber / CFArray. Logically identical to the
/// libIOKit path; structurally we walk the binary tree directly via
/// `osbinary::OsBinary`.

/// Same semantics as `read_agx_used_mib`, but takes a pre-parsed
/// property blob instead of an IOKit registry entry. Looks up
/// `PerformanceStatistics["In use system memory"]`, a CFNumber giving
/// bytes, and returns it rounded down to MiB.
pub(crate) fn read_agx_used_mib_blob(bin: &crate::osbinary::OsBinary<'_>) -> Option<u64> {
    let root = bin.root()?;
    let perf = bin.find_dict(root, b"PerformanceStatistics")?;
    let used = bin.find_dict(perf, b"In use system memory")?;
    let bytes = bin.as_number(used)?;
    Some(bytes >> 20)
}

/// Same semantics as `read_user_client`, but takes a pre-parsed
/// AGXDeviceUserClient property blob. Extracts the creator's PID and
/// the sum of `AppUsage[*].accumulatedGPUTime` (in nanoseconds).
pub(crate) fn read_user_client_blob(bin: &crate::osbinary::OsBinary<'_>) -> Option<(u32, u64)> {
    let root = bin.root()?;
    let creator = bin.find_dict(root, b"IOUserClientCreator")?;
    let creator_bytes = bin.as_string(creator)?;
    let pid = parse_creator_pid_bytes(creator_bytes)?;

    let mut total_ns = 0u64;
    if let Some(usage) = bin.find_dict(root, b"AppUsage") {
        bin.for_each_array(usage, |item| {
            if let Some(acc) = bin.find_dict(item, b"accumulatedGPUTime") {
                if let Some(ns) = bin.as_number(acc) {
                    total_ns = total_ns.saturating_add(ns);
                }
            }
        });
    }
    Some((pid, total_ns))
}

/// Parses the bytes of `IOUserClientCreator` (emitted by xnu's
/// `IOUserClient.cpp` as `pid %d, %s`) and returns NNN. The trailing
/// `", <name>"` is not validated — we stop at the first non-digit so
/// the field's exact width doesn't matter.
fn parse_creator_pid_bytes(bytes: &[u8]) -> Option<u32> {
    if !bytes.starts_with(b"pid ") { return None; }
    let mut n: u32 = 0;
    let mut any = false;
    for &b in &bytes[4..] {
        if !(b'0'..=b'9').contains(&b) { break; }
        n = n.checked_mul(10)?.checked_add((b - b'0') as u32)?;
        any = true;
    }
    if any { Some(n) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── OSSerializeBinary blob builders for synthetic-data tests ─────
    //
    // CI's macos-14-arm64 runners don't have AGX hardware, so we
    // exercise `read_agx_used_mib_blob` and `read_user_client_blob`
    // against hand-built blobs that mirror the real IORegistry
    // shapes. The helpers below match xnu's `OSSerializeBinary` writer.

    const MAGIC: [u8; 4] = [0xd3, 0x00, 0x00, 0x00];
    const TY_DICT: u8    = 0x01;
    const TY_ARRAY: u8   = 0x02;
    const TY_NUMBER: u8  = 0x04;
    const TY_SYMBOL: u8  = 0x08;
    const TY_STRING: u8  = 0x09;

    fn emit_tag(out: &mut Vec<u8>, ty: u8, operand: u32, eoc: bool) {
        let raw = ((ty as u32) << 24) | (operand & 0x00ff_ffff)
                | (if eoc { 0x8000_0000 } else { 0 });
        out.extend_from_slice(&raw.to_le_bytes());
    }
    fn emit_symbol(out: &mut Vec<u8>, s: &[u8], eoc: bool) {
        emit_tag(out, TY_SYMBOL, s.len() as u32 + 1, eoc);
        out.extend_from_slice(s);
        out.push(0);
        while out.len() % 4 != 0 { out.push(0); }
    }
    fn emit_string(out: &mut Vec<u8>, s: &[u8], eoc: bool) {
        emit_tag(out, TY_STRING, s.len() as u32, eoc);
        out.extend_from_slice(s);
        while out.len() % 4 != 0 { out.push(0); }
    }
    fn emit_number(out: &mut Vec<u8>, value: u64, eoc: bool) {
        emit_tag(out, TY_NUMBER, 64, eoc);
        out.extend_from_slice(&(value as u32).to_le_bytes());
        out.extend_from_slice(&((value >> 32) as u32).to_le_bytes());
    }

    /// Hand-build an AGXAccelerator-shaped dict and verify that
    /// `read_agx_used_mib_blob` extracts the expected MiB value.
    /// Tests the full chain: nested-dict lookup, Number decode, and
    /// the bytes-to-MiB shift.
    #[test]
    fn read_agx_used_mib_blob_synthetic() {
        // { "PerformanceStatistics" : { "In use system memory" : 12451840u64 } }
        // 12451840 bytes >> 20 = 11 MiB.
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC);
        emit_tag(&mut b, TY_DICT, 1, false);          // root
        emit_symbol(&mut b, b"PerformanceStatistics", false);
        emit_tag(&mut b, TY_DICT, 1, true);           // EOC on the value
        emit_symbol(&mut b, b"In use system memory", false);
        emit_number(&mut b, 12_451_840, true);

        let mut ckpts = vec![0u32; 32];
        let bin = crate::osbinary::OsBinary::parse(&b, &mut ckpts).expect("parse");
        assert_eq!(read_agx_used_mib_blob(&bin), Some(11));
    }

    /// Hand-build an AGXDeviceUserClient-shaped dict and verify that
    /// `read_user_client_blob` extracts the creator PID and sums
    /// `AppUsage[*].accumulatedGPUTime` correctly.
    #[test]
    fn read_user_client_blob_synthetic() {
        // { "IOUserClientCreator" : "pid 619, WindowServer",
        //   "AppUsage" : [ { "accumulatedGPUTime" : 1e9 },
        //                  { "accumulatedGPUTime" : 5e8 } ] }
        // → (619, 1_500_000_000).
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC);
        emit_tag(&mut b, TY_DICT, 2, false);
        emit_symbol(&mut b, b"IOUserClientCreator", false);
        emit_string(&mut b, b"pid 619, WindowServer", false);
        emit_symbol(&mut b, b"AppUsage", false);
        emit_tag(&mut b, TY_ARRAY, 2, true);
        emit_tag(&mut b, TY_DICT, 1, false);
        emit_symbol(&mut b, b"accumulatedGPUTime", false);
        emit_number(&mut b, 1_000_000_000, true);
        emit_tag(&mut b, TY_DICT, 1, true);
        emit_symbol(&mut b, b"accumulatedGPUTime", false);
        emit_number(&mut b, 500_000_000, true);

        let mut ckpts = vec![0u32; 64];
        let bin = crate::osbinary::OsBinary::parse(&b, &mut ckpts).expect("parse");
        assert_eq!(read_user_client_blob(&bin), Some((619, 1_500_000_000)));
    }

    /// `read_user_client_blob` must return `None` if `IOUserClientCreator`
    /// is missing — that's a userclient that didn't come from a
    /// process (e.g., kernel-internal helper), and we shouldn't claim
    /// a fake PID for it.
    #[test]
    fn read_user_client_blob_missing_creator_returns_none() {
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC);
        emit_tag(&mut b, TY_DICT, 1, false);
        emit_symbol(&mut b, b"AppUsage", false);
        emit_tag(&mut b, TY_ARRAY, 0, true);
        let mut ckpts = vec![0u32; 16];
        let bin = crate::osbinary::OsBinary::parse(&b, &mut ckpts).expect("parse");
        assert_eq!(read_user_client_blob(&bin), None);
    }

    /// `read_user_client_blob` must return `Some((pid, 0))` if AppUsage
    /// is missing entirely — that's a Metal client that's been opened
    /// but hasn't dispatched any work yet.
    #[test]
    fn read_user_client_blob_missing_appusage_returns_zero_ns() {
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC);
        emit_tag(&mut b, TY_DICT, 1, false);
        emit_symbol(&mut b, b"IOUserClientCreator", false);
        emit_string(&mut b, b"pid 1234, app", true);
        let mut ckpts = vec![0u32; 16];
        let bin = crate::osbinary::OsBinary::parse(&b, &mut ckpts).expect("parse");
        assert_eq!(read_user_client_blob(&bin), Some((1234, 0)));
    }

    /// Pure unit test for the percentage math. No IOKit, no Metal — covers
    /// the contract every interesting tick depends on: first-tick zero,
    /// counter reset, divide-by-zero guard, saturation, and the typical
    /// ~50% case. If this passes, the only remaining failure mode for the
    /// per-process GPU% column is bad input from IOKit (covered by the
    /// `agx_tracks_spin` ignored e2e test).
    #[test]
    fn compute_pct_cases() {
        let s = 1_000_000_000u64; // one second in ns
        // First-tick: prev unknown → 0 (avoids reporting whole-program
        // accumulated time on first observation).
        assert_eq!(compute_pct(0, 5 * s, s), 0);
        // No-progress: counter unchanged → 0.
        assert_eq!(compute_pct(s, s, s), 0);
        // Counter reset (driver restart): now_ns < prev_ns → 0.
        assert_eq!(compute_pct(2 * s, s, s), 0);
        // dt_ns == 0 guard (caller bug; don't crash).
        assert_eq!(compute_pct(s, 2 * s, 0), 0);
        // 50% over 1s: 0.5s of GPU time spent.
        assert_eq!(compute_pct(s, s + s / 2, s), 50);
        // 100% over 1s.
        assert_eq!(compute_pct(s, 2 * s, s), 100);
        // 200% (parallel queues) clamps to 100.
        assert_eq!(compute_pct(s, 3 * s, s), 100);
        // Tiny activity rounds down to 0; avoids dispatching mostly-idle
        // PIDs into the visibility set.
        assert_eq!(compute_pct(s, s + 1_000, s), 0);
        // Saturation safety: huge delta shouldn't wrap. (prev=1 to dodge
        // the prev==0 first-tick guard above.)
        assert_eq!(compute_pct(1, u64::MAX, s), 100);
    }

    /// Pure unit test for `parse_creator_pid_bytes`. The IOKit format
    /// `"pid <N>, <name>"` is built by xnu's `IOUserClient.cpp`; we
    /// don't lex past the first non-digit so the trailing `, name` is
    /// whatever.
    #[test]
    fn parse_creator_pid_cases() {
        for &(input, expect) in &[
            (&b"pid 619, WindowServer\0"[..], Some(619u32)),
            (&b"pid 1, launchd\0"[..],         Some(1)),
            (&b"pid 4294967295, x\0"[..],      Some(u32::MAX)),
            // Overflow past u32::MAX → checked_mul fails → None.
            (&b"pid 4294967296, x\0"[..],      None),
            (&b"pid 0, scheduler\0"[..],       Some(0)),  // unrealistic but well-formed
            // Doesn't start with "pid ".
            (&b"PID 619, X\0"[..],             None),
            (&b"\0"[..],                       None),
            (&b"pid\0"[..],                    None),  // missing space
            // No digits after "pid ".
            (&b"pid abc, X\0"[..],             None),
            (&b"pid , X\0"[..],                None),
        ] {
            let got = parse_creator_pid_bytes(input);
            assert_eq!(got, expect, "input={:?}", core::str::from_utf8(input).unwrap_or("?"));
        }
    }

    /// Smoke test: AGX init populates n_gpus=1 on Apple Silicon, total_mib
    /// matches `sysctl hw.memsize`, and populate() runs without crashing
    /// even on the first tick (where every PID is "new" and reports 0%).
    #[test]
    fn agx_init_and_first_tick() {
        let _g = crate::arena::test_lock();
        crate::arena::scope(|frame| {
            let state = init(frame);
            // GitHub Actions macOS runners are virtualized — no AGXAccelerator
            // in the IORegistry. Skip the GPU-dependent assertions there;
            // init() returning a valid empty State is itself the no-GPU smoke.
            if !state.has_gpu() {
                eprintln!("no AGXAccelerator found; skipping GPU smoke test");
                return;
            }
            assert!(state.total_mib > 1024, "implausible total_mib: {}", state.total_mib);
            let prev: FSpan<(u32, u64)> = frame.empty();
            let (entries, _new_prev, totals) = populate(&state, &prev, frame, 1_000_000_000);
            // First tick: prev is empty so no entries qualify yet.
            assert_eq!(entries.len(), 0);
            let (n, _util, used, total) = totals;
            assert_eq!(n, 1);
            assert_eq!(total, state.total_mib);
            // Used memory comes straight from PerformanceStatistics, so
            // it must be nonzero even on the first tick — the kernel
            // always has some MiB in use. This is the tripwire for the
            // parse-failure-reads-as-zero bug class (a too-small parse
            // buffer once zeroed this permanently).
            assert!(used > 0, "GPU used-memory parsed as 0 MiB");
        });
    }

    /// Two-tick test: spawn `target/release/spin --gpu 0.3`, populate
    /// once to seed prev, sleep 1s, populate again and assert spin's PID
    /// shows up at ≥ 5%. Skipped under `cargo test` because the test
    /// binary doesn't depend on the spin workspace member (we'd have to
    /// add a dev-dependency); manual test run is:
    ///   cargo build --release -p spin
    ///   cargo test --release -- --ignored agx_tracks_spin
    #[test]
    #[ignore = "spawns spin; run after `cargo build --release -p spin`"]
    fn agx_tracks_spin() {
        use std::process::Command;
        use std::thread::sleep;
        use std::time::Duration;

        let _g = crate::arena::test_lock();
        // Workspace-relative path to the spin binary built in release mode.
        let spin_bin = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target/release/spin");
        let mut child = Command::new(&spin_bin)
            .args(["--cpu", "0.0", "--gpu", "0.3"])
            .spawn()
            .unwrap_or_else(|e| panic!("spawn {}: {e}", spin_bin.display()));
        // Give Metal time to initialise + first dispatch to land.
        sleep(Duration::from_millis(800));
        crate::arena::scope(|frame| {
            let state = init(frame);
            // Tick 1: seed prev. Allocate prev as an empty FSpan; populate
            // returns the next-tick prev FSpan, which we feed into tick 2.
            // Inside the scope we can't rotate via init.replace2 (that's
            // the run loop's job), so we just hold the FSpan by value.
            let prev0: FSpan<(u32, u64)> = frame.empty();
            let (_e0, prev1, _t0) = populate(&state, &prev0, frame, 1_000_000_000);
            sleep(Duration::from_secs(1));
            let (entries, _prev2, _t1) = populate(&state, &prev1, frame, 1_000_000_000);
            let pid = child.id();
            let entry = entries.iter().find(|(p, _)| *p == pid);
            child.kill().ok();
            let _ = child.wait();
            let entry = entry.unwrap_or_else(|| {
                let summary: Vec<_> =
                    entries.iter().map(|(p, g)| (*p, g.sm_pct)).collect();
                panic!("spin pid {pid} not in {summary:?}");
            });
            assert!(entry.1.sm_pct >= 5, "spin pct too low: {}", entry.1.sm_pct);
        });
    }
}
