#![cfg_attr(not(test), no_std)]
#![cfg_attr(not(test), no_main)]
// In test builds the `extern "C" fn main` entry point (and thus `run` and
// everything it transitively calls) is gated out, leaving most of the
// binary's code unreachable. Silence the resulting dead-code noise so real
// warnings stay visible.
#![cfg_attr(test, allow(dead_code))]

//! Minimal customisable process monitor.
//! Works on Linux (via /proc) and macOS (via sysctl + libproc).
//! Designed for narrow terminals (~69 cols).

extern crate alloc;

// arena exposes a few API methods (remaining, is_empty, etc.) that aren't
// used by the binary but are exercised by the module's own tests. Silence
// the dead_code warnings for the module as a whole.
#[allow(dead_code)]
mod arena;
mod bytes;
mod gpu;
mod map;
mod platform;
mod sort;
#[macro_use]
mod twrite;
// `src/start.rs` is our `_start` replacement. Active only on the
// libc-free production build (the `cargo ltop` / `cargo stack`
// aliases, which set `--cfg=libc_free` in rustflags). Other Linux
// builds — plain `cargo build --release`, used by `cargo test` —
// go through glibc's `crt1.o` → `__libc_start_main` → our `main`
// below.
#[cfg(all(not(test), libc_free))]
mod start;
#[cfg(target_os = "linux")]
mod sandbox;
mod syscall;
mod tty;
mod zeroable;
#[cfg(target_os = "macos")]
mod mac_sys;
#[cfg(target_os = "macos")]
mod osbinary;

use crate::arena::{FBuilder, FNest, FReserve, FSpan, FSmallStr, FVec, Frame};
// FBox backs the Linux /proc-reader scratch buffers (dirent, stat_raw,
// cmdline, meminfo). The macOS collect path sizes its argv buffer at
// runtime with an FVec instead (KERN_PROCARGS2 blobs vary per pid), so
// FBox is Linux-only.
#[cfg(target_os = "linux")]
use crate::arena::FBox;
#[cfg(any(target_os = "linux", not(test)))]
use crate::arena::FStr;
use crate::bytes::{f1_wide, ieq, pad_left, pad_right, pad_zero, u32d};
// `repeat`, `Instant` are only used by `fn run` (gated
// `#[cfg(not(test))]`); gating the imports too avoids unused-import
// warnings in the test build.
#[cfg(not(test))]
use crate::bytes::repeat;
use crate::gpu::GpuUsage;
use crate::map::{Map, Set};
#[cfg(not(test))]
use crate::platform::Instant;
#[cfg(not(test))]
use core::time::Duration;
use crate::twrite::{Put, TinyWriter};

const CPU_THRESH: f32 = 5.0;     // show if CPU% >= this
const MEM_MB_THRESH: u64 = 128;  // show if RSS >= this MB
const GPU_THRESH: u32 = 5;       // show if GPU% >= this (Apple Silicon only;
                                 // NVIDIA path always passes through whatever
                                 // pids the driver reports)

/// Number of data columns the GPU section contributes when has_gpu:
/// Linux/NVIDIA shows GPU% + GMEM; macOS/AGX shows just GPU%.
#[cfg(target_os = "macos")] const GPU_DATA_COLS: usize = 1;
#[cfg(not(target_os = "macos"))] const GPU_DATA_COLS: usize = 2;
/// Blank cell for a row with `p.gpu = None`. `GPU_DATA_COLS * 7` spaces
/// (one separator + 6 char width per column). Bound at compile time so
/// emit_proc_row writes a single literal rather than computing widths.
#[cfg(target_os = "macos")]
const GPU_BLANK: &[u8] = b"       ";   // 7
#[cfg(not(target_os = "macos"))]
const GPU_BLANK: &[u8] = b"              "; // 14
/// Header text for the GPU column(s), appended after `RSS` when has_gpu.
#[cfg(target_os = "macos")]
const GPU_HEADER: &[u8] = b"   GPU%";
#[cfg(not(target_os = "macos"))]
const GPU_HEADER: &[u8] = b"   GPU%   GMEM";

// ── Terminal ─────────────────────────────────────────────────────────────────

/// Number of online logical CPUs, or 1 as a last-resort fallback.
#[cfg(target_os = "linux")]
fn num_cpus_online() -> f64 {
    // `sched_getaffinity(0, size, set)` fills `set` with the affinity
    // bitmask of the current process. popcount = effective nproc.
    // 1024 CPUs (16 × u64) is well beyond anything we'll encounter.
    let mut set = [0u64; 16];
    let ret = syscall::sched_getaffinity_self(&mut set);
    if ret > 0 {
        let n: u32 = set.iter().map(|w| w.count_ones()).sum();
        if n > 0 { n as f64 } else { 1.0 }
    } else {
        1.0
    }
}

/// Number of online logical CPUs, or 1 as a last-resort fallback.
#[cfg(not(target_os = "linux"))]
fn num_cpus_online() -> f64 {
    mac_sys::hw_activecpu() as f64
}

// ── Tree layout ──────────────────────────────────────────────────────────────

/// Position of a node in the display tree: depth and the "is_last" bit for
/// each level from root down to this node. Bit i of `mask` is 1 iff the
/// ancestor (or self) at level i+1 is the last child at its level.
/// Renders as the box-drawing prefix on demand; no heap allocation.
/// (mask is u32 — supports up to 32 levels of nesting, far beyond any real tree.)
#[derive(Clone, Copy)]
struct Indent {
    depth: u8,
    mask: u32,
}

impl Indent {
    fn is_root(self) -> bool { self.depth == 0 }
    /// Rendered depth is clamped to the mask's 32 levels: deeper
    /// chains (nested containers/CI wrappers) draw a depth-32 prefix
    /// instead of wrapping the mask shift (panic in debug, stale bits
    /// in release). `width` and `put` clamp identically so the cmd
    /// column stays aligned.
    fn render_depth(self) -> u32 { (self.depth as u32).min(32) }
    fn width(self) -> usize { self.render_depth() as usize * 2 }
}

impl Put for Indent {
    fn put<W: TinyWriter + ?Sized>(&self, w: &mut W) {
        let depth = self.render_depth();
        if depth == 0 { return; }
        for i in 0..depth - 1 {
            let is_last = (self.mask >> i) & 1 != 0;
            w.put_bytes(if is_last { b"  " } else { "│ ".as_bytes() });
        }
        let is_last = (self.mask >> (depth - 1)) & 1 != 0;
        w.put_bytes(if is_last { "└─".as_bytes() } else { "├─".as_bytes() });
    }
}

/// Build a pid → index-into-procs `Map` for the current tick, internally
/// allocating from `frame`. Used by both `tree_layout` and `filter_with_
/// children` to walk ppid → parent-index chains. PIDs are unique this
/// tick so the freeze-time merge closure never fires.
///
/// Returns `Map<'a, u32>` with the arena's lifetime; the underlying bytes
/// stay in the arena until scope exit.
fn build_pid_to_idx<'a>(
    frame: &Frame<'a>,
    procs: &[ProcInfo<'_>],
) -> Map<'a, u32> {
    let mut buf: FVec<'a, (u32, u32)> = frame.vec("pid_to_idx", procs.len() as u32);
    buf.extend(procs.iter().enumerate().map(|(i, p)| (p.pid, i as u32)));
    map::freeze_map_owned(buf, |_, _| ())
}

/// Children adjacency: `children.bucket(p)` lists the proc positions
/// whose parent is at position `p`. Stored as a single CSR-style
/// [`FNest`] — one offsets array and one flat data array, no per-parent
/// FVec headers.
///
/// Parent-of-i is looked up via `pid_to_idx.get(&procs[i].ppid)`; procs
/// whose ppid isn't tracked are tree roots and contribute no edge.
fn build_children<'a>(
    frame: &mut Frame<'a>,
    procs: &[ProcInfo<'_>],
    pid_to_idx: &Map<'_, u32>,
) -> FNest<'a, u32> {
    frame.nest("children", procs.len() as u32, || {
        procs.iter().enumerate().filter_map(|(i, p)| {
            pid_to_idx.get(&p.ppid).map(|&pa| (pa, i as u32))
        })
    })
}

/// Flatten process list into tree order with compact position info.
/// Assumes `procs` is pre-sorted by pid. Children naturally come out
/// pid-ordered because we push in procs-order.
fn tree_layout<'a>(
    frame: &mut Frame<'a>,
    procs: &[ProcInfo<'_>],
) -> FSpan<'a, (Indent, u32)> {
    // Wrap the whole build in `compact` so scratch allocations (pid_to_idx,
    // counts inside build_children, children outer+inner, and tree_walk's
    // DFS stack) are reclaimed before we return — only the result FSpan
    // persists.
    frame.compact("tree_layout", |frame| {
        let n = procs.len() as u32;
        let pid_to_idx = build_pid_to_idx(frame, procs);
        let children = build_children(frame, procs, &pid_to_idx);

        // Iterative DFS. The recursive version was unbounded in depth
        // (proc tree can chain 20+ levels: systemd → sshd → bash → sh →
        // shell wrappers → tool → child) and those frames crossed our
        // stack page budget. Explicit work-stack sized at procs.len()
        // — at any one time the stack holds at most siblings-from-every-
        // ancestor pending, which is bounded by n.
        //
        // WorkItem = (idx, depth, parent_mask, is_last). u8 depth, u32
        // for both idx and mask; tuple packs to 12 B.
        let mut stack: FVec<(u32, u8, u32, bool)> = frame.vec("tree/stack", n);
        let mut result = frame.reserve::<(Indent, u32)>("tree", n);
        for (i, p) in procs.iter().enumerate() {
            if pid_to_idx.get(&p.ppid).is_none() {
                tree_walk(&children, i as u32, &mut result, &mut stack);
            }
        }
        result.freeze()
    })
}

/// Iterative pre-order DFS from `root_idx`. Emits `(Indent, idx)` into
/// `out` in the same order a recursive visit would. `stack` is an
/// arena-backed scratch buffer reused across roots; it's cleared on
/// entry so the caller doesn't have to.
fn tree_walk(
    children: &FNest<'_, u32>,
    root_idx: u32,
    out: &mut FReserve<'_, '_, (Indent, u32)>,
    stack: &mut FVec<'_, (u32, u8, u32, bool)>,
) {
    // is_last=true for the root — matches the recursive signature's
    // initial call. mask accumulates which ancestors were last-child.
    stack.clear();
    let _ = stack.push((root_idx, 0, 0, true));
    while let Some((idx, depth, parent_mask, is_last)) = stack.pop() {
        let mask = if depth > 0 && depth <= 32 && is_last {
            parent_mask | (1u32 << (depth - 1))
        } else {
            parent_mask
        };
        // Capacity is >= procs.len() (sized in tree_layout), so push
        // never overflows.
        let _ = out.push((Indent { depth, mask }, idx));
        // Push children in reverse order: LIFO pop then yields them in
        // forward order, matching the recursive visit.
        let kids = children.bucket(idx as usize);
        let last_i = kids.len().saturating_sub(1);
        for (i, &kid) in kids.iter().enumerate().rev() {
            let _ = stack.push((kid, depth.saturating_add(1), mask, i == last_i));
        }
    }
}

/// CPU percent over the wall interval (100% = one core), using the low
/// 32 bits of cumulative CPU ticks stored in `prev_times` (zero if the
/// pid is new). The caller pushes `(pid, total as u32)` into the next-
/// tick buffer and tracks the common sample instant at the tick level.
///
/// `dt` is the wall-clock interval between this tick and the prev
/// tick's sample. Short or zero `dt` (first tick, clock glitch) →
/// return 0 rather than divide.
///
/// `ticks_per_sec` is clock_ticks on Linux, 1e9 on macOS (counter is ns).
///
/// Truncating cumulative ticks to 31 bits is sound because we only
/// ever compute the delta, and `wrapping_sub` reconstructs it as long
/// as the real delta fits in `2^(32 + SHIFT - 1)` counter units —
/// see the `Packed::SHIFT` doc for the per-platform ceiling (~10M
/// cores on Linux, ~137 cores on macOS, both at 2 s steady-state).
///
/// `age_secs` is the process-identity check: prev matches by pid only,
/// and on a fork-heavy host a pid can be reused between ticks — the
/// newcomer's small counter minus the dead process's large one wraps
/// to a near-2^31 delta (CPU% in the millions, forced visible,
/// inherited hysteresis). A process younger than the sample interval
/// cannot have been the one observed last tick, which catches every
/// reuse: the old owner was alive at the prev sample, so its
/// replacement was necessarily born inside the interval.
fn compute_cpu(
    prev_entry: Option<Packed>,
    age_secs: u32, total: u64, ticks_per_sec: f64, dt: f64,
) -> f32 {
    if dt <= 0.01 || (age_secs as f64) < dt { return 0.0; }
    prev_entry.map(|p| {
        // Delta in the shifted domain (u31 wrapping), then
        // `<< SHIFT` to recover the real counter delta.
        let new_bits = ((total >> Packed::SHIFT) as u32) & Packed::CPU_MASK;
        let delta_bits = new_bits.wrapping_sub(p.cpu_bits()) & Packed::CPU_MASK;
        let delta = (delta_bits as f64) * (1u64 << Packed::SHIFT) as f64;
        (delta / ticks_per_sec / dt * 100.0) as f32
    }).unwrap_or(0.0)
}

fn emit_proc_row<W: TinyWriter + ?Sized>(
    out: &mut W,
    procs: &[ProcInfo<'_>],
    tree: &[(Indent, u32)],
    ti: usize,
    width: usize,
    gpu_total_mib: u64,
    has_gpu: bool,
) {
    let (indent, idx) = tree[ti];
    let p = &procs[idx as usize];
    let (r, g, b) = row_color(p.cpu, p.rss_kib, &p.gpu, gpu_total_mib);
    // Layout: pid(7) + N data cols(1+6 each) + 2 spaces + cmd.
    //   N = 3 base (cpu/rss/age) + 1 (GPU%) on macOS or + 2 (GPU%/GMEM) on
    //   Linux when has_gpu. macOS has no per-process GPU memory column —
    //   Apple Silicon's unified memory is already counted in RSS.
    let gpu_cols = if has_gpu { GPU_DATA_COLS } else { 0 };
    let n_data_cols = 3 + gpu_cols;
    let stats_w = 7 + n_data_cols * 7;
    let cmd_w = width.saturating_sub(stats_w + 2 + indent.width());
    let cmd = truncate(p.display.as_slice(), cmd_w);
    twrite!(out, FgOpen(r, g, b),
        pad_left(p.pid, 7), " ", f1_wide(p.cpu as f64, 6), " ", format_mem(p.rss_kib));
    if has_gpu {
        match &p.gpu {
            #[cfg(target_os = "macos")]
            Some(gpu) => twrite!(out, " ", pad_right(gpu.sm_pct, 6)),
            #[cfg(not(target_os = "macos"))]
            // GPU mem is reported in MiB (NonZeroU32); << 10 converts to KiB.
            // mem_mib < 4 Ti ensures the shift doesn't overflow u32 on any
            // current GPU.
            Some(gpu) => twrite!(out, " ", pad_right(gpu.sm_pct, 6), " ",
                                 format_mem(gpu.mem_mib.get() << 10)),
            // GPU_BLANK is GPU_DATA_COLS * 7 spaces.
            None => out.put_bytes(GPU_BLANK),
        }
    }
    twrite!(out, " ", format_age(p.age_secs()), "  ", indent, cmd);
    out.put_bytes(FG_CLOSE_EOL);
}

/// Render the process tree, fitting within `max_rows` by truncating children
/// per root group (never roots). Each group gets a share of the child budget
/// proportional to its size; overflow becomes a "... (N more)" line.
///
/// `tree` and `roots` are computed by the caller — doing so here would
/// require `&mut Frame`, but the caller has already locked the frame
/// by opening the stdout span we're writing into.
fn render_tree<W: TinyWriter + ?Sized>(
    out: &mut W,
    procs: &[ProcInfo<'_>],
    tree: &[(Indent, u32)],
    roots: &[usize],
    max_rows: usize,
    width: usize,
    gpu_total_mib: u64,
    has_gpu: bool,
) {
    // Roots themselves can outnumber the terminal (a wide build in a
    // short tmux split): emitting them all would push the frame past
    // the height and scroll the 🌳 header off the top every tick —
    // the exact bug class the child elision budget exists to prevent.
    // The overflow gets no marker row: the cap itself is what protects
    // the header, the situation is a degenerate layout, and a second
    // emit_elision_row call site forces it out of line (~150 B) for a
    // row that would only ever show in a shoebox terminal.
    let n_roots = roots.len().min(max_rows);
    // Budget: roots always shown, remaining lines distributed to children.
    let child_budget = max_rows.saturating_sub(roots.len());
    let kids_in = |k: usize| -> usize {
        let end = roots.get(k + 1).copied().unwrap_or(tree.len());
        end - roots[k] - 1
    };
    let total_children: usize = (0..roots.len()).map(kids_in).sum();

    // Proportional allocation: each group gets a fair slot of child_budget;
    // if slot can't fit all children, one line of the slot becomes "...".
    let mut allocated = 0;
    for k in 0..n_roots {
        let root_ti = roots[k];
        let kids = kids_in(k);
        let slot = if k + 1 == roots.len() {
            child_budget.saturating_sub(allocated)
        } else if total_children > 0 {
            // Integer floor; child_budget (≤ terminal rows) × kids
            // (≤ procs) stays far below usize overflow.
            child_budget * kids / total_children
        } else { 0 };
        let (limit, used) = if slot >= kids {
            (kids, kids)
        } else if slot > 0 {
            (slot - 1, slot)
        } else {
            (0, 0)
        };
        allocated += used;

        emit_proc_row(out, procs, &tree, root_ti, width, gpu_total_mib, has_gpu);
        for ti in root_ti + 1..root_ti + 1 + limit {
            emit_proc_row(out, procs, &tree, ti, width, gpu_total_mib, has_gpu);
        }
        // Emit the elision only when the slot actually reserved a row for it
        // (`used > limit`). `limit < kids` alone would fire even at slot=0 —
        // which spends a line the budget never earmarked, pushing the frame
        // past the terminal height and scrolling the header off the top.
        if used > limit {
            // Per-channel max over elided children: red (mem) and blue (cpu/gpu)
            // each pick the loudest elided row. The elision inherits the same
            // palette as a real row, so peaks aren't hidden by the truncation.
            let (mut rmax, mut bmax) = (0u8, 0u8);
            for ti in root_ti + 1 + limit..root_ti + 1 + kids {
                let p = &procs[tree[ti].1 as usize];
                let (r, _, b) = row_color(p.cpu, p.rss_kib, &p.gpu, gpu_total_mib);
                rmax = rmax.max(r); bmax = bmax.max(b);
            }
            emit_elision_row(out, kids - limit, has_gpu, (rmax, 0, bmax));
        }
    }
}

/// Render an elision marker ("N hidden children") as a regular tree row.
/// Blank data cells, depth-1 `└─` tree prefix, cmd cell becomes
/// `N more...`. Column-aligned with `emit_proc_row`, so the marker
/// reads as "last-sibling under this root" rather than a floating
/// annotation. `color` is the per-channel max of the elided children's
/// `row_color`s, so the elision's hue tracks peak load under the root.
fn emit_elision_row<W: TinyWriter + ?Sized>(out: &mut W, more: usize, has_gpu: bool, color: (u8, u8, u8)) {
    let (r, g, b) = color;
    // Column widths (all blank): pid(7) + sp + cpu(6) + sp + mem(6) +
    //   [GPU_DATA_COLS * 7 if has_gpu] + sp + age(6) + 2 sp.
    // Linux: 30 / 44. macOS: 30 / 37. `repeat` instead of literal space
    // runs — the three static strings cost ~110 B of rodata.
    // The depth-1 last-sibling prefix is constant ("└─") — going
    // through Indent's Put impl would keep a second call site alive
    // for its ~190 B monomorph.
    let blanks = 30 + if has_gpu { GPU_DATA_COLS * 7 } else { 0 };
    twrite!(out, FgOpen(r, g, b), bytes::repeat(' ', blanks),
            "└─", u32d(more as u32), " more...");
    out.put_bytes(FG_CLOSE_EOL);
}

// ── Entry point ───────────────────────────────────────────────────────────────

/// Packed flag bits returned by `parse_args` and consumed by `run`. Using a
/// bitmask (rather than a struct or `bool`s) lets both entry paths — the
/// glibc `main` wrapper below and the libc-free custom `_start` in
/// `src/start.rs` — thread the parsed result through a single scalar
/// register (`%rdi` on SysV), with no pointer chasing or memory stash.
#[cfg(not(test))] pub(crate) const FLAG_ONCE: u64 = 1 << 0;

/// Parse argv into a flag bitmask. Handles `--check` by emitting `ok\n`
/// and `exit_group(0)` — it never returns in that case. Called from both
/// entry paths (`main` on glibc, `ltop_entry` on the production build)
/// so argv parsing lives in exactly one place regardless of how we got here.
///
/// # Safety
/// `argv` must point to an `argc`-element array of NUL-terminated C strings.
#[cfg(not(test))]
pub(crate) unsafe fn parse_args(argc: i32, argv: *const *const u8) -> u64 {
    if platform::has_arg(argc, argv, b"--check") {
        syscall::write_once(1, b"ok\n");
        syscall::exit_group(0);
    }
    let mut flags: u64 = 0;
    if platform::has_arg(argc, argv, b"--once") { flags |= FLAG_ONCE; }
    flags
}

/// Traditional `extern "C" fn main` entry. Only compiled on targets
/// where we still go through libc's `__libc_start_main` (macOS, plus
/// glibc Linux builds like `cargo build --release`). The libc-free
/// production build has a custom `_start` (`src/start.rs`) that calls
/// `parse_args` itself and tail-jumps into `run`.
#[cfg(all(not(test), not(libc_free)))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn main(argc: i32, argv: *const *const u8) -> i32 {
    let flags = unsafe { parse_args(argc, argv) };

    // On macOS: relocate SP to the top page of the stack VMA, then
    // tail-jump to `mac_run_then_exit`. The wrapper drops every dirty
    // page below us (libdyld's argv/env / init frames, ~16 KB worth
    // typically), runs `unmap_idle_state` and the tick loop in a
    // single fresh page, and exits via raw `_exit` so libdyld's own
    // `exit(3)` (atexit handlers, os_log flush) never runs against
    // the pages we just reclaimed.
    //
    // Doing this BEFORE `unmap_idle_state` matters: that function
    // walks ~50 VM regions and would otherwise dirty ~16 KB of stack
    // in the page below the libdyld-set sp, contributing a second
    // dirty page that's harder to reclaim.
    //
    // Linux does the analogous dance from `_start` (`src/start.rs`);
    // on mac we have to do it from inside `main` because libdyld
    // owns the entry point.
    #[cfg(target_os = "macos")]
    {
        let cur_sp: u64;
        unsafe { core::arch::asm!("mov {}, sp", out(reg) cur_sp); }
        // We materialise pid in a u64 local first because passing
        // `syscall::getpid() as u64` directly through `in("x1")` of
        // a `noreturn` asm block can leave x1 holding a stale value
        // (the compiler picks x1 as a scratch reg for an unrelated
        // expression in the same statement and never re-loads it).
        // Burning a stack slot for the materialised value sidesteps
        // it, at the cost of one extra `str`.
        let pid_u64 = syscall::getpid() as u32 as u64;
        if let Some((_lo, hi)) = mac_sys::region_containing(pid_u64 as i32, cur_sp) {
            let new_sp = hi - 16; // SysV/AAPCS 16-byte aligned, top of last page
            let target = mac_run_then_exit as *const () as u64;
            unsafe {
                core::arch::asm!(
                    "mov sp, {sp}",
                    "br {target}",
                    sp = in(reg) new_sp,
                    target = in(reg) target,
                    in("x0") flags,
                    in("x1") pid_u64,
                    options(noreturn, nostack),
                );
            }
        }
        // Fallback: region lookup failed. Skip the SP relocation but
        // still do the libmalloc / libSystem reclaim before running.
        let _ = mac_sys::unmap_idle_state(pid_u64 as i32);
    }

    run(flags);
    #[cfg(target_os = "macos")]
    syscall::exit_group(0);
    #[cfg(not(target_os = "macos"))]
    0
}

/// Tail-jumped from `main` after SP has been relocated to the top page
/// of the stack VMA. Drops the libdyld-leftover pages below us, runs
/// `unmap_idle_state` (libmalloc / libSystem reclaim) and the tick
/// loop, then exits. Never returns.
#[cfg(all(not(test), not(libc_free), target_os = "macos"))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mac_run_then_exit(flags: u64, pid: i32) -> ! {
    const PAGE: u64 = 16384;
    let cur_sp: u64;
    unsafe { core::arch::asm!("mov {}, sp", out(reg) cur_sp); }
    // Walk downward from the page below ours, `MADV_FREE_REUSABLE`
    // each one until the kernel returns -ENOMEM (off the VMA bottom).
    // `MADV_FREE_REUSABLE` is the strongest Darwin reclaim and the
    // analogue of Linux's `MADV_DONTNEED`; plain `MADV_FREE` only
    // marks pages freeable and the kernel keeps them resident until
    // memory pressure hits.
    let mut p = (cur_sp & !(PAGE - 1)).wrapping_sub(PAGE);
    while p != 0 {
        let r = unsafe {
            syscall::madvise(p as *mut u8, PAGE as usize, syscall::MADV_FREE_REUSABLE)
        };
        if r != 0 { break; }
        p = p.wrapping_sub(PAGE);
    }
    let _ = mac_sys::unmap_idle_state(pid);
    run(flags);
    syscall::exit_group(0)
}

#[cfg(not(test))]
pub(crate) fn run(flags: u64) {
    let once = flags & FLAG_ONCE != 0;
    // `_SC_CLK_TCK` is ABI-fixed at 100 regardless of kernel HZ — no
    // sysconf roundtrip needed (a libc wrapper that touches TLS for
    // errno, and the production build has no TLS). The page size is
    // real state though: /proc reports RSS in kernel pages and aarch64
    // kernels ship 4 K/16 K/64 K, so it comes from auxv AT_PAGESZ via
    // `syscall::page`.
    #[cfg(target_os = "linux")]
    let page_size: u64 = syscall::page::get() as u64;
    // Unused on mac: RSS arrives in bytes (pti_resident_size).
    #[cfg(target_os = "macos")]
    let page_size: u64 = 0;
    let clock_ticks: f64 = 100.0;
    let num_cpus = num_cpus_online();

    // Interactive mode: raw termios + hidden cursor, restored on exit via guard.
    // For --once, skip both — no 'q' polling and no cursor flicker to hide.
    let _guard = (!once).then(tty::enter_raw_mode);

    let boot = boot_time();

    arena::scope(|init| {
        // Initialise the GPU driver once, at the top of the init scope. On
        // no-GPU hosts this returns a State with empty gpus; every downstream
        // call stays a no-op, so nothing else special-cases the absence.
        let rm_state = gpu::init(init);
        let has_gpu = rm_state.has_gpu();

        // One-shot startup is done; lock down the syscall surface for the
        // tick loop. Not under arena-trace: the tracer writes every arena
        // event to ./arena-trace.log (fd > 2), and the seccomp write
        // filter only allows fds 1 and 2 — the first traced tick would
        // die with SIGSYS mid-frame, silently truncating the trace the
        // arena-svg workflow consumes. An instrumented build isn't a
        // production binary; it runs unsandboxed.
        #[cfg(all(target_os = "linux", libc_free, not(feature = "arena-trace")))]
        sandbox::install();

        // Two adjacent cross-tick FSpans, both rotated each iteration via
        // `init.replace2`. Order matters: gpu_prev sits below prev in the
        // bump, and the closure must produce the new gpu_prev BEFORE the
        // new cpu prev so replace2's memmoves land them in their slots.
        //
        // gpu_prev: `(pid, accumulatedGPUTime_ns)` snapshot from last tick
        // (used by macOS to diff into per-process %). On Linux the type
        // alias `GpuPrevElem` resolves to `()`, so the slot is zero bytes
        // and replace2's first-span memmove compiles to nothing.
        //
        // prev: `(pid + last-tick visible bit, low 32 bits of cumulative
        // CPU counter)` per pid, sorted by masked pid so binary search
        // ignores the flag.
        let mut gpu_prev: FSpan<gpu::GpuPrevElem> = init.empty();
        let mut prev: FSpan<(u32, Packed)> = init.empty();
        // Wall time of the previous tick's sample — used to derive `dt`
        // for CPU% computation. Initialised to "now" so the first tick
        // sees dt ≈ 0 and compute_cpu returns 0 (prev is empty
        // anyway, so the value is academic).
        let mut prev_tick_time = Instant::now();
        // Sleep budget until the next tick, in 50 ms poll steps. First
        // tick is short (300 ms) so CPU% — which needs a delta between
        // two samples — populates on the second render in a third of a
        // second rather than two. Set to 40 (2 s) after the first loop.
        // At CLK_TCK=100 a 70% process accrues ~21 clock ticks in 300 ms,
        // so the ±1-tick quantization noise is ~±5% of the reading —
        // accurate enough for a "this is roughly what it's doing" read.
        let mut next_steps: u32 = 6;

        'main: loop {
            let now = Instant::now();
            let dt = now.duration_since(prev_tick_time).as_secs_f64();
            let (gpu_prev_new, prev_new) = init.replace2(
                "gpu/agx/prev", gpu_prev, "prev", prev,
                |tick, gpu_prev, prev| {
                let wall_now = now_epoch();
                let (width, height) = tty::term_size();

                // The RM ioctl scratch (see rm::populate's doc for the
                // current byte count) comes from `tick` via a sub-scope
                // inside populate — reclaimed before we return, so the
                // rest of the tick gets the bytes back. The small
                // gpu/procs FVec backing `gpu.procs` is sized exactly
                // (n_gpus * MAX_PROCS_PER_GPU) and also lives in `tick`.
                //
                // gpu::collect produces (gpu_info, new_gpu_prev) — new_gpu_prev
                // sits at a known offset above the rotated old gpu_prev/prev
                // slots, with gpu_info's transient `entries` FSpan above it.
                // Both will be relocated by replace2 once this closure returns.
                let dt_ns = (dt * 1e9) as u64;
                let (gpu, new_gpu_prev) = gpu::collect(&rm_state, gpu_prev, tick, dt_ns);
                let mut gpu_pids_buf: FVec<u32> =
                    tick.vec("gpu_pids", gpu.procs.len() as u32);
                // GPU_THRESH applies on macOS (where every per-process
                // entry already has a percentage from `accumulatedGPUTime`
                // delta); NVIDIA's RM path stores 0% for any process that
                // didn't happen to be the latest GPUMON sample, so the
                // filter would hide PIDs that hold real GPU memory. Keep
                // the filter at 0 there.
                let pid_thresh = if cfg!(target_os = "macos") { GPU_THRESH } else { 0 };
                gpu_pids_buf.extend(
                    gpu.procs.iter()
                        .filter(|(_, gp)| gp.sm_pct >= pid_thresh)
                        .map(|(pid, _)| *pid),
                );
                let gpu_pids = map::freeze_set(&mut gpu_pids_buf);

                // next: tick-scope FVec populated in lockstep with
                // procs during collect_procs. Initial visible bit is 0;
                // filter_with_children flips it on per procs index during
                // its retain pass. Sorted + frozen into an FSpan at the
                // end of the tick — that's the closure's return value
                // and becomes next tick's prev.
                // Wrap prev's sorted FSpan as a Map so lookups are
                // `prev.get(&pid)` — binary search, clean pid key.
                let prev_map: Map<'_, Packed> = Map::new(prev.as_slice());
                let (mut procs, mut next) = collect_procs(
                    page_size, clock_ticks, boot, wall_now, dt,
                    &prev_map, gpu_pids, tick,
                );
                // GPU usage attaches BEFORE the filter: the coalesce
                // pass inside filter_with_children folds descendants'
                // stats into their app root, and for Electron apps the
                // GPU numbers live on helper processes the fold hides.
                for p in procs.iter_mut() {
                    if let Some(usage) = gpu.usage(p.pid) {
                        p.gpu = Some(usage);
                    }
                }
                filter_with_children(tick, &mut procs, &prev_map, &mut next);
                // Delegate to sort.rs's in-place heapsort (via map.rs's
                // shared re-export) instead of `sort_unstable_by_key` — the
                // latter monomorphises Rust's generic quicksort on ProcInfo
                // for ~2 KB. ProcInfo is `repr(C)` with pid at offset 0, so
                // the shared u32-key sorter works directly.
                // Proof of the sorter's key-at-offset-0 precondition
                // (repr(C) makes it stable; the assert makes it loud).
                const { assert!(core::mem::offset_of!(ProcInfo, pid) == 0) };
                map::sort_by_u32_key(&mut procs);

                let la = load_avg(num_cpus);
                let mi = mem_info(tick);
                let (lr, _, lb) = frac_color(0.0, la.frac);
                let (mr, _, mb) = frac_color(mi.frac(), 0.0);
                let gpu_total_mib = gpu.total_mib;
                let gpu_mem_frac = gpu.mem_frac();
                let max_rows = height.saturating_sub(4);

                // Tree layout allocates from `tick`; do it BEFORE opening
                // the stdout span (which locks `tick` for the duration).
                // Both tree and roots live until end-of-tick; the span's
                // bytes allocate above them and never alias.
                let tree = tree_layout(tick, &procs);
                let roots: FSpan<usize> = tick.collect(
                    "roots",
                    tree.iter().enumerate()
                        .filter_map(|(i, (ind, _))| ind.is_root().then_some(i)),
                );

                // Render the whole frame into an arena span and ship it
                // to fd 1 in one write(2) loop.
                let out: FStr = tick.span("tty/stdout", |out| {
                    out.put_bytes(CURSOR_HOME);
                    twrite!(out, "🌳 ",
                        fg(0, 0, 255, "load:"), " ", fg(lr, 0, lb, &la), "  ",
                        fg(255, 0, 0, "mem:"),  " ", fg(mr, 0, mb, &mi));
                    if has_gpu {
                        // Red ← memory pressure, blue ← compute pressure.
                        // n_gpus is non-zero when has_gpu is true.
                        let util_frac = (gpu.total_util as f64 / gpu.n_gpus as f64) / 100.0;
                        let (gr, _, gb) = frac_color(gpu_mem_frac, util_frac);
                        // Blue label, matching `load:` — both are compute-pressure
                        // dimensions, just on different processors. The text
                        // "gpu:" / "load:" disambiguates them.
                        twrite!(out, "  ", fg(0, 0, 255, "gpu:"), " ", fg(gr, 0, gb, &gpu));
                    }
                    out.put_bytes(EOL);
                    line(out, repeat('─', width));
                    // Column headers. Widths (incl. separator spaces):
                    // pid(7+1)=8; cpu(6+1)=7; rss(6+1)=7; gpu%(6+1)=7;
                    // gmem(6+1)=7; age(6+2)=8; command.
                    out.put_bytes(BOLD);
                    out.put_bytes(b"PID       CPU%    RSS");
                    if has_gpu { out.put_bytes(GPU_HEADER); }
                    out.put_bytes(b"    AGE  COMMAND");
                    out.put_bytes(FG_CLOSE_EOL);
                    line(out, repeat('─', width));

                    if procs.is_empty() {
                        line(out, "(no matching processes)");
                    }
                    render_tree(out, &procs, &tree, &roots, max_rows, width, gpu_total_mib, has_gpu);
                    // Strip the trailing `\n` from the last `line()` so the
                    // cursor lands on the last content row, not below it.
                    // Emitting `\n` at the bottom scrolls the terminal by 1,
                    // pushing the 🌳 header off the top every tick. A single
                    // `\n` is emitted at exit (below) so the shell prompt
                    // lands on its own line after both `--once` and `q`.
                    out.pop();
                    out.put_bytes(CLEAR_TO_END);
                });
                drop(gpu);  // release borrow on gpu_scratch so it's reusable next tick
                tty::write_stdout(out.as_slice());

                // Sort next by pid and freeze into the FSpan replace2 will
                // slide down to prev's old slot. `new_gpu_prev` is the
                // sibling result that lands in gpu_prev's slot.
                map::sort_by_u32_key(next.as_mut_slice());
                (new_gpu_prev, next.into_fspan())
            });
            gpu_prev = gpu_prev_new;
            prev = prev_new;
            prev_tick_time = now;

            if once { break 'main; }
            // Tick scope closed: every per-tick allocation has been rewound
            // in logical terms. Now physically drop the pages we touched back
            // to the kernel so our RSS tracks the idle working set, not the
            // tick peak. Skipped when we're bailing — freeing right before
            // exit is pointless.
            arena::uncommit_tail();

            // Poll for 'q' while sleeping until the next tick. First tick
            // interval is 500 ms so CPU% — which needs a delta between two
            // `/proc` samples — becomes valid on the second render in half
            // a second instead of two. Steady-state is 2 s, restoring the
            // 50/tick `next_steps` that the CPU quantization was tuned for.
            for _ in 0..next_steps {
                // 0x03/0x1c are Ctrl-C/Ctrl-\: raw mode disables their
                // signal-char role (tty::enter_raw_mode), so they arrive
                // as bytes and quit cleanly through RawModeGuard.
                if matches!(tty::read_one_stdin(), Some(b'q' | 0x03 | 0x1c)) { break 'main; }
                platform::sleep(Duration::from_millis(50));
            }
            next_steps = 40;
        }
    });
    // Both exit paths (`--once` completion, `q` keypress) land here. Frames
    // deliberately omit the trailing `\n` so the 🌳 header doesn't scroll;
    // emit it once on the way out so the shell prompt lands below the last
    // rendered row, not appended to it.
    tty::write_stdout(b"\n");
}

/// Emit `content` followed by the ANSI "erase to end of line + newline"
/// trailer that every rendered row ends with. Keeps the frame coherent
/// when content doesn't reach the right margin.
fn line<P: Put, W: TinyWriter + ?Sized>(out: &mut W, content: P) {
    content.put(out);
    out.put_bytes(EOL);
}

// ── ANSI escape constants ──────────────────────────────────────────────────
//
// Named to keep them out of call sites where they'd read as magic.
//   FgClose — reset-all, closes a `FgOpen` span.
//   Eol     — erase-to-end-of-row + '\n'; used at the end of every rendered
//             line so short content doesn't leave stale cells from the
//             previous tick on-screen.

const FG_CLOSE: &[u8] = b"\x1b[0m";
const EOL: &[u8] = b"\x1b[K\n";
const FG_CLOSE_EOL: &[u8] = b"\x1b[0m\x1b[K\n";
const CURSOR_HOME: &[u8] = b"\x1b[H";
const CLEAR_TO_END: &[u8] = b"\x1b[J";
const BOLD: &[u8] = b"\x1b[1m";

/// ANSI 24-bit foreground-color open — writes `\x1b[38;2;R;G;Bm`.
/// Close with [`FG_CLOSE`] or wrap via [`fg`] (which does both).
struct FgOpen(u8, u8, u8);
impl Put for FgOpen {
    fn put<W: TinyWriter + ?Sized>(&self, w: &mut W) {
        w.put_bytes(b"\x1b[38;2;");
        u32d(self.0 as u32).put(w);
        w.put_byte(b';');
        u32d(self.1 as u32).put(w);
        w.put_byte(b';');
        u32d(self.2 as u32).put(w);
        w.put_byte(b'm');
    }
}

/// Wrap `content` in an ANSI 24-bit foreground-color escape plus reset.
struct Fg<P>(u8, u8, u8, P);
impl<P: Put> Put for Fg<P> {
    fn put<W: TinyWriter + ?Sized>(&self, w: &mut W) {
        FgOpen(self.0, self.1, self.2).put(w);
        self.3.put(w);
        w.put_bytes(FG_CLOSE);
    }
}
fn fg<P: Put>(r: u8, g: u8, b: u8, content: P) -> Fg<P> { Fg(r, g, b, content) }


// ── Process info struct ───────────────────────────────────────────────────────

// `repr(C)` pins pid at offset 0 so `map::sort_by_u32_key` works on
// `&mut [ProcInfo]`. All fields have align 4 (rss_kib is u32, not u64),
// so declaration-order layout needs no padding — size is 36 bytes.
//
// `age_visible` packs a bool and a u31 age counter into one u32 to avoid
// the 4-byte padding a separate `bool` would force. Bit 31 is `visible`,
// bits 0..31 are age in seconds (68 years max — any real process fits).
//
// `visible` has two phases:
//   1. collect_procs sets it to whether this proc passes the direct CPU/
//      mem/GPU thresholds on its own.
//   2. filter_with_children may turn it on for CPU-active descendants of
//      visible ancestors (the BFS pass).
// After filter_with_children, every surviving ProcInfo has visible = true.
#[repr(C)]
#[derive(Default)]
struct ProcInfo<'a> {
    pid: u32,
    ppid: u32,
    cpu: f32,        // % of one core; f32 (7-digit precision) covers 0–10000
    // KiB, not bytes — drops the struct's max-alignment from 8 (u64) to 4.
    // u32 covers up to 4 PiB of RSS per process, which is fine for any
    // realistic machine. All display thresholds are power-of-2 so they
    // shift cleanly: 1 MiB = 1 << 10 kib, 1 GiB = 1 << 20 kib.
    rss_kib: u32,
    age_visible: u32,
    gpu: Option<GpuUsage>,
    display: FSmallStr<'a>,
}

impl<'a> ProcInfo<'a> {
    const VISIBLE_BIT: u32 = 1 << 31;
    fn pack(age_secs: u32, visible: bool) -> u32 {
        debug_assert!(age_secs < Self::VISIBLE_BIT, "age_secs overflows 31 bits");
        age_secs | ((visible as u32) << 31)
    }
    fn age_secs(&self) -> u32 { self.age_visible & !Self::VISIBLE_BIT }
    fn visible(&self) -> bool { self.age_visible & Self::VISIBLE_BIT != 0 }
    fn set_visible(&mut self, v: bool) {
        if v { self.age_visible |= Self::VISIBLE_BIT; }
        else { self.age_visible &= !Self::VISIBLE_BIT; }
    }
}

/// Cross-tick per-pid payload, stored as the value of a `Map<u32, Packed>`
/// keyed by pid. Bit 31 carries last tick's visibility (for CPU-threshold
/// hysteresis); bits 0..31 carry the **upper 31 bits of the cumulative CPU
/// counter**. Shifting cpu right by 1 costs one bit of resolution (2
/// ticks on Linux, 2 ns on macOS — irrelevant at display scale) and
/// gives us a free bit for the flag.
#[derive(Clone, Copy, Default, Debug)]
#[repr(transparent)]
struct Packed(u32);

impl Packed {
    const VISIBLE: u32 = 1 << 31;
    const CPU_MASK: u32 = !Self::VISIBLE;  // bits 0..30, 31 bits

    /// Right-shift applied to the cumulative cpu counter before packing
    /// its low bits into u31. Must be ≥ 1 (to free bit 31 for the
    /// visibility flag). The amount above 1 trades precision for
    /// overflow headroom: the representable per-tick delta is
    /// `2^(32 + SHIFT - 1)` counter units.
    ///
    /// Linux `CLK_TCK=100`: a 2 s interval at 100% is ~200 ticks per
    /// core. `SHIFT=1` fits ~10M cores — never overflows.
    ///
    /// macOS Mach absolute time (24 MHz on M1+): a 2 s interval at 100%
    /// is ~48e6 ticks per core. `SHIFT=1` fits ~89 cores; `SHIFT=7`
    /// raises the ceiling to ~5700 cores, far beyond any Mac. Precision
    /// loss is 128 ticks ≈ 5 µs per sample (< 0.001% at 2 s tick),
    /// well below display resolution.
    #[cfg(target_os = "linux")]
    const SHIFT: u32 = 1;
    #[cfg(target_os = "macos")]
    const SHIFT: u32 = 7;

    /// Pack `total` (the raw cumulative CPU counter) + visibility bit
    /// into one u32. The stored low 31 bits are `(total >> SHIFT)`
    /// truncated to u32. `compute_cpu` reverses the shift via
    /// `wrapping_sub` + `<< SHIFT`.
    fn new(visible: bool, total: u64) -> Self {
        let bits = ((total >> Self::SHIFT) as u32) & Self::CPU_MASK;
        let v = if visible { Self::VISIBLE } else { 0 };
        Packed(bits | v)
    }

    fn visible(self) -> bool { self.0 & Self::VISIBLE != 0 }
    fn cpu_bits(self) -> u32 { self.0 & Self::CPU_MASK }
    fn set_visible(&mut self) { self.0 |= Self::VISIBLE; }
}

/// Keep processes that directly qualify, plus descendants with CPU >= threshold.
/// Hysteresis: a child needs CHILD_CPU_THRESH to start appearing, but only
/// CHILD_CPU_KEEP_THRESH to stay visible once previously shown. This keeps
/// processes hovering around the boundary from flickering in and out.
/// When an intermediate process is filtered out, reparent its children to
/// the nearest surviving ancestor.
const CHILD_CPU_THRESH: f32 = 1.0;
const CHILD_CPU_KEEP_THRESH: f32 = 0.3;

/// Electron names an app's service processes after the app itself:
/// "Code Helper (Renderer)" under "Code", "Claude Helper (GPU)" under
/// "Claude". So "is this proc a boring shard of that ancestor" is a
/// name test — no per-app list, and byte compare keeps it case
/// sensitive ("Claude Helper" never folds into the "claude" CLI, whose
/// children are exactly what this monitor exists to show).
fn is_helper_of(kid: &[u8], root: &[u8]) -> bool {
    let n = root.len();
    kid.len() >= n + 7 && kid[..n] == *root && kid[n..n + 7] == *b" Helper"
}

/// Nearest ancestor of `idx` that `idx` is named a helper of, via a
/// flat ppid-chain walk (cheaper in bytes than a subtree traversal).
/// The hop cap bounds ppid cycles (possible via a pid-reuse race
/// during the sequential scan); a self-cycle needs no extra check
/// because `is_helper_of(n, n)` is always false.
fn coalesce_root(procs: &[ProcInfo], pid_to_idx: &Map<'_, u32>, idx: usize) -> Option<usize> {
    let name = procs[idx].display.as_slice();
    let mut a = procs[idx].ppid;
    for _ in 0..64 {
        let &pa = pid_to_idx.get(&a)?;
        let pa = pa as usize;
        if is_helper_of(name, procs[pa].display.as_slice()) { return Some(pa); }
        a = procs[pa].ppid;
    }
    None
}

/// Coalesce Electron helper subtrees into their app's row — apps like
/// Code or Linear are one logical thing sprawled over a main +
/// zygote/helper process tree. Every helper-named proc (per
/// [`is_helper_of`]) folds its cpu/rss/gpu into its app root, hiding
/// itself; the root surfaces if anything folded in was visible.
/// Non-helpers never fold — a `claude` CLI spawned by a plugin helper
/// keeps its own subtree. Runs BEFORE the visibility BFS, which makes
/// that survivor's reparenting free: its helper parent is already
/// hidden (with cpu/rss zeroed, the thresholds can't resurface it), so
/// the BFS hangs the survivor off the app root like any other visible
/// proc under an invisible parent.
///
/// Out of line: inlined into `run`'s tick closure the walk pays ~100
/// bytes of register spills against that huge frame.
#[inline(never)]
fn coalesce_helpers(procs: &mut [ProcInfo], pid_to_idx: &Map<'_, u32>) {
    for idx in 0..procs.len() {
        let Some(r) = coalesce_root(procs, pid_to_idx, idx) else { continue };
        let k = &mut procs[idx];
        let vis = k.visible();
        let (cpu, rss, gpu) = (k.cpu, k.rss_kib, k.gpu.take());
        k.set_visible(false);
        k.cpu = 0.0;
        k.rss_kib = 0;
        let p = &mut procs[r];
        if vis { p.set_visible(true); }
        p.cpu += cpu;
        p.rss_kib = p.rss_kib.saturating_add(rss);
        if let Some(g) = gpu {
            #[cfg(target_os = "macos")]
            {
                let s0 = p.gpu.take().map(|h| h.sm_pct).unwrap_or(0);
                p.gpu = Some(GpuUsage { sm_pct: s0 + g.sm_pct });
            }
            #[cfg(not(target_os = "macos"))]
            {
                let (s0, m0) = p.gpu.take()
                    .map(|h| (h.sm_pct, h.mem_mib.get())).unwrap_or((0, 0));
                let mem = m0.saturating_add(g.mem_mib.get());
                // mem > 0: g.mem_mib is NonZero, so the sum is too.
                if let Some(mem_mib) = core::num::NonZeroU32::new(mem) {
                    p.gpu = Some(GpuUsage { sm_pct: s0 + g.sm_pct, mem_mib });
                }
            }
        }
    }
}

fn filter_with_children<'a>(
    frame: &mut Frame<'a>,
    procs: &mut FVec<'a, ProcInfo<'a>>,
    prev: &Map<'_, Packed>,
    next: &mut FVec<'_, (u32, Packed)>,
) {
    // Wrap scratch (pid_to_idx, children, bfs_queue) in a sub-scope so
    // they're reclaimed when filter returns — otherwise their bytes
    // linger in the tick arena alongside tree_layout's own copies.
    // `procs` and `next` are 'a-scoped outer buffers; mutating them
    // from inside the sub-scope is fine because neither mutation needs
    // frame allocation — just &mut on the buffers.
    // The procs[i]/next[i] lockstep is maintained independently by both
    // platforms' collect_procs (push placeholder first, overwrite at i);
    // an early `continue` inserted before either overwrite would desync
    // them with no compile-time signal. Tripwire for dev builds.
    debug_assert!(
        procs.len() == next.len()
            && procs.iter().zip(next.iter()).all(|(p, n)| p.pid == n.0),
        "collect_procs pushed procs and next out of lockstep",
    );
    frame.scope(|sub| {
        let n = procs.len() as u32;
        let pid_to_idx = build_pid_to_idx(sub, procs);
        let children = build_children(sub, procs, &pid_to_idx);

        coalesce_helpers(procs.as_mut_slice(), &pid_to_idx);

        // BFS from directly qualifying roots. Always traverse descendants so
        // we can reach high-CPU grandchildren under quiet parents; only mark
        // visible when the per-child threshold passes (with hysteresis). As
        // we descend, carry "last visible ancestor on this path" alongside
        // each queue entry — it's `kid` once `kid` becomes visible, otherwise
        // inherited from the parent. When a kid becomes/stays visible under
        // an invisible parent, rewrite its ppid straight to that last-visible
        // ancestor's pid. Same end state as the old two-pass (BFS then
        // walk-up reparent) for ~6 fewer lines and no O(chain) walk for deep
        // invisible chains.
        let mut queue: FVec<(u32, u32)> = sub.vec("bfs_queue", n);
        // Enqueue-once guard, one byte per proc. Without it a ppid
        // cycle (possible via a pid-reuse race during the sequential
        // /proc scan) loops this traversal forever, and a node that is
        // both a seed and someone's child gets its subtree pushed
        // twice — silently overflowing the n-capacity queue. Marked
        // nodes still get threshold/reparent treatment when their
        // parent pops; only the re-enqueue is skipped (their own
        // children were already processed with identical state).
        let mut enqueued: FVec<u8> = sub.vec("bfs_enqueued", n);
        for (i, p) in procs.iter().enumerate() {
            let vis = p.visible();
            let _ = enqueued.push(vis as u8);
            if vis { let _ = queue.push((i as u32, i as u32)); }
        }
        while let Some((i, last_vis)) = queue.pop() {
            // Snapshot the ancestor state the inner loop needs, so the
            // body can hold a single `&mut` to the kid without fighting
            // the borrow checker over other indices into `procs`.
            let i_visible = procs[i as usize].visible();
            let last_vis_pid = procs[last_vis as usize].pid;
            for &kid in children.bucket(i as usize).iter() {
                let k = &mut procs[kid as usize];
                let was_visible = prev.get(&k.pid).map(|p| p.visible()).unwrap_or(false);
                let thresh = if was_visible { CHILD_CPU_KEEP_THRESH } else { CHILD_CPU_THRESH };
                if k.cpu >= thresh { k.set_visible(true); }
                if k.visible() && !i_visible { k.ppid = last_vis_pid; }
                let new_last = if k.visible() { kid } else { last_vis };
                let seen = &mut enqueued.as_mut_slice()[kid as usize];
                if *seen == 0 {
                    *seen = 1;
                    let _ = queue.push((kid, new_last));
                }
            }
        }

        // Compact in place, keeping only visible procs. `next` was pushed in
        // lockstep with procs during collect_procs, so it has the same length
        // and pid order — use a parallel index `i` to flip the visible flag
        // on this tick's entries as we retain the visible procs.
        let mut i = 0usize;
        procs.retain_in_place(|p| {
            if p.visible() { next[i].1.set_visible(); }
            i += 1;
            p.visible()
        });
    });
}

// ── Wall clock ────────────────────────────────────────────────────────────────

fn now_epoch() -> u64 {
    // `clock_gettime(CLOCK_REALTIME)` is the modern replacement for
    // `gettimeofday`; same wall-clock seconds, no microseconds field we'd
    // throw away.
    syscall::get_clock(syscall::CLOCK_REALTIME).tv_sec as u64
}

// ── Small /proc file reader ───────────────────────────────────────────────────

/// Read a small (< buf.len()) /proc file in one syscall. Returns the text on
/// success. Avoids std::fs::read_to_string, which pulls in ~1.5 KB of
/// io::Error + read_to_end machinery for us.
#[cfg(target_os = "linux")]
pub(crate) fn read_small_file<'a>(path: &core::ffi::CStr, buf: &'a mut [u8]) -> Option<&'a [u8]> {
    let fd = syscall::open_cstr(path, syscall::O_RDONLY | syscall::O_CLOEXEC);
    if fd < 0 { return None; }
    let n = syscall::read_buf(fd, buf);
    syscall::close(fd);
    if n <= 0 { return None; }
    Some(&buf[..n as usize])
}

// ── Boot time ─────────────────────────────────────────────────────────────────

#[cfg(target_os = "linux")]
fn boot_time() -> u64 {
    // Derive boot epoch as `now - uptime`. /proc/uptime's first field is
    // the seconds-since-boot as an ASCII float. Avoids `sysinfo(2)`,
    // whose libc wrapper touches TLS for errno, and whose 256-byte
    // `struct sysinfo` we'd otherwise put on the stack for one number.
    let mut buf = [0u8; 64];
    let Some(text) = read_small_file(c"/proc/uptime", &mut buf) else { return 0; };
    // /proc/uptime: "1234.56 9876.54\n". First field's integer part.
    let first = bytes::split_ascii_whitespace(text).next().unwrap_or(b"0");
    let whole = first.split(|&b| b == b'.').next().unwrap_or(b"0");
    let uptime = parse_u64_ascii(whole).unwrap_or(0);
    now_epoch().saturating_sub(uptime)
}

#[cfg(target_os = "macos")]
fn boot_time() -> u64 {
    // sysctl kern.boottime (CTL_KERN=1, KERN_BOOTTIME=21) → struct timeval
    // { tv_sec: i64, tv_usec: i32 } on arm64. We take only tv_sec.
    #[repr(C)]
    struct Timeval { tv_sec: i64, tv_usec: i32 }
    let mut mib = [1, 21];
    let mut tv = Timeval { tv_sec: 0, tv_usec: 0 };
    mac_sys::sysctl_read(&mut mib, &mut tv);
    tv.tv_sec as u64
}


// ── Process collection: macOS ─────────────────────────────────────────────────

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn collect_procs<'a>(
    _page_size: u64,
    _clock_ticks: f64,
    _boot: u64,
    wall_now: u64,
    dt: f64,
    prev: &Map<'_, Packed>,
    gpu_pids: Set<'_>,
    frame: &mut Frame<'a>,
) -> (FVec<'a, ProcInfo<'a>>, FVec<'a, (u32, Packed)>) {
    use mac_sys::*;
    let my_pid = platform::getpid();
    // pti_total_{user,system} are in Mach absolute time units
    // (TASK_ABSOLUTETIME_INFO), the same clock as mach_absolute_time().
    // On M1+ the timebase is {numer=125, denom=3} → 24 MHz; NOT nanoseconds.
    let mach_tps = syscall::mach_ticks_per_sec();

    // Query required buffer size, then read all PIDs into scratch that
    // lives only as long as we need to seed `procs`. `procs` is allocated
    // first (at the bottom of the bump); `pids_raw` is allocated inside
    // a sub-scope above it, so by the time the sub-scope exits the i32
    // pid buffer is rewound and the only allocation that survives into
    // pass 2 is `procs` itself. We can't use `frame.compact` (which
    // would also size `procs` exactly to its filled length) because
    // ProcInfo holds an FSmallStr and isn't `Copy`.
    // `capacity` is a pid count, not bytes — the wrapper already divided
    // by 4. +32 slots covers pids spawned between the query and the fill.
    let capacity = proc_listallpids_query();
    if capacity <= 0 { return (frame.vec("procs", 0), frame.vec("next", 0)); }
    let n_slots = (capacity as usize + 32) as u32;
    let mut procs: FVec<ProcInfo> = frame.vec("procs", n_slots);
    frame.scope(|sub| {
        let mut pids_raw = sub.zeros::<i32>("pids_raw", n_slots as usize);
        let count = (proc_listallpids_into(&mut pids_raw[..]) as usize)
            .min(n_slots as usize);
        for &pid in &pids_raw[..count] {
            let pid_u = pid as u32;
            if pid <= 0 || pid_u == my_pid { continue; }
            let _ = procs.push(ProcInfo { pid: pid_u, ..Default::default() });
        }
    });
    let mut next: FVec<(u32, Packed)> = frame.vec("next", procs.len() as u32);

    let mut comm_buf = [0u8; 64];
    for i in 0..procs.len() {
        let pid_u = procs[i].pid;
        let pid = pid_u as i32;
        let _ = next.push((pid_u, Packed::new(false, 0)));

        let mut ti = ProcTaskInfo::default();
        if proc_pidtaskinfo(pid, &mut ti) <= 0 { continue; }

        let rss_bytes = ti.pti_resident_size;
        let total_ticks = ti.pti_total_user + ti.pti_total_system;
        let (age_secs, ppid) = macos_proc_age_ppid(pid, wall_now);
        let cpu = compute_cpu(prev.get(&pid_u).copied(), age_secs, total_ticks, mach_tps, dt);
        next[i] = (pid_u, Packed::new(false, total_ticks));

        proc_name_into(pid, &mut comm_buf);
        let end = comm_buf.iter().position(|&b| b == 0).unwrap_or(comm_buf.len());
        let comm: &[u8] = &comm_buf[..end];

        // Per-pid `compact`: allocate args_buf (4 KB, transient scratch)
        // above the eventual display string, build display, then shift
        // display down so args_buf is reclaimed before the next pid.
        // 4 KB covers typical argv+envp (~2 KB); pids whose KERN_PROCARGS2
        // exceeds 4 KB get an empty args list and may be skipped or
        // displayed without argv detail.
        //
        // Inlines `classify_proc` here because args borrows from args_buf
        // and the display string must come from the same Frame as
        // args_buf — which is `f` inside the closure, not the outer
        // `frame`. The classification booleans cross the closure boundary
        // via `&mut` captures.
        let rss_mb = rss_bytes / (1024 * 1024);
        let mut skip = false;
        let mut visible = false;
        let display_span: FSpan<u8> = frame.compact("display", |f| {
            // Size the args buffer to this pid's actual argv+envp blob (capped
            // at ARG_MAX). KERN_PROCARGS2 only returns argv when the buffer
            // spans the whole blob, but most processes are ~2 KB — only a big
            // environment (Lake/lean during a Mathlib build) allocates more, so
            // the arena's steady-state peak stays small. `cap == 0` (query
            // failed or out of range) leaves args empty → render by `comm`.
            let cap = match macos_args_blob_size(pid_u) {
                Some(n) if (4..=MACOS_ARG_MAX).contains(&n) => n,
                _ => 0,
            };
            let mut args_buf = f.vec::<u8>("args_buf", cap as u32);
            let mut args_out: [&[u8]; 64] = [&[][..]; 64];
            let n_args = if cap > 0 {
                macos_proc_args_into(pid_u, args_buf.fill_zeroed(), &mut args_out)
            } else {
                0
            };
            let args = &args_out[..n_args];

            if is_noise(comm, args) {
                skip = true;
                return f.empty::<u8>();
            }
            let always_show = is_lean_or_lake(comm, args);
            let mem_qualifies = rss_mb >= MEM_MB_THRESH && !is_idle_noise(comm, args);
            visible = always_show || cpu >= CPU_THRESH || mem_qualifies
                || gpu_pids.contains(&pid_u);
            let is_related = always_show || is_lean_related(comm, args);
            f.str("display", |b| build_display_into(b, comm, args, is_related))
        });
        if skip {
            // Same treatment as the Linux collect path: noise stays
            // hidden (cpu zeroed, no display) but keeps its real ppid
            // so it doesn't sever the ancestry chain for busy
            // non-noise descendants.
            procs[i] = ProcInfo {
                pid: pid_u, ppid, cpu: 0.0, rss_kib: (rss_bytes >> 10) as u32,
                age_visible: ProcInfo::pack(age_secs, false),
                gpu: None,
                display: FSmallStr::default(),
            };
            continue;
        }

        procs[i] = ProcInfo {
            pid: pid_u, ppid, cpu, rss_kib: (rss_bytes >> 10) as u32,
            age_visible: ProcInfo::pack(age_secs, visible),
            gpu: None,
            display: display_span.into_small_lossy(),
        };
    }
    (procs, next)
}

/// macOS caps a process's argv+envp at `ARG_MAX` (256 KiB) at exec time,
/// so that's the largest blob `KERN_PROCARGS2` can return. The arena has
/// room for one transient buffer that big inside the per-pid `compact`.
#[cfg(target_os = "macos")]
const MACOS_ARG_MAX: usize = 1 << 18;

/// Size of the `KERN_PROCARGS2` blob for `pid` (argc + exec_path + argv +
/// envp), or `None` if the query fails. Lets the caller size the transient
/// args buffer to exactly this pid, so ordinary processes don't pay for a
/// worst-case buffer just so the rare big-environment one fits.
#[cfg(target_os = "macos")]
fn macos_args_blob_size(pid: u32) -> Option<usize> {
    let mut mib = [1, 49, pid as i32];
    mac_sys::sysctl_size(&mut mib)
}

/// Read KERN_PROCARGS2 for a PID into `buf`, then fill `out` with
/// [argv0, argv1, ...] as byte slices borrowing from `buf`. Returns
/// the number of slices written. `out` is a stack-array of 64 slots.
/// The blob's leading exec_path is skipped so the layout matches the
/// Linux side exactly — the classifiers (`find_script_arg`'s
/// `.skip(1)`, `is_lean_related`'s `.take(3)`) and `build_display_into`
/// all assume `args[0]` is argv[0].
///
/// `buf` must be sized by the caller to the full blob (argc + exec_path +
/// argv + **envp**) — see `macos_args_blob_size`. We only want argv, but
/// the kernel won't hand back a front-only slice: a buffer shorter than the
/// whole region yields the envp *tail* (xnu `sysctl_procargsx` copies
/// `copy_end - buflen`), so under-sizing loses argv entirely. Hence the
/// size-query-then-alloc dance in the caller rather than a fixed buffer.
///
/// Returns slices via `out` rather than a `Vec<&[u8]>`; the Vec growth path
/// pulled `RawVec<&[u8]>::grow_one` + `grow_amortized` (~325 B of alloc
/// machinery) into the binary even though our global allocator aborts.
#[cfg(target_os = "macos")]
fn macos_proc_args_into<'buf>(
    pid: u32,
    buf: &'buf mut [u8],
    out: &mut [&'buf [u8]; 64],
) -> usize {
    // sysctl kern.procargs2 (CTL_KERN=1, KERN_PROCARGS2=49) for pid. `buf`
    // already matches the queried size, so this single read captures argv.
    let mut mib = [1, 49, pid as i32];
    let Some(size) = mac_sys::sysctl_read_bytes(&mut mib, buf) else { return 0; };
    if size < 4 { return 0; }

    // Layout: [argc: i32] [exec_path\0] [null padding] [argv[0]\0] [argv[1]\0] ...
    let argc = i32::from_ne_bytes(buf[..4].try_into().unwrap_or([0; 4])).max(0) as usize;
    let want = argc.min(out.len());     // argv entries only, clamped to 64
    let mut i = 4usize;
    // Skip exec_path and the null padding that follows it.
    while i < size && buf[i] != 0 { i += 1; }
    while i < size && buf[i] == 0 { i += 1; }
    let mut n = 0usize;
    while i < size && n < want {
        let start = i;
        while i < size && buf[i] != 0 { i += 1; }
        if i > start {
            out[n] = &buf[start..i];
            n += 1;
        }
        i += 1;
    }
    n
}

/// Get process age in seconds + PPID via PROC_PIDTBSDINFO.
/// struct proc_bsdinfo: pbi_ppid (u32) at byte offset 16,
///                      pbi_start_tvsec (u64) at byte offset 120.
#[cfg(target_os = "macos")]
fn macos_proc_age_ppid(pid: i32, wall_now: u64) -> (u32, u32) {
    const BUF_SIZE: usize = 232; // PROC_PIDTBSDINFO_SIZE
    const PPID_OFF: usize = 16;  // offset of pbi_ppid in proc_bsdinfo
    const TVSEC_OFF: usize = 120; // offset of pbi_start_tvsec in proc_bsdinfo

    // Stack array, not `vec![0u8; BUF_SIZE]` — 232 B is well under the
    // 1 KB rule-of-thumb for stack allocation, and keeping it off the
    // heap means no RawVec machinery in the binary.
    let mut buf = [0u8; BUF_SIZE];
    let ret = mac_sys::proc_pidbsdinfo(pid, &mut buf);
    if ret < (TVSEC_OFF + 8) as i32 { return (0, 0); }
    let ppid = u32::from_ne_bytes(buf[PPID_OFF..PPID_OFF + 4].try_into().unwrap());
    let start = u64::from_ne_bytes(buf[TVSEC_OFF..TVSEC_OFF + 8].try_into().unwrap());
    (wall_now.saturating_sub(start) as u32, ppid)
}

// ── Process collection: Linux ─────────────────────────────────────────────────

#[cfg(target_os = "linux")]
#[allow(clippy::too_many_arguments)]
fn collect_procs<'a>(
    page_size: u64,
    clock_ticks: f64,
    boot: u64,
    wall_now: u64,
    dt: f64,
    prev: &Map<'_, Packed>,
    gpu_pids: Set<'_>,
    frame: &mut Frame<'a>,
) -> (FVec<'a, ProcInfo<'a>>, FVec<'a, (u32, Packed)>) {
    let my_pid = platform::getpid();

    // Pass 1: enumerate /proc and spawn a ProcInfo stub per pid (other
    // fields zeroed). `frame.span` sizes the FSpan to the actual live
    // proc count without the caller having to count first. `pid_dir`
    // is kept alive through both passes because its fd is reused for
    // per-pid openat() calls in pass 2.
    //
    // `..mem::zeroed()` is sound for ProcInfo because every field has
    // a valid all-bits-zero value: u32/f32 → 0, Option<GpuUsage> with
    // a NonZeroU32 niche → None, FStr = FSpan<u8> with offset=0/len=0
    // → a valid empty slice.
    // 4 KB getdents64 scratch buffer — lives in the arena for the tick,
    // not on the stack. `platform::PidDir` borrows from it for the
    // duration of this call.
    let mut dirent_buf: FBox<[u64; platform::DIRENT_BUF_U64S]> =
        frame.alloc_zeroed("proc/dirent_buf");
    let Some(mut pid_dir) = platform::PidDir::open(c"/proc", &mut dirent_buf)
        else { return (frame.vec("procs", 0), frame.vec("next", 0)); };
    let proc_fd = pid_dir.fd();
    let procs_stub: FSpan<ProcInfo> = frame.span("procs", |b| {
        while let Some(pid) = pid_dir.next() {
            if pid != my_pid {
                b.push(ProcInfo { pid, ..Default::default() });
            }
        }
    });

    // Convert to FVec so filter_with_children can `retain_in_place`.
    // Capacity == len, so no growth possible — just shrinkage when
    // non-visible procs get dropped.
    let mut procs: FVec<ProcInfo> = procs_stub.into_fvec();

    // `next` is sized to the exact pid count, allocated after pass 1
    // so it lives through pass 2 / filter / render but with no
    // over-capacity. Push a placeholder at the top of each iteration
    // so procs[i] and next[i] stay in lockstep even when stat reads
    // fail (otherwise filter_with_children's index mirror breaks).
    let mut next: FVec<(u32, Packed)> = frame.vec("next", procs.len() as u32);

    // Pass 2: reused per-pid buffers. Max written to path is
    // "{pid}/cmdline\0" = ~19 bytes; 32 is safe.
    let mut path: FVec<u8> = frame.vec("path", 32);
    // stat_raw and cmdline_chunk are each 4 KB and only needed for a single
    // per-pid read; they live inside per-iteration sub-scopes so they're
    // reclaimed before display_fallback accumulates and tree layout runs.
    // Keeping them in outer scope would put 8 KB on the critical path all
    // the way through rendering.
    // Stat parse results, copied out of the sub-scope so stat_raw can
    // be reclaimed. `comm` is the process name from /proc/<pid>/stat
    // (max TASK_COMM_LEN = 16 bytes including NUL).
    struct Parsed {
        ppid: u32,
        total_ticks: u64,
        age_secs: u32,
        rss_bytes: u64,
        comm_buf: [u8; 16],
        comm_len: u8,
    }

    for i in 0..procs.len() {
        let pid = procs[i].pid;
        // Placeholder; stat success below overwrites with real total.
        let _ = next.push((pid, Packed::new(false, 0)));

        path.clear();
        twrite!(&mut path, u32d(pid), "/stat\0");
        // stat_raw lives only for this read; any references into it
        // (comm, field slices) must be copied/parsed out before the
        // sub-scope returns.
        let parsed: Option<Parsed> = frame.scope(|sub| {
            let mut stat_raw: FBox<[u8; 4096]> = sub.alloc_zeroed("proc/stat_raw");
            // Raw open+read+close: a single `read()` syscall gets the whole stat
            // file (kernel-generated /proc files never short-read under 4K), saving
            // the EOF-probe read that `read_to_string` would do.
            // Path is "{pid}/stat\0" — digits are NUL-free, so the
            // terminator is by construction unique. Skip the runtime
            // memchr scan that `from_bytes_with_nul` would cost.
            let path_cstr = unsafe { core::ffi::CStr::from_bytes_with_nul_unchecked(&path) };
            let fd = syscall::openat_cstr(proc_fd, path_cstr, syscall::O_RDONLY | syscall::O_CLOEXEC);
            if fd < 0 { return None; }
            let n = syscall::read_buf(fd, &mut stat_raw[..]);
            syscall::close(fd);
            if n <= 0 { return None; }
            let (comm, after_comm) = parse_stat(&stat_raw[..n as usize]);

            // Parse the needed /proc/<pid>/stat fields via byte-level iterator.
            let mut fields = after_comm.split(|b: &u8| b.is_ascii_whitespace())
                .filter(|s| !s.is_empty());
            let ppid = fields.nth(1).and_then(parse_u64_ascii).unwrap_or(0) as u32;
            let utime = fields.nth(9).and_then(parse_u64_ascii).unwrap_or(0);
            let stime = fields.next().and_then(parse_u64_ascii).unwrap_or(0);
            let total_ticks = utime + stime;
            let starttime = fields.nth(6).and_then(parse_u64_ascii).unwrap_or(0);
            let start_epoch = boot + starttime / clock_ticks as u64;
            let age_secs = wall_now.saturating_sub(start_epoch) as u32;
            let rss_pages = fields.nth(1).and_then(parse_u64_ascii).unwrap_or(0);
            let rss_bytes = rss_pages * page_size;

            let mut comm_buf = [0u8; 16];
            let comm_len = comm.len().min(15);
            comm_buf[..comm_len].copy_from_slice(&comm[..comm_len]);
            Some(Parsed { ppid, total_ticks, age_secs, rss_bytes, comm_buf, comm_len: comm_len as u8 })
        });
        // sub-scope exited: stat_raw reclaimed, only parsed values + comm_buf remain.

        let Some(parsed) = parsed else { continue; };
        let Parsed { ppid, total_ticks, age_secs, rss_bytes, comm_buf, comm_len } = parsed;
        let comm: &[u8] = &comm_buf[..comm_len as usize];
        let prev_entry = prev.get(&pid).copied();
        let cpu = compute_cpu(prev_entry, age_secs, total_ticks, clock_ticks, dt);
        // Overwrite the placeholder pushed above with the real
        // total (visible=false; filter_with_children flips it on
        // per-entry during its retain pass).
        next[i] = (pid, Packed::new(false, total_ticks));

        // Fast path: processes that can't possibly show up in the visible set
        // (too quiet, no GPU, not lean/lake) skip the cmdline read and get a
        // minimal display. They still go into the output so filter_with_children
        // can resolve tree parents through them.
        let rss_mb = rss_bytes / (1024 * 1024);
        // The hysteresis band: a previously-visible child stays shown
        // down to CHILD_CPU_KEEP_THRESH, so it needs its argv display
        // too — gating on CHILD_CPU_THRESH alone made kept rows in the
        // 0.3..1.0% band flicker back to bare comm.
        let kept_visible = cpu >= CHILD_CPU_KEEP_THRESH
            && prev_entry.map(|p| p.visible()).unwrap_or(false);
        let may_qualify = cpu >= CHILD_CPU_THRESH || kept_visible
            || rss_mb >= MEM_MB_THRESH
            || gpu_pids.contains(&pid) || matches!(comm, b"lean" | b"lake");
        if !may_qualify {
            let display = frame.str("display_fallback", |b| b.extend_from_slice(comm)).into_small();
            procs[i] = ProcInfo { pid, ppid, cpu, rss_kib: (rss_bytes >> 10) as u32,
                                  age_visible: ProcInfo::pack(age_secs, false),
                                  gpu: None, display };
            continue;
        }

        path.clear();
        twrite!(&mut path, u32d(pid), "/cmdline\0");
        // Read cmdline + build display under `frame.compact` so the
        // cmdline bytes, plus the cmdline_chunk scratch buffer, are
        // all reclaimed as soon as the display is returned. Zero
        // persistent arena cost for cmdline or scratch.
        let mut keep = false;
        let mut noise = false;
        let mut visible = false;
        // Linux permits argv+envp up to RLIMIT_STACK/4 (~2 MB at the
        // default 8 MB stack) — far beyond the 512 KB arena, and a huge
        // C++ link line is exactly what this monitor watches. Only the
        // first MAX_ARGS arguments can affect the display, so cap the
        // read; the macOS path is capped the same way via MACOS_ARG_MAX.
        const CMDLINE_MAX: usize = 32 * 1024;
        let display: FSmallStr = frame.compact("display", |inner| {
            let mut cmdline_chunk: FBox<[u8; 4096]> = inner.alloc_zeroed("proc/cmdline_chunk");
            let cmdline: FStr = inner.str("cmdline", |b| {
                // Path is "{pid}/cmdline\0" — digits are NUL-free, so the
                // trailing 0 is the only NUL by construction.
                let path_cstr = unsafe { core::ffi::CStr::from_bytes_with_nul_unchecked(&path) };
                let fd = syscall::openat_cstr(proc_fd, path_cstr, syscall::O_RDONLY | syscall::O_CLOEXEC);
                if fd < 0 { return; }
                let chunk: &mut [u8; 4096] = &mut cmdline_chunk;
                loop {
                    let n = syscall::read_buf(fd, chunk);
                    if n <= 0 { break; }
                    b.extend_from_slice(&chunk[..n as usize]);
                    if (n as usize) < chunk.len() { break; } // short read = EOF
                    if b.len() >= CMDLINE_MAX { break; }
                }
                syscall::close(fd);
            });
            // Split at NUL into a stack array. args slices borrow
            // from cmdline; valid until the compact closure exits.
            const MAX_ARGS: usize = 64;
            let mut args_arr: [&[u8]; MAX_ARGS] = [&[]; MAX_ARGS];
            let mut args_len = 0;
            for s in cmdline.as_slice().split(|&b| b == 0).filter(|s| !s.is_empty()) {
                if args_len >= MAX_ARGS { break; }
                args_arr[args_len] = s;
                args_len += 1;
            }
            // A capped read can end mid-argument; a partial trailing
            // arg could masquerade as the script/module name, so drop it.
            if cmdline.len() >= CMDLINE_MAX && cmdline.last() != Some(&0) {
                args_len = args_len.saturating_sub(1);
            }
            let args = &args_arr[..args_len];
            if is_noise(comm, args) {
                noise = true;
                return inner.empty::<u8>();
            }
            let always_show = is_lean_or_lake(comm, args);
            let is_related = always_show || is_lean_related(comm, args);
            let mem_qualifies = rss_mb >= MEM_MB_THRESH && !is_idle_noise(comm, args);
            visible = always_show || cpu >= CPU_THRESH || mem_qualifies
                || gpu_pids.contains(&pid);
            keep = true;
            inner.str("display", |b| build_display_into(b, comm, args, is_related))
        }).into_small_lossy();
        // Noise procs stay permanently hidden but keep their real ppid:
        // a zeroed stub severs the ancestry chain, so a busy non-noise
        // grandchild under a visible terminal could never be reached by
        // filter_with_children's traversal. cpu is zeroed — it's what
        // the child-visibility rule keys on — and the display is empty
        // (never rendered).
        if keep || noise {
            procs[i] = ProcInfo { pid, ppid,
                                  cpu: if keep { cpu } else { 0.0 },
                                  rss_kib: (rss_bytes >> 10) as u32,
                                  age_visible: ProcInfo::pack(age_secs, visible),
                                  gpu: None,
                                  display: if keep { display } else { FSmallStr::default() } };
        }
    }
    (procs, next)
}

#[cfg(target_os = "linux")]
fn parse_stat(stat: &[u8]) -> (&[u8], &[u8]) {
    let open = stat.iter().position(|&b| b == b'(').unwrap_or(0);
    // No ')' (truncated/garbled read) → empty comm and fields; the
    // caller skips the process. Defaulting to stat.len() would make
    // the `close + 1..` slice below panic-abort the monitor.
    let Some(close) = stat.iter().rposition(|&b| b == b')') else {
        return (&[], &[]);
    };
    let comm = &stat[open + 1..close];
    let after = stat[close + 1..]
        .iter().position(|b| !b.is_ascii_whitespace())
        .map_or(&[][..], |i| &stat[close + 1 + i..]);
    (comm, after)
}

/// Parse an ASCII unsigned integer from a byte slice. Returns None on empty
/// input or any non-digit byte. Avoids str::parse<u64>'s generic machinery.
#[cfg(target_os = "linux")]
fn parse_u64_ascii(s: &[u8]) -> Option<u64> {
    if s.is_empty() { return None; }
    let mut n: u64 = 0;
    for &b in s {
        let d = b.wrapping_sub(b'0');
        if d >= 10 { return None; }
        n = n.checked_mul(10)?.checked_add(d as u64)?;
    }
    Some(n)
}

// ── Load average (POSIX getloadavg — works on Linux and macOS) ───────────────

/// Linux: the kernel already prints /proc/loadavg as "%.2f %.2f %.2f
/// ..." — keep the first three fields verbatim instead of parsing all
/// three to f64 and re-rendering them, which kept an `F2` formatter
/// monomorph alive in the binary just for this line. Only the first
/// value is parsed (for the header color fraction).
#[cfg(target_os = "linux")]
struct LoadAvg { text: [u8; 24], len: u8, frac: f64 }
#[cfg(target_os = "linux")]
impl Put for LoadAvg {
    fn put<W: TinyWriter + ?Sized>(&self, w: &mut W) {
        if self.len == 0 { w.put_byte(b'?'); return; }
        w.put_bytes(&self.text[..self.len as usize]);
    }
}

/// macOS: loads arrive as doubles (vm_loadavg's fixed-point scaled),
/// so they're formatted with `f2`.
#[cfg(target_os = "macos")]
struct LoadAvg { loads: Option<[f64; 3]>, frac: f64 }
#[cfg(target_os = "macos")]
impl Put for LoadAvg {
    fn put<W: TinyWriter + ?Sized>(&self, w: &mut W) {
        let Some(l) = self.loads else { w.put_byte(b'?'); return; };
        twrite!(w, bytes::f2(l[0]), " ", bytes::f2(l[1]), " ", bytes::f2(l[2]));
    }
}

#[cfg(target_os = "linux")]
fn load_avg(num_cpus: f64) -> LoadAvg {
    // `getloadavg` is a libc function that reads /proc/loadavg for us
    // and formats the three floats. Do the read + parse ourselves to
    // keep off the TLS path and the libc dependency.
    // /proc/loadavg: "0.12 0.34 0.56 1/123 456\n".
    let mut buf = [0u8; 128];
    let Some(text) = read_small_file(c"/proc/loadavg", &mut buf) else {
        return LoadAvg { text: [0; 24], len: 0, frac: 0.0 };
    };
    // First three whitespace-separated fields, verbatim. Their end is
    // the 3rd space's position (fields are single-space separated).
    let mut spaces = 0usize;
    let mut end = 0usize;
    for (i, &b) in text.iter().enumerate() {
        if b == b' ' {
            spaces += 1;
            if spaces == 3 { end = i; break; }
        }
    }
    if end == 0 || end > 24 {
        return LoadAvg { text: [0; 24], len: 0, frac: 0.0 };
    }
    let mut out = [0u8; 24];
    out[..end].copy_from_slice(&text[..end]);
    // loads[0] alone drives the color fraction: "N.DD" — accumulate
    // the digits (skipping the dot) into hundredths. Avoids the f64
    // parser (Grisu + Dragon, ~10 KB of .text) and a slice::Split
    // monomorph.
    let mut hundredths: u64 = 0;
    for &b in &text[..end] {
        if b == b' ' { break; }
        let d = b.wrapping_sub(b'0');
        if d < 10 { hundredths = hundredths * 10 + d as u64; }
    }
    let load0 = hundredths as f64 * 0.01;
    LoadAvg { text: out, len: end as u8, frac: (load0 / num_cpus).min(1.0) }
}

#[cfg(not(target_os = "linux"))]
fn load_avg(num_cpus: f64) -> LoadAvg {
    match mac_sys::vm_loadavg() {
        Some(loads) => LoadAvg { loads: Some(loads), frac: (loads[0] / num_cpus).min(1.0) },
        None => LoadAvg { loads: None, frac: 0.0 },
    }
}

// ── Memory info ───────────────────────────────────────────────────────────────

struct MemInfo { used_bytes: u64, total_bytes: u64 }

impl MemInfo {
    fn frac(&self) -> f64 {
        if self.total_bytes > 0 { self.used_bytes as f64 / self.total_bytes as f64 } else { 0.0 }
    }
}

impl Put for MemInfo {
    fn put<W: TinyWriter + ?Sized>(&self, w: &mut W) {
        if self.total_bytes == 0 { w.put_byte(b'?'); return; }
        let gib = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;
        twrite!(w,
            f1_wide(gib(self.used_bytes), 0), "G / ",
            f1_wide(gib(self.total_bytes), 0), "G used");
    }
}

#[cfg(target_os = "linux")]
fn mem_info(frame: &mut Frame<'_>) -> MemInfo {
    // 4 KB /proc/meminfo read buffer — arena, not stack (>1 KB rule).
    frame.scope(|inner| {
        let mut buf: FBox<[u8; 4096]> = inner.alloc_zeroed("meminfo/buf");
        let Some(content) = read_small_file(c"/proc/meminfo", &mut buf[..]) else {
            return MemInfo { used_bytes: 0, total_bytes: 0 };
        };
        let mut total_kib = 0u64;
        let mut avail_kib = 0u64;
        let nth = |line: &[u8], n: usize| -> u64 {
            bytes::split_ascii_whitespace(line).nth(n).and_then(parse_u64_ascii).unwrap_or(0)
        };
        for line in bytes::split_lines(content) {
            if line.starts_with(b"MemTotal:") {
                total_kib = nth(line, 1);
            } else if line.starts_with(b"MemAvailable:") {
                avail_kib = nth(line, 1);
            }
        }
        MemInfo {
            used_bytes: total_kib.saturating_sub(avail_kib) * 1024,
            total_bytes: total_kib * 1024,
        }
    })
}

/// macOS: total RAM from sysctl HW_MEMSIZE; free+inactive from host_statistics64.
/// vm_statistics64 starts with 4 natural_t (u32) fields: free, active, inactive, wire.
/// HOST_VM_INFO64_COUNT = sizeof(vm_statistics64_data_t)/sizeof(int) = 38.
#[cfg(target_os = "macos")]
fn mem_info(_frame: &mut Frame<'_>) -> MemInfo {
    // Total physical memory: sysctl hw.memsize (CTL_HW=6, HW_MEMSIZE=24).
    let mut total: u64 = 0;
    let mut mib_hw = [6, 24];
    mac_sys::sysctl_read(&mut mib_hw, &mut total);
    if total == 0 { return MemInfo { used_bytes: 0, total_bytes: 0 }; }

    // VM page stats via host_statistics64(HOST_VM_INFO64). vm[0]=free,
    // vm[1]=active, vm[2]=inactive (all natural_t = u32).
    let page = mac_sys::hw_pagesize() as u64;
    let avail = match mac_sys::vm_statistics64() {
        Some(vm) => (vm[0] as u64 + vm[2] as u64) * page,  // free + inactive
        None => 0,
    };
    MemInfo {
        used_bytes: total.saturating_sub(avail),
        total_bytes: total,
    }
}

// ── Process classification ────────────────────────────────────────────────────

/// ASCII-case-insensitive substring check. Zero allocation.
// ── Byte-level string helpers ────────────────────────────────────────────────
//
// Everything we look at is either kernel /proc text (strictly ASCII per
// proc(5)) or process argv (may have non-UTF-8 bytes, which we only ever
// pass through to the terminal). We never need the general-Unicode rules
// `str::*` assumes, so these helpers take `&[u8]` throughout — no UTF-8
// decoder, no CharSearcher, no grapheme tables.

fn icontains(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() { return true; }
    if haystack.len() < needle.len() { return false; }
    haystack.windows(needle.len()).any(|w| ieq(w, needle))
}

/// File name component of a path — everything after the last `/`, or the
/// whole input if it has no separator.
fn basename(s: &[u8]) -> &[u8] {
    match s.iter().rposition(|&b| b == b'/') {
        Some(i) => &s[i + 1..],
        None => s,
    }
}

/// `haystack.rfind(needle)` but for byte slices.
fn rfind_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() { return Some(haystack.len()); }
    if haystack.len() < needle.len() { return None; }
    (0..=haystack.len() - needle.len()).rev()
        .find(|&i| &haystack[i..i + needle.len()] == needle)
}

/// macOS daemons that are perfectly fine to ignore unless they're actively
/// burning CPU (or GPU). Most run with hundreds of MB resident — Spotlight
/// indexes its caches; mediaanalysisd keeps photo embeddings hot — but
/// they're idle in steady state. Filtering them out via `is_noise` would
/// hide a runaway corespotlightd that's eating a core; passing the RSS
/// threshold lets every one of them on screen permanently. Threading them
/// through this predicate gates only the RSS path: CPU% (or GPU%) above
/// threshold still surfaces them.
#[cfg(target_os = "macos")]
fn is_idle_noise(comm: &[u8], args: &[&[u8]]) -> bool {
    let name = args.first().copied().map(basename).unwrap_or(comm);
    icontains(name, b"1password")                  // incl. its helpers
        || ieq(name, b"corespotlightd")
        || ieq(name, b"managedcorespotlightd")
        || ieq(name, b"spotlightknowledged")
        || ieq(name, b"mediaanalysisd")
        || ieq(name, b"photolibraryd")
        || ieq(name, b"photoanalysisd")
        || ieq(name, b"sharingd")
        || ieq(name, b"contactsd")
        || ieq(name, b"finder")
        || ieq(name, b"loginwindow")
        || ieq(name, b"netnewswire")
        || ieq(name, b"preview")
        || ieq(name, b"things3")
}
#[cfg(not(target_os = "macos"))]
fn is_idle_noise(_comm: &[u8], _args: &[&[u8]]) -> bool { false }

fn is_noise(comm: &[u8], args: &[&[u8]]) -> bool {
    // macOS daemons / UI processes that clutter the display.
    // Flat or-chain rather than `NOISE.iter().any(|s| ieq(name, s))` over
    // a `const NOISE: &[&[u8]]` — the latter emits one 16-byte
    // (ptr, len) slot per entry into __DATA,__const on Mach-O, each
    // with a rebase fixup. An or-chain materializes each literal's
    // (ptr, len) in registers at the call site (adrp+add+mov) with
    // zero static-storage fixups. Same byte cost in __TEXT,__const
    // for the string data; 16 × N bytes saved in __DATA,__const and
    // N rebase entries dropped.
    let name = args.first().copied().map(basename).unwrap_or(comm);
    icontains(name, b"chrome")                     // Chrome and helpers
        || icontains(name, b"widget")              // macOS UI widgets
        || name.starts_with(b"com.apple.")         // Bundle-ID names
        || name.starts_with(b"amazon-")              // AWS agent processes
        || ieq(name, b"ssm-session-worker")
        || ieq(name, b"spotlight")
        || ieq(name, b"springboard")
        || ieq(name, b"newstoday2")
        || ieq(name, b"newsscoringservice")
        || ieq(name, b"screentimeagent")
        || ieq(name, b"appleaccountd")
        || ieq(name, b"siriinferenced")
        || ieq(name, b"siriactionsd")
        || ieq(name, b"callservicesd")
        || ieq(name, b"calaccessd")
        || ieq(name, b"amsengagementd")
        || ieq(name, b"chronod")
        || ieq(name, b"remindd")
        || ieq(name, b"routined")
        || ieq(name, b"textunderstandingd")
        || ieq(name, b"corespeechd")
        || ieq(name, b"characterpalette")          // emoji picker's glyph cache
        // BlastDoor content-parsing sandboxes (Messages/IDS/Hubble/…):
        // one suffix check covers the whole family.
        || name.ends_with(b"BlastDoorService")
}

fn is_lean_or_lake(comm: &[u8], args: &[&[u8]]) -> bool {
    let base = args.first().copied().map(basename).unwrap_or(b"");
    matches!(comm, b"lean" | b"lake") || matches!(base, b"lean" | b"lake")
}

/// `"lean" | "lake" | "certificate"` substring check, inlined as an
/// or-chain to avoid a `const LEAN_KEYWORDS: &[&[u8]]`-style static
/// array (each entry would cost a __DATA,__const fixup on Mach-O).
fn has_lean_keyword(s: &[u8]) -> bool {
    icontains(s, b"lean") || icontains(s, b"lake") || icontains(s, b"certificate")
}

fn is_lean_related(comm: &[u8], args: &[&[u8]]) -> bool {
    has_lean_keyword(comm) || args.iter().take(3).any(|&a| has_lean_keyword(a))
}

fn is_interpreter(base: &[u8]) -> bool {
    base.get(..6).is_some_and(|p| ieq(p, b"python"))
        || ieq(base, b"ruby")
        || ieq(base, b"node")
        || ieq(base, b"perl")
        || ieq(base, b"lua")
}

/// Find the first non-flag argument after the interpreter, i.e. the script path.
/// Skips arguments to flags like -c (inline script), -m (module), -W, etc.
fn find_script_arg<'a>(args: &[&'a [u8]]) -> Option<&'a [u8]> {
    let mut skip_next = false;
    for &arg in args.iter().skip(1) {
        if skip_next { skip_next = false; continue; }
        if arg.starts_with(b"-") {
            // Value-taking interpreter flags: the short ones compare as
            // one byte each instead of five slice-literal memcmps.
            if (arg.len() == 2 && matches!(arg[1], b'c' | b'm' | b'W' | b'X' | b'Q'))
                || arg == b"--check-hash-based-pycs"
            { skip_next = true; }
            continue;
        }
        return Some(basename(arg));
    }
    None
}

fn find_cc_source<'a>(args: &[&'a [u8]]) -> Option<&'a [u8]> {
    // Or-chain instead of `const EXTS: &[&[u8]]` — see is_noise for
    // why: `const: &[&[u8]]` emits one rebase fixup per entry.
    let has_cxx_ext = |a: &[u8]| {
        a.ends_with(b".cc")
            || a.ends_with(b".cpp")
            || a.ends_with(b".cxx")
            || a.ends_with(b".c")
            || a.ends_with(b".C")
            || a.ends_with(b".c++")
    };
    for &arg in args {
        if arg.starts_with(b"-") { continue }
        if has_cxx_ext(arg) {
            return Some(basename(arg));
        }
    }
    None
}

fn build_display_into(out: &mut FBuilder<'_, '_, u8>,
                      comm: &[u8], args: &[&[u8]], is_related: bool) {
    // Compiler processes: show "cc1plus foo.cc" instead of the full path.
    if matches!(comm, b"cc1plus" | b"cc1") {
        if let Some(src) = find_cc_source(args) {
            out.extend_from_slice(comm);
            out.extend_from_slice(b" ");
            out.extend_from_slice(src);
            return;
        }
    }
    if !is_related {
        if let Some(&arg0) = args.first() {
            let base = basename(arg0);
            // For interpreters (python, ruby, node, ...), show the script name instead.
            if is_interpreter(base) {
                if let Some(script) = find_script_arg(args) {
                    out.extend_from_slice(script);
                    return;
                }
            }
            out.extend_from_slice(base);
            return;
        }
        out.extend_from_slice(comm);
        return;
    }
    out.extend_from_slice(lean_label(args));
    if let Some(f) = find_lean_file(args) {
        out.extend_from_slice(b" ");
        out.extend_from_slice(f);
    }
}

fn lean_label<'a>(args: &[&'a [u8]]) -> &'a [u8] {
    let any = |needle: &[u8]| args.iter().any(|a| icontains(a, needle));
    if any(b"--worker") { return b"lean"; }
    if any(b"--server") { return b"lean --server"; }
    if any(b"lake") && any(b"serve") { return b"lake serve"; }
    if any(b"lean.rs") || any(b"debug/lean") { return b"lean-mcp"; }
    if any(b"certificate") { return b"certificate"; }
    for &arg in args {
        if has_lean_keyword(arg) {
            return basename(arg);
        }
    }
    args.first().copied().map(basename).unwrap_or(b"?")
}

fn find_lean_file<'a>(args: &[&'a [u8]]) -> Option<&'a [u8]> {
    for &arg in args {
        let path = arg.strip_prefix(b"file://").unwrap_or(arg);
        if path.ends_with(b".lean") {
            // Show path relative to the project root (look for /AKS/ component).
            return Some(match rfind_bytes(path, b"/AKS/") {
                Some(idx) => &path[idx + 1..],   // "AKS/Graph/Regular.lean"
                None => basename(path),
            });
        }
        // Module names like AKS.Graph.Regular — Lean identifiers are ASCII,
        // so a first-byte ASCII uppercase check is sufficient.
        if !arg.starts_with(b"-") && !arg.iter().any(|&b| b == b'/') {
            if let Some(dot) = arg.iter().position(|&b| b == b'.') {
                if arg[..dot].first().is_some_and(|b| b.is_ascii_uppercase()) {
                    return Some(arg);
                }
            }
        }
    }
    None
}

// ── Formatting helpers ────────────────────────────────────────────────────────

/// Produces exactly 6 chars (right-aligned number + unit letter); no allocation.
struct FormatAge(u32);
fn format_age(secs: u32) -> FormatAge { FormatAge(secs) }
impl Put for FormatAge {
    fn put<W: TinyWriter + ?Sized>(&self, w: &mut W) {
        let secs = self.0;
        if secs >= 86400 {
            twrite!(w, pad_right(secs / 86400, 2), "d", pad_zero((secs % 86400) / 3600, 2), "h");
        } else if secs >= 3600 {
            twrite!(w, pad_right(secs / 3600, 2), "h", pad_zero((secs % 3600) / 60, 2), "m");
        } else if secs >= 60 {
            twrite!(w, pad_right(secs / 60, 2), "m", pad_zero(secs % 60, 2), "s");
        } else {
            twrite!(w, pad_right(secs, 5), "s");
        }
    }
}

/// Produces exactly 6 chars (right-aligned number + unit letter); no allocation.
/// Integer-only arithmetic — avoids pulling in core::fmt::float (Grisu +
/// Dragon, ~10 KB). Input is KiB so the threshold shifts are biased by 10.
struct FormatMem(u32);
fn format_mem(kib: u32) -> FormatMem { FormatMem(kib) }
impl Put for FormatMem {
    fn put<W: TinyWriter + ?Sized>(&self, w: &mut W) {
        let kib = self.0;
        if kib >= 1 << 20 {  // 1 GiB
            // Cast to u64 for the * 10 — overflows u32 above ~429 GiB.
            let tenths = (kib as u64 * 10 + (1 << 19)) >> 20;
            twrite!(w, pad_right((tenths / 10) as u32, 3), ".", u32d((tenths % 10) as u32), "G");
        } else if kib >= 1 << 10 {
            twrite!(w, pad_right(kib >> 10, 5), "M");
        } else {
            twrite!(w, pad_right(kib, 5), "K");
        }
    }
}

struct Truncate<'a> { s: &'a [u8], max: usize }
fn truncate(s: &[u8], max: usize) -> Truncate<'_> { Truncate { s, max } }
impl Put for Truncate<'_> {
    fn put<W: TinyWriter + ?Sized>(&self, w: &mut W) {
        if self.s.len() <= self.max { return self.s.put(w); }
        // Back up from the raw byte offset to a UTF-8 boundary (first byte
        // whose top bits aren't 10xxxxxx) so we don't split a codepoint.
        let target = if self.max > 3 { self.max - 3 } else { self.max };
        let mut n = target;
        while n > 0 && (self.s[n] & 0xC0) == 0x80 { n -= 1; }
        self.s[..n].put(w);
        if self.max > 3 { w.put_bytes(b"..."); }
    }
}

/// Red ← memory pressure; blue ← CPU pressure.
///
/// All but `cpu` are integers, so we stay in integer arithmetic as much
/// as possible. This avoids the NaN-correcting `f64::max`/`min` dance
/// (~48 B each) and the SSE2 u64→f64 split-high/low magic that the
/// compiler emits for the `gpu_total_mib as f64` cast. `cpu` is the
/// only genuinely-float input and goes through the one remaining float
/// pipeline.
fn row_color(cpu: f32, rss_kib: u32, gpu: &Option<GpuUsage>, gpu_total_mib: u64) -> (u8, u8, u8) {
    // Redness saturates at 4 GiB of RSS (4 << 20 KiB). One formula for
    // both platforms; the `mut` is only exercised on Linux (per-process
    // GPU memory feeds redness below), so mac allows the unused_mut.
    #[cfg_attr(target_os = "macos", allow(unused_mut))]
    let mut r = (rss_kib as u64 * 255 / (4u64 << 20)).min(255) as u8;
    // Blueness saturates at 100% CPU. Truncate-to-u32 + integer
    // scale keeps us off the f64 pipeline (cvt/divsd/mulsd/maxsd/
    // cvttsd2si, ~40 B); CPU inputs come from integer /proc/stat
    // ticks so the sub-1% precision loss is below visual threshold.
    let mut b = ((cpu as u32).min(100) * 255 / 100) as u8;
    if let Some(g) = gpu {
        // GPU memory contributes to redness on Linux only — Apple Silicon's
        // unified memory is already counted in RSS (and reflected in `r`
        // above), so a separate channel would double-count.
        #[cfg(not(target_os = "macos"))]
        if gpu_total_mib > 0 {
            let gpu_r = (g.mem_mib.get() as u64 * 255 / gpu_total_mib).min(255) as u8;
            r = r.max(gpu_r);
        }
        #[cfg(target_os = "macos")]
        let _ = gpu_total_mib;
        let gpu_b = (g.sm_pct as u64 * 255 / 100).min(255) as u8;
        b = b.max(gpu_b);
    }
    (r, 0, b)
}

fn frac_color(red_frac: f64, blue_frac: f64) -> (u8, u8, u8) {
    ((red_frac.min(1.0) * 255.0) as u8, 0, (blue_frac.min(1.0) * 255.0) as u8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::FSmallStr;

    fn proc_info(pid: u32, ppid: u32, cpu: f32, visible: bool) -> ProcInfo<'static> {
        ProcInfo {
            pid, ppid, cpu, rss_kib: 1024,
            age_visible: ProcInfo::pack(100, visible),
            gpu: None,
            display: FSmallStr::default(),
        }
    }

    #[test]
    fn packed_round_trips_at_boundaries() {
        for total in [0u64, 1 << Packed::SHIFT, u64::MAX] {
            let mut p = Packed::new(false, total);
            assert!(!p.visible());
            assert_eq!(p.cpu_bits(), ((total >> Packed::SHIFT) as u32) & Packed::CPU_MASK);
            let bits = p.cpu_bits();
            p.set_visible();
            assert!(p.visible());
            assert_eq!(p.cpu_bits(), bits, "set_visible must not disturb cpu bits");
        }
        assert!(Packed::new(true, 7).visible());
    }

    #[test]
    fn truncate_edges() {
        let put = |s: &[u8], max: usize| {
            let mut v = Vec::new();
            truncate(s, max).put(&mut v);
            v
        };
        assert_eq!(put(b"hello", 5), b"hello");        // exact fit
        assert_eq!(put(b"hello!", 5), b"he...");       // ellipsis within max
        assert_eq!(put(b"hello", 3), b"hel");          // too narrow for "..."
        assert_eq!(put(b"hello", 0), b"");
        // Never splits a UTF-8 codepoint: 🌳 is 4 bytes at offset 1.
        let v = put("a🌳bc".as_bytes(), 6);
        assert_eq!(v, b"a...");
    }

    #[test]
    fn compute_cpu_ignores_reused_pid() {
        // prev knew pid 42 with a large counter; a young process now
        // owns the pid. Matching them would wrap to a huge delta.
        let prev_entry = Some(Packed::new(false, u32::MAX as u64));
        let young = compute_cpu(prev_entry, /*age_secs=*/0, 100, 100.0, 2.0);
        assert_eq!(young, 0.0, "process younger than dt cannot match prev");
        let old = compute_cpu(prev_entry, /*age_secs=*/60, u32::MAX as u64 + (200 << Packed::SHIFT), 100.0, 2.0);
        assert!(old > 0.0, "same-identity delta should survive");
    }

    /// A hidden intermediate (e.g. noise) with a real ppid must not
    /// sever the chain: the busy grandchild becomes visible and is
    /// reparented to the nearest visible ancestor. This is the
    /// regression test for noise stubs that used to keep ppid = 0.
    #[test]
    fn filter_reaches_through_hidden_parent() {
        let _g = crate::arena::test_lock();
        crate::arena::scope(|frame| {
            let mut procs: FVec<ProcInfo> = frame.vec("t/procs", 8);
            let _ = procs.push(proc_info(100, 1, 6.0, true));    // visible terminal
            let _ = procs.push(proc_info(200, 100, 0.0, false)); // hidden noise
            let _ = procs.push(proc_info(300, 200, 2.0, false)); // busy grandchild
            let mut next: FVec<(u32, Packed)> = frame.vec("t/next", 8);
            for p in procs.iter() { let _ = next.push((p.pid, Packed::new(false, 0))); }
            let prev = crate::map::Map::new(&[]);
            filter_with_children(frame, &mut procs, &prev, &mut next);
            let pids: Vec<(u32, u32)> = procs.iter().map(|p| (p.pid, p.ppid)).collect();
            assert_eq!(pids, vec![(100, 1), (300, 100)],
                       "grandchild visible and reparented to the terminal");
        });
    }

    #[test]
    fn helper_name_match() {
        assert!(is_helper_of(b"Code Helper", b"Code"));
        assert!(is_helper_of(b"Code Helper (Renderer)", b"Code"));
        assert!(!is_helper_of(b"Code", b"Code"));
        assert!(!is_helper_of(b"CodeHelper", b"Code"));
        assert!(!is_helper_of(b"Code Helper", b"Code Helper (Plugin)"));
        assert!(!is_helper_of(b"claude", b"Code"));
        assert!(!is_helper_of(b"Claude Helper (GPU)", b"claude"));
    }

    /// Electron helper subtrees fold into their app's row by name (no
    /// per-app list): descendants named "<App> Helper*" sum cpu/rss/gpu
    /// into the root and disappear, visible or not. A non-helper
    /// descendant (a `claude` CLI under a plugin helper) never folds:
    /// it keeps its own subtree, and the BFS reparents it to the app
    /// root because its helper parent is hidden by the time the BFS
    /// runs.
    #[test]
    fn filter_coalesces_helper_subtrees() {
        let _g = crate::arena::test_lock();
        crate::arena::scope(|frame| {
            #[cfg(target_os = "macos")]
            let gpu = Some(GpuUsage { sm_pct: 30 });
            #[cfg(not(target_os = "macos"))]
            let gpu = Some(GpuUsage {
                sm_pct: 30,
                mem_mib: core::num::NonZeroU32::new(64).unwrap(),
            });

            let code = frame.str("t/n1", |b| b.extend_from_slice(b"Code")).into_small();
            let renderer =
                frame.str("t/n2", |b| b.extend_from_slice(b"Code Helper (Renderer)")).into_small();
            let plugin =
                frame.str("t/n3", |b| b.extend_from_slice(b"Code Helper (Plugin)")).into_small();
            let cli = frame.str("t/n4", |b| b.extend_from_slice(b"claude")).into_small();
            let mut procs: FVec<ProcInfo> = frame.vec("t/procs", 8);
            let _ = procs.push(ProcInfo { pid: 100, ppid: 1, cpu: 1.0, rss_kib: 100,
                age_visible: ProcInfo::pack(10, true), gpu: None, display: code });
            let _ = procs.push(ProcInfo { pid: 200, ppid: 100, cpu: 3.0, rss_kib: 1024,
                age_visible: ProcInfo::pack(10, false), gpu, display: renderer });
            let _ = procs.push(ProcInfo { pid: 300, ppid: 100, cpu: 2.0, rss_kib: 1024,
                age_visible: ProcInfo::pack(10, false), gpu: None, display: plugin });
            let _ = procs.push(ProcInfo { pid: 400, ppid: 300, cpu: 6.0, rss_kib: 100,
                age_visible: ProcInfo::pack(10, true), gpu: None, display: cli });
            let _ = procs.push(proc_info(500, 400, 6.0, true));   // CLI child stays

            let mut next: FVec<(u32, Packed)> = frame.vec("t/next", 8);
            for p in procs.iter() { let _ = next.push((p.pid, Packed::new(false, 0))); }
            let prev = crate::map::Map::new(&[]);
            filter_with_children(frame, &mut procs, &prev, &mut next);

            let pids: Vec<(u32, u32)> = procs.iter().map(|p| (p.pid, p.ppid)).collect();
            assert_eq!(pids, vec![(100, 1), (400, 100), (500, 400)],
                       "helpers folded; CLI reparented to the surfaced root");
            let root = &procs.as_slice()[0];
            assert_eq!(root.cpu, 6.0, "1 + 3 + 2");
            assert_eq!(root.rss_kib, 100 + 2 * 1024);
            let g = root.gpu.as_ref().expect("gpu folded up");
            assert_eq!(g.sm_pct, 30);
            #[cfg(not(target_os = "macos"))]
            assert_eq!(g.mem_mib.get(), 64);
        });
    }

    /// A ppid cycle (pid-reuse race during the /proc scan) must not
    /// hang the traversal — this looped forever before the
    /// enqueue-once guard.
    #[test]
    fn filter_survives_ppid_cycle() {
        let _g = crate::arena::test_lock();
        crate::arena::scope(|frame| {
            let mut procs: FVec<ProcInfo> = frame.vec("t/procs", 8);
            let _ = procs.push(proc_info(10, 20, 9.0, true)); // visible, in-cycle
            let _ = procs.push(proc_info(20, 10, 2.0, false)); // cycle partner
            let mut next: FVec<(u32, Packed)> = frame.vec("t/next", 8);
            for p in procs.iter() { let _ = next.push((p.pid, Packed::new(false, 0))); }
            let prev = crate::map::Map::new(&[]);
            filter_with_children(frame, &mut procs, &prev, &mut next);
            assert!(procs.iter().any(|p| p.pid == 10), "seed survives");
        });
    }

    /// More visible roots than terminal rows must not overflow the
    /// frame (it scrolled the header off every tick).
    #[test]
    fn render_tree_caps_roots_at_max_rows() {
        let procs: Vec<ProcInfo> = (0..6).map(|i| proc_info(100 + i, 1, 6.0, true)).collect();
        let tree: Vec<(Indent, u32)> =
            (0..6).map(|i| (Indent { depth: 0, mask: 0 }, i as u32)).collect();
        let roots: Vec<usize> = (0..6).collect();
        let count_rows = |max_rows: usize| {
            let mut out = Vec::new();
            render_tree(&mut out, &procs, &tree, &roots, max_rows, 80, 0, false);
            out.windows(FG_CLOSE_EOL.len()).filter(|w| *w == FG_CLOSE_EOL).count()
        };
        assert_eq!(count_rows(3), 3, "capped at the terminal height");
        assert_eq!(count_rows(6), 6, "exact fit");
        assert_eq!(count_rows(10), 6);
        assert_eq!(count_rows(0), 0);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_stat_without_close_paren_returns_empty() {
        assert_eq!(parse_stat(b"123 (comm-with-no-close"), (&[][..], &[][..]));
        let (comm, after) = parse_stat(b"123 (a b) R 4 5");
        assert_eq!(comm, b"a b");
        assert_eq!(after, b"R 4 5");
    }

    /// Indent glyphs past depth 32 must not wrap the mask shift
    /// (debug builds panicked; release read stale bits).
    #[test]
    fn indent_survives_depth_past_mask_width() {
        let mut v = Vec::new();
        Indent { depth: 40, mask: u32::MAX }.put(&mut v);
        assert!(!v.is_empty());
    }

    /// `macos_proc_args_into` must yield `[argv0, argv1, ...]` with the
    /// KERN_PROCARGS2 blob's leading exec_path skipped — the Linux
    /// layout every classifier assumes. Spawn a child whose argv[0]
    /// differs from its executable path so the two are distinguishable
    /// (a plain spawn has argv[0] == exec_path and can't catch an
    /// off-by-one).
    #[cfg(target_os = "macos")]
    #[test]
    fn macos_args_skip_exec_path() {
        use std::os::unix::process::CommandExt;
        let mut child = std::process::Command::new("/bin/sleep")
            .arg0("ltop-argv0-probe")
            .arg("30")
            .spawn()
            .expect("spawn sleep");
        let pid = child.id();
        let cap = super::macos_args_blob_size(pid).expect("KERN_PROCARGS2 size");
        let mut buf = vec![0u8; cap];
        let mut out: [&[u8]; 64] = [&[][..]; 64];
        let n = super::macos_proc_args_into(pid, &mut buf, &mut out);
        let got: Vec<Vec<u8>> = out[..n].iter().map(|s| s.to_vec()).collect();
        child.kill().ok();
        child.wait().ok();
        assert_eq!(got, vec![b"ltop-argv0-probe".to_vec(), b"30".to_vec()]);
    }
}
