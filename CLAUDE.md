# ltop 🌳

Minimal process monitor (top replacement) in Rust. For monitoring builds (Lean/Lake, C++), GPU experiments, and similar workloads. Per-platform footprint table in README's Footprint section; `tests/footprint.rs` enforces it.

The 🌳 reads two ways, both load-bearing:
- **Process tree**: the `└─`/`├─`/`│` DFS tree in the COMMAND column is the thing `ltop` does that `top` doesn't.
- **Bonsai**: the project is shaped by repeated pruning. Zero heap allocations, zero libc code in production (Linux), one LC_LOAD_DYLIB on mac. Each step trims something (a libc dep, a stdlib feature, a generic monomorphisation, a Vec, a malloc chain, a committed page) while user-visible function stays the same. Most commits in the log are a bud snipped, not a feature grafted on.

Keep that spirit. When a change adds bytes or RSS, the bar is "worth the bytes," not "it works." When a change can remove something instead, do that.

## Research before implementing uncertain subsystems

When about to write code against a subsystem you're not sure about — undocumented kernel ABIs, obscure libc semantics, platform-specific FFI, low-level codegen details — **do the research first, before writing the code**. Read primary sources (xnu, linux kernel, libc headers/source, rustc internals) via `WebFetch` / `WebSearch`; pin down every numeric constant, calling convention, argument aliasing, and error semantic. Write down what you find in a markdown doc if the work is multi-phase.

The cost of guessing and iterating via CI (or locally with long edit-build-test loops) is much higher than the cost of 20 minutes of reading. "This silently returned 0 bytes" is often 5 primary-source lines away from "because the `pid` slot carries the `type` selector when `callnum == LISTPIDS`." A solid research pass also lets the model recognise gotchas it wouldn't otherwise have thought to check — deprecated trap indices, top-bit option flags, struct fields that look ignorable but aren't.

Use judgement: not everything needs research. A pure Rust refactor, a clear stdlib API change, or a typo fix doesn't. But anything that's going to talk to a kernel interface or an ABI boundary probably does. `mac-min.md` is an example of what this looks like end-to-end for a genuinely hard case.

Corollary: when debugging a failure in such a subsystem, **first read the authoritative source for the behavior** before iterating. If the test says "returns 0" and the docs say "returns bytes written", go find the actual kernel dispatch code — don't just try different arg orderings.

## Running

```bash
cargo ltop               # minimal static build + run; args after `--`
cargo ltop -- --once
cargo run --release      # dev (glibc dynamic)
```

`cargo ltop` is the alias in `.cargo/config.toml`: minimal static build, std rebuilt via `-Z build-std`, target = custom spec at `.cargo/static-linux-{x86_64,aarch64}.json`. `rust-toolchain.toml` pins nightly + `rust-src`; rustup auto-installs. Zero per-host setup.

## Structure

Standard cargo layout. `Cargo.toml` at root, source in `src/`. Only runtime dep is `libc` for its constant/type definitions on glibc test builds (production links no libc on Linux, only libSystem on mac). `build.rs` probes `libnvidia-ml.so.1` and emits its link arg for test builds only.

`#![cfg_attr(not(test), no_std)]` + `no_main`: production uses `core` directly. Test builds get std for the harness + `gpu/nvml.rs`.

- `src/main.rs` — proc collection / filtering / tree layout / rendering; provides `extern "C" fn main` in non-test builds.
- `src/lib.rs` — re-exports so doc-tests (incl. arena `compile_fail`) run under `cargo test`.
- `src/arena.rs` — bump allocator with generative-lifetime scoping. Dedicated section below.
- `src/twrite.rs` — `Put` + `TinyWriter` + `twrite!`. Replaces `write!`/`Display`/`core::fmt::Arguments` on the render path.
- `src/bytes.rs` — `Put`-impl helpers (`U32d`, `PadRight`/`PadLeft`/`PadZero`, `F1Wide`, `F2`, `Repeat`) + byte-slice iterators (`split_ascii_whitespace`, `split_lines`, `ieq`).
- `src/sort.rs` — in-place heapsort by u32-at-offset-0 keys; replaces musl `qsort` (~600 B; word-wise swap drops `compiler_builtins::memcpy`).
- `src/zeroable.rs` — marker trait making `Frame::alloc_zeroed` safe at the call site.
- `src/platform.rs` — libc-backed std replacements (`Instant`, `PidDir`, `sleep`, `getpid`), `#[global_allocator]` (`Abort`), `#[panic_handler]`, `rust_eh_personality` stub.
- `src/start.rs` — custom `_start` for libc-free Linux (relocates SP, walks `madvise(DONTNEED)` to find the stack VMA, tail-jumps to `ltop_main`).
- `src/syscall.rs` — raw `svc #0x80` / `syscall` wrappers, Linux + macOS branches.
- `src/sandbox.rs` — Linux-only seccomp-bpf install. Hand-encoded `sock_filter[]` allowing only the tick-loop syscall set (`read`/`write`/`close`/`ioctl`/`madvise`/`nanosleep`/`clock_gettime`/`exit_group`/`openat`/`getdents64`/`restart_syscall`/`rt_sigreturn`); deny action is `KILL_PROCESS` by default, `RET_TRAP` + SIGSYS-handler-prints-syscall-nr under the `sandbox-trap` cargo feature. Installed in `run()` after `gpu::init` and before the tick loop, gated on `cfg(libc_free)` (glibc-dynamic builds aren't allowlisted). Argument filtering (`write` fd, `openat` flags, `ioctl` cmd) and Landlock path restriction are implemented — sandbox.rs's module doc is the authoritative description, including its remaining future work. macOS has no in-process equivalent that doesn't break the no-libSystem-symbols invariant; a sandbox-exec wrapper would be the equivalent.
- `src/mac_sys.rs` — Darwin-only: hand-rolled Mach traps, `mach_msg2` MIG, IOKit calls, `unmap_idle_state`. `cfg(target_os = "macos")`.
- `src/osbinary.rs` — pure-byte decoder for `OSSerializeBinary` (xnu's IOKit reply format). macOS-only.
- `src/tty.rs` — `enter_raw_mode`, `term_size`, `write_stdout` (raw `write(2)` from a caller-supplied `FStr`), `read_one_stdin`. Render buffer lives per-tick in `tick.span("tty/stdout", …)`.
- `src/map.rs` — u32-keyed flat Map/Set: sort+dedup an arena FVec, binary search to read.
- `src/gpu/{mod,rm,nvml,agx}.rs` — shared GPU types + per-platform backends. `rm.rs` talks to `/dev/nvidiactl` via raw RM ioctls (no libnvidia-ml/libcuda); `nvml.rs` is reference-only under `cfg(all(test, has_nvml))`; `agx.rs` is macOS AGX through `mac_sys` MIG.

## Build profiles

- `release` (default) — glibc dynamic, used for `cargo test` (NVML linked). ~640 KB.
- `min` — `opt-level="z"` + LTO + 1 codegen unit + `panic=abort` + strip. Invoked via `cargo ltop`. The alias also carries `-Z build-std=core,alloc,std,panic_abort -Z build-std-features=compiler-builtins-mem`; `[target.static-linux-*]` adds `-Cpanic=immediate-abort`, `-Cforce-unwind-tables=no`, `-Crelocation-model=static`, `-Clink-arg=-no-pie`, `-Clink-arg=-nostartfiles`, `--cfg=libc_free`. Target-scoped not profile-scoped so build scripts compile for the host normally. Lesson 6 has the rationale chain.
- `min-dbg` — same as `min` with `strip=false` for `nm`/`objdump`.
- `min-stack` — same as `min` with `strip=false` + `debug=2`. Built by `cargo stack` so `tools/stack-svg` / `tools/inline-bytes` can attribute frames + `.text` bytes back to source.

## Testing

`cargo test --release` runs arena/FVec/... unit tests plus four end-to-end GPU tests comparing RM vs NVML (`rm_matches_nvml` asserts n_gpus, total_mib, util, per-PID mem_mib/sm_pct within tolerance). GPU tests are `#[cfg_attr(not(has_gpu), ignore)]`; `build.rs` probes `/dev/nvidiactl`. On non-GPU hosts force with `cargo test -- --ignored`.

To diff TUI output after a refactor: `cargo ltop -- --once` prints one frame and exits. For more, `script -q -c "timeout 4.5 target/release/ltop" out.txt` captures three (first tick at 300 ms for CPU%-delta, then 2 s steady-state).

For manual tree/color/elision checks, `tools/spin` spawns N children burning configurable CPU (and GPU on mac) fractions — e.g. `cargo run --release -p spin -- --children 8 --cpu 0.5` in one terminal, `cargo ltop` in another.

## Architecture notes

- `collect_procs` (per-platform) writes non-noise processes into a caller-supplied `FVec<ProcInfo>`; `filter_with_children` BFS-marks visibility and compacts via `FVec::retain_in_place`. Both platforms must stay in sync.
- Per-type display names (lean, cc1plus, interpreters) live in `build_display_into` — add new cases there.
- Tree rendering + vertical-overflow logic in `render_tree`, via `emit_proc_row` + per-group child budgets.
- GPU driver state (`rm::State` from `gpu::init(init)`) allocates once at the outermost scope and builds an exact-sized `FSpan<Gpu>`; no-GPU hosts get an empty `FSpan` and downstream ops are natural no-ops. Per-tick: the RM ioctl buffer in a `rm::populate` sub-scope (reclaimed on return; rm.rs's populate doc is the single source of truth for its size) and per-process entries in a tick-scope FVec.

## Arena allocator

`src/arena.rs`: one 512 KB static `.bss` arena, bump-allocated, single-threaded, scope-rewinding via `ScopeGuard`. Four public container types (sizes include the brand `PhantomData`):

- **`FBox<'id, T>`** (4 B): owning handle to one `T`. Built via `frame.alloc_zeroed("label")` (requires `Zeroable`) or `frame.alloc(value)`.
- **`FSpan<'id, T>`** (8 B): frozen slice. `Deref<Target=[T]>`. Built via `frame.span("label", |b| ...)`, `frame.compact("label", |sub| ...)`, `FReserve::freeze`, or `frame.empty()`. `FStr = FSpan<u8>`.
- **`FVec<'id, T>`** (12 B): mutable arena vector, `push`/`pop`/`clear`/`retain_in_place`. `frame.vec("label", capacity)` — capacity fixed, length `0..=capacity`.
- **`FReserve<'a, 'id, T>`** (12 B): build-then-freeze. Push, then `.freeze()` trims the unused tail into an `FSpan`; drop unfreeze rewinds. The `&mut Frame` lockout makes "still at top of bump" structural.
- Plus `FSmallStr` (length-prefixed inline-or-arena byte string for `ProcInfo.display`), `FBuilder` (in-progress contents of a `span`/`compact`), `FNest` (n-bucket cluster, used by `freeze_map`/`freeze_set`).

Every allocation carries a `Label` (string literal, ZST when `arena-trace` feature is off). The `'id` brand from the `for<'id> FnOnce(&mut Frame<'id>)` HRTB on `arena::scope` is invariant; leaking out of scope is a compile error verified by `compile_fail` doc-tests. Frame methods split by mutability: `alloc_*` and `vec` take `&self` so cross-tick buffers coexist with sub-scopes; `span` / `reserve` / `compact` / `replace2` / `nest` / `scope` take `&mut self`, locking out other allocations during the build (makes freeze semantics sound). `compact` and `replace2` run their build closures under a *fresh* sub-brand and rebrand the returned spans internally, so stashing a build-scope allocation past the rewind is a compile error too (`compile_fail`-tested).

The production tick uses `init.replace2` to rotate two adjacent cross-tick `FSpan`s in-place: the closure receives `(tick, gpu_prev, prev)`, builds replacements above its working set, and `replace2` slides them down into the original slots so cross-tick state doesn't bump the high-water mark each iteration. Inside that closure, transient state (`procs: FVec`, `out: FStr` via `tick.span("tty/stdout", …)` then `tty::write_stdout(&out)`) lives until `replace2` returns.

`ScopeGuard::drop` and `FReserve::drop` unwind paths are elided in production (`panic = abort`); they're load-bearing under `cargo test` (std + unwinding) so a panicking closure rewinds cleanly.

## Test serialisation

One global arena, one offset cell — `cargo test`'s default parallelism would race. Tests that allocate grab `arena::test_lock()` (`cfg(test)`-only `Mutex`); that includes the `cfg(has_gpu)` tests in `gpu/rm.rs` since `rm::init(frame)` allocates.

## Bytes everywhere (not `&str`)

`comm`, argv, and arena-stored display strings are `&[u8]` / `FVec<u8>`, not `&str` / `String`. We do byte-level ASCII compare and terminal output — no case folding, locale, or graphemes. `&str` paid for CharSearcher / TwoWaySearcher / UTF-8 decoder / Debug grapheme tables we never used.

The render path doesn't go through `core::fmt` at all. `src/twrite.rs` defines `Put` + `TinyWriter` + the `twrite!` macro: each call site monomorphises to direct `put_bytes`/`put_byte`, no `Formatter::pad` (~900 B), no `Arguments` staging. Rendered types (`FormatMem`, `FormatAge`, `Truncate`, `Fg`, `Indent`) implement `Put`; `&str` / `&[u8]` / integers / `char` have stock impls. `write!(out, "{:>5}", n)` becomes `twrite!(out, pad_right(n, 5))`. UTF-8 validation for arbitrary bytes (the `cmd` column) lives in `<[u8] as Put>::put` — ~100 B `match`-driven Unicode Table 3-7, replaces `from_utf8` (~470 B) + `from_utf8_lossy`. `src/bytes.rs` ships the `Put` helper kit (`U32d`, `PadRight`/`PadLeft`/`PadZero`, `F1Wide`, `F2`, `Repeat`) plus byte-slice iterators (`split_ascii_whitespace`, `split_lines`, `ieq`).

## Style & allocation philosophy

Zero heap allocations in production: the `#[global_allocator]` is `Abort`; any `alloc`/`dealloc` terminates. Everything routes through the arena or the stack. Don't reintroduce `Vec`/`String`/`Box` without a concrete reason; if you do need heap, re-enable the allocator first.

Patterns worth preserving:

- **`Put` impls + `twrite!`, not `Display` / `fmt::from_fn`.** See "Bytes everywhere" — `core::fmt` is off the render path.
- **Bake fixed widths into the writer.** `format_mem` ends with `pad_right(kib >> 10, 5)` + a unit byte; `pad_right` does its own digit emission, no `f.pad` double-pass.
- **Borrow rather than own.** `args: &[&[u8]]` borrowed from a per-iteration cmdline buffer beats `Vec<Vec<u8>>`. `basename`, `find_script_arg`, `lean_label`, `find_lean_file` all return `&[u8]`.
- **Compact value types over rendered bytes.** `Indent { depth: u8, mask: u32 }` (8 B `Copy`) replaces `String` tree prefixes; renders via `Put`.
- **Single reusable buffers per tick.** `collect_procs::stat_raw` is one `FBox<[u8; 4096]>` per tick; `path` / inner per-pid `FVec<u8>`s allocate fresh in the tick scope. Cross-tick state (`prev`, `gpu_prev`) is rotated as `FSpan`s via `init.replace2`, not swapped FVecs.
- **Stack arrays for short-lived small collections.** `args: [&[u8]; 64]` inside the per-pid loop beats a reused `FVec` — stack alloc is free and inner-slice lifetimes auto-scope to the iteration.
- **>1 KB goes on the arena, not the stack.** Anything bigger than 1 KB belongs in an arena sub-scope, never a bare `[T; N]` on the stack. The arena gives `arena-trace` visibility, scope-exit rewind, and `madvise(DONTNEED)` on uncommitted pages — invisible for stack arrays. `tools/stack-svg` flags >1 KB rects inside real frames as the first place to look. Counter-cases are rare (usually fixed kernel-ABI payloads that must be on the exact frame).
- **Pack bools into a u32 field.** A `bool` field costs 1 B + 3 B padding; stealing a bit from a u32 costs nothing. See `ProcInfo.age_visible` (bit 31 = visible, bits 0..31 = seconds) with `pack()` / `visible()` / `set_visible()` methods.
- **Comments describe HEAD, not the diff.** A comment that says "X NOT Y because…" or "fixed from Y to X" tells the change's story; six months later nobody knows what Y was and the comment is dead weight. The diff explanation belongs in the commit message; the source file documents what's there. Often the right HEAD-comment is *no comment at all* — surrounding structure (per-arch submodules, named consts) already documents the fact.

Break these when the data must outlive the producer (e.g. `ProcInfo.display` in the arena across ticks).

## Binary-size lessons

The production binary is a few tens of KB (`tests/footprint.rs` enforces the exact per-target thresholds; README's Footprint section has the table). Lessons that generalise — each is a rule, not a story:

1. **Direct syscalls beat std when behaviour matches.** `available_parallelism` (~10 KB of cgroup parsers) → `sysconf(_SC_NPROCESSORS_ONLN)`. `read_to_string` (~1.5 KB) → `open`+`read`+`close` (~200 B). Keep stdlib for setup; syscalls in kernel-bound helpers.
2. **Generics that sneak in tables.** `str::split_whitespace` pulls Unicode tables; `split_ascii_whitespace` doesn't. `{:?}` pulls grapheme tables; `{}` doesn't. ASCII case-fold: `b.eq_ignore_ascii_case(c)` not the `c…` form. `nm --size-sort` before/after every refactor.
3. **Measure `.text`, not file size.** Loaders page-align LOAD segments; a 50 B trim can move the file by 0/4/8/16 KB depending where the section end lands. `size target/.../min/ltop` for the real number. Quote text deltas in commit messages.
4. **`f64` formatting pulls Grisu+Dragon (~10 KB).** Rewrite every `{:.1}`/`{:.2}` site as `(x * 10 + 0.5) as u64` → `{}.{}`. No accuracy change for our range.
5. **`#[inline(never)]` only saves bytes when there are multiple call sites.** Single callers get re-inlined. Don't bother.
6. **The Linux shrinking pipeline.** Custom `.cargo/static-linux-{x86_64,aarch64}.json` (no crt fallbacks, `env = "gnu"` exactly — empty breaks `libc`-crate compile, `"musl"` drags wrong syscalls into glibc tests) + `opt-level="z"` + LTO + 1 codegen unit + `panic = "abort"` + `-Cpanic=immediate-abort` (kills backtrace stack ≈ 120 KB) + `-Cforce-unwind-tables=no` + `-Crelocation-model=static` + `-no-pie` (no `.rela.dyn`/`.got`/`.plt`/`.dynamic` ≈ 9 KB) + `-Clink-self-contained=no` + `-Z build-std-features=compiler-builtins-mem` (weak Rust `mem*`; +300 B but kills the libc dep).
7. **`#![no_std]` + `no_main` + custom `extern "C" fn main`** drops std entirely (~13 KB of `OnceLock<ReentrantLock<...>>` + rt::cleanup + std_detect). `platform.rs` fills the gaps. `#![cfg_attr(not(test), no_std)]` keeps tests on std for the NVML harness.
8. **Aborting `#[global_allocator]`.** After every `Vec`/`String`/`Box` routes through the arena, swap libc Malloc for an `Abort` stub. Linker drops libc malloc; an accidental `Vec::new` aborts at first use instead of silently working.
9. **Libc functions that implicitly malloc.** ~7 KB of libc malloc lingered after #8 because `opendir` calls `calloc` for its `DIR`. Replaced with raw `getdents64` + inline `dirent64` parsing. After every alloc cull, `nm` for residual malloc symbols.
10. **Bypass `core::fmt` on the render path.** `<str as Display>::fmt` is `f.pad(self)` (~900 B unused); `pad_integral` (~470 B) fires on any `{}` over an integer; `core::fmt::write` is ~445 B. `src/twrite.rs`'s `Put`/`TinyWriter`/`twrite!` lower to direct `put_bytes`/`put_byte`. `src/bytes.rs` ships `U32d`/`PadRight`/`PadLeft`/`PadZero` (10 B stack buffer), `F1Wide`/`F2`, `Repeat` as `Put` impls. The render path doesn't reference `core::fmt`.
11. **`core::str::from_utf8` is ~470 B.** `/proc` reads compare bytes anyway — `read_small_file` returns `&[u8]`. UTF-8 validation for Display lives in `<[u8] as Put>::put` (~100 B `match`-driven Unicode Table 3-7: 4-byte UTF-8 + surrogate/overlong reject + recovery; invalid bytes emit `?`).
12. **Iterator closures monomorphise per call site.** `haystack.split(|b| b == b',')` in three functions = three `Split::next` specialisations. Promote shared patterns to `src/bytes.rs` helpers (`split_ascii_whitespace`, `split_lines`). ~1 KB.
13. **`-Wl,-z,noseparate-code`** merges R and R+X LOADs (default splits for W^X — moot with PIE off on a small read-only monitor). One page.
14. **`.cargo/rodata-first.ld` (`INSERT AFTER .note.gnu.build-id`)** drops `.rodata` into the ELF-header page (otherwise zero-padded), fusing three R-flag LOADs into two. `tests/stack_frames.rs::rodata_before_text` guards layout.
15. **Post-strip cleanup.** `strip = true` leaves `.stack_sizes` (1.1 KB), `.comment`, section headers, `.shstrtab` — non-LOAD bytes only `readelf` cares about. `.cargo/link-and-strip.sh` strips them, profile-aware (skips on `min-stack` so `stack-svg` keeps `.stack_sizes`).
16. **Per-page `madvise` is its own VMA detector.** `src/start.rs::ltop_entry` walks up from `sp_page+PAGE` calling `madvise(DONTNEED)`; succeeds inside the stack VMA, `-ENOMEM` past it. (Per-page probes dodge the multi-page caveat that one unmapped page kills the whole range.) Stack VMA is isolated by `stack_guard_gap` so the walk can't wander adjacent. `ltop_main` walks down to drop kernel pre-main pages. Sp relocates to `vma_top - 16` (AAPCS-aligned, ~4080 B in-page headroom) so the tick's peak frame fits in one page — `tests/stack_frames.rs::steady_state_stack_rss_one_page` enforces. ~35 syscalls / 315 μs at startup.
17. **`size`-test before celebrating.** Many "obvious" refactors compile to identical machine code (LTO had it) or worse (LTO loses cross-crate folding). Same `text` = no change.
18. **Rvalue static promotion bites `const X: &[&[u8]]`.** Iterating stores each `(ptr, len)` as a static — on Mach-O each is a `__DATA,__const` rebase fixup. Or-chains (`cond(x, b"a") || …`) materialise each literal in registers via `adrp+add+mov`, zero static storage. At opt-z rustc folds the common-prefix scan, no dup. Eliminating these tables on mac emptied `__DATA,__const`.
19. **`ld64` packs `__TEXT` sections at the *end* of its segment.** Even when content fits one 16 KiB page, ld64 emits 2 pages with full-page zero-pad. No linker flag fixes it. `tools/mac-pack` shifts `__TEXT` sections + `__DATA` + `__LINKEDIT` by exactly −0x4000 in vmaddr and re-signs ad-hoc. Single-page shift is load-bearing: source and target both move 4 pages, encoded page-delta unchanged. Currently passthroughs since IOKit-free MIG pushed `__text` past one page; auto-collapses if it shrinks below ~12 KB.
20. **`ptr::copy` lowers to `compiler_builtins::memmove` (~364 B).** Arena compaction (`Frame::replace`/`compact`) always knows direction at write time — replace with in-crate `copy_down(src, dst, len)` (forward byte-loop with `read_volatile`/`write_volatile` to defeat LLVM's loop-idiom rewrite). Same trick for `ptr::swap_nonoverlapping::<u8>`: word-wise swap if `T` is 4-aligned and a multiple of 4.
21. **`u64::ilog10` pulls `__udivti3` (u128 division) on arm64.** Our digit counter takes `u32` — a compare-ladder (`if n < 10 …`) is smaller and drops the 740 B helper.
22. **`Vec<T>` pulls `alloc::raw_vec::*` even with an aborting allocator** (~560 B of `RawVecInner::grow_amortized` + friends). The types are in the monomorphised call graph; linker can't dead-strip even though `alloc` aborts. Audit `Vec`/`vec![]` on the production path; use `FBox<[T; N]>`/`FVec<T>` or stack arrays. `nm`-per-function-size + `llvm-objdump --disassemble-symbols` in `mac-arm.yml` finds callers.
23. **`checked_*().expect(...)` on hot helpers can be load-bearing for downstream codegen, even with `panic=immediate-abort`.** `arena::bump`'s `aligned.checked_add(size).expect(...)` → plain `+` (same follow-up `assert!(end <= SIZE)`) cost +384 B on Linux aarch64 only — x86_64 + mac arm64 unchanged. The `expect` told LLVM the result was exactly `aligned + size` with no wrap, an ordering invariant downstream pointer math used to elide bounds checks. Plain `+` wraps silently; the assert catches the real failure but loses that proof. When a helper feeds many call sites, keep `checked_*().expect(...)` even if the panic is dead. Per-CGU sizes don't show it — LTO does; measure the linked binary.

## Arena-migration lessons

Converting 700+ allocations/tick to zero is more about API ergonomics than size:

1. **Generative lifetimes make scope discipline a compile-time property.** The `for<'id> FnOnce(&mut Frame<'id>)` HRTB on `scope` brands every allocation with a lifetime that can only exist inside the closure. Returning an arena allocation out is a compile error, verified by `compile_fail` doc-tests. This is what makes the arena safe without runtime refcounts.
2. **Type names should match Rust's semantic expectations.** `FSpan` = frozen view (Rust "Span" connotes that), `FVec` = mutable (push/pop, like `Vec`). Don't reuse "Vec" for "frozen length".
3. **Build-then-freeze: pick the flavour that matches the access pattern.** `frame.span(|b| { push... })` for one-shot construction in a closure; `frame.reserve(n)` returning an `FReserve` for imperative builds with `?`-propagating error paths (drop unfreezes / rewinds the arena cleanly).
4. **Thread `&State` instead of `Once<static>` when the state has a lifetime.** A `Once<State>` cached an arena pointer that the next test's scope rewound; tests would re-read a dead pointer. Pass it through the call chain — extra params, but makes the lifetime visible in the type.
5. **Represent "nothing to do" as an empty value, not `Option<T>`.** `rm::init` returns a `State` with `FSpan<Gpu>::empty()` on no-GPU hosts. Every consumer (`populate`, `Scratch::collect`, render) is already a natural no-op on an empty span; `Option` would force every site to handle `None`.
6. **In-place compaction beats drain-to-new-buffer.** `FVec::retain_in_place` compacts an arena FVec in place (closure returns keep/discard, kept moved forward via `ptr::copy`, discarded `drop_in_place`'d). Saves ~36 KB of arena pressure vs. filling a second FVec; unsafe scoped to one method.
7. **`#[cfg_attr(not(has_gpu), ignore)]` for environment-dependent tests.** `build.rs` probes `/dev/nvidiactl` and sets `cfg(has_gpu)`; tests auto-run on GPU hosts, stay ignored elsewhere. Beats `cargo test -- --ignored` invocations people forget.
8. **Keep arena `Drop` impls unwind-correct even though `panic = abort` elides them in production.** `cargo test` is std + unwinding, so a panicking builder closure rewinds cleanly — that's why tests don't leak the arena.
9. **Size the arena pessimistically.** Peak usage ~160 KB, declared 512 KB. Untouched `.bss` pages don't commit to RSS, so oversize costs virtual address space only. VmSize grows; VmRSS tracks the working set.

## Visualisation tools

Three tools in `tools/` for understanding where bytes go. All consume the `min-stack` profile (`cargo stack`); on mac, `dsymutil` first since DWARF lives in the dSYM.

- **`arena-svg`** (`--features arena-trace`; feature-off cost is zero, the `Label` ZST DCEs out): space-time diagram of arena allocs/rewinds. x = cumulative bytes (clamped to [16 B, 32 KB] per event), y = arena offset. **Regenerate `arena-trace.svg` whenever memory layout changes**; `arena-svg` prints the new peak on stderr — quote it in the commit message.
- **`stack-svg`**: per-function frame sizes from `.stack_sizes`, call graph from `objdump -d -C`, inlined-subroutine decomposition from DWARF. DFS from `main`; cycles break on second visit (`↻`); `--bound ltop::tree_walk=N` unrolls N levels first. Inlined sub-rects paint dashed-outline on their containing real frame so opt-z+LTO merges still attribute back to source. Unknown callees (non-Rust without `-Zemit-stack-sizes`) shown as `?`.
- **`inline-bytes`**: `.text`-bytes counterpart. Walks `DW_TAG_inlined_subroutine` to attribute bytes back to source — useful when one LTO-merged symbol (`run::{closure#0}::{closure#0}` at 10 KB) holds a dozen inlined callees and `nm` only sees the umbrella.

```
cargo stack
# Linux
./target/release/stack-svg   target/static-linux-x86_64/min-stack/ltop stack-trace.svg
./target/release/inline-bytes target/static-linux-x86_64/min-stack/ltop
# Mac (dSYM holds DWARF)
dsymutil target/aarch64-apple-darwin/min-stack/ltop
./target/release/inline-bytes target/aarch64-apple-darwin/min-stack/ltop.dSYM/Contents/Resources/DWARF/ltop
```

## Platform support

Linux (via `/proc`) and macOS (via sysctl + libproc FFI). Platform-specific code is gated with `#[cfg(target_os = ...)]`. GPU support is Nvidia-only via RM ioctls on `/dev/nvidiactl`; gracefully absent when that file can't be opened.

### macOS min binary — layout, packer, runtime reclaim

Mac `min` is ~33 KB; only LC_LOAD_DYLIB is libSystem (dyld enforces). Four layered shrinkages:

1. **No libSystem CALLS in source.** Every syscall = raw `svc #0x80` (`syscall.rs` Darwin branch); Mach RPC = hand-rolled `mach_msg2_trap` (`mac_sys.rs`). In-crate `bzero` in `platform.rs::rt` covers LLVM's one implicit codegen import. Parity tests (`mac_sys::tests`, `syscall::tests`) compare every raw call to its libSystem reference per CI run.
2. **No IOKit / CoreFoundation in the link.** AGX backend speaks raw IOKit MIG (subsystem 2800-series: `io_iterator_next` 2802, `io_registry_entry_get_child_iterator` 2813, `io_registry_entry_get_properties_bin` 2878, `io_registry_entry_get_property_bin` 2879 — single-property fetch, see `mac-property-bin.md`, `io_service_get_matching_services_bin` 2881). Reply blobs decoded by `src/osbinary.rs` (xnu's `OSSerializeBinary` format). `IOServiceMatching("AGXAccelerator")` is hand-encoded as a 48-B static `AGX_MATCHING_BLOB` — kernel parses it identically to libIOKit's CF round-trip. Routine IDs sourced from libIOKit's actual `bl _mach_msg2_internal` sites; the SDK's `iokitmig.h` is a one-line stub.
3. **`__DATA,__const` emptied via source-side fixup removal** (lesson 18). Or-chains instead of `const X: &[&[u8]]`; `twrite::put_arg` by value; arena `FBox` instead of `Vec` on the mac collect path. ld64 auto-assigns `filesize=0` to `__DATA` → 16 KB file page drops.
4. **`tools/mac-pack` post-link rewriter** would collapse `__TEXT` 2 → 1 page when content fits (lesson 19). Currently passthroughs since IOKit-free MIG pushed `__text` past one page.

When you touch any of this: parity tests fail closed on libSystem/libIOKit divergence; `osbinary` synthetic-blob tests catch encoder/decoder regressions on the AGX-free CI runner; `tests/stack_frames.rs` + `--once` smoke catches layout regressions. The mac-arm CI workflow dumps `otool -l`, `dyld_info -fixups`, and per-function `nm` sizes — look there first when size regresses.

**Inspecting Apple frameworks**: IOKit/CoreFoundation aren't on disk; they live in the dyld shared cache. Disassemble via `dyld_info -arch arm64e -disassemble /System/Library/Frameworks/IOKit.framework/Versions/A/IOKit` — the tool resolves the symbolic path through the cache. No `-disassemble-symbols` filter; `awk`/`grep` for the function you want.

**Never hardcode `mach_task_self()`.** It's the global `mach_task_self_` libdyld writes during startup. macOS 26 changed the accepted value (rejects others as `MACH_SEND_INVALID_DEST` = 0x10000003), so every `mach_vm_deallocate` / `mach_port_deallocate` silently failed and IOKit OOL buffers leaked ~9 × 112 KB per tick (Mach VM growth, not malloc — `leaks` reported constant). Right answer: `task_self_trap` (Mach trap #-28) once, cache, route every task-port arg through `cached_task_self()`. `OolBuffer::Drop` does a plain `assert_eq!(kr, 0)` under `cfg(test)` (not `debug_assert!`, which `cargo test --release` strips) — catches the next ABI shift in CI.

**`unmap_idle_state` reclaims everything dyld's init dirtied.** Process bring-up dirties ~288 KB libmalloc zone arenas, ~320 KB writable shared-cache pages (`__DATA_DIRTY`, `__OBJC_RW`, dirty shlib `__DATA`), ~128 KB RO subdylib mappings. `mac_sys::unmap_idle_state` walks VM regions (`proc_pidregioninfo`) and `mach_vm_deallocate`s anything not in the keep-list = four runtime anchors: binary `__TEXT` (`unmap_idle_state as *const ()`), arena `.bss` (`arena::extent()`, range-overlap because the kernel splits it across two VM regions), active stack frame (`mov sp`), virtual reservations ≥ 32 MiB (Memory Tag 22 = dyld's malloc pool — freeing it SIGKILLs us). Plus an SP-relocation dance in `main` (`mov sp; br mac_run_then_exit`) so the call runs in the top stack page; `mac_run_then_exit` then `madvise(MADV_FREE_REUSABLE)`s downward to reclaim argv/env/init pages, leaving 1 dirty page. phys_footprint 944 → 224 KB; region count 254 → 31. Returning from `main` after reclaim SIGSEGVs at `exit(3)` (atexit touches reclaimed pages) — exit via `syscall::exit_group(0)`.

Caveats:
- `pri_user_tag` / `pri_pages_dirtied` are 0 on macOS 26 `proc_pidregioninfo`. Keep-list filter only consults `pri_protection`, `pri_address`, `pri_size`.
- Kernel-as-witness invariant rests on: zero LC_LOAD_DYLIB-imported symbols (`nm -u` empty), no atexit/GCD/signal handlers/Mach notification ports, single-threaded. Break any of those and the kernel SIGSEGVs at the access — louder than corrupted state.

**`mach_vm_deallocate` is a placebo on dyld-shared-cache submap entries.** The 224 KB floor is structural: `vm_map_delete` un-permanents and "removes" the submap entry (`KERN_SUCCESS`; `proc_pidregioninfo` reports gone) but doesn't walk the leaf PTEs nested into the shared submap. An `ldrb` from a "freed" shared-cache `__TEXT` byte still succeeds, and the kernel still charges us for those PTEs. **`kr == 0` doesn't mean unmapped on shared submaps; the test is "does access fault?"** Reclaiming the residual 160 KB needs `VM_MAP_REMOVE_IMMUTABLE` (kernel-internal) or pmap-unnest (doesn't exist) — neither reachable without SIP off + signed entitlements. Nothing to claw back.

Diagnostic for "phys_footprint growing, `leaks` constant": the leak is in Mach VM, not malloc. `vmmap -summary` shows it as `VM_ALLOCATE` (tag 0). Stub `main` to sleep+exit, compare malloc counts vs. a real run — same count means dyld init owns all malloc activity and the delta is pure Mach VM. Then instrument `vm_deallocate` to write `kr` to fd 2; non-zero = silent rejection.
