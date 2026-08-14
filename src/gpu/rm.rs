//! Direct Resource Manager (RM) backend — talks to the NVIDIA kernel driver
//! via ioctls on /dev/nvidiactl and /dev/nvidia<N>, bypassing libnvidia-ml
//! and libcuda. Protocol references: NVIDIA/open-gpu-kernel-modules
//! (headers under src/common/sdk/nvidia/inc/ and kernel-open/common/inc/).
//!
//! Init: CHECK_VERSION_STR → CARD_INFO → RM_ALLOC NV01_ROOT, then per-GPU
//! open(/dev/nvidia<N>) + RM_ALLOC NV01_DEVICE_0 + NV20_SUBDEVICE_0.
//! Per-tick collect: three RM_CONTROL queries (FB_GET_INFO, PERFMON_UTIL,
//! GET_PIDS+GET_PID_INFO) against the cached subdevice handles.


use super::GpuProc;
use crate::arena::{FBox, FReserve, FSpan, FVec, Frame};
use crate::syscall;

/// Bare-byte "DBG: …\n" → fd 2. Cold paths only (RM init failures
/// on a misbehaving driver). Takes a string literal so we `concat!`
/// the prefix / suffix at compile time — no runtime formatter, no
/// `fmt::Write` chain, no `Formatter::pad_integral`. Loses the
/// formatted `ret=…` / `status=…` values that the old version
/// included; those live in the NVIDIA kernel driver's dmesg anyway.
macro_rules! dbg_eprintln {
    ($msg:literal) => {
        crate::syscall::write_once(2, concat!("DBG: ", $msg, "\n").as_bytes());
    };
}

/// RAII close of an fd. Replaces std's `OwnedFd`, which pulls in the
/// io_error path we otherwise don't need.
struct FdGuard(i32);
impl Drop for FdGuard {
    fn drop(&mut self) { syscall::close(self.0); }
}

/// Open `path` read+write with CLOEXEC. Returns None on failure.
fn open_rw(path: &core::ffi::CStr) -> Option<FdGuard> {
    let fd = syscall::open_cstr(path, syscall::O_RDWR | syscall::O_CLOEXEC);
    if fd < 0 { None } else { Some(FdGuard(fd)) }
}

// ── Protocol constants ───────────────────────────────────────────────────────

const NV_IOCTL_MAGIC: u32 = b'F' as u32;

// Escape codes — see src/nvidia/arch/nvalloc/unix/include/nv_escape.h
const NV_ESC_RM_ALLOC: u32 = 0x2B;
const NV_ESC_RM_CONTROL: u32 = 0x2A;

// Base-offset ioctls — see kernel-open/common/inc/nv-ioctl-numbers.h
const NV_IOCTL_BASE: u32 = 200;
const NV_ESC_CARD_INFO: u32 = NV_IOCTL_BASE + 0;
const NV_ESC_CHECK_VERSION_STR: u32 = NV_IOCTL_BASE + 10;

const NV_MAX_DEVICES: usize = 32;

// Classes — see src/common/sdk/nvidia/inc/class/
const NV01_ROOT: u32 = 0x0;
const NV01_DEVICE_0: u32 = 0x80;
const NV20_SUBDEVICE_0: u32 = 0x2080;

// Control commands — see src/common/sdk/nvidia/inc/ctrl/
const NV2080_CTRL_CMD_FB_GET_INFO: u32 = 0x2080_1301;
const NV2080_CTRL_FB_INFO_INDEX_TOTAL_RAM_SIZE: u32 = 0x08; // includes reserved (KB)
const NV2080_CTRL_FB_INFO_INDEX_HEAP_FREE: u32 = 0x16;      // free (KB)

const NV2080_CTRL_CMD_PERF_GET_GPUMON_PERFMON_UTIL_SAMPLES_V2: u32 = 0x2080_2096;
const NV2080_CTRL_PERF_GPUMON_SAMPLE_COUNT_PERFMON_UTIL: usize = 72;
const NV_SUBPROC_NAME_MAX_LENGTH: usize = 100;

const NV2080_CTRL_CMD_GPU_GET_PIDS: u32 = 0x2080_018d;
const NV2080_CTRL_GPU_GET_PIDS_MAX_COUNT: usize = 950;
const NV2080_CTRL_GPU_GET_PIDS_ID_TYPE_CLASS: u32 = 0;

const NV2080_CTRL_CMD_GPU_GET_PID_INFO: u32 = 0x2080_018e;
const NV2080_CTRL_GPU_GET_PID_INFO_MAX_COUNT: usize = 200;
const NV2080_CTRL_GPU_PID_INFO_INDEX_VIDEO_MEMORY_USAGE: u32 = 0;

// Handle scheme: 0xc1d0_XXYY, XX = object-kind, YY = GPU index.
const fn device_handle(i: u32) -> u32 { 0xc1d0_0100 | (i & 0xff) }
const fn subdevice_handle(i: u32) -> u32 { 0xc1d0_0200 | (i & 0xff) }

// Version-check modes — kernel-open/common/inc/nv-ioctl.h
const NV_RM_API_VERSION_CMD_STRICT: u32 = 0;
const NV_RM_API_VERSION_STRING_LENGTH: usize = 64;

// Process-name buffer size — src/common/sdk/nvidia/inc/nvlimits.h
const NV_PROC_NAME_MAX_LENGTH: usize = 100;

/// Linux ioctl number for a read+write command of the given magic, nr, size.
/// Encodes (dir << 30) | (size << 16) | (magic << 8) | nr, where dir=3 means
/// _IOC_READ(2) | _IOC_WRITE(1). The result is a u64 the way the raw
/// `ioctl(2)` syscall consumes it.
const fn iowr(magic: u32, nr: u32, size: u32) -> u64 {
    ((3 << 30) | (size << 16) | (magic << 8) | nr) as u64
}

// ── Struct layouts (must match driver headers exactly) ───────────────────────

// kernel-open/common/inc/nv-ioctl.h: nv_ioctl_rm_api_version_t
#[repr(C)]
struct NvIoctlRmApiVersion {
    cmd: u32,
    reply: u32,
    version_string: [u8; NV_RM_API_VERSION_STRING_LENGTH],
}
unsafe impl crate::zeroable::Zeroable for NvIoctlRmApiVersion {}

// kernel-open/common/inc/nv-ioctl.h: nv_pci_info_t
#[repr(C)]
#[derive(Default, Clone, Copy)]
struct NvPciInfo {
    domain: u32,
    bus: u8,
    slot: u8,
    function: u8,
    vendor_id: u16,
    device_id: u16,
}

// kernel-open/common/inc/nv-ioctl.h: nv_ioctl_card_info_t (72 bytes per entry).
// CARD_INFO ioctl payload is an NV_MAX_DEVICES-element array. repr(C) handles
// all the inter-field padding the C struct needs.
#[repr(C)]
#[derive(Default, Clone, Copy)]
struct NvIoctlCardInfo {
    valid: u8,
    pci_info: NvPciInfo,
    gpu_id: u32,
    interrupt_line: u16,
    reg_address: u64,
    reg_size: u64,
    fb_address: u64,
    fb_size: u64,
    minor_number: u32,
    dev_name: [u8; 10],
}
// All fields are primitive integer types; every-byte-zero is a valid value.
unsafe impl crate::zeroable::Zeroable for NvIoctlCardInfo {}

// src/common/sdk/nvidia/inc/nvos.h: NVOS21_PARAMETERS
// pAllocParms has NV_ALIGN_BYTES(8) → forces 8-alignment; first 16 bytes
// (4×u32) are already aligned so no padding needed there.
#[repr(C)]
struct Nvos21Parameters {
    h_root: u32,
    h_object_parent: u32,
    h_object_new: u32,
    h_class: u32,
    p_alloc_parms: u64,
    params_size: u32,
    status: u32,
}

// src/common/sdk/nvidia/inc/class/cl0000.h: NV0000_ALLOC_PARAMETERS
// pOsPidInfo is NV_ALIGN_BYTES(8); repr(C) auto-pads before it.
// [u8; 100] has no Default impl, so we spell out the zero value.
#[repr(C)]
struct Nv0000AllocParameters {
    h_client: u32,
    process_id: u32,
    process_name: [u8; NV_PROC_NAME_MAX_LENGTH],
    p_os_pid_info: u64,
}

impl Default for Nv0000AllocParameters {
    fn default() -> Self {
        Self { h_client: 0, process_id: 0,
               process_name: [0; NV_PROC_NAME_MAX_LENGTH], p_os_pid_info: 0 }
    }
}

// src/common/sdk/nvidia/inc/nvos.h: NVOS54_PARAMETERS (RM_CONTROL).
// Observed 32 bytes via strace. `flags` may be padding on older drivers; zero
// either way.
#[repr(C)]
struct Nvos54Parameters {
    h_client: u32,
    h_object: u32,
    cmd: u32,
    flags: u32,
    params: u64,
    params_size: u32,
    status: u32,
}

// src/common/sdk/nvidia/inc/class/cl0080.h: NV0080_ALLOC_PARAMETERS
#[repr(C)]
#[derive(Default)]
struct Nv0080AllocParameters {
    device_id: u32,
    h_client_share: u32,
    h_target_client: u32,
    h_target_device: u32,
    flags: u32,
    va_space_size: u64,
    va_start_internal: u64,
    va_limit_internal: u64,
    va_mode: u32,
}

// src/common/sdk/nvidia/inc/class/cl2080.h: NV2080_ALLOC_PARAMETERS
#[repr(C)]
struct Nv2080AllocParameters {
    sub_device_id: u32,
}

// src/common/sdk/nvidia/inc/ctrl/ctrlxxxx.h: NVXXXX_CTRL_XXX_INFO
#[repr(C)]
#[derive(Default, Clone, Copy)]
struct NvFbInfo {
    index: u32,
    data: u32,
}

// ctrl2080gpumon.h: NV2080_CTRL_GPUMON_SAMPLE (base of every gpumon sample)
#[repr(C, align(8))]
#[derive(Default, Clone, Copy)]
struct NvGpumonSample {
    time_stamp: u64,
}

// ctrl2080perf.h: NV2080_CTRL_PERF_GPUMON_ENGINE_UTIL_SAMPLE (128 bytes).
// subProcessName[100] ends at offset 116; repr(C) auto-pads 4 bytes before
// p_os_pid_info's 8-alignment. No Default derive because [u8; 100] has none.
#[repr(C, align(8))]
#[derive(Clone, Copy)]
struct NvEngineUtilSample {
    util: u32,
    vgpu_scale: u32,
    proc_id: u32,
    sub_process_id: u32,
    sub_process_name: [u8; NV_SUBPROC_NAME_MAX_LENGTH],
    p_os_pid_info: u64,
}

// ctrl2080perf.h: NV2080_CTRL_PERF_GPUMON_PERFMON_UTIL_SAMPLE (776 bytes)
#[repr(C, align(8))]
#[derive(Clone, Copy)]
struct NvPerfmonUtilSample {
    base: NvGpumonSample,
    fb: NvEngineUtilSample,
    gr: NvEngineUtilSample,       // ← graphics/SM utilisation; this is what we want
    nvenc: NvEngineUtilSample,
    nvdec: NvEngineUtilSample,
    nvjpg: NvEngineUtilSample,
    nvofa: NvEngineUtilSample,
}

// ctrl2080gpu.h: NV2080_CTRL_GPU_GET_PIDS_PARAMS (3812 bytes)
#[repr(C)]
struct NvGetPidsParams {
    id_type: u32,
    id: u32,
    pid_tbl_count: u32,
    pid_tbl: [u32; NV2080_CTRL_GPU_GET_PIDS_MAX_COUNT],
}

// ctrl2080gpu.h: NV2080_CTRL_GPU_PID_INFO_VIDEO_MEMORY_USAGE_DATA (48 bytes)
#[repr(C, align(8))]
#[derive(Default, Clone, Copy)]
struct NvVideoMemoryUsageData {
    mem_private: u64,
    mem_shared_owned: u64,
    mem_shared_duped: u64,
    protected_mem_private: u64,
    protected_mem_shared_owned: u64,
    protected_mem_shared_duped: u64,
}

// ctrl2080gpu.h: NV2080_CTRL_SMC_SUBSCRIPTION_INFO (8 bytes)
#[repr(C)]
#[derive(Default, Clone, Copy)]
struct NvSmcSubscriptionInfo {
    compute_instance_id: u32,
    gpu_instance_id: u32,
}

// ctrl2080gpu.h: NV2080_CTRL_GPU_PID_INFO (72 bytes). `data` is a union; the
// VIDEO_MEMORY_USAGE variant is 48 bytes — the biggest (only?) variant, so
// embedding it directly gives the right size.
#[repr(C, align(8))]
#[derive(Default, Clone, Copy)]
struct NvPidInfo {
    pid: u32,
    index: u32,
    result: u32,
    // 4 bytes padding before data (u64-aligned)
    data: NvVideoMemoryUsageData,
    smc_subscription: NvSmcSubscriptionInfo,
}

// ctrl2080gpu.h: NV2080_CTRL_GPU_GET_PID_INFO_PARAMS (14408 bytes)
#[repr(C)]
struct NvGetPidInfoParams {
    pid_info_list_count: u32,
    // 4 bytes padding before pid_info_list (u64-aligned via PidInfo)
    pid_info_list: [NvPidInfo; NV2080_CTRL_GPU_GET_PID_INFO_MAX_COUNT],
}

// ctrl2080perf.h: NV2080_CTRL_PERF_GET_GPUMON_PERFMON_UTIL_SAMPLES_V2_PARAMS
// (~55 KB)
#[repr(C)]
struct NvPerfmonUtilParams {
    sample_type: u8,
    buf_size: u32,
    count: u32,
    tracker: u32,
    samples: [NvPerfmonUtilSample; NV2080_CTRL_PERF_GPUMON_SAMPLE_COUNT_PERFMON_UTIL],
}

// SAFETY for each of the three scratch param types below: they're
// compositions of `#[repr(C)]` integer/array data. No references,
// enums, NonZero, or niche-bearing fields — all-bits-zero is a valid
// initial state for the ioctl to fill in.
unsafe impl crate::zeroable::Zeroable for NvPerfmonUtilParams {}
unsafe impl crate::zeroable::Zeroable for NvGetPidsParams {}
unsafe impl crate::zeroable::Zeroable for NvGetPidInfoParams {}

// src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080fb.h: V1 uses an external
// pointer to the list. fbInfoList is NV_DECLARE_ALIGNED(NvP64, 8), so repr(C)
// inserts 4 bytes of padding after the u32 counter.
#[repr(C)]
struct Nv2080CtrlFbGetInfoParams {
    fb_info_list_size: u32,
    fb_info_list: u64,  // NvP64 pointer to [NvFbInfo; N]
}

// ── Global state ─────────────────────────────────────────────────────────────

struct Gpu {
    _dev_fd: FdGuard,
    subdevice: u32,
}

/// Opened NVIDIA driver state — a ctl fd (absent on no-driver hosts), a
/// client handle, and an exact-sized arena-allocated list of GPUs. No-GPU
/// hosts get a State with `gpus` empty and `ctl_fd: None`; every call path
/// stays a no-op in that case, so callers never need to special-case it.
/// Lifetime-branded to the arena scope it was initialised in.
pub struct State<'id> {
    ctl_fd: Option<FdGuard>,
    client: u32,
    gpus: FSpan<'id, Gpu>,
}

impl<'id> State<'id> {
    /// Are we connected to a usable GPU? Equivalent to `!gpus.is_empty()`.
    pub fn has_gpu(&self) -> bool { !self.gpus.is_empty() }
}

/// Tight upper bound on entries pushed into `procs` per GPU per tick:
/// one (pid, sm_pct) from `perfmon_util` plus up to
/// `NV2080_CTRL_GPU_GET_PID_INFO_MAX_COUNT` from `proc_memory`. The
/// caller sizes the buffer as `n_gpus * MAX_PROCS_PER_GPU` — exact, no
/// over-allocation.
pub const MAX_PROCS_PER_GPU: u32 = 1 + NV2080_CTRL_GPU_GET_PID_INFO_MAX_COUNT as u32;

/// Open the NVIDIA driver, allocate a root client, and register one device +
/// subdevice per attached GPU. The returned State's `gpus` list is allocated
/// from `frame` and sized exactly to the attached-GPU count — no MAX_GPUS
/// constant or padding. On hosts without a usable driver, returns a State
/// with `ctl_fd: None` and empty `gpus`; downstream calls are no-ops.
pub fn init<'a>(frame: &mut Frame<'a>) -> State<'a> {
    try_init(frame).unwrap_or_else(|| State {
        ctl_fd: None,
        client: 0,
        gpus: frame.span("gpu/gpus", |_| {}),
    })
}

fn try_init<'a>(frame: &mut Frame<'a>) -> Option<State<'a>> {
    let ctl_fd = open_rw(c"/dev/nvidiactl")?;
    let fd = ctl_fd.0;

    check_version(frame, fd)?;
    // SYS_PARAMS is a one-shot global init that EBUSYs if the first nvidia
    // process already set it — skipped with no observed ill effect.
    let n_gpus = count_attached_gpus(frame, fd)?;
    let mut root = Nv0000AllocParameters {
        process_id: crate::platform::getpid(),
        ..Default::default()
    };
    root.process_name[..4].copy_from_slice(b"ltop");
    let client = rm_alloc(fd, 0, 0, 0, NV01_ROOT, &mut root)?;

    // Reserve exact space for `n_gpus` Gpu entries; any early return
    // (open_rw / alloc_device failure) drops the reserve, which rewinds
    // the arena and closes any FdGuards already pushed.
    let mut gpus_res: FReserve<Gpu> = frame.reserve("gpu/gpus", n_gpus as u32);
    // "/dev/nvidia%d\0" — u32 maxes out at 10 digits, plus prefix + NUL.
    let mut path = [0u8; 32];
    for i in 0..n_gpus as u32 {
        // Reusable NUL-terminated path buffer. ASCII-only, so treat the
        // filled prefix as a &CStr-equivalent via raw pointer.
        let n = {
            let mut w = CharBuf::new(&mut path);
            crate::twrite!(&mut w, "/dev/nvidia", crate::bytes::u32d(i as u32), "\0");
            w.len
        };
        let fd_path = unsafe { core::ffi::CStr::from_ptr(path[..n].as_ptr() as *const _) };
        let dev_fd = open_rw(fd_path)?;
        alloc_device(fd, client, i)?;
        alloc_subdevice(fd, client, i)?;
        let _ = gpus_res.push(Gpu { _dev_fd: dev_fd, subdevice: subdevice_handle(i) });
    }

    Some(State { ctl_fd: Some(ctl_fd), client, gpus: gpus_res.freeze() })
}

/// Bounded writer over a `&mut [u8]`. Used to format a small path into a
/// stack buffer without going through `String` / `alloc::format!` just to
/// immediately hand the bytes to `libc::open`.
struct CharBuf<'a> { buf: &'a mut [u8], len: usize }
impl<'a> CharBuf<'a> {
    fn new(buf: &'a mut [u8]) -> Self { Self { buf, len: 0 } }
}
impl crate::twrite::TinyWriter for CharBuf<'_> {
    fn put_bytes(&mut self, b: &[u8]) {
        let end = self.len + b.len();
        if end > self.buf.len() { return; } // silent truncation; caller sized the buffer
        self.buf[self.len..end].copy_from_slice(b);
        self.len = end;
    }
    fn put_byte(&mut self, b: u8) {
        if self.len < self.buf.len() {
            self.buf[self.len] = b;
            self.len += 1;
        }
    }
}

fn count_attached_gpus(frame: &mut Frame<'_>, fd: i32) -> Option<usize> {
    // 32 × 72 B = 2304 B — over the >1 KB → arena rule, so allocate it in
    // a sub-scope and let the arena reclaim it when this call returns.
    // `alloc_zeroed` relies on the `Zeroable` impl asserting that all-
    // bits-zero is a valid `[NvIoctlCardInfo; N]`.
    frame.scope(|inner| {
        let mut cards: FBox<[NvIoctlCardInfo; NV_MAX_DEVICES]> =
            inner.alloc_zeroed("gpu/cards");
        let req = iowr(
            NV_IOCTL_MAGIC,
            NV_ESC_CARD_INFO,
            core::mem::size_of_val(&*cards) as u32,
        );
        // `ioctl` returns 0 on success, -errno on failure. We log the
        // negative return directly — no `__errno_location` reach-in.
        let ret = unsafe { syscall::ioctl(fd, req, cards.as_mut_ptr() as *mut _) };
        if ret != 0 {
            dbg_eprintln!("CARD_INFO ioctl failed");
            return None;
        }
        Some(cards.iter().filter(|c| c.valid != 0).count())
    })
}

fn alloc_device(fd: i32, client: u32, index: u32) -> Option<u32> {
    let mut alloc = Nv0080AllocParameters { device_id: index, ..Default::default() };
    rm_alloc(fd, client, client, device_handle(index), NV01_DEVICE_0, &mut alloc)
}

fn alloc_subdevice(fd: i32, client: u32, index: u32) -> Option<u32> {
    let mut alloc = Nv2080AllocParameters { sub_device_id: 0 };
    rm_alloc(fd, client, device_handle(index), subdevice_handle(index), NV20_SUBDEVICE_0, &mut alloc)
}

/// RM_ALLOC. Returns the driver-assigned handle (for non-zero `h_new`, just
/// `h_new` echoed back; for h_new=0, the driver picks).
fn rm_alloc<T>(fd: i32, h_root: u32, h_parent: u32, h_new: u32, class: u32, alloc_params: &mut T) -> Option<u32> {
    let mut p = Nvos21Parameters {
        h_root, h_object_parent: h_parent, h_object_new: h_new, h_class: class,
        p_alloc_parms: alloc_params as *mut _ as u64,
        params_size: core::mem::size_of::<T>() as u32,
        status: 0,
    };
    let req = iowr(NV_IOCTL_MAGIC, NV_ESC_RM_ALLOC, core::mem::size_of::<Nvos21Parameters>() as u32);
    let ret = unsafe { syscall::ioctl(fd, req, &mut p as *mut _ as *mut _) };
    if ret != 0 || p.status != 0 {
        dbg_eprintln!("RM_ALLOC failed");
        return None;
    }
    Some(p.h_object_new)
}

fn rm_control<T>(fd: i32, h_client: u32, h_object: u32, cmd: u32, params: &mut T) -> Option<()> {
    let mut p = Nvos54Parameters {
        h_client, h_object, cmd, flags: 0,
        params: params as *mut _ as u64,
        params_size: core::mem::size_of::<T>() as u32,
        status: 0,
    };
    let req = iowr(NV_IOCTL_MAGIC, NV_ESC_RM_CONTROL, core::mem::size_of::<Nvos54Parameters>() as u32);
    let ret = unsafe { syscall::ioctl(fd, req, &mut p as *mut _ as *mut _) };
    if ret != 0 || p.status != 0 {
        dbg_eprintln!("RM_CONTROL failed");
        return None;
    }
    Some(())
}

fn check_version(frame: &mut Frame<'_>, fd: i32) -> Option<()> {
    // Read the driver version file and pick out the numeric-dotted token.
    // Keep the token borrowed from `file_buf` — no heap allocation.
    //
    // "NVRM version: NVIDIA UNIX x86_64 Kernel Module  570.211.01  ..."
    //
    // Byte-level scan for "first-char digit, contains '.'" — avoids chars()
    // / contains(char) which pull in the UTF-8 decoder and CharSearcher
    // (~500 B). The version string is always ASCII.
    //
    // Both scratch buffers (file_buf at 256 B, v at 72 B) live in the
    // arena for the duration of this call — not in our stack frame. Each
    // one is well under the >1 KB → arena rule on its own, but grouping
    // them keeps the call's stack contribution near-zero and matches the
    // pattern we use for larger one-shot buffers elsewhere.
    frame.scope(|inner| {
        let mut file_buf: FBox<[u8; 256]> = inner.alloc_zeroed("check_version/file_buf");
        let content =
            crate::read_small_file(c"/proc/driver/nvidia/version", &mut file_buf[..])?;
        let driver_version = crate::bytes::split_ascii_whitespace(content).find(|t| {
            t.first().is_some_and(|x| x.is_ascii_digit()) && t.iter().any(|&b| b == b'.')
        })?;

        let mut v: FBox<NvIoctlRmApiVersion> = inner.alloc_zeroed("check_version/v");
        v.cmd = NV_RM_API_VERSION_CMD_STRICT;
        let n = driver_version.len().min(NV_RM_API_VERSION_STRING_LENGTH - 1);
        v.version_string[..n].copy_from_slice(&driver_version[..n]);

        // Non-zero ioctl return = version mismatch; the driver writes its own
        // version into `version_string` for diagnostics. `reply` == 1 on success
        // is just a "request understood" flag, not a pass/fail signal.
        let req = iowr(
            NV_IOCTL_MAGIC,
            NV_ESC_CHECK_VERSION_STR,
            core::mem::size_of::<NvIoctlRmApiVersion>() as u32,
        );
        if unsafe { syscall::ioctl(fd, req, &mut *v as *mut _ as *mut _) } != 0 {
            dbg_eprintln!("CHECK_VERSION_STR failed (driver mismatch?)");
            return None;
        }
        Some(())
    })
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Snapshot this tick: returns the sort+deduped `(pid, GpuProc)`
/// FSpan alongside per-GPU totals (n_gpus, total_util, total_used_mib,
/// total_mib).
///
/// Everything runs inside a single `frame.compact`: the 55 KB
/// `rm/util` buffer (phase 1) and `gpu/procs/raw` (phase 2) never
/// coexist, so the peak arena footprint is max(55 KB, 9.6 KB + 18 KB)
/// ≈ 55 KB, down from 65 KB when the old code held `gpu/procs/raw`
/// open across phase 1's `rm/util` sub-scope.
///
/// * **Phase 1**: for each GPU, query FB memory then run
///   PERFMON_UTIL_SAMPLES in its own sub-scope. The 55 KB util buffer
///   is transient; the only survivors are tiny per-GPU stashes
///   (`samples` = 12 B × n_gpus, `valid` = 1 B × n_gpus) pre-allocated
///   in the compact's inner frame.
/// * **Phase 2**: allocate `gpu/procs/raw` (9.6 KB worst case), push
///   the stashed samples, then run GET_PIDS + GET_PID_INFO per GPU
///   (rm/pids + rm/pid_info = 18 KB transient). Sort + dedup + freeze
///   to the exact post-dedup FSpan that compact slides down to its
///   entry offset.
pub fn populate<'id>(state: &State<'_>, frame: &mut Frame<'id>)
    -> (FSpan<'id, (u32, GpuProc)>, (u32, u32, u64, u64))
{
    // No-driver / no-GPU hosts: empty FSpan, zero totals.
    let Some(ctl_fd) = state.ctl_fd.as_ref() else {
        return (frame.empty(), (0, 0, 0, 0));
    };
    let fd = ctl_fd.0;
    let n_gpus_total = state.gpus.len();

    // Captured by the compact closure via `&mut` so the outer caller
    // can return them alongside the FSpan.
    let mut n_gpus = 0u32;
    let mut total_util = 0u32;
    let mut total_used_mib = 0u64;
    let mut total_mib = 0u64;

    let entries: FSpan<'id, (u32, GpuProc)> = frame.compact("gpu/procs", |inner| {
        // Per-GPU stashes live in the compact's inner scope: reclaimed
        // when compact rewinds, so only the final dedup'd FSpan
        // escapes. Sized exactly to n_gpus, no MAX_GPUS cap.
        let mut samples: FSpan<Option<(u32, u32)>> = inner.zeros("samples", n_gpus_total);
        let mut valid: FSpan<bool> = inner.zeros("valid", n_gpus_total);

        // Phase 1: fb_memory + perfmon per GPU. rm/util (55 KB) is the
        // only arena resident during this loop beyond the tiny samples
        // + valid stashes.
        for (i, gpu) in state.gpus.iter().enumerate() {
            let Some((used, tot)) = fb_memory(fd, state.client, gpu.subdevice) else { continue };
            total_used_mib += used;
            total_mib += tot;
            total_util += inner.scope(|s| {
                let mut util = s.alloc_zeroed::<NvPerfmonUtilParams>("rm/util");
                perfmon_util(fd, state.client, gpu.subdevice, &mut util, &mut samples[i])
            });
            valid[i] = true;
            n_gpus += 1;
        }

        // Phase 2: staging FVec + per-GPU proc_memory sub-scopes.
        let mut buf: FVec<(u32, GpuProc)> =
            inner.vec("gpu/procs/raw", n_gpus * MAX_PROCS_PER_GPU);
        for &sample in samples.iter() {
            if let Some((pid, sm_pct)) = sample {
                let _ = buf.push((pid, GpuProc { sm_pct, mem_mib: 0 }));
            }
        }
        for (i, gpu) in state.gpus.iter().enumerate() {
            if !valid[i] { continue; }
            inner.scope(|s| {
                let mut pids = s.alloc_zeroed::<NvGetPidsParams>("rm/pids");
                let mut pid_info = s.alloc_zeroed::<NvGetPidInfoParams>("rm/pid_info");
                proc_memory(fd, state.client, gpu.subdevice, &mut pids, &mut pid_info, &mut buf);
            });
        }
        let _ = crate::map::freeze_map(&mut buf, |a, b| {
            a.mem_mib += b.mem_mib;
            a.sm_pct += b.sm_pct;
        });
        buf.into_fspan()
    });

    (entries, (n_gpus, total_util, total_used_mib, total_mib))
}

/// Query this subdevice's heap and return (used_mib, total_mib).
fn fb_memory(fd: i32, client: u32, subdevice: u32) -> Option<(u64, u64)> {
    let mut list = [
        NvFbInfo { index: NV2080_CTRL_FB_INFO_INDEX_TOTAL_RAM_SIZE, data: 0 },
        NvFbInfo { index: NV2080_CTRL_FB_INFO_INDEX_HEAP_FREE, data: 0 },
    ];
    let mut params = Nv2080CtrlFbGetInfoParams {
        fb_info_list_size: list.len() as u32,
        fb_info_list: list.as_mut_ptr() as u64,
    };
    rm_control(fd, client, subdevice, NV2080_CTRL_CMD_FB_GET_INFO, &mut params)?;
    let (total_kib, free_kib) = (list[0].data as u64, list[1].data as u64);
    // saturating: a driver quirk reporting HEAP_FREE > TOTAL_RAM_SIZE
    // (SMC-partitioned subdevices, transient alloc races) must clamp to
    // 0 used, not wrap to ~u64::MAX and blow up the rendered totals.
    Some((total_kib.saturating_sub(free_kib) >> 10, total_kib >> 10))
}

/// Pull the latest GPUMON graphics-engine sample: returns this GPU's
/// utilisation percent. If the sample's proc_id is set, also stashes a
/// (proc_id, sm_pct) pair into `sample_out` for the caller to merge
/// into the procs buffer after the util buffer has been freed (so the
/// 55 KB util allocation never coexists with the procs staging vec).
/// GPUMON reports utilisation in hundredths of a percent (7200 = 72.00%);
/// we divide by 100 to match NVML's integer-percent convention.
fn perfmon_util(fd: i32, client: u32, subdevice: u32, buf: &mut NvPerfmonUtilParams,
                sample_out: &mut Option<(u32, u32)>) -> u32 {
    // Driver rejects with INVALID_ARGUMENT unless count matches the ring
    // capacity exactly (empirically — it returns however many are actually
    // populated, not up-to-count). We only read samples[count-1] afterwards.
    *buf = unsafe { core::mem::zeroed() };
    buf.count = NV2080_CTRL_PERF_GPUMON_SAMPLE_COUNT_PERFMON_UTIL as u32;
    buf.buf_size = (core::mem::size_of::<NvPerfmonUtilSample>()
                  * NV2080_CTRL_PERF_GPUMON_SAMPLE_COUNT_PERFMON_UTIL) as u32;
    if rm_control(fd, client, subdevice,
                  NV2080_CTRL_CMD_PERF_GET_GPUMON_PERFMON_UTIL_SAMPLES_V2, buf).is_none() { return 0; }
    // `count` is kernel-written output. The VF handler rejects any
    // bufSize that isn't exactly the 72-slot ring, so count > ring is
    // only reachable via driver/firmware corruption — but the clamp is
    // free and turns that abort (index panic under panic=immediate-
    // abort) into a stale-sample read, matching pid_tbl_count's clamp.
    let n = (buf.count as usize).min(NV2080_CTRL_PERF_GPUMON_SAMPLE_COUNT_PERFMON_UTIL);
    if n == 0 { return 0; }
    let gr = &buf.samples[n - 1].gr;
    let pct = gr.util / 100;
    if gr.proc_id != 0 {
        *sample_out = Some((gr.proc_id, pct));
    }
    pct
}

/// Push (pid, {mem_mib}) entries for every process using this subdevice,
/// via GET_PIDS + GET_PID_INFO. NVML's getComputeRunningProcesses reports
/// the mem_shared_owned field (CUDA allocations owned by the PID).
fn proc_memory(fd: i32, client: u32, subdevice: u32,
               pids: &mut NvGetPidsParams, pid_info: &mut NvGetPidInfoParams,
               procs: &mut FVec<'_, (u32, GpuProc)>) {
    pids.id_type = NV2080_CTRL_GPU_GET_PIDS_ID_TYPE_CLASS;
    pids.id = NV20_SUBDEVICE_0;
    pids.pid_tbl_count = 0;
    if rm_control(fd, client, subdevice, NV2080_CTRL_CMD_GPU_GET_PIDS, pids).is_none() { return; }
    let n = (pids.pid_tbl_count as usize).min(NV2080_CTRL_GPU_GET_PID_INFO_MAX_COUNT);
    if n == 0 { return; }

    pid_info.pid_info_list_count = n as u32;
    for i in 0..n {
        pid_info.pid_info_list[i] = NvPidInfo {
            pid: pids.pid_tbl[i],
            index: NV2080_CTRL_GPU_PID_INFO_INDEX_VIDEO_MEMORY_USAGE,
            ..Default::default()
        };
    }
    if rm_control(fd, client, subdevice, NV2080_CTRL_CMD_GPU_GET_PID_INFO, pid_info).is_none() { return; }

    for entry in &pid_info.pid_info_list[..n] {
        let mib = (entry.data.mem_shared_owned >> 20) as u32;
        if mib > 0 {
            let _ = procs.push((entry.pid, GpuProc { sm_pct: 0, mem_mib: mib }));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pin every kernel-ABI struct against the byte size its driver
    /// header documents (the literal each struct's comment quotes). A
    /// reordered or mis-typed field stays Zeroable-sound but silently
    /// truncates/misaligns the ioctl payload — garbage GPU stats with
    /// no other signal. Same pattern as sandbox.rs's
    /// kernel_struct_layout. Runs on every host, no GPU needed.
    #[test]
    fn struct_sizes_match_driver_headers() {
        use core::mem::size_of;
        assert_eq!(size_of::<NvIoctlCardInfo>(), 72);
        assert_eq!(size_of::<NvEngineUtilSample>(), 128);
        assert_eq!(size_of::<NvPerfmonUtilSample>(), 776);
        assert_eq!(size_of::<NvGetPidsParams>(), 3812);
        assert_eq!(size_of::<NvVideoMemoryUsageData>(), 48);
        assert_eq!(size_of::<NvSmcSubscriptionInfo>(), 8);
        assert_eq!(size_of::<NvPidInfo>(), 72);
        assert_eq!(size_of::<NvGetPidInfoParams>(), 14408);
        // V2 params: 16-byte header (u8 + 3 pad + 3×u32) + the 72-slot ring.
        assert_eq!(size_of::<NvPerfmonUtilParams>(),
                   16 + 72 * size_of::<NvPerfmonUtilSample>());
    }

    // Requires an NVIDIA driver + /dev/nvidiactl access. build.rs probes
    // for /dev/nvidiactl and sets cfg(has_gpu) when present, so these auto-
    // run on GPU hosts. On non-GPU hosts they stay ignored by default.
    // Force-run on a non-GPU host with `cargo test -- --ignored`.
    //
    // Each test grabs `arena::test_lock` so it serialises against the arena
    // unit tests (which also touch the global arena). `cargo test` runs tests
    // in parallel by default; without the lock, arena offset assertions can
    // race with rm_matches_nvml's arena::scope calls.
    #[test]
    #[cfg_attr(not(has_gpu), ignore)]
    fn rm_init_succeeds() {
        let _g = crate::arena::test_lock();
        crate::arena::scope(|frame| {
            let state = super::init(frame);
            assert!(state.has_gpu(), "RM init should succeed on a GPU host");
            eprintln!("DBG: RM initialised with {} GPU(s)", state.gpus.len());
        });
    }

    #[test]
    #[cfg_attr(not(has_gpu), ignore)]
    fn rm_subdevice_control_works() {
        let _g = crate::arena::test_lock();
        crate::arena::scope(|frame| {
            let state = super::init(frame);
            let ctl_fd = state.ctl_fd.as_ref().expect("driver").0;
            #[repr(C)] struct GetId { gpu_id: u32 }
            let mut p = GetId { gpu_id: 0 };
            let ok = rm_control(ctl_fd, state.client, state.gpus[0].subdevice,
                                0x2080_0142, &mut p);
            eprintln!("RM subdevice GET_ID ok={:?} gpu_id=0x{:x}", ok.is_some(), p.gpu_id);
            assert!(ok.is_some());
        });
    }

    #[test]
    #[cfg_attr(not(has_gpu), ignore)]
    fn rm_root_control_works() {
        // NV0000_CTRL_CMD_GPU_GET_ATTACHED_IDS exercises a RM_CONTROL on the
        // root client, confirming the client handle from RM_ALLOC is valid.
        const CMD: u32 = 0x201;
        const MAX: usize = 32;
        const INVALID: u32 = 0xffff_ffff;
        #[repr(C)] struct Params { gpu_ids: [u32; MAX] }
        let _g = crate::arena::test_lock();
        crate::arena::scope(|frame| {
            let state = super::init(frame);
            let ctl_fd = state.ctl_fd.as_ref().expect("driver").0;
            let mut p = Params { gpu_ids: [INVALID; MAX] };
            let ok = rm_control(ctl_fd, state.client, state.client, CMD, &mut p);
            let n = p.gpu_ids.iter().take_while(|&&g| g != INVALID).count();
            eprintln!("RM root GET_ATTACHED_IDS ok={:?} n={}", ok.is_some(), n);
            assert!(ok.is_some());
        });
    }

    #[test]
    #[cfg_attr(not(has_gpu), ignore)]
    fn rm_matches_nvml() {
        let _g = crate::arena::test_lock();
        let merge = |a: &mut GpuProc, b: &mut GpuProc| {
            a.mem_mib += b.mem_mib;
            a.sm_pct += b.sm_pct;
        };
        crate::arena::scope(|frame| {
            let mut nvml_buf: FVec<(u32, GpuProc)> = frame.vec("test/nvml", 2048);
            let (nv_n, nv_util, nv_used, nv_total) = super::super::nvml::populate(&mut nvml_buf);
            let nvml_procs = crate::map::freeze_map(&mut nvml_buf, merge);

            let state = super::init(frame);
            let (rm_entries, (rm_n, rm_util, rm_used, rm_total)) =
                super::populate(&state, frame);
            // populate already sort+deduped; wrap as Map directly.
            let rm_procs = crate::map::Map::new(rm_entries.as_slice());
            let _ = &merge;  // nvml path still uses freeze_map's merge

            eprintln!("RM:   n={rm_n} total_mib={rm_total} used_mib={rm_used} util={rm_util}");
            eprintln!("NVML: n={nv_n} total_mib={nv_total} used_mib={nv_used} util={nv_util}");
            assert_eq!(rm_n, nv_n);
            assert!(rm_total.abs_diff(nv_total) < 100, "total_mib mismatch");
            // Utilisation sampling happens at slightly different times; a few
            // percent per GPU is expected noise.
            let n = nv_n.max(1);
            assert!(rm_util.abs_diff(nv_util) < 20 * n,
                "util diff {} (RM={rm_util}, NVML={nv_util}, n={n} gpus)",
                rm_util.abs_diff(nv_util));

            // Per-process: every NVML PID must appear in RM with matching mem_mib.
            for (pid, nvml_p) in nvml_procs.iter() {
                let rm_p = rm_procs.get(pid).unwrap_or_else(|| panic!("pid {pid} missing in RM"));
                eprintln!("  pid {} NVML={} MiB / {}% RM={} MiB / {}%",
                    pid, nvml_p.mem_mib, nvml_p.sm_pct, rm_p.mem_mib, rm_p.sm_pct);
                assert_eq!(nvml_p.mem_mib, rm_p.mem_mib, "pid {pid} mem mismatch");
            }
        });
    }
}
