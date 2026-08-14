//! Darwin-specific FFI. Sibling of [`syscall`][crate::syscall] — whereas
//! that module wraps POSIX calls with Linux/macOS parity, this module
//! holds the Darwin-only surface: `libproc` flavors of `proc_info`,
//! `sysctl` MIB callers, Mach host statistics, and the
//! `struct proc_taskinfo` layout.
//!
//! Every wrapper takes references or slices so caller code in
//! `main.rs` / `arena.rs` holds no `unsafe` blocks of its own. Each
//! underlying call goes through raw `svc #0x80` syscalls in
//! [`syscall::sysctl`][crate::syscall::sysctl] /
//! [`syscall::proc_info`][crate::syscall::proc_info] — libSystem is
//! not in the call chain.

// ── Mach traps (Flavor A, phase 3) ─────────────────────────────────────────
//
// Negative trap numbers in x16 route the svc #0x80 into the Mach-trap
// table instead of the BSD-syscall table. Indices come from xnu's
// `osfmk/kern/syscall_sw.c`.
//
// Uses `mach_msg2_trap` (-47), NOT the deprecated `mach_msg_trap` (-31).
// The newer trap has a completely different packed-arg convention —
// see `mac-min.md` for the full rationale and the xnu source pointers.

const MACH_REPLY_PORT_TRAP:  i64 = -26;
const MACH_HOST_SELF_TRAP:   i64 = -29;
const MACH_MSG2_TRAP:        i64 = -47;

#[inline]
unsafe fn mach_host_self() -> u32 {
    use core::arch::asm;
    let port: u64;
    unsafe {
        asm!(
            "svc #0x80",
            in("x16") MACH_HOST_SELF_TRAP,
            lateout("x0") port,
            options(nostack),
        );
    }
    port as u32
}

#[inline]
unsafe fn mach_reply_port() -> u32 {
    use core::arch::asm;
    let port: u64;
    unsafe {
        asm!(
            "svc #0x80",
            in("x16") MACH_REPLY_PORT_TRAP,
            lateout("x0") port,
            options(nostack),
        );
    }
    port as u32
}

/// Cached receive-right reply port, allocated lazily. One port per
/// process; the kernel auto-generates a fresh MAKE_SEND_ONCE right
/// each time we pass the port name in `msgh_local_port`. Sidesteps
/// receive-vs-send right type confusion around `mach_port_deallocate`
/// (which SIGKILLs on a receive port).
fn cached_reply_port() -> u32 {
    use core::sync::atomic::{AtomicU32, Ordering};
    static PORT: AtomicU32 = AtomicU32::new(0);
    let p = PORT.load(Ordering::Relaxed);
    if p != 0 { return p; }
    let new = unsafe { mach_reply_port() };
    PORT.store(new, Ordering::Relaxed);
    new
}

/// `cargo test` runs the parity tests on parallel threads, but every
/// MIG sender shares the one cached reply port and no reply parser
/// validates msgh_id — two concurrent calls can dequeue each other's
/// replies (observed as rare flaky failures). Production is
/// single-threaded with one outstanding RPC, so serialisation is
/// test-only: each sender holds this lock across its send/receive
/// pair. Same pattern as `arena::test_lock`.
#[cfg(test)]
fn mig_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// `mach_msg2_trap` with all 8 packed args. Each 64-bit arg carries
/// two 32-bit message-header fields (low and high halves).
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn mach_msg2(
    data:                        *mut u8,
    options:                     u64,
    msgh_bits_and_send_size:     u64,
    msgh_remote_and_local_port:  u64,
    msgh_voucher_and_id:         u64,
    desc_count_and_rcv_name:     u64,
    rcv_size_and_priority:       u64,
    timeout:                     u64,
) -> i32 {
    use core::arch::asm;
    let ret: i64;
    unsafe {
        asm!(
            "svc #0x80",
            in("x16") MACH_MSG2_TRAP,
            inlateout("x0") data as u64 => ret,
            in("x1") options,
            in("x2") msgh_bits_and_send_size,
            in("x3") msgh_remote_and_local_port,
            in("x4") msgh_voucher_and_id,
            in("x5") desc_count_and_rcv_name,
            in("x6") rcv_size_and_priority,
            in("x7") timeout,
            options(nostack),
        );
    }
    ret as i32
}

// libproc `proc_info` sub-call numbers. One `callnum` + `flavor` pair
// picks the behavior; see `xnu/bsd/sys/proc_info.h`.
const PROC_INFO_CALL_LISTPIDS: i32 = 1;
const PROC_INFO_CALL_PIDINFO:  i32 = 2;
// Flavor for `proc_listallpids`: all processes.
const PROC_ALL_PIDS: u32 = 1;
// Flavors for `proc_pidinfo`.
pub const PROC_PIDTASKINFO: i32 = 4;
pub const PROC_PIDTBSDINFO: i32 = 3;
pub const PROC_PIDREGIONINFO: i32 = 7;
// Flavor for `host_statistics64`.
pub const HOST_VM_INFO64: i32 = 4;

/// Mirror of `struct proc_regioninfo` from `<sys/proc_info.h>`. C lays
/// it out with 4 bytes of padding before `pri_address` to give the u64
/// natural alignment; `repr(C)` reproduces that. Total size is 96 B
/// (`PROC_PIDREGIONINFO_SIZE`).
///
/// Caveat: on macOS 26 the kernel populates everything except
/// `pri_user_tag`, which is reported as 0 even for clearly-tagged
/// regions like libmalloc's. We compensate by ignoring user_tag and
/// filtering on the fields the kernel does fill in (protection,
/// pages_dirtied, address, size).
#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct ProcRegionInfo {
    pub pri_protection:                u32,  // bitmask: 1=R, 2=W, 4=X
    pub pri_max_protection:            u32,
    pub pri_inheritance:               u32,
    pub pri_flags:                     u32,
    pub pri_offset:                    u64,
    pub pri_behavior:                  u32,
    pub pri_user_tag:                  u32,
    pub pri_pages_resident:            u32,
    pub pri_pages_shared_now_private:  u32,
    pub pri_pages_swapped_out:         u32,
    pub pri_pages_dirtied:             u32,
    pub pri_ref_count:                 u32,
    pub pri_shadow_depth:              u32,
    pub pri_share_mode:                u32,
    pub pri_private_pages_resident:    u32,
    pub pri_shared_pages_resident:     u32,
    pub pri_obj_id:                    u32,
    pub pri_depth:                     u32,
    pub pri_address:                   u64,
    pub pri_size:                      u64,
}

/// Mirror of `struct proc_taskinfo` from <sys/proc_info.h>.
/// pti_resident_size is in bytes; pti_total_{user,system} in Mach absolute
/// time units (TASK_ABSOLUTETIME_INFO), i.e. mach_absolute_time() ticks —
/// 24 MHz on M1+ (numer=125, denom=3). Use mach_ticks_per_sec() to convert.
#[repr(C)]
#[derive(Default)]
pub struct ProcTaskInfo {
    pub pti_virtual_size:   u64,
    pub pti_resident_size:  u64,
    pub pti_total_user:     u64,
    pub pti_total_system:   u64,
    pub pti_threads_user:   u64,
    pub pti_threads_system: u64,
    pub pti_policy:             i32,
    pub pti_faults:             i32,
    pub pti_pageins:            i32,
    pub pti_cow_faults:         i32,
    pub pti_messages_sent:      i32,
    pub pti_messages_received:  i32,
    pub pti_syscalls_mach:      i32,
    pub pti_syscalls_unix:      i32,
    pub pti_csw:                i32,
    pub pti_threadnum:          i32,
    pub pti_numrunning:         i32,
    pub pti_priority:           i32,
}

// ── Safe wrappers ───────────────────────────────────────────────────────────
//
// Each wrapper takes references or slices so the pointer + length
// can't drift apart; the unsafe FFI is isolated inside the wrapper
// body. `#[inline]` so the wrapper is call-overhead-free after LTO.

/// `proc_listallpids` in query-size mode (null buffer). Returns the
/// **pid count**, matching libSystem's `proc_listallpids` semantics
/// (the underlying `__proc_info` syscall returns bytes-written;
/// libSystem divides by `sizeof(pid_t)` = 4 to turn that into a pid
/// count, and our wrapper does the same).
///
/// Argument aliasing: `proc_listpids(type, typeinfo, buf, bufsize)`
/// routes through `proc_info` with the `pid` slot carrying `type` and
/// the `flavor` slot carrying `typeinfo` — so we pass
/// `PROC_ALL_PIDS=1` as `pid` and `0` as `flavor`, NOT the other way
/// around. Easy to get wrong because the names "pid" and "flavor"
/// suggest a literal meaning that only applies to the PIDINFO callnum.
#[inline]
pub fn proc_listallpids_query() -> i32 {
    let bytes = unsafe {
        crate::syscall::proc_info(
            PROC_INFO_CALL_LISTPIDS, PROC_ALL_PIDS as i32, 0, 0,
            core::ptr::null_mut(), 0,
        )
    };
    if bytes < 0 { bytes } else { bytes / core::mem::size_of::<i32>() as i32 }
}

/// `proc_listallpids` filling `buf` with at most `buf.len()` pids.
/// Returns the number of pids written (matching libSystem).
#[inline]
pub fn proc_listallpids_into(buf: &mut [i32]) -> i32 {
    let bytes = unsafe {
        crate::syscall::proc_info(
            PROC_INFO_CALL_LISTPIDS, PROC_ALL_PIDS as i32, 0, 0,
            buf.as_mut_ptr() as *mut core::ffi::c_void,
            core::mem::size_of_val(buf) as i32,
        )
    };
    if bytes < 0 { bytes } else { bytes / core::mem::size_of::<i32>() as i32 }
}

/// `proc_pidinfo(pid, PROC_PIDTASKINFO, …)` into `ti`.
#[inline]
pub fn proc_pidtaskinfo(pid: i32, ti: &mut ProcTaskInfo) -> i32 {
    unsafe {
        crate::syscall::proc_info(
            PROC_INFO_CALL_PIDINFO, pid, PROC_PIDTASKINFO as u32, 0,
            ti as *mut _ as *mut core::ffi::c_void,
            core::mem::size_of::<ProcTaskInfo>() as i32,
        )
    }
}

/// `proc_pidinfo(pid, PROC_PIDTBSDINFO, …)` into `buf` as raw bytes.
/// The `proc_bsdinfo` struct has no safe Rust mirror here (we only
/// extract `pbi_ppid` + `pbi_start_tvsec` at fixed offsets), so a
/// `&mut [u8]` destination is the honest interface.
#[inline]
pub fn proc_pidbsdinfo(pid: i32, buf: &mut [u8]) -> i32 {
    unsafe {
        crate::syscall::proc_info(
            PROC_INFO_CALL_PIDINFO, pid, PROC_PIDTBSDINFO as u32, 0,
            buf.as_mut_ptr() as *mut core::ffi::c_void,
            buf.len() as i32,
        )
    }
}

/// Write the process's full name (up to 32 bytes, NUL-terminated) into
/// `buf`. Returns the byte count written (excluding trailing NUL), or
/// 0 on failure.
///
/// libSystem's `proc_name()` reads `pbi_name` out of the full
/// `proc_bsdinfo` (32-char field), not `pbsi_comm` out of
/// `proc_bsdshortinfo` (16-char field). The 16-char path would
/// truncate Rust test-binary names like `ltop-9d55a7a5af1ddafb` (21
/// chars); the 32-char path matches the reference exactly.
#[inline]
pub fn proc_name_into(pid: i32, buf: &mut [u8]) -> i32 {
    // proc_bsdinfo layout (xnu/bsd/sys/proc_info.h):
    //   [48..64] pbi_comm   char[MAXCOMLEN=16]
    //   [64..96] pbi_name   char[2*MAXCOMLEN=32]
    // pbi_name is what libSystem returns; reach into the full struct
    // via proc_pidbsdinfo (the same call already used for age/ppid).
    const PBI_NAME_OFFSET: usize = 64;
    const PBI_NAME_MAX:    usize = 32;
    let mut bsd = [0u8; 232]; // PROC_PIDTBSDINFO_SIZE
    let n = proc_pidbsdinfo(pid, &mut bsd);
    if n < (PBI_NAME_OFFSET + PBI_NAME_MAX) as i32 { return 0; }
    let name = &bsd[PBI_NAME_OFFSET..PBI_NAME_OFFSET + PBI_NAME_MAX];
    let end = name.iter().position(|&b| b == 0).unwrap_or(PBI_NAME_MAX);
    let copy = end.min(buf.len().saturating_sub(1));
    buf[..copy].copy_from_slice(&name[..copy]);
    if copy < buf.len() { buf[copy] = 0; }
    copy as i32
}

/// `sysctl` size-query: pass a MIB, get back the required byte count
/// for the sysctl value, or `None` on failure.
#[inline]
pub fn sysctl_size(mib: &mut [i32]) -> Option<usize> {
    let mut size = 0usize;
    let ret = unsafe {
        crate::syscall::sysctl(
            mib.as_mut_ptr(), mib.len() as u32,
            core::ptr::null_mut(), &mut size,
            core::ptr::null_mut(), 0,
        )
    };
    if ret == 0 { Some(size) } else { None }
}

/// `sysctl` read-into: fill `buf` (up to its capacity) from the MIB.
/// Returns the actual byte count written, or `None` on failure.
#[inline]
pub fn sysctl_read_bytes(mib: &mut [i32], buf: &mut [u8]) -> Option<usize> {
    let mut sz = buf.len();
    let ret = unsafe {
        crate::syscall::sysctl(
            mib.as_mut_ptr(), mib.len() as u32,
            buf.as_mut_ptr() as *mut core::ffi::c_void, &mut sz,
            core::ptr::null_mut(), 0,
        )
    };
    if ret == 0 { Some(sz) } else { None }
}

/// `sysctl` read-scalar: fill `out` (of type `T`) from the MIB.
/// Returns the syscall's raw return code (0 on success).
#[inline]
pub fn sysctl_read<T>(mib: &mut [i32], out: &mut T) -> i32 {
    let mut sz = core::mem::size_of::<T>();
    unsafe {
        crate::syscall::sysctl(
            mib.as_mut_ptr(), mib.len() as u32,
            out as *mut _ as *mut core::ffi::c_void, &mut sz,
            core::ptr::null_mut(), 0,
        )
    }
}

/// `host_statistics64(HOST_VM_INFO64)` read as `[u32; 38]` (i.e. the
/// `vm_statistics64_data_t` layout). Returns `None` on failure.
///
/// Built by hand on top of `mach_msg2_trap`. Message layout from
/// xnu's `osfmk/mach/mach_host.defs`: 24-byte header + 8-byte NDR tag
/// + flavor/count (request) or retcode/count/data[] (reply).
/// `mach_msg2_trap` takes the header fields as packed u64 args rather
/// than reading them from the message buffer; the buffer itself
/// carries only the body (NDR tag onward) from the kernel's POV,
/// though we allocate space for the full header+body+trailer because
/// the kernel writes the reply header back.
pub fn vm_statistics64() -> Option<[u32; 38]> {
    // Mach message constants (xnu `osfmk/mach/message.h`).
    const MACH_MSG_TYPE_COPY_SEND:       u32 = 19;
    const MACH_MSG_TYPE_MAKE_SEND_ONCE:  u32 = 21;
    // MACH_MSGH_BITS(remote, local) = remote | (local << 8).
    const MSGH_BITS: u32 = MACH_MSG_TYPE_COPY_SEND | (MACH_MSG_TYPE_MAKE_SEND_ONCE << 8);

    // mach_msg2 option bits. The MACH64_MACH_MSG2 top bit is required
    // to signal "use the new packed-args dispatch" — missing it makes
    // the kernel misinterpret our args. MACH64_SEND_KOBJECT_CALL is
    // required for RPCs to kernel-object ports; the host port is one
    // such port (see xnu's `ipc_kobject.c` dispatch).
    const MACH64_MACH_MSG2:        u64 = 0x8000_0000_0000_0000;
    const MACH64_SEND_KOBJECT_CALL: u64 = 0x0000_0002_0000_0000;
    const MACH64_SEND_MSG:         u64 = 0x1;
    const MACH64_RCV_MSG:          u64 = 0x2;

    // host subsystem base msgid = 200; host_statistics64 is routine
    // offset 19 (see comment in mac-min.md for the full routine list).
    const HOST_STATISTICS64_ID: u32 = 219;
    const HOST_VM_INFO64_COUNT: u32 = 38;

    // NDR_record_0 on arm64 LE: int_rep=NDR_INT_LITTLE_ENDIAN=1, rest 0.
    const NDR_LE: [u8; 8] = [0, 0, 0, 0, 1, 0, 0, 0];

    // Unified request/reply buffer layout. Request fills 40 bytes;
    // reply overlaps at offsets 24+ with retcode/count/data.
    #[repr(C)]
    struct Msg {
        msgh_bits:         u32,   // 0
        msgh_size:         u32,   // 4
        msgh_remote_port:  u32,   // 8
        msgh_local_port:   u32,   // 12
        msgh_voucher_port: u32,   // 16
        msgh_id:           u32,   // 20
        ndr:               [u8; 8], // 24
        word0:             i32,   // 32  flavor / retcode
        word1:             u32,   // 36  count-in / count-out
        data:              [u32; 38], // 40  reply data
        trailer:           [u8; 32], // 192  kernel-written trailer
    }
    const REQ_SIZE: u32 = 40;                  // header + NDR + flavor + count
    const RCV_MAX:  u32 = 40 + 38 * 4 + 32;    // + data + trailer = 224

    let host = unsafe { mach_host_self() };
    #[cfg(test)]
    let _mig_serial = mig_test_lock();
    let reply = cached_reply_port();
    if host == 0 || reply == 0 { return None; }

    let mut msg = Msg {
        msgh_bits:         MSGH_BITS,
        msgh_size:         REQ_SIZE,
        msgh_remote_port:  host,
        msgh_local_port:   reply,
        msgh_voucher_port: 0,
        msgh_id:           HOST_STATISTICS64_ID,
        ndr:               NDR_LE,
        word0:             HOST_VM_INFO64,
        word1:             HOST_VM_INFO64_COUNT,
        data:              [0; 38],
        trailer:           [0; 32],
    };

    // Pack args per struct mach_msg2_trap_args: low 32 bits in the
    // first-named field, high 32 bits in the second-named field.
    let options = MACH64_MACH_MSG2 | MACH64_SEND_KOBJECT_CALL
                | MACH64_SEND_MSG | MACH64_RCV_MSG;
    let msgh_bits_and_send_size    = (MSGH_BITS as u64) | ((REQ_SIZE as u64) << 32);
    let msgh_remote_and_local_port = (host as u64) | ((reply as u64) << 32);
    let msgh_voucher_and_id        = 0u64 | ((HOST_STATISTICS64_ID as u64) << 32);
    let desc_count_and_rcv_name    = 0u64 | ((reply as u64) << 32);
    let rcv_size_and_priority      = RCV_MAX as u64; // priority = 0 in upper half

    let kr = unsafe {
        mach_msg2(
            &mut msg as *mut _ as *mut u8,
            options,
            msgh_bits_and_send_size,
            msgh_remote_and_local_port,
            msgh_voucher_and_id,
            desc_count_and_rcv_name,
            rcv_size_and_priority,
            0, // timeout: MACH_MSG_TIMEOUT_NONE
        )
    };

    if kr != 0 { return None; }
    if msg.word0 != 0 { return None; } // RPC-level retcode
    Some(msg.data)
}

// ── IOKit master port (libIOKit-free Phase 1) ───────────────────────────────
//
// Both libIOKit and libCoreFoundation get dropped from the link line by
// reaching IOKit through raw Mach IPC. Phase 1 of `mac-iokit-free.md` is
// just two MIG calls: `host_get_io_master` (so we can produce the master
// port that IOKit RPCs require) and `mach_port_deallocate` (so we can
// release the send rights they hand back).
//
// Both reuse `mach_msg2`'s packed-arg trap below. Each is its own
// hand-rolled message buffer because the layouts differ — adding a
// generic `mig_call` helper before we have multiple callers would be
// premature; the call sites are short enough that copy-paste is clearer
// than abstraction. Phase 2 picks up the pattern and Phase 3 may extract
// a helper if the duplication actually hurts.

/// `host_get_io_master(mach_host_self(), &out)` — MIG routine 205 on the
/// `mach_host` subsystem (base 200). Returns the IOKit "main port" (a
/// Mach send right) — the value `kIOMainPortDefault` / `kIOMasterPortDefault`
/// stand in for via libIOKit's lazy substitution. Required as the first
/// argument to every IOKit MIG call; passing 0 directly to those RPCs
/// does not work the way it does through the cover library.
///
/// Cached across calls; one MIG round-trip per process. Returns 0 on
/// failure (callers treat that as "no IOKit available", same as the
/// libIOKit path treats `IOServiceGetMatchingServices` returning a
/// non-success kern_return).
pub fn io_master_port() -> u32 {
    use core::sync::atomic::{AtomicU32, Ordering};
    static MASTER: AtomicU32 = AtomicU32::new(0);
    let p = MASTER.load(Ordering::Relaxed);
    if p != 0 { return p; }
    let new = unsafe { host_get_io_master_uncached() };
    if new != 0 {
        MASTER.store(new, Ordering::Relaxed);
    }
    new
}

#[inline(never)]
unsafe fn host_get_io_master_uncached() -> u32 {
    // ── Mach message constants ────────────────────────────────────────
    const MACH_MSG_TYPE_COPY_SEND:       u32 = 19;
    const MACH_MSG_TYPE_MAKE_SEND_ONCE:  u32 = 21;
    const MSGH_BITS: u32 =
        MACH_MSG_TYPE_COPY_SEND | (MACH_MSG_TYPE_MAKE_SEND_ONCE << 8);
    const MACH_MSGH_BITS_COMPLEX: u32 = 0x8000_0000;

    const MACH64_MACH_MSG2:         u64 = 0x8000_0000_0000_0000;
    const MACH64_SEND_KOBJECT_CALL: u64 = 0x0000_0002_0000_0000;
    const MACH64_SEND_MSG:          u64 = 0x1;
    const MACH64_RCV_MSG:           u64 = 0x2;

    const HOST_GET_IO_MASTER_ID: u32 = 205;

    // Request: just header (24 bytes) — no body args.
    // Reply: header (24) + descriptor count (4) + port descriptor (12)
    // + kernel-written trailer (32 bytes). On error the body is NDR (8)
    // + kern_return_t (4) instead — distinguish via the COMPLEX bit in
    // `msgh_bits` of the reply header.
    #[repr(C)]
    struct Msg {
        msgh_bits:         u32,        // 0
        msgh_size:         u32,        // 4
        msgh_remote_port:  u32,        // 8
        msgh_local_port:   u32,        // 12
        msgh_voucher_port: u32,        // 16
        msgh_id:           u32,        // 20
        // Reply body: success laid out as a port descriptor (3 u32s
        // after the count); error as NDR + retcode (3 u32s). Either
        // way 16 bytes covers both shapes; the trailer follows in the
        // remaining slack.
        body:              [u32; 4],   // 24
        trailer:           [u8; 32],   // 40
    }
    const REQ_SIZE: u32 = 24;
    const RCV_MAX:  u32 = 24 + 4 + 12 + 32;     // 72

    let host  = unsafe { mach_host_self() };
    #[cfg(test)]
    let _mig_serial = mig_test_lock();
    let reply = cached_reply_port();
    if host == 0 || reply == 0 { return 0; }

    let mut msg = Msg {
        msgh_bits:         MSGH_BITS,
        msgh_size:         REQ_SIZE,
        msgh_remote_port:  host,
        msgh_local_port:   reply,
        msgh_voucher_port: 0,
        msgh_id:           HOST_GET_IO_MASTER_ID,
        body:              [0; 4],
        trailer:           [0; 32],
    };

    let options = MACH64_MACH_MSG2 | MACH64_SEND_KOBJECT_CALL
                | MACH64_SEND_MSG  | MACH64_RCV_MSG;
    let msgh_bits_and_send_size    = (MSGH_BITS as u64) | ((REQ_SIZE as u64) << 32);
    let msgh_remote_and_local_port = (host as u64)      | ((reply as u64) << 32);
    let msgh_voucher_and_id        = (HOST_GET_IO_MASTER_ID as u64) << 32;
    let desc_count_and_rcv_name    = (reply as u64) << 32;
    let rcv_size_and_priority      = RCV_MAX as u64;

    let kr = unsafe {
        mach_msg2(
            &mut msg as *mut _ as *mut u8,
            options,
            msgh_bits_and_send_size,
            msgh_remote_and_local_port,
            msgh_voucher_and_id,
            desc_count_and_rcv_name,
            rcv_size_and_priority,
            0,
        )
    };
    if kr != 0 { return 0; }

    // Success replies have COMPLEX set and a descriptor; error replies
    // are simple with NDR + kern_return_t in the body. We don't need to
    // inspect kern_return_t — we just want a valid port or 0.
    if msg.msgh_bits & MACH_MSGH_BITS_COMPLEX == 0 { return 0; }
    let descriptor_count = msg.body[0];
    if descriptor_count != 1 { return 0; }
    msg.body[1]                                            // port descriptor's `name` field
}

/// `mach_port_deallocate(mach_task_self(), port)` — MIG routine 6 on the
/// `mach_port` subsystem (base 3200, msgh_id = 3206). Releases one
/// send-right reference on `port`, the libIOKit-free counterpart to
/// `IOObjectRelease`.
///
/// Returns 0 on success, -errno on failure (kern_return_t flattened
/// the same way our syscall wrappers do). Callers usually ignore it.
pub fn mach_port_deallocate(port: u32) -> i32 {
    if port == 0 { return 0; }
    unsafe { mach_port_deallocate_raw(port) }
}

#[inline(never)]
unsafe fn mach_port_deallocate_raw(port: u32) -> i32 {
    const MACH_MSG_TYPE_COPY_SEND:       u32 = 19;
    const MACH_MSG_TYPE_MAKE_SEND_ONCE:  u32 = 21;
    const MSGH_BITS: u32 =
        MACH_MSG_TYPE_COPY_SEND | (MACH_MSG_TYPE_MAKE_SEND_ONCE << 8);

    const MACH64_MACH_MSG2:         u64 = 0x8000_0000_0000_0000;
    const MACH64_SEND_KOBJECT_CALL: u64 = 0x0000_0002_0000_0000;
    const MACH64_SEND_MSG:          u64 = 0x1;
    const MACH64_RCV_MSG:           u64 = 0x2;

    const MACH_PORT_DEALLOCATE_ID: u32 = 3206;

    // Request: header (24) + NDR (8) + port-name (4) = 36 bytes.
    // Reply: simple — header (24) + NDR (8) + kern_return (4) = 36 bytes.
    #[repr(C)]
    struct Msg {
        msgh_bits:         u32,
        msgh_size:         u32,
        msgh_remote_port:  u32,
        msgh_local_port:   u32,
        msgh_voucher_port: u32,
        msgh_id:           u32,
        ndr:               [u8; 8],
        /// Request: the port name to deallocate. Reply: the MIG
        /// RetCode — mig_reply_error_t is header(24) + NDR(8) +
        /// kern_return_t, so the code lands at offset 32, right here.
        port_name_or_retcode: u32,
        trailer:           [u8; 36],
    }
    const REQ_SIZE: u32 = 36;
    const RCV_MAX:  u32 = 36 + 32;       // simple reply + trailer

    const NDR_LE: [u8; 8] = [0, 0, 0, 0, 1, 0, 0, 0];

    // Target port: our own task port. We used to assume the well-known
    // name `0x103` here, but on macOS 26 the kernel rejects that value
    // and `mach_msg2` returns `MACH_SEND_INVALID_DEST` (= 0x10000003) —
    // every `mach_port_deallocate` was silently leaking the IOKit
    // send-right (and the kernel object it referenced). The portable
    // fix is `task_self_trap`, cached.
    let task_port_name: u32 = cached_task_self();

    #[cfg(test)]
    let _mig_serial = mig_test_lock();
    let reply = cached_reply_port();
    if reply == 0 { return -1; }

    let mut msg = Msg {
        msgh_bits:         MSGH_BITS,
        msgh_size:         REQ_SIZE,
        msgh_remote_port:  task_port_name,
        msgh_local_port:   reply,
        msgh_voucher_port: 0,
        msgh_id:           MACH_PORT_DEALLOCATE_ID,
        ndr:               NDR_LE,
        port_name_or_retcode: port,
        trailer:           [0; 36],
    };

    let options = MACH64_MACH_MSG2 | MACH64_SEND_KOBJECT_CALL
                | MACH64_SEND_MSG  | MACH64_RCV_MSG;
    let msgh_bits_and_send_size    = (MSGH_BITS as u64) | ((REQ_SIZE as u64) << 32);
    let msgh_remote_and_local_port = (task_port_name as u64) | ((reply as u64) << 32);
    let msgh_voucher_and_id        = (MACH_PORT_DEALLOCATE_ID as u64) << 32;
    let desc_count_and_rcv_name    = (reply as u64) << 32;
    let rcv_size_and_priority      = RCV_MAX as u64;

    let kr = unsafe {
        mach_msg2(
            &mut msg as *mut _ as *mut u8,
            options,
            msgh_bits_and_send_size,
            msgh_remote_and_local_port,
            msgh_voucher_and_id,
            desc_count_and_rcv_name,
            rcv_size_and_priority,
            0,
        )
    };
    if kr != 0 { return kr; }
    msg.port_name_or_retcode as i32
}

// ── IOKit MIG calls (Phase 2 of mac-iokit-free.md) ──────────────────────────
//
// Five IOKit MIG routines, all sent via mach_msg2 with the same packed-arg
// pattern as host_statistics64 / host_get_io_master. Routine IDs were
// confirmed by disassembling libIOKit's wrappers in the dyld shared cache
// (the Xcode SDK's iokitmig.h is a one-line stub these days, so the source
// of truth is the binary itself):
//
//   io_iterator_next                       = 2802
//   io_registry_entry_get_child_iterator   = 2813
//   io_registry_entry_get_properties_bin   = 2878  (OOL reply)
//   io_registry_entry_get_property_bin     = 2879  (single property, OOL reply;
//                                                   layout notes in mac-property-bin.md)
//   io_service_get_matching_services_bin   = 2881  (binary serialised matching dict)
//
// The `_bin` variants take the matching dict / receive properties as raw
// kOSSerializeBinary blobs — exactly what we'll parse ourselves in Phase 3.
// libIOKit picks `_bin` over the legacy XML routine when the global
// `_gIOKitLibSerializeOptions` flag has bit 0 set (default on modern macOS);
// we hardcode the `_bin` path because we control the encoder.

/// Hand-encoded `IOServiceMatching("AGXAccelerator")` as a kOSSerializeBinary
/// blob. The cover function returns a `CFDictionary { "IOProviderClass" =
/// "AGXAccelerator" }`; libIOKit's `IOServiceGetMatchingServices` then
/// serialises that dict via `IOCFSerialize` and feeds the bytes into
/// `io_service_get_matching_services_bin`. We skip the CF round-trip and
/// pin the bytes at compile time.
///
/// Layout (4-byte words, little-endian):
///   magic               D3 00 00 00     (kOSSerializeBinarySignature)
///   Dict, count=1       01 00 00 01
///   Symbol, len=16      08 00 00 10     ("IOProviderClass\0")
///   "IOProviderClass\0"
///   String|EOC, len=14  09 00 00 8E     ("AGXAccelerator")
///   "AGXAccelerator" + 2-byte tail pad to 4-byte boundary
pub static AGX_MATCHING_BLOB: [u8; 48] = [
    // OSSerializeBinary signature.
    0xd3, 0x00, 0x00, 0x00,
    // Dictionary, 1 pair, no EOC (this is the top-level container).
    0x01, 0x00, 0x00, 0x01,
    // Symbol, length 16 bytes (incl trailing NUL).
    0x10, 0x00, 0x00, 0x08,
    b'I', b'O', b'P', b'r', b'o', b'v', b'i', b'd',
    b'e', b'r', b'C', b'l', b'a', b's', b's', 0x00,
    // String, length 14, end-of-collection bit set (last item of the
    // enclosing dict). Stored little-endian as 0x0e, 0x00, 0x00, 0x89.
    0x0e, 0x00, 0x00, 0x89,
    b'A', b'G', b'X', b'A', b'c', b'c', b'e', b'l',
    b'e', b'r', b'a', b't', b'o', b'r', 0x00, 0x00,
];

/// `io_service_get_matching_services_bin(master, blob)` — MIG routine 2881.
/// Sends the matching dict as inline binary data; receives a Mach send
/// right to an `io_iterator_t`. Returns the iterator port, or 0 on
/// failure. Caller eventually releases the port via `mach_port_deallocate`.
///
/// The blob is capped at 4095 bytes (libIOKit's threshold for the inline
/// vs OOL variant); ours is 48.
pub fn io_service_get_matching_services_bin(master: u32, blob: &[u8]) -> u32 {
    if master == 0 || blob.len() >= 4096 { return 0; }
    unsafe { iokit_iter_call(2881, master, Some(blob)) }
}

/// `io_iterator_next(iter)` — MIG routine 2802. Pops one Mach send right
/// off the iterator. Returns 0 when the iterator is exhausted (kernel
/// signals this via `kIOReturnNoDevice` in the retcode, which our code
/// rolls into a 0 return). Caller releases the popped port via
/// `mach_port_deallocate`.
pub fn io_iterator_next(iter: u32) -> u32 {
    if iter == 0 { return 0; }
    unsafe { iokit_iter_call(2802, iter, None) }
}

/// `io_registry_entry_get_child_iterator(entry, plane)` — MIG routine 2813.
/// Allocates a child iterator over `entry`'s descendants in the given
/// `plane` (e.g., `b"IOService\0"`). Returns the iterator port or 0 on
/// failure. `plane` is sent as an `io_name_t` (NUL-terminated, ≤128 B).
pub fn io_registry_entry_get_child_iterator(entry: u32, plane: &[u8]) -> u32 {
    if entry == 0 || plane.is_empty() || plane.len() > 128 { return 0; }
    // The plane-name MIG arg is `char[128]` — fixed buffer, NUL-terminated.
    // We zero-fill and the caller passes a slice that ends with NUL.
    unsafe { iokit_get_child_iter(entry, plane) }
}

/// `io_registry_entry_get_properties_bin(entry)` — MIG routine 2878.
/// The kernel serialises the entry's `OSDictionary` to kOSSerializeBinary
/// and returns it as an out-of-line memory descriptor. We wrap the
/// returned address+size in an `OolBuffer` whose `Drop` calls
/// `vm_deallocate` so the kernel-mapped pages get released even on
/// early returns.
pub fn io_registry_entry_get_properties_bin(entry: u32) -> Option<OolBuffer> {
    if entry == 0 { return None; }
    unsafe { iokit_get_props_inner(2878, entry, None) }
}

/// `io_registry_entry_get_property_bin(entry, "", name, 0)` — MIG
/// routine 2879. The kernel serialises the single named property (via
/// `IOCopyPropertyCompatible` when the plane is empty) as the root of
/// a kOSSerializeBinary blob and returns it out-of-line — same reply
/// shape as `get_properties_bin`, but ~25× smaller for the AGX root
/// whose full table is dominated by IOReportLegend. `name` is a
/// NUL-terminated byte string ≤ 128 B (the kernel's `io_name_t`
/// ceiling). Layout research in mac-property-bin.md.
pub fn io_registry_entry_get_property_bin(entry: u32, name: &[u8]) -> Option<OolBuffer> {
    if entry == 0 || name.is_empty() || name.len() > 128 { return None; }
    unsafe { iokit_get_props_inner(2879, entry, Some(name)) }
}

/// RAII wrapper around an out-of-line memory descriptor returned by an
/// IOKit MIG call. The kernel allocates pages in our address space; we
/// own them until `vm_deallocate` releases them.
pub struct OolBuffer {
    addr: u64,
    size: u32,
}

impl OolBuffer {
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: kernel-allocated, valid for `size` bytes until Drop runs.
        unsafe { core::slice::from_raw_parts(self.addr as *const u8, self.size as usize) }
    }
}

impl Drop for OolBuffer {
    fn drop(&mut self) {
        if self.addr != 0 && self.size != 0 {
            let kr = unsafe { vm_deallocate(self.addr, self.size as u64) };
            // A non-zero kr here means the kernel rejected our deallocate
            // and the OOL buffer just leaked into our address space.
            // Historically this fired on macOS 26 because we were spelling
            // mach_task_self() as the well-known name `0x103` instead of
            // the value returned by `task_self_trap`; if it fires again
            // it's a similar ABI regression worth a yelp + a test. We
            // hard-fail in tests (catches regressions in CI; `cargo test
            // --release` strips `debug_assert!`) and silently leak in
            // production — better a slow-growing process than a TUI
            // that aborts over a kernel ABI shift.
            #[cfg(test)]
            assert_eq!(kr, 0, "vm_deallocate failed: kr=0x{:x}", kr);
            let _ = kr;
        }
    }
}

/// Walk every VM region in our task and `mach_vm_deallocate` the ones
/// we don't need, identified by *forward computation*: enumerate
/// what we keep (binary `__TEXT` + binary `__DATA` + arena `__bss` +
/// active stack frame + huge virtual reservations) and free the rest.
/// All five "keep" categories use anchors — addresses we can compute
/// at runtime that the keep-filter range-tests against — rather than
/// relying on protection bits or share modes.
///
/// What gets freed: libmalloc's ~288 KB of zone arenas
/// (`libSystem_initializer` sets these up even though our
/// `#[global_allocator]` is `Abort`), ~320 KB of writable shared-
/// cache pages that dyld dirtied during binding (`__DATA_DIRTY`,
/// `__OBJC_RW`, "unused but dirty shlib `__DATA`"), every libSystem
/// subdylib's `__TEXT`/`__LINKEDIT` (we never call into them — every
/// syscall is raw `svc #0x80`, every Mach call is hand-rolled
/// `mach_msg2`), the Apple commpage (arm64 reads `cntvct_el0` for
/// `mach_absolute_time` directly, not the commpage), and dyld's own
/// shared-cache `__DATA_CONST` / `__TPRO_CONST` once binding is
/// done. All file-backed COW pages can re-fault from the kernel
/// cache mapping if anyone ever does access them; private pages
/// (libmalloc) SIGSEGV at the access site. Either way the kernel
/// reports the violation rather than letting the process silently
/// corrupt state.
///
/// Effect on macOS 26 steady-state `phys_footprint`:
///   ~944 KB (libmalloc zones intact)
///   ~688 KB (libmalloc reclaimed, shared cache kept)
///   ~368 KB (libmalloc + writable shared-cache pages reclaimed)
///   ~240 KB (read-only catch-all replaced with explicit code anchor)
/// Region count drops 254 → 31 over the same path; `page table in
/// kernel` shrinks 224 → 160 KB as the kernel reclaims leaf
/// translation pages once enough VA goes away. Memory Tag 22 (dyld's
/// internal pool, 64 MB virtual / 16 KB dirty) IS load-bearing —
/// dyld touches it during the tick loop and SIGKILLs us if we free
/// it. We catch it via the `huge_reservation` filter.
///
/// Safety rests on three invariants we already established for the
/// libmalloc-only version:
///   1. We import zero LC_LOAD_DYLIB symbols (`nm -u` is empty), so
///      no lazy stubs exist that could trampoline back into a
///      now-unmapped library `__DATA` page.
///   2. We exit via raw `svc #0x80 / SYS_exit`; no atexit, no
///      libdyld teardown, no Objective-C destructors.
///   3. We're single-threaded with no signal handlers / GCD queues /
///      Mach notification ports — nothing async can fault on a
///      reclaimed page.
/// Any future change that breaks one of these will SIGSEGV at the
/// access site, exactly the kernel-as-witness property we want.
///
/// `pri_user_tag` would normally be the obvious "is this libmalloc?"
/// filter, but on macOS 26 the kernel reports 0 for every region in
/// `proc_pidregioninfo` — same for `pri_pages_dirtied`. We work
/// around this by enumerating regions to *keep* using only the
/// fields the kernel does populate reliably (protection, address,
/// size). Anything not on the keep list is fair game.
///
/// Returns the number of bytes unmapped.
#[cfg(target_os = "macos")]
pub fn unmap_idle_state(pid: i32) -> u64 {
    // ── Anchors we must not deallocate ────────────────────────────────────
    // Three runtime addresses cover everything our process owns:
    //   * `arena::extent()` — the arena lives in `__bss` (the
    //     512 KB `Storage`), and other zero-init statics like
    //     `cached_task_self`'s `TASK` AtomicU32 land in the same
    //     `__DATA,__bss` section. The kernel maps that whole
    //     section as one VM region; range-overlap against the
    //     arena's `(base, size)` keeps it.
    //   * `unmap_idle_state as *const ()` — code anchor, in
    //     `__TEXT`. Dropping the catch-all `read_only` filter
    //     would otherwise free our own code.
    //   * `mov sp` — the active stack frame's region.
    // We deliberately do NOT carry an anchor for `__DATA,__data`:
    // the production build has zero initialised writable statics
    // (every cache global is `AtomicU32::new(0)` → `__bss`), so
    // `__data` is empty and ld64 emits no file backing for the
    // `__DATA` segment — saving the 16 KB page that an
    // 8-byte anchor static would otherwise force. If anything
    // ever writes to `__data` and the kernel materialises a fresh
    // VM region for it, this code would (correctly) try to free
    // that region; the regression would surface as either an
    // immediate SIGSEGV in `--once` or a `kr != 0` debug-assert
    // somewhere downstream.
    let (arena_ptr, arena_len) = crate::arena::extent();
    let arena_lo = arena_ptr as u64;
    let arena_hi = arena_lo + arena_len as u64;
    let code_anchor = unmap_idle_state as *const () as u64;
    let stack_anchor: u64;
    unsafe { core::arch::asm!("mov {}, sp", out(reg) stack_anchor); }
    // (Used to keep all of [0x180000000, ∞) — the dyld shared-cache
    // range — out of fear of breaking libdyld/libobjc state. Dropped:
    // the read-only filter already keeps the COW-shared __TEXT /
    // __DATA_CONST / __TPRO_CONST pages, and the `huge_reservation`
    // filter still catches Memory Tag 22 / STACK GUARD. So all the
    // *writable* dirty regions in the shared-cache range are now
    // eligible for free if our `nm -u`-empty / no-libSystem-calls
    // invariants hold for them too.)

    let mut addr: u64 = 0;
    let mut total: u64 = 0;
    // ~50 regions is typical, 256 is paranoid.
    for _ in 0..256 {
        let mut info = ProcRegionInfo::default();
        let ret = unsafe { crate::syscall::proc_info(
            PROC_INFO_CALL_PIDINFO,
            pid,
            PROC_PIDREGIONINFO as u32,
            addr,
            &mut info as *mut _ as *mut core::ffi::c_void,
            core::mem::size_of::<ProcRegionInfo>() as i32,
        ) };
        if ret <= 0 { break; }
        let region_end = info.pri_address + info.pri_size;
        let next = region_end;

        // ── KEEP filter ───────────────────────────────────────────────────
        // Three runtime anchors: the arena's full byte range (which
        // covers the entire `__DATA,__bss` VM region the kernel maps
        // for our binary, including small bss statics adjacent to
        // ARENA), our binary's `__TEXT` (`code_anchor`), and the
        // active stack frame.
        let contains = |a: u64| info.pri_address <= a && a < region_end;
        let overlaps  = |lo: u64, hi: u64| lo < region_end && info.pri_address < hi;
        let contains_arena = overlaps(arena_lo, arena_hi);
        let contains_code  = contains(code_anchor);
        let contains_stack = contains(stack_anchor);
        // Big virtual reservations: Memory Tag 22 (dyld's malloc pool, 64
        // MB virtual, ~16 KB dirty) is load-bearing — dyld touches it
        // during the tick loop and SIGKILLs us if we free it. STACK
        // GUARD (56 MB, prot=0) is caught by `read_only` separately,
        // but it doesn't hurt to also catch it here.
        let huge_reservation = info.pri_size >= 32 * 1024 * 1024;
        // (We don't keep Apple's commpage here even though it's at a
        // known low address — `mach_absolute_time` on arm64 reads
        // `cntvct_el0` directly, not the commpage, and we issue every
        // syscall via raw `svc #0x80` rather than libSystem's stubs.
        // If we ever called something that actually reads the
        // commpage, the kernel would re-fault the read-only mapping
        // for us — they're file-backed COW from the kernel.)

        let keep = contains_arena || contains_code
                || contains_stack || huge_reservation;

        if !keep {
            let kr = unsafe { vm_deallocate(info.pri_address, info.pri_size) };
            if kr == 0 { total += info.pri_size; }
        }
        addr = next;
    }
    total
}

/// Look up the VM region containing `addr` via `proc_pidregioninfo` and
/// return `(region_start, region_end)`. Used by mac startup to find
/// the stack region's bounds so we can relocate SP near the top page
/// (1-page steady-state stack) and madvise the lower libdyld leftovers
/// off the books. Returns `None` if `addr` is unmapped.
#[cfg(target_os = "macos")]
pub fn region_containing(pid: i32, addr: u64) -> Option<(u64, u64)> {
    let mut info = ProcRegionInfo::default();
    let ret = unsafe { crate::syscall::proc_info(
        PROC_INFO_CALL_PIDINFO,
        pid,
        PROC_PIDREGIONINFO as u32,
        addr,
        &mut info as *mut _ as *mut core::ffi::c_void,
        core::mem::size_of::<ProcRegionInfo>() as i32,
    ) };
    if ret <= 0 { return None; }
    // proc_pidregioninfo returns the region containing `addr` OR (if
    // `addr` is in a hole) the next region above. Filter out the
    // latter: we only want the actual region we asked about.
    if addr < info.pri_address || addr >= info.pri_address + info.pri_size {
        return None;
    }
    Some((info.pri_address, info.pri_address + info.pri_size))
}

/// Our own task port via `task_self_trap` (Mach trap #-28), cached on
/// first call. Spells `mach_task_self()` without linking libSystem,
/// which writes the value to the global `mach_task_self_` during dyld
/// init.
///
/// Never assume the well-known name `0x103` (the value of that global
/// on some macOS versions): macOS 26 rejects it as
/// `MACH_SEND_INVALID_DEST` (`= 0x10000003`), silently failing every
/// `vm_deallocate` / `mach_port_deallocate` and leaking the IOKit OOL
/// buffers / send-rights (`phys_footprint` grew ~3 MB every few
/// seconds). The trap is the only ABI-stable spelling.
fn cached_task_self() -> u32 {
    use core::sync::atomic::{AtomicU32, Ordering};
    static TASK: AtomicU32 = AtomicU32::new(0);
    let cached = TASK.load(Ordering::Relaxed);
    if cached != 0 { return cached; }
    let port: u64;
    unsafe {
        core::arch::asm!(
            "svc #0x80",
            in("x16") -28i64,
            lateout("x0") port,
            options(nostack),
        );
    }
    let port = port as u32;
    TASK.store(port, Ordering::Relaxed);
    port
}

/// `mach_vm_deallocate(mach_task_self(), addr, size)` via the Mach trap
/// `_kernelrpc_mach_vm_deallocate_trap` (#-12). Direct trap, no MIG —
/// xnu's `osfmk/kern/syscall_sw.c` exposes this as a fastpath for the
/// allocate/deallocate pair.
///
/// Mach traps return `kern_return_t` directly in x0 — no carry-flag
/// error convention (that's the BSD syscall ABI). Success is 0; any
/// other value is a kern error code (KERN_INVALID_ARGUMENT, etc.).
/// libsyscall's `_mach_vm_deallocate_trap` is literally
/// `mov x16,#-12; svc #0x80; ret` for this reason.
#[inline]
unsafe fn vm_deallocate(addr: u64, size: u64) -> i32 {
    use core::arch::asm;
    const VM_DEALLOC_TRAP: i64 = -12;
    let ret: i64;
    let task = cached_task_self() as u64;
    unsafe {
        asm!(
            "svc #0x80",
            in("x16") VM_DEALLOC_TRAP,
            inlateout("x0") task => ret,
            in("x1") addr,
            in("x2") size,
            options(nostack),
        );
    }
    ret as i32
}

/// Shared inner for routines whose request is just a port (with optional
/// inline binary data) and reply is a single port descriptor — covers
/// `io_iterator_next`, `io_service_get_matching_services_bin`. The
/// `data` parameter switches between the two: `None` ⇒ no body
/// (iterator_next, request size 24); `Some(blob)` ⇒ NDR + length + data
/// padded to 4 bytes (matching_services_bin, request size 36 + padded blob).
#[inline(never)]
unsafe fn iokit_iter_call(msgh_id: u32, target: u32, data: Option<&[u8]>) -> u32 {
    const MACH_MSG_TYPE_COPY_SEND:       u32 = 19;
    const MACH_MSG_TYPE_MAKE_SEND_ONCE:  u32 = 21;
    const MSGH_BITS: u32 =
        MACH_MSG_TYPE_COPY_SEND | (MACH_MSG_TYPE_MAKE_SEND_ONCE << 8);
    const MACH_MSGH_BITS_COMPLEX: u32 = 0x8000_0000;

    const MACH64_MACH_MSG2:         u64 = 0x8000_0000_0000_0000;
    const MACH64_SEND_KOBJECT_CALL: u64 = 0x0000_0002_0000_0000;
    const MACH64_SEND_MSG:          u64 = 0x1;
    const MACH64_RCV_MSG:           u64 = 0x2;

    // Largest request: header(24) + NDR(8) + length(4) + 4096-byte blob = 4132.
    // Largest reply: header(24) + body_count(4) + port_descriptor(12) + trailer(32) = 72.
    // 4-KB stack buffer covers both with slack — well under the >1 KB rule's
    // arena guidance, but acceptable for a tick-rate routine with no
    // recursion below it.
    const BUF_SIZE: usize = 4096 + 64;
    let mut buf = [0u8; BUF_SIZE];

    #[cfg(test)]
    let _mig_serial = mig_test_lock();
    let reply = cached_reply_port();
    if reply == 0 { return 0; }

    // Header.
    let req_size: u32 = match data {
        None => 24,
        Some(d) => {
            // NDR (8 zero bytes — left as-is from buf init) + length (4) + padded data.
            let pad = (d.len() + 3) & !3;
            let off_len = 24 + 8;
            let off_data = off_len + 4;
            // Defensive: every public caller already bounds the blob,
            // but the inner `unsafe fn` shouldn't trust them. Reject
            // anything that wouldn't fit in `buf` rather than panic on
            // an OOB write inside `copy_from_slice`.
            if off_data + pad > BUF_SIZE { return 0; }
            buf[off_len..off_len + 4].copy_from_slice(&(d.len() as u32).to_le_bytes());
            buf[off_data..off_data + d.len()].copy_from_slice(d);
            (off_data + pad) as u32
        }
    };
    buf[0..4].copy_from_slice(&MSGH_BITS.to_le_bytes());
    buf[4..8].copy_from_slice(&req_size.to_le_bytes());
    buf[8..12].copy_from_slice(&target.to_le_bytes());
    buf[12..16].copy_from_slice(&reply.to_le_bytes());
    // voucher_port = 0 (already zero from buf init).
    buf[20..24].copy_from_slice(&msgh_id.to_le_bytes());

    let options = MACH64_MACH_MSG2 | MACH64_SEND_KOBJECT_CALL
                | MACH64_SEND_MSG  | MACH64_RCV_MSG;
    let msgh_bits_and_send_size    = (MSGH_BITS as u64) | ((req_size as u64) << 32);
    let msgh_remote_and_local_port = (target as u64) | ((reply as u64) << 32);
    let msgh_voucher_and_id        = (msgh_id as u64) << 32;
    let desc_count_and_rcv_name    = (reply as u64) << 32;
    let rcv_size_and_priority      = BUF_SIZE as u64;

    let kr = unsafe {
        mach_msg2(
            buf.as_mut_ptr(),
            options,
            msgh_bits_and_send_size,
            msgh_remote_and_local_port,
            msgh_voucher_and_id,
            desc_count_and_rcv_name,
            rcv_size_and_priority,
            0,
        )
    };
    if kr != 0 { return 0; }

    let reply_bits = u32::from_le_bytes(buf[0..4].try_into().unwrap());
    if reply_bits & MACH_MSGH_BITS_COMPLEX == 0 { return 0; }
    let descriptor_count = u32::from_le_bytes(buf[24..28].try_into().unwrap());
    if descriptor_count != 1 { return 0; }
    // Port descriptor: name (4) at body+4, then pad/disposition/type.
    u32::from_le_bytes(buf[28..32].try_into().unwrap())
}

/// `io_registry_entry_get_child_iterator` — request layout differs from
/// the iterator/matching pattern: header + NDR + 4-byte zero pad +
/// length + plane bytes (zero-padded to 4). Reply is the same single-
/// port-descriptor shape, so we share the parser tail.
#[inline(never)]
unsafe fn iokit_get_child_iter(entry: u32, plane: &[u8]) -> u32 {
    const MACH_MSG_TYPE_COPY_SEND:       u32 = 19;
    const MACH_MSG_TYPE_MAKE_SEND_ONCE:  u32 = 21;
    const MSGH_BITS: u32 =
        MACH_MSG_TYPE_COPY_SEND | (MACH_MSG_TYPE_MAKE_SEND_ONCE << 8);
    const MACH_MSGH_BITS_COMPLEX: u32 = 0x8000_0000;

    const MACH64_MACH_MSG2:         u64 = 0x8000_0000_0000_0000;
    const MACH64_SEND_KOBJECT_CALL: u64 = 0x0000_0002_0000_0000;
    const MACH64_SEND_MSG:          u64 = 0x1;
    const MACH64_RCV_MSG:           u64 = 0x2;

    const ID: u32 = 2813;

    // Layout: header(24) + NDR(8) + zero-pad(4) + length(4) + plane(≤128, padded to 4).
    // Max request: 24 + 8 + 4 + 4 + 128 = 168 bytes. Reply: 72 max.
    let mut buf = [0u8; 256];

    #[cfg(test)]
    let _mig_serial = mig_test_lock();
    let reply = cached_reply_port();
    if reply == 0 { return 0; }

    // Defensive: the public wrapper already enforces plane.len() ≤ 128
    // (the kernel's `io_name_t` ceiling), but the inner unsafe fn shouldn't
    // trust callers. Reject rather than panic on an OOB write below.
    if plane.len() > 128 { return 0; }
    let pad = (plane.len() + 3) & !3;
    buf[40..40 + plane.len()].copy_from_slice(plane);
    buf[36..40].copy_from_slice(&(plane.len() as u32).to_le_bytes());
    let req_size = (40 + pad) as u32;

    buf[0..4].copy_from_slice(&MSGH_BITS.to_le_bytes());
    buf[4..8].copy_from_slice(&req_size.to_le_bytes());
    buf[8..12].copy_from_slice(&entry.to_le_bytes());
    buf[12..16].copy_from_slice(&reply.to_le_bytes());
    buf[20..24].copy_from_slice(&ID.to_le_bytes());

    let options = MACH64_MACH_MSG2 | MACH64_SEND_KOBJECT_CALL
                | MACH64_SEND_MSG  | MACH64_RCV_MSG;
    let kr = unsafe {
        mach_msg2(
            buf.as_mut_ptr(),
            options,
            (MSGH_BITS as u64) | ((req_size as u64) << 32),
            (entry as u64) | ((reply as u64) << 32),
            (ID as u64) << 32,
            (reply as u64) << 32,
            buf.len() as u64,
            0,
        )
    };
    if kr != 0 { return 0; }

    let reply_bits = u32::from_le_bytes(buf[0..4].try_into().unwrap());
    if reply_bits & MACH_MSGH_BITS_COMPLEX == 0 { return 0; }
    let descriptor_count = u32::from_le_bytes(buf[24..28].try_into().unwrap());
    if descriptor_count != 1 { return 0; }
    u32::from_le_bytes(buf[28..32].try_into().unwrap())
}

/// Shared sender for the two property-fetch routines, whose replies
/// are byte-identical (a single OOL descriptor holding a
/// kOSSerializeBinary blob — see `parse_ool_reply`):
///
///   2878 `get_properties_bin` (`name` = None) — request is just the
///        header (24 bytes).
///   2879 `get_property_bin` (`name` = Some) — header + NDR + two MIG
///        counted strings + options. We always send plane = "" (⇒
///        direct property lookup, no plane iteration) and options = 0,
///        matching libIOKit's IORegistryEntryCreateCFProperty. Wire
///        layout pinned from the local libIOKit MIG stub — see
///        mac-property-bin.md:
///
///   header(24) + NDR(8)
///   + [offset 0 (4) | planeCnt (4) | "" padded (4)]     — plane
///   + [offset 0 (4) | nameCnt (4)  | name padded to 4]  — property name
///   + options (4)
#[inline(never)]
unsafe fn iokit_get_props_inner(id: u32, entry: u32, name: Option<&[u8]>) -> Option<OolBuffer> {
    const MACH_MSG_TYPE_COPY_SEND:       u32 = 19;
    const MACH_MSG_TYPE_MAKE_SEND_ONCE:  u32 = 21;
    const MSGH_BITS: u32 =
        MACH_MSG_TYPE_COPY_SEND | (MACH_MSG_TYPE_MAKE_SEND_ONCE << 8);

    const MACH64_MACH_MSG2:         u64 = 0x8000_0000_0000_0000;
    const MACH64_SEND_KOBJECT_CALL: u64 = 0x0000_0002_0000_0000;
    const MACH64_SEND_MSG:          u64 = 0x1;
    const MACH64_RCV_MSG:           u64 = 0x2;

    // Max request: 52 fixed + 4 (empty plane) + 128 (name) = 184. The
    // reply (header + body_count + OOL_desc + NDR + size + trailer =
    // 88, capped at 128 in the receive) reuses the same buffer.
    let mut buf = [0u8; 256];

    #[cfg(test)]
    let _mig_serial = mig_test_lock();
    let reply = cached_reply_port();
    if reply == 0 { return None; }

    let req_size: u32 = match name {
        None => 24,
        Some(name) => {
            // Public wrapper enforces this; the inner fn shouldn't
            // trust callers.
            if name.is_empty() || name.len() > 128 { return None; }
            // Plane "" at 32..44: offset 0, count 1 (just the NUL),
            // 4 zero bytes. NDR at 24..32 stays zeroed, as elsewhere.
            buf[36..40].copy_from_slice(&1u32.to_le_bytes());
            // Property name at 44..: offset 0, count (incl NUL),
            // padded bytes, then options = 0 (already zeroed).
            let pad = (name.len() + 3) & !3;
            buf[48..52].copy_from_slice(&(name.len() as u32).to_le_bytes());
            buf[52..52 + name.len()].copy_from_slice(name);
            (52 + pad + 4) as u32
        }
    };

    buf[0..4].copy_from_slice(&MSGH_BITS.to_le_bytes());
    buf[4..8].copy_from_slice(&req_size.to_le_bytes());
    buf[8..12].copy_from_slice(&entry.to_le_bytes());
    buf[12..16].copy_from_slice(&reply.to_le_bytes());
    buf[20..24].copy_from_slice(&id.to_le_bytes());

    let options = MACH64_MACH_MSG2 | MACH64_SEND_KOBJECT_CALL
                | MACH64_SEND_MSG  | MACH64_RCV_MSG;
    let kr = unsafe {
        mach_msg2(
            buf.as_mut_ptr(),
            options,
            (MSGH_BITS as u64) | ((req_size as u64) << 32),
            (entry as u64) | ((reply as u64) << 32),
            (id as u64) << 32,
            (reply as u64) << 32,
            128,
            0,
        )
    };
    if kr != 0 { return None; }
    parse_ool_reply(buf[..128].try_into().unwrap())
}

/// Shared tail for MIG replies whose payload is a single OOL memory
/// descriptor (`io_buf_ptr_t, physicalcopy`): `get_properties_bin`
/// (2878) and `get_property_bin` (2879) have byte-identical replies.
///
/// Reply layout (after header at offset 0):
///   offset 24: descriptor count (4 bytes, == 1)
///   offset 28: OOL descriptor (16 bytes)
///                addr (8) + flags (4, type byte at +3) + size (4)
///   offset 44: NDR (8 bytes)
///   offset 52: redundant size field (4 bytes)
///   offset 56: trailer (32 bytes the kernel writes)
fn parse_ool_reply(buf: &[u8; 128]) -> Option<OolBuffer> {
    const MACH_MSGH_BITS_COMPLEX: u32 = 0x8000_0000;
    let reply_bits = u32::from_le_bytes(buf[0..4].try_into().unwrap());
    if reply_bits & MACH_MSGH_BITS_COMPLEX == 0 { return None; }
    let descriptor_count = u32::from_le_bytes(buf[24..28].try_into().unwrap());
    if descriptor_count != 1 { return None; }
    // OOL descriptor at body offset 28..44:
    //   addr (8) at 28..36
    //   flags (4) at 36..40 — type byte at 39 (MACH_MSG_OOL_DESCRIPTOR = 1)
    //   size (4) at 40..44
    let addr = u64::from_le_bytes(buf[28..36].try_into().unwrap());
    let dtype = buf[39];
    let size = u32::from_le_bytes(buf[40..44].try_into().unwrap());
    if dtype != 1 || addr == 0 || size == 0 { return None; }
    Some(OolBuffer { addr, size })
}

// Everything below that used to go through libSystem's `sysconf` and
// `getloadavg` is backed by sysctl anyway. Routing directly saves the
// libSystem round-trip and keeps the "no libSystem calls" budget for
// Flavor A.

/// `hw.pagesize` — system page size. 16 KB on Apple silicon, 4 KB on
/// Intel macs. Fallback to 16 KB on error (present-day arm64 default).
#[inline]
pub fn hw_pagesize() -> i32 {
    let mut mib = [6, 7]; // CTL_HW, HW_PAGESIZE
    let mut n: i32 = 0;
    if sysctl_read(&mut mib, &mut n) == 0 && n > 0 { n } else { 16384 }
}

/// `hw.activecpu` — number of currently-online logical CPUs. Matches
/// what `sysconf(_SC_NPROCESSORS_ONLN)` does in libSystem.
#[inline]
pub fn hw_activecpu() -> i32 {
    let mut mib = [6, 25]; // CTL_HW, HW_AVAILCPU
    let mut n: i32 = 0;
    if sysctl_read(&mut mib, &mut n) == 0 && n > 0 { n } else { 1 }
}

/// `vm.loadavg` — the three load averages. Darwin returns a
/// `struct loadavg { fixpt_t ldavg[3]; long fscale; }`; fscale is the
/// divisor used to recover fractional loads from the fixpt counters
/// (typically 2048).
#[inline]
pub fn vm_loadavg() -> Option<[f64; 3]> {
    #[repr(C)]
    struct LoadAvg { ldavg: [u32; 3], fscale: i64 }
    let mut mib = [2, 2]; // CTL_VM, VM_LOADAVG
    let mut la = LoadAvg { ldavg: [0; 3], fscale: 0 };
    if sysctl_read(&mut mib, &mut la) != 0 || la.fscale <= 0 { return None; }
    let scale = la.fscale as f64;
    Some([
        la.ldavg[0] as f64 / scale,
        la.ldavg[1] as f64 / scale,
        la.ldavg[2] as f64 / scale,
    ])
}

/// `madvise` flag: "these anonymous pages are no longer needed; the
/// kernel can drop them now, and a subsequent access will zero-fault
/// a fresh page". Semantically the Darwin counterpart of Linux's
/// `MADV_DONTNEED` for anonymous memory — unlike plain `MADV_FREE`,
/// which on Darwin is lazy and only reclaims under memory pressure.
/// Value defined in Darwin's `<sys/mman.h>`.
pub const MADV_FREE_REUSABLE: i32 = 7;

/// `madvise(addr, len, MADV_FREE_REUSABLE)` — eagerly return a range of
/// anonymous pages to the kernel so RSS drops now, not under pressure.
/// Subsequent writes zero-fault. Callers must ensure `addr` and `len`
/// are page-aligned; misalignment yields `EINVAL`.
#[inline]
pub fn madvise_free_reusable(addr: *mut u8, len: usize) -> i32 {
    unsafe { crate::syscall::madvise(addr, len, MADV_FREE_REUSABLE) }
}

// ── Parity tests ───────────────────────────────────────────────────────────
//
// Each raw-syscall wrapper in this module is checked against its
// libSystem equivalent (via `libc::*` / the `extern "C"` declarations
// we keep for comparison). Catches wrong syscall numbers, struct
// offsets, MIB tuples, or flavor constants — any of which would yield
// garbage data at runtime otherwise.

#[cfg(test)]
mod tests {
    use super::*;

    // libSystem reference shims — only linked in tests. The production
    // path uses the raw-syscall wrappers above.
    unsafe extern "C" {
        fn proc_listallpids(buffer: *mut libc::c_void, buffersize: libc::c_int) -> libc::c_int;
        fn proc_pidinfo(
            pid: libc::c_int, flavor: libc::c_int, arg: u64,
            buffer: *mut libc::c_void, buffersize: libc::c_int,
        ) -> libc::c_int;
        fn mach_host_self() -> libc::c_uint;
        fn host_statistics64(
            host: libc::c_uint, flavor: libc::c_int,
            info: *mut libc::c_int, count: *mut libc::c_uint,
        ) -> libc::c_int;
    }

    #[test]
    fn hw_pagesize_matches_sysconf() {
        let libc_ps = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        let ours = hw_pagesize() as i64;
        assert_eq!(ours, libc_ps, "page size: ours={ours} libc={libc_ps}");
    }

    #[test]
    fn hw_activecpu_matches_sysconf() {
        let libc_n = unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) };
        let ours = hw_activecpu() as i64;
        assert_eq!(ours, libc_n, "activecpu: ours={ours} libc={libc_n}");
    }

    #[test]
    fn vm_loadavg_matches_getloadavg() {
        let mut libc_loads = [0.0f64; 3];
        let n = unsafe { libc::getloadavg(libc_loads.as_mut_ptr(), 3) };
        assert_eq!(n, 3, "getloadavg returned {n}");
        let Some(ours) = vm_loadavg() else { panic!("vm_loadavg returned None") };
        // Load averages drift between the two calls by a tiny amount;
        // they should agree within ~5%.
        for i in 0..3 {
            let diff = (ours[i] - libc_loads[i]).abs();
            let bound = libc_loads[i].abs() * 0.05 + 0.01;
            assert!(diff < bound,
                "loadavg[{i}] ours={} libc={} diff={diff} bound={bound}",
                ours[i], libc_loads[i]);
        }
    }


    #[test]
    fn proc_listallpids_matches_libsystem() {
        let libc_count = unsafe { proc_listallpids(core::ptr::null_mut(), 0) };
        let ours_count = proc_listallpids_query();
        assert!(libc_count > 0 && ours_count > 0,
                "counts: libc={libc_count} ours={ours_count}");
        assert!((libc_count - ours_count).abs() < 20,
                "counts differ: libc={libc_count} ours={ours_count}");

        let mut ours_buf = vec![0i32; ours_count as usize + 32];
        let filled = proc_listallpids_into(&mut ours_buf);
        assert!(filled > 0, "fill returned {filled}");
        let my_pid = std::process::id() as i32;
        assert!(ours_buf[..filled as usize].contains(&my_pid),
                "proc list missing self pid {my_pid}");
    }

    #[test]
    fn proc_pidtaskinfo_matches_libsystem() {
        let pid = std::process::id() as i32;
        let mut ti_libc = ProcTaskInfo::default();
        let n_libc = unsafe {
            proc_pidinfo(
                pid, PROC_PIDTASKINFO, 0,
                &mut ti_libc as *mut _ as *mut libc::c_void,
                core::mem::size_of::<ProcTaskInfo>() as i32,
            )
        };
        let mut ti_ours = ProcTaskInfo::default();
        let n_ours = proc_pidtaskinfo(pid, &mut ti_ours);
        assert_eq!(n_libc, n_ours, "return: libc={n_libc} ours={n_ours}");
        // Most struct fields drift between two synchronous calls
        // (threadnum in a parallel test runner, page-fault counters,
        // rusage tallies). Compare only pti_policy (constant for
        // normal processes) and verify both non-zero struct returns.
        assert_eq!(ti_libc.pti_policy, ti_ours.pti_policy, "policy differs");
        assert!(ti_libc.pti_resident_size > 0 && ti_ours.pti_resident_size > 0,
                "rss zero: libc={} ours={}",
                ti_libc.pti_resident_size, ti_ours.pti_resident_size);
    }

    /// Verify the units of pti_total_user by burning CPU for a known wall
    /// interval and checking the delta. Passes if the delta is within ±20%
    /// of the expected value.
    // libc 0.2.185 deprecated mach_absolute_time in favour of the `mach2`
    // crate; we don't add deps for a test helper.
    #[allow(deprecated)]
    #[test]
    fn pti_total_user_units() {
        let pid = std::process::id() as i32;
        let mut ti0 = ProcTaskInfo::default();
        assert!(proc_pidtaskinfo(pid, &mut ti0) > 0);
        let t0 = unsafe { libc::mach_absolute_time() };

        // Busy-loop for ~50 ms: gives a reliable non-zero CPU delta.
        let start = std::time::Instant::now();
        let mut x: u64 = 1;
        while start.elapsed() < std::time::Duration::from_millis(50) {
            for _ in 0..4096 { x = x.wrapping_mul(6364136223846793005).wrapping_add(1); }
            std::hint::black_box(x);
        }

        let t1 = unsafe { libc::mach_absolute_time() };
        let mut ti1 = ProcTaskInfo::default();
        assert!(proc_pidtaskinfo(pid, &mut ti1) > 0);

        let user_delta = ti1.pti_total_user.wrapping_sub(ti0.pti_total_user);
        let mach_delta = t1.wrapping_sub(t0);

        // If pti_total_user is in the same units as mach_absolute_time, the
        // ratio user_delta/mach_delta should be ≈ 1.0 (close to 1 core busy).
        // If pti_total_user is in nanoseconds and mach_absolute_time is in
        // Mach ticks (24 MHz on M1+), ratio would be ≈ 41.67 (ns >> ticks).
        let ratio = user_delta as f64 / mach_delta as f64;
        eprintln!("pti_total_user_units: user_delta={user_delta} mach_delta={mach_delta} ratio={ratio:.3}");

        // Should be between 0.5 and 1.5 — i.e., same clock family,
        // not off by 40× (which would indicate nanoseconds vs. ticks).
        assert!(ratio > 0.5 && ratio < 1.5,
            "ratio {ratio:.3} out of [0.5, 1.5]: pti_total_user units mismatch");
    }

    #[test]
    fn proc_pidbsdinfo_matches_libsystem() {
        let pid = std::process::id() as i32;
        let mut libc_buf = [0u8; 232];
        let n_libc = unsafe {
            proc_pidinfo(
                pid, PROC_PIDTBSDINFO, 0,
                libc_buf.as_mut_ptr() as *mut libc::c_void,
                libc_buf.len() as i32,
            )
        };
        let mut our_buf = [0u8; 232];
        let n_ours = proc_pidbsdinfo(pid, &mut our_buf);
        assert_eq!(n_libc, n_ours, "return: libc={n_libc} ours={n_ours}");
        // Bytes through offset 128 are stable (ppid, pgid, status, flags,
        // xstatus, pid, pbi_comm — all set-once fields). After that
        // there are uid/gid/rusage fields that also shouldn't change
        // between two synchronous calls in the same process.
        assert_eq!(libc_buf, our_buf, "struct bytes differ");
    }

    #[test]
    fn proc_name_into_matches_libsystem() {
        unsafe extern "C" {
            fn proc_name(pid: libc::c_int, buffer: *mut libc::c_void, buffersize: u32) -> libc::c_int;
        }
        let pid = std::process::id() as i32;
        let mut libc_buf = [0u8; 64];
        unsafe {
            proc_name(
                pid, libc_buf.as_mut_ptr() as *mut libc::c_void, libc_buf.len() as u32,
            );
        }
        let libc_end = libc_buf.iter().position(|&b| b == 0).unwrap_or(0);
        let libc_name = &libc_buf[..libc_end];

        let mut our_buf = [0u8; 64];
        proc_name_into(pid, &mut our_buf);
        let our_end = our_buf.iter().position(|&b| b == 0).unwrap_or(0);
        let our_name = &our_buf[..our_end];

        assert_eq!(libc_name, our_name,
                   "name: libc={:?} ours={:?}",
                   core::str::from_utf8(libc_name).unwrap_or("?"),
                   core::str::from_utf8(our_name).unwrap_or("?"));
    }

    #[test]
    fn mach_host_self_trap_nonzero() {
        let port = unsafe { super::mach_host_self() };
        assert_ne!(port, 0, "mach_host_self_trap returned 0");
    }

    #[test]
    fn cached_reply_port_is_stable() {
        let a = super::cached_reply_port();
        let b = super::cached_reply_port();
        assert_ne!(a, 0, "reply port was 0");
        assert_eq!(a, b, "cached_reply_port should be idempotent");
    }

    #[test]
    fn vm_statistics64_matches_host_statistics64() {
        // libSystem reference.
        let mut lib = [0u32; 38];
        let mut count = 38u32;
        let ret = unsafe {
            let host = mach_host_self();
            host_statistics64(
                host, HOST_VM_INFO64,
                lib.as_mut_ptr() as *mut libc::c_int, &mut count,
            )
        };
        assert_eq!(ret, 0, "libSystem host_statistics64 returned {ret}");
        assert_eq!(count, 38, "libSystem count = {count}");

        // Our raw Mach RPC.
        let ours = vm_statistics64().expect("vm_statistics64 returned None");

        // free + inactive pages (the quantity mem_info() actually uses)
        // drifts slightly between the two calls; compare within 5%.
        let lib_avail = lib[0] as u64 + lib[2] as u64;
        let our_avail = ours[0] as u64 + ours[2] as u64;
        let diff = if lib_avail > our_avail { lib_avail - our_avail }
                   else { our_avail - lib_avail };
        let bound = lib_avail / 20 + 1024;
        assert!(diff < bound,
                "avail pages: lib={lib_avail} ours={our_avail} diff={diff} bound={bound}");
        assert!(ours[0] > 0, "free pages = 0");
    }

    #[test]
    fn sysctl_hw_memsize_matches_libc() {
        let mut mib = [6i32, 24]; // CTL_HW, HW_MEMSIZE
        let mut libc_mem: u64 = 0;
        let mut sz = core::mem::size_of::<u64>();
        unsafe {
            libc::sysctl(
                mib.as_mut_ptr(), 2,
                &mut libc_mem as *mut _ as *mut libc::c_void, &mut sz,
                core::ptr::null_mut(), 0,
            );
        }
        let mut mib2 = [6i32, 24];
        let mut ours_mem: u64 = 0;
        sysctl_read(&mut mib2, &mut ours_mem);
        assert_eq!(libc_mem, ours_mem,
                   "hw.memsize: libc={libc_mem} ours={ours_mem}");
    }

    /// Our `io_master_port()` should produce a port we can use as the
    /// first argument to libIOKit's `IOServiceGetMatchingServices` and
    /// get the same iterator behaviour as passing `kIOMainPortDefault`
    /// (which is `0`, with libIOKit substituting the real master port
    /// internally). Both calls must succeed and yield iterators with
    /// the same number of entries — that's parity with the libIOKit
    /// path without requiring any specific service to be present.
    ///
    /// On a machine with no AGX hardware (CI, virtualized macOS) the
    /// counts are still equal — both yield 0 entries. The test
    /// genuinely catches a wrong port: a bad port value would either
    /// fail the kern_return check or return a different count.
    #[test]
    fn io_master_port_matches_iokit_default() {
        // libIOKit reference shims — test-only. Once the gpu/agx
        // production code stopped referencing these symbols, build.rs's
        // global `cargo:rustc-link-lib=framework=IOKit` no longer
        // reached the bin-test compilation reliably; mark the link
        // attribute locally on each extern block that needs it.
        #[link(name = "IOKit", kind = "framework")]
        unsafe extern "C" {
            fn IOServiceMatching(name: *const libc::c_char) -> *const core::ffi::c_void;
            fn IOServiceGetMatchingServices(
                main_port: u32, matching: *const core::ffi::c_void,
                existing: *mut u32,
            ) -> i32;
            fn IOIteratorNext(iter: u32) -> u32;
            fn IOObjectRelease(obj: u32) -> i32;
        }

        // CFDictionary references are owned by the caller; we leak
        // them for the duration of the test.
        let count_via = |master: u32| -> Option<u32> {
            let dict = unsafe {
                IOServiceMatching(b"AGXAccelerator\0".as_ptr() as *const _)
            };
            if dict.is_null() { return None; }
            let mut iter: u32 = 0;
            let kr = unsafe {
                IOServiceGetMatchingServices(master, dict, &mut iter)
            };
            // IOServiceGetMatchingServices consumes the dict — don't release.
            if kr != 0 { return None; }
            let mut n = 0u32;
            loop {
                let svc = unsafe { IOIteratorNext(iter) };
                if svc == 0 { break; }
                unsafe { IOObjectRelease(svc); }
                n += 1;
            }
            unsafe { IOObjectRelease(iter); }
            Some(n)
        };

        let ours_port = io_master_port();
        assert_ne!(ours_port, 0, "io_master_port() returned 0");

        let lib_count = count_via(0)
            .expect("kIOMainPortDefault path failed");
        let our_count = count_via(ours_port)
            .expect("our explicit master port failed");
        assert_eq!(
            lib_count, our_count,
            "AGXAccelerator count via our master ({our_count}) != \
             count via kIOMainPortDefault ({lib_count})"
        );
    }

    /// `io_master_port()` is cached — repeated calls return the same
    /// integer. Cheap sanity check that the static AtomicU32 is wired
    /// correctly.
    #[test]
    fn io_master_port_is_cached() {
        let a = io_master_port();
        let b = io_master_port();
        assert_eq!(a, b);
        assert_ne!(a, 0);
    }

    /// `mach_port_deallocate` releases a send right. We test it the
    /// way `IOObjectRelease` is normally exercised: get a port (an
    /// AGX accelerator's registry-entry port via libIOKit), then
    /// release it via *our* MIG call. Returns 0 on success. If there
    /// is no AGX (CI / virtualized macOS), we substitute a port we
    /// own — the IOKit master port itself, since calling
    /// `mach_port_deallocate` on a known-valid port should still
    /// succeed (it just releases one ref; libIOKit retains its own).
    #[test]
    fn mach_port_deallocate_returns_zero() {
        // Get a *fresh* IOKit master send right via the uncached path
        // and deallocate that one — `io_master_port()`'s cached value
        // is shared with every other test, and decrementing its
        // refcount can race a parallel test into MACH_SEND_INVALID_DEST
        // when the kernel sees a 1→0 transition before the other
        // thread's mach_msg2 looks up the name. (We deliberately pick
        // a port we know exists rather than `IOIteratorNext`-derived
        // ports because those require AGX hardware to test on CI.)
        let port = unsafe { host_get_io_master_uncached() };
        assert_ne!(port, 0);
        let kr = mach_port_deallocate(port);
        // kern_return_t = 0 on success. KERN_INVALID_RIGHT (17) would
        // mean we passed a bad port name. Anything else is a bug.
        assert_eq!(kr, 0, "mach_port_deallocate kr = {kr}");
    }

    /// The failure side of the tripwire: this test's whole point is
    /// catching leaks via a nonzero RetCode, which only works if we
    /// read the RetCode from the right reply offset (it once read the
    /// trailer and reported 0 unconditionally). Deallocating a
    /// receive-right name is a deterministic, harmless MIG-level error
    /// — the kernel answers KERN_INVALID_RIGHT (17) and the receive
    /// right survives.
    #[test]
    fn mach_port_deallocate_reports_real_errors() {
        let recv = super::cached_reply_port();
        assert_ne!(recv, 0);
        let kr = mach_port_deallocate(recv);
        assert_eq!(kr, 17, "expected KERN_INVALID_RIGHT, got {kr}");
    }

    // ── Phase 2: IOKit MIG parity tests ─────────────────────────────
    //
    // Compare each of our four IOKit RPC wrappers against the libIOKit
    // cover function it replaces. CI's virtualised macOS has no AGX
    // hardware, so we look for any registry entry that exists on every
    // mac (the root `IOService` plane root), not just AGXAccelerator.

    #[link(name = "IOKit", kind = "framework")]
    unsafe extern "C" {
        fn IOServiceMatching(name: *const libc::c_char) -> *const core::ffi::c_void;
        fn IOServiceGetMatchingServices(
            main_port: u32, matching: *const core::ffi::c_void, existing: *mut u32,
        ) -> i32;
        fn IOIteratorNext(iter: u32) -> u32;
        fn IOObjectRelease(obj: u32) -> i32;
        fn IORegistryEntryGetChildIterator(
            entry: u32, plane: *const libc::c_char, iter: *mut u32,
        ) -> i32;
    }

    /// Our `io_service_get_matching_services_bin` with the hand-encoded
    /// AGX blob must yield the same iterator results as libIOKit's
    /// `IOServiceGetMatchingServices` with `IOServiceMatching("AGXAccelerator")`.
    /// Comparing iteration counts works on AGX hosts (count > 0) and on
    /// virtualised hosts (count == 0); a non-zero diff means our blob
    /// doesn't match the kernel's parsed matching dict.
    #[test]
    fn io_service_get_matching_services_bin_matches_iokit() {
        let master = io_master_port();
        assert_ne!(master, 0);

        let count_via_lib = || -> u32 {
            let dict = unsafe { IOServiceMatching(b"AGXAccelerator\0".as_ptr() as *const _) };
            assert!(!dict.is_null());
            let mut iter: u32 = 0;
            let kr = unsafe {
                IOServiceGetMatchingServices(0, dict, &mut iter)
            };
            assert_eq!(kr, 0);
            let mut n = 0u32;
            loop {
                let svc = unsafe { IOIteratorNext(iter) };
                if svc == 0 { break; }
                unsafe { IOObjectRelease(svc); }
                n += 1;
            }
            unsafe { IOObjectRelease(iter); }
            n
        };

        let count_via_ours = || -> Option<u32> {
            let iter = io_service_get_matching_services_bin(master, &AGX_MATCHING_BLOB);
            if iter == 0 { return None; }
            let mut n = 0u32;
            loop {
                let svc = io_iterator_next(iter);
                if svc == 0 { break; }
                mach_port_deallocate(svc);
                n += 1;
            }
            mach_port_deallocate(iter);
            Some(n)
        };

        let lib = count_via_lib();
        // CI runs on virtualised macOS without AGX, so libIOKit returns 0
        // matches. Without an AGX device to anchor a comparison the
        // parity check isn't informative — skip. The synthetic-blob
        // tests (`agx_matching_blob_decodes_to_expected_dict` and
        // friends in osbinary) cover the encoder side regardless of
        // hardware.
        if lib == 0 {
            eprintln!("no AGXAccelerator matches libIOKit; skipping bin parity");
            return;
        }
        let ours = count_via_ours()
            .expect("io_service_get_matching_services_bin returned 0 on a host \
                     where libIOKit found AGX — likely a routine-ID mismatch \
                     against this macOS version");
        assert_eq!(lib, ours,
                   "AGX iterator count: libIOKit={lib} ours={ours} \
                    (blob: {} bytes)", AGX_MATCHING_BLOB.len());
    }

    /// Capture an AGXAccelerator (or any matching service) port via
    /// libIOKit and verify our `io_registry_entry_get_properties_bin`
    /// returns a non-empty kOSSerializeBinary blob whose magic and
    /// top-level Dictionary tag match. Skipped if the host has no
    /// AGX (CI / virtualised macOS).
    #[test]
    fn io_registry_entry_get_properties_bin_returns_serialised_dict() {
        let dict = unsafe { IOServiceMatching(b"AGXAccelerator\0".as_ptr() as *const _) };
        assert!(!dict.is_null());
        let mut iter: u32 = 0;
        let kr = unsafe {
            IOServiceGetMatchingServices(0, dict, &mut iter)
        };
        assert_eq!(kr, 0);
        let entry = unsafe { IOIteratorNext(iter) };
        unsafe { IOObjectRelease(iter); }
        if entry == 0 {
            eprintln!("no AGXAccelerator on this host; skipping");
            return;
        }

        let buf = io_registry_entry_get_properties_bin(entry)
            .expect("get_properties_bin returned None");
        let bytes = buf.as_bytes();
        assert!(!bytes.is_empty(), "properties blob is empty");
        // kOSSerializeBinary signature: 0xD3 0x00 0x00 0x00 (4-byte magic).
        assert_eq!(&bytes[0..4], &[0xd3, 0x00, 0x00, 0x00],
                   "wrong magic: got {:02x?}", &bytes[0..4.min(bytes.len())]);
        // Next u32 should be a Dictionary tag (top byte 0x01).
        let top_tag = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(top_tag & 0x7f00_0000, 0x0100_0000,
                   "top-level tag isn't a Dictionary: 0x{top_tag:08x}");
        unsafe { IOObjectRelease(entry); }
        // OolBuffer Drop releases the kernel-allocated bytes.
    }

    /// Cross-check MIG 2879 against 2878 on a service every Mac
    /// (including virtualised CI) has: IOPlatformExpertDevice's
    /// IOPlatformUUID. The single-property blob's root String must
    /// byte-match the same key inside the full properties dict — pins
    /// routine ID 2879, the two-counted-string request layout, and the
    /// shared OOL reply parse against the live kernel.
    #[test]
    fn io_registry_entry_get_property_bin_matches_properties_bin() {
        let dict = unsafe { IOServiceMatching(b"IOPlatformExpertDevice\0".as_ptr() as *const _) };
        assert!(!dict.is_null());
        let mut iter: u32 = 0;
        let kr = unsafe { IOServiceGetMatchingServices(0, dict, &mut iter) };
        assert_eq!(kr, 0);
        let entry = unsafe { IOIteratorNext(iter) };
        unsafe { IOObjectRelease(iter); }
        assert_ne!(entry, 0, "IOPlatformExpertDevice should exist on every Mac");

        // Reference: the full table via 2878, IOPlatformUUID extracted.
        let all = io_registry_entry_get_properties_bin(entry)
            .expect("get_properties_bin returned None");
        let mut ckpts_all = vec![0u32; 512];
        let bin_all = crate::osbinary::OsBinary::parse(all.as_bytes(), &mut ckpts_all)
            .expect("full-table blob should parse");
        let root_all = bin_all.root().unwrap();
        let want = bin_all.find_dict(root_all, b"IOPlatformUUID")
            .and_then(|v| bin_all.as_string(v))
            .expect("IOPlatformUUID missing from the full property table")
            .to_vec();

        // Single property via 2879 — the String is the blob's root.
        let one = io_registry_entry_get_property_bin(entry, b"IOPlatformUUID\0")
            .expect("get_property_bin returned None — routine-ID or request-\
                     layout mismatch against this macOS version");
        let mut ckpts_one = vec![0u32; 16];
        let bin_one = crate::osbinary::OsBinary::parse(one.as_bytes(), &mut ckpts_one)
            .expect("single-property blob should parse");
        let got = bin_one.root()
            .and_then(|r| bin_one.as_string(r))
            .expect("2879 blob root should be a String");
        assert_eq!(got, &want[..], "IOPlatformUUID: 2879 vs 2878 mismatch");

        // A missing property must come back None (MIG error reply has
        // no OOL descriptor), not garbage.
        assert!(io_registry_entry_get_property_bin(entry, b"NoSuchLtopProperty\0").is_none());
        unsafe { IOObjectRelease(entry); }
    }

    /// Our `io_registry_entry_get_child_iterator` on an AGXAccelerator
    /// should yield the same number of children as libIOKit's
    /// `IORegistryEntryGetChildIterator`.
    #[test]
    fn io_registry_entry_get_child_iterator_matches_iokit() {
        let dict = unsafe { IOServiceMatching(b"AGXAccelerator\0".as_ptr() as *const _) };
        assert!(!dict.is_null());
        let mut iter: u32 = 0;
        let kr = unsafe {
            IOServiceGetMatchingServices(0, dict, &mut iter)
        };
        assert_eq!(kr, 0);
        let entry = unsafe { IOIteratorNext(iter) };
        unsafe { IOObjectRelease(iter); }
        if entry == 0 {
            eprintln!("no AGXAccelerator on this host; skipping");
            return;
        }

        let plane = b"IOService\0";

        // libIOKit reference.
        let mut lib_iter: u32 = 0;
        let kr = unsafe {
            IORegistryEntryGetChildIterator(entry, plane.as_ptr() as *const _, &mut lib_iter)
        };
        assert_eq!(kr, 0);
        let mut lib_count = 0u32;
        loop {
            let c = unsafe { IOIteratorNext(lib_iter) };
            if c == 0 { break; }
            unsafe { IOObjectRelease(c); }
            lib_count += 1;
        }
        unsafe { IOObjectRelease(lib_iter); }

        // Our raw MIG.
        let our_iter = io_registry_entry_get_child_iterator(entry, plane);
        assert_ne!(our_iter, 0);
        let mut our_count = 0u32;
        loop {
            let c = io_iterator_next(our_iter);
            if c == 0 { break; }
            mach_port_deallocate(c);
            our_count += 1;
        }
        mach_port_deallocate(our_iter);

        unsafe { IOObjectRelease(entry); }
        assert_eq!(lib_count, our_count,
                   "child count: libIOKit={lib_count} ours={our_count}");
    }
}
