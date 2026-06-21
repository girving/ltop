//! Terminal I/O: helpers for writing a rendered frame to fd 1 via a
//! single `write(2)` loop, an unbuffered stderr writer (fd 2), and a
//! non-blocking one-byte read from stdin (fd 0). Bypasses
//! `std::io::Stdout` and its
//! `OnceLock<ReentrantLock<RefCell<LineWriter<StdoutRaw>>>>` chain —
//! we're single-threaded, nothing else writes to stdout, and we don't
//! need line buffering.
//!
//! Stdout buffering lives in the caller: render a frame into an
//! [`arena::FStr`](crate::arena::FStr) via `frame.span("tty/stdout", |b|
//! { write!(b, ...) })`, then pass the frozen bytes to [`write_stdout`].
//! The buffer's lifetime is exactly the render phase — no cross-tick
//! FVec pinned at init scope.

use crate::syscall;

/// Linux kernel `struct termios` (NCCS = 19, no `c_ispeed`/`c_ospeed`).
/// This is what the TCGETS/TCSETS ioctls read and write — 36 bytes.
/// glibc's userspace `struct termios` is 60 bytes with extra baud-rate
/// fields that TCGETS never populates; defining our own saves us pulling
/// the `libc` crate just for a type shim, and avoids carrying the 24 B
/// of trailing garbage across the save/restore pair.
///
/// macOS `struct termios` (arm64/LP64, NCCS=20): 72 bytes.
/// tcflag_t = unsigned long = u64 on LP64; c_cc[20] padded to 8-byte
/// boundary before c_ispeed/c_ospeed. Used with TIOCGETA/TIOCSETA
/// (0x40487413/0x80487414). <sys/termios.h>: ICANON=0x100, ECHO=0x8.
#[cfg(target_os = "linux")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Termios {
    c_iflag: u32,
    c_oflag: u32,
    c_cflag: u32,
    c_lflag: u32,
    c_line: u8,
    c_cc: [u8; 19],
}

// Linux c_lflag bits and c_cc[] indices. Magic numbers from
// <asm-generic/termbits.h>; Linux-specific but so are TCGETS/TCSETS.
#[cfg(target_os = "linux")]
const ICANON: u32 = 0o0000002;
#[cfg(target_os = "linux")]
const ECHO:   u32 = 0o0000010;
#[cfg(target_os = "linux")]
const VTIME: usize = 5;
#[cfg(target_os = "linux")]
const VMIN:  usize = 6;
#[cfg(target_os = "linux")]
const TCGET: u64 = syscall::TCGETS;
#[cfg(target_os = "linux")]
const TCSET: u64 = syscall::TCSETS;

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Termios {
    c_iflag: u64, c_oflag: u64, c_cflag: u64, c_lflag: u64,
    c_cc: [u8; 20], _pad: [u8; 4],
    c_ispeed: i64, c_ospeed: i64,
}
// macOS c_lflag bits and c_cc[] indices. <sys/termios.h>.
#[cfg(target_os = "macos")]
const ICANON: u64 = 0x100;
#[cfg(target_os = "macos")]
const ECHO:   u64 = 0x8;
#[cfg(target_os = "macos")]
const VTIME: usize = 17;
#[cfg(target_os = "macos")]
const VMIN:  usize = 16;
#[cfg(target_os = "macos")]
const TCGET: u64 = syscall::TIOCGETA;
#[cfg(target_os = "macos")]
const TCSET: u64 = syscall::TIOCSETA;

/// `ioctl(fd, cmd, &mut Termios)` — wraps the raw `syscall::ioctl` so
/// every termios get/set in this module is one safe call. The kernel
/// termios ioctls (TCGETS/TCSETS on Linux, TIOCGETA/TIOCSETA on mac)
/// read or write exactly the bytes of a `Termios`, so handing them
/// `&mut Termios` is sound.
#[inline]
fn ioctl_termios(fd: i32, cmd: u64, t: &mut Termios) -> i64 {
    // SAFETY: `t` is a valid `&mut Termios`; the kernel reads/writes
    // exactly its layout.
    unsafe { syscall::ioctl(fd, cmd, t as *mut _ as *mut _) }
}

/// Put stdin into raw mode (ICANON + ECHO off, non-blocking reads)
/// and hide the cursor. The returned guard restores both on drop.
/// `tcgetattr` / `tcsetattr` are library wrappers around the platform
/// ioctl (TCGETS/TCSETS on Linux, TIOCGETA/TIOCSETA on macOS); we issue
/// the ioctl directly to avoid pulling in libc::tcgetattr.
pub fn enter_raw_mode() -> RawModeGuard {
    // fd 0 = stdin. Avoids io::stdin() and its OnceLock-backed global.
    const FD: i32 = 0;
    let mut orig: Termios = Termios::default();
    ioctl_termios(FD, TCGET, &mut orig);
    let mut raw = orig;
    raw.c_lflag &= !(ICANON | ECHO);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    ioctl_termios(FD, TCSET, &mut raw);
    syscall::write_all(1, b"\x1b[?25l");
    RawModeGuard { fd: FD, orig }
}

/// RAII: restores the terminal to its pre-`enter_raw_mode` state when
/// dropped (normal return, `break 'main`, or unwinding-if-it-ever-
/// happens). Keeps the restore paired with the enter in one place.
pub struct RawModeGuard { fd: i32, orig: Termios }
impl Drop for RawModeGuard {
    fn drop(&mut self) {
        let mut t = self.orig;
        ioctl_termios(self.fd, TCSET, &mut t);
        syscall::write_all(1, b"\x1b[?25h");
    }
}

/// `(width, height)` in columns/rows of the controlling terminal on
/// fd 1, or `(80, 24)` if the ioctl fails (fd not a tty, redirected
/// output, etc.).
pub fn term_size() -> (usize, usize) {
    // `struct winsize` — same layout as libc's, kept local so we don't
    // pull the type in only for this one ioctl.
    #[repr(C)]
    #[derive(Default)]
    struct Winsize {
        row: u16,
        col: u16,
        xpixel: u16,
        ypixel: u16,
    }
    let mut ws = Winsize::default();
    // SAFETY: TIOCGWINSZ writes a struct winsize to the destination; `ws`
    // is a valid `&mut Winsize` and the layout matches.
    let ret = unsafe { syscall::ioctl(1, syscall::TIOCGWINSZ, &mut ws as *mut _ as *mut _) };
    if ret == 0 {
        let w = if ws.col > 0 { ws.col as usize } else { 80 };
        let h = if ws.row > 0 { ws.row as usize } else { 24 };
        (w, h)
    } else {
        (80, 24)
    }
}

/// Write `bytes` to fd 1, retrying on partial writes. Silently drops
/// bytes if `write(2)` errors — matches the old Stdout::flush behaviour.
pub fn write_stdout(bytes: &[u8]) {
    let mut off = 0;
    while off < bytes.len() {
        let n = syscall::write_all(1, &bytes[off..]);
        if n <= 0 { break; }
        off += n as usize;
    }
}

/// Non-blocking one-byte read from stdin (expects the fd to be in raw
/// non-canonical mode with VMIN=0 VTIME=0). Returns the byte on success.
pub fn read_one_stdin() -> Option<u8> {
    let mut b = 0u8;
    (syscall::read_buf(0, core::slice::from_mut(&mut b)) == 1).then_some(b)
}
