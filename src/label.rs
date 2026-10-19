//! Labels: a process may name its own COMMAND cell by writing one line
//! to `$XDG_CACHE_HOME/ltop/status/<pid>.<start>` (else
//! `~/.cache/ltop/status/…`). README "Labels" is the protocol; this is
//! the reader.
//!
//! `<start>` is the process start time as one decimal integer: on Linux
//! `/proc/<pid>/stat` field 22 (clock ticks since boot), on macOS
//! `p_starttime` as `tv_sec * 1000000 + tv_usec` (microseconds since
//! the epoch). Pid plus start time names one process for good, so a
//! file left behind by a dead process can't label its pid's next owner.
//!
//! The directory is opened once, from `parse_args` while the environment
//! is still mapped; the fd rides to `run` in the upper half of the flag
//! word, and on Linux the sandbox grants read and unlink beneath it.

use crate::arena::{FBox, Frame};
use crate::{syscall, ProcInfo};
use crate::platform::{PidDir, NO_START};

/// Longest label we read, in bytes. The rest of the file is ignored.
pub(crate) const LABEL_MAX: usize = 200;

/// Open the label directory named by the environment that follows
/// `argv`'s NULL. Returns the directory fd, or a negative value when it
/// is missing.
///
/// # Safety
/// `argv` must be the process's `argc`-element argv, followed by NULL
/// and the NULL-terminated envp array, as the kernel lays them out.
pub(crate) unsafe fn open_dir(argc: i32, argv: *const *const u8) -> i32 {
    let mut env = unsafe { argv.add(argc as usize + 1) };
    let mut home: *const u8 = core::ptr::null();
    let mut xdg: *const u8 = core::ptr::null();
    loop {
        let e = unsafe { *env };
        if e.is_null() { break; }
        if let Some(v) = unsafe { value(e, b"HOME=") } { home = v; }
        else if let Some(v) = unsafe { value(e, b"XDG_CACHE_HOME=") } { xdg = v; }
        env = unsafe { env.add(1) };
    }
    // XDG_CACHE_HOME stands in for `$HOME/.cache`; empty counts as unset
    // (XDG base-dir spec). Open the base, then the rest beneath it.
    const REST: &core::ffi::CStr = c".cache/ltop/status";
    let (base, rest) = if !xdg.is_null() && unsafe { *xdg } != 0 {
        (xdg, &REST.to_bytes_with_nul()[7..])
    } else {
        (home, REST.to_bytes_with_nul())
    };
    if base.is_null() { return -1; }
    let flags = syscall::O_RDONLY | syscall::O_DIRECTORY | syscall::O_CLOEXEC;
    unsafe {
        let b = syscall::open(base, flags);
        // A failed open's -ENOENT is -2, which is AT_FDCWD on macOS.
        if b < 0 { return b; }
        let fd = syscall::openat(b, rest.as_ptr(), flags);
        syscall::close(b);
        fd
    }
}

/// If the C string `e` starts with `key`, the rest of it.
#[inline(never)]
unsafe fn value(e: *const u8, key: &[u8]) -> Option<*const u8> {
    for (i, &k) in key.iter().enumerate() {
        if unsafe { *e.add(i) } != k { return None; }
    }
    Some(unsafe { e.add(key.len()) })
}

/// Show each label whose pid and start time match a live process as
/// that process's COMMAND (when the process is among `procs`), and
/// unlink every label whose process is gone or whose start time
/// differs. Names that aren't `<digits>.<digits>` (writers' temp files)
/// are left alone. A negative `dir` is a no-op.
pub(crate) fn apply<'a>(dir: i32, frame: &mut Frame<'a>, procs: &mut [ProcInfo<'a>]) {
    // One 1 KB scratch: dirents in the first half, file reads in the second.
    let mut scratch: FBox<[u64; 128]> = frame.alloc_zeroed("label/scratch");
    let (ents, buf) = scratch.split_at_mut(64);
    // SAFETY: 64 initialised u64s viewed as 512 bytes.
    let buf = unsafe { core::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, 512) };
    let Some(mut d) = PidDir::at(dir, c".", ents) else { return; };
    while let Some(pid) = d.next() {
        if d.start == NO_START { continue; }
        if start_time(pid, d.name, buf) != Some(d.start) {
            unsafe { syscall::unlinkat(dir, d.name); }
            continue;
        }
        let Some(p) = procs.iter_mut().find(|p| p.pid == pid) else { continue; };
        let n = unsafe {
            let fd = syscall::openat(dir, d.name, syscall::O_RDONLY | syscall::O_NONBLOCK | syscall::O_CLOEXEC);
            let n = syscall::read_buf(fd, &mut buf[..LABEL_MAX]);
            syscall::close(fd);
            n
        };
        if n <= 0 { continue; }
        p.display = frame.str("label", |b| {
            // One line; control bytes (C0 and DEL) dropped so the label
            // can't move the cursor or recolour the screen.
            for &c in &buf[..n as usize] {
                if c == b'\n' { break; }
                if c >= b' ' && c != 0x7f { b.push(c); }
            }
        }).into_small_lossy();
    }
}

/// Start time, in the label encoding, of the process a label named
/// `name` (`<pid>.<start>`) claims; `None` if it's gone.
#[cfg(target_os = "linux")]
fn start_time(_pid: u32, name: *const u8, buf: &mut [u8]) -> Option<u64> {
    // "/proc/<pid>/stat" in the tail of `buf`, read into the head.
    let (head, path) = buf.split_at_mut(480);
    path[..6].copy_from_slice(b"/proc/");
    let mut i = 6;
    loop {
        let c = unsafe { *name.add(i - 6) };
        if c == b'.' { break; }
        if i > 16 { return None; }
        path[i] = c;
        i += 1;
    }
    path[i..i + 6].copy_from_slice(b"/stat\0");
    // SAFETY: the pid digits are NUL-free; the only NUL is the terminator.
    let path = unsafe { core::ffi::CStr::from_bytes_with_nul_unchecked(&path[..i + 6]) };
    let stat = crate::read_small_file(path, head)?;
    let (_, after_comm) = crate::parse_stat(stat);
    crate::bytes::split_ascii_whitespace(after_comm).nth(19).and_then(crate::parse_u64_ascii)
}

/// Start time, in the label encoding, of the process a label named
/// `name` (`<pid>.<start>`) claims; `None` if it's gone.
/// `proc_bsdinfo.pbi_start_tvsec`/`_tvusec` (u64s at bytes 120/128)
/// carry the same `p_start` that `kinfo_proc.kp_proc.p_starttime` does.
#[cfg(target_os = "macos")]
fn start_time(pid: u32, _name: *const u8, buf: &mut [u8]) -> Option<u64> {
    if crate::mac_sys::proc_pidbsdinfo(pid as i32, buf) < 136 { return None; }
    let word = |o: usize| u64::from_ne_bytes(buf[o..o + 8].try_into().unwrap());
    Some(word(120) * 1_000_000 + word(128))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena;
    use std::ffi::CString;
    use std::fs;
    use std::os::fd::AsRawFd;
    use std::path::PathBuf;

    /// This process's start time, read the way a writer would: Linux
    /// `/proc/self/stat` through std, macOS libc's `proc_pidinfo`.
    fn my_start() -> u64 {
        #[cfg(target_os = "linux")]
        {
            let stat = fs::read_to_string("/proc/self/stat").unwrap();
            let after = &stat[stat.rfind(')').unwrap() + 2..];
            after.split(' ').nth(19).unwrap().parse().unwrap()
        }
        #[cfg(target_os = "macos")]
        unsafe {
            let mut info: libc::proc_bsdinfo = core::mem::zeroed();
            let size = core::mem::size_of_val(&info) as i32;
            let n = libc::proc_pidinfo(std::process::id() as i32, libc::PROC_PIDTBSDINFO, 0,
                                       &mut info as *mut _ as *mut libc::c_void, size);
            assert_eq!(n, size);
            info.pbi_start_tvsec * 1_000_000 + info.pbi_start_tvusec
        }
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("ltop-label-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    /// Run `apply` over `dir` with one proc row per pid in `pids`, each
    /// pre-set to the display "orig"; return the rows' displays.
    fn run(dir: &PathBuf, pids: &[u32]) -> Vec<Vec<u8>> {
        let _g = arena::test_lock();
        let dirf = fs::File::open(dir).unwrap();
        arena::scope(|f| {
            let mut procs = f.vec::<ProcInfo>("procs", pids.len() as u32);
            for &pid in pids {
                let display = f.str("orig", |b| b.extend_from_slice(b"orig")).into_small();
                let _ = procs.push(ProcInfo { pid, display, ..Default::default() });
            }
            apply(dirf.as_raw_fd(), f, &mut procs);
            procs.iter().map(|p| p.display.as_slice().to_vec()).collect()
        })
    }

    #[test]
    fn start_time_matches_writer_view() {
        let mut buf = [0u8; 512];
        let me = std::process::id();
        let name = format!("{me}.1\0");
        assert_eq!(start_time(me, name.as_ptr(), &mut buf), Some(my_start()));
        assert_eq!(start_time(2000000000, b"2000000000.1\0".as_ptr(), &mut buf), None);
    }

    #[test]
    fn matching_label_replaces_command() {
        let d = temp_dir("match");
        let me = std::process::id();
        let name = d.join(format!("{me}.{}", my_start()));
        fs::write(&name, "stamp serve :7468 · 3 stamps\nsecond line\n").unwrap();
        let shown = run(&d, &[1, me]);
        assert_eq!(shown[0], b"orig");
        assert_eq!(shown[1], "stamp serve :7468 · 3 stamps".as_bytes());
        assert!(name.exists(), "a live label must stay");
        fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn stale_labels_are_ignored_and_unlinked() {
        let d = temp_dir("stale");
        let me = std::process::id();
        // Start time off by one: the pid's owner isn't the writer.
        let wrong = d.join(format!("{me}.{}", my_start() + 1));
        // No such process: pid_max is below 2^22 on Linux, 99999 on macOS.
        let gone = d.join("2000000000.5");
        // Not label-shaped: left alone (a writer's temp file, say).
        let temp = d.join(format!(".{me}.5.tmp"));
        let other = d.join("notes");
        for p in [&wrong, &gone, &temp, &other] { fs::write(p, "x").unwrap(); }
        let shown = run(&d, &[me, 2000000000]);
        assert_eq!(shown, vec![b"orig".to_vec(), b"orig".to_vec()]);
        assert!(!wrong.exists() && !gone.exists());
        assert!(temp.exists() && other.exists());
        fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn control_characters_are_stripped() {
        let d = temp_dir("ctl");
        let me = std::process::id();
        fs::write(d.join(format!("{me}.{}", my_start())),
                  b"a\x1b[2Jb\tc\x07d\x7fe\r\xc3\xa9").unwrap();
        assert_eq!(run(&d, &[me])[0], "a[2Jbcde\u{e9}".as_bytes());
        fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn oversized_label_is_truncated() {
        let d = temp_dir("big");
        let me = std::process::id();
        fs::write(d.join(format!("{me}.{}", my_start())), vec![b'x'; 5000]).unwrap();
        assert_eq!(run(&d, &[me])[0], vec![b'x'; LABEL_MAX]);
        fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn missing_directory_is_fine() {
        let _g = arena::test_lock();
        arena::scope(|f| {
            let display = f.str("orig", |b| b.extend_from_slice(b"orig")).into_small();
            let mut procs = [ProcInfo { pid: 1, display, ..Default::default() }];
            apply(-1, f, &mut procs);
            assert_eq!(procs[0].display.as_slice(), b"orig");
        });
        assert!(open(&[("HOME", "/nonexistent-ltop-home")]) < 0);
        assert!(open(&[]) < 0);
    }

    /// `open_dir` over a synthetic argv/envp block.
    fn open(env: &[(&str, &str)]) -> i32 {
        let strs: Vec<CString> = std::iter::once("ltop".to_string())
            .chain(env.iter().map(|(k, v)| format!("{k}={v}")))
            .map(|s| CString::new(s).unwrap()).collect();
        let mut ptrs: Vec<*const u8> = vec![strs[0].as_ptr() as *const u8, core::ptr::null()];
        ptrs.extend(strs[1..].iter().map(|s| s.as_ptr() as *const u8));
        ptrs.push(core::ptr::null());
        unsafe { open_dir(1, ptrs.as_ptr()) }
    }

    #[test]
    fn open_dir_prefers_xdg_cache_home() {
        let d = temp_dir("env");
        let (home, xdg) = (d.join("home"), d.join("xdg"));
        fs::create_dir_all(home.join(".cache/ltop/status")).unwrap();
        fs::create_dir_all(xdg.join("ltop/status")).unwrap();
        fs::write(home.join(".cache/ltop/status/h"), "").unwrap();
        fs::write(xdg.join("ltop/status/x"), "").unwrap();
        let (h, x) = (home.to_str().unwrap(), xdg.to_str().unwrap());
        // Which directory came back: look for its marker file.
        let has = |fd: i32, f: &str| {
            assert!(fd >= 0, "open_dir failed: {fd}");
            let c = CString::new(f).unwrap();
            let r = syscall::openat_cstr(fd, &c, syscall::O_RDONLY);
            syscall::close(fd);
            if r >= 0 { syscall::close(r); }
            r >= 0
        };
        assert!(has(open(&[("HOME", h)]), "h"));
        assert!(has(open(&[("HOME", h), ("XDG_CACHE_HOME", x)]), "x"));
        assert!(has(open(&[("XDG_CACHE_HOME", x), ("HOME", h)]), "x"));
        assert!(has(open(&[("XDG_CACHE_HOME", ""), ("HOME", h)]), "h"));
        fs::remove_dir_all(&d).unwrap();
    }
}
