#!/bin/sh
# Linker wrapper: invoke the system `cc` to link, then strip the
# sections the loader never reads out of the finished binary.
#
# Cargo's `strip = true` profile option runs `strip --strip-all`,
# which removes debug info and symbol tables — but leaves behind:
#   .stack_sizes       1150 B  (-Zemit-stack-sizes metadata; only
#                                tools/stack-svg consumes it, and
#                                only via the min-stack profile)
#   .comment             52 B  (rustc version string)
#   section headers    ~640 B  (one 64-byte entry per section)
#   .shstrtab            95 B  (section-header name table)
# None of these are in any LOAD segment, so the loader doesn't map
# them; they're purely there for `readelf`/`nm`/`objdump`. Removing
# them saves ~2 KB of file size for every `cargo ltop` / `cargo stack`
# / `cargo install ltop`.
#
# Ordering matters: run `cc` first, then strip. Removing sections
# before/during link would break the link itself.

set -e

# Find the `-o OUTPUT` path in the argv so we can pick the right
# cross-toolchain per target and know what to strip after link.
out=""
prev=""
for arg in "$@"; do
    if [ "$prev" = "-o" ]; then
        out="$arg"
        break
    fi
    prev="$arg"
done

# Select a cross-toolchain based on the target triple embedded in the
# output path (target/<triple>/<profile>/…). On a host-arch build
# (static-linux-x86_64 on x86_64, static-linux-aarch64 on aarch64),
# plain `cc`/`strip` are correct. On a cross build we use the
# GNU-named binutils/gcc that ship with the Debian cross packages.
# macOS host: no GNU cross-gcc, so cross-link with the rust-lld that
# ships in the pinned toolchain (GNU flavor) and strip with
# rust-objcopy. rustc hands us cc-driver argv (gnu-cc linker-flavor);
# translate to ld-style: `-Wl,a,b` unwraps to `a b`, and
# -no-pie / -nostartfiles / -nodefaultlibs / -m64 are driver-only
# no-ops for a raw linker (ld links non-PIE by default and never adds
# crt objects or default libs). Args go through an @response-file so
# plain POSIX sh can rebuild the list. This branch exists for local
# dev + size measurement from a mac; lld's layout differs from GNU
# ld's by a few bytes, so CI's GNU output remains the authoritative
# footprint number.
if [ "$(uname -s)" = "Darwin" ]; then
    host=$(rustc -vV | sed -n 's/^host: //p')
    bindir="$(rustc --print sysroot)/lib/rustlib/$host/bin"
    # Prefer real GNU binutils when installed (`brew install
    # x86_64-elf-binutils aarch64-elf-binutils`): GNU ld's layout is
    # what CI ships and what the README/footprint numbers describe, so
    # a GNU-linked local binary is byte-comparable. Otherwise fall back
    # to an LLVM toolchain — the active toolchain's bundled rust-lld
    # (zero setup), or Homebrew's (`brew install lld llvm`) when a
    # nightly ships rust-lld/rust-objcopy with a broken libLLVM @rpath.
    # lld's layout differs from GNU's by ~100-200 B (8-byte fast
    # build-id vs sha1, padding), so lld numbers are only good for
    # same-linker deltas.
    # The *-linux-gnu toolchains carry the linux emulations CI links
    # with (the bare-ELF `aarch64elf` lays the first LOAD out without
    # the ELF header, costing a full 64 K of max-page offset padding —
    # bare-elf x86_64 is layout-compatible, so it stays as a fallback).
    # aarch64: brew tap messense/macos-cross-toolchains &&
    #          brew install aarch64-unknown-linux-gnu
    # x86_64:  brew install x86_64-linux-gnu-binutils
    case "$out" in
        */static-linux-aarch64/*)
            gnu=aarch64-linux-gnu; emu=aarch64linux ;;
        *)
            if command -v x86_64-linux-gnu-ld >/dev/null 2>&1; then
                gnu=x86_64-linux-gnu
            else
                gnu=x86_64-elf
            fi
            emu=elf_x86_64 ;;
    esac
    if command -v "$gnu-ld" >/dev/null 2>&1; then
        LLD() { "$gnu-ld" "$@"; }
    elif ( "$bindir/rust-lld" -flavor gnu --version ) >/dev/null 2>&1; then
        LLD() { "$bindir/rust-lld" -flavor gnu "$@"; }
    elif command -v ld.lld >/dev/null 2>&1; then
        LLD() { ld.lld "$@"; }
    else
        echo "link-and-strip.sh: no GNU-flavor linker (brew install $gnu-binutils, or lld)" >&2
        exit 1
    fi
    if command -v "$gnu-strip" >/dev/null 2>&1; then
        # Same invocation as the Linux branch, so the output matches CI's.
        STRIP_EXTRA() {
            "$gnu-strip" \
                --strip-section-headers \
                --remove-section=.stack_sizes \
                --remove-section=.comment \
                "$1" 2>/dev/null || true
        }
    elif ( "$bindir/rust-objcopy" --version ) >/dev/null 2>&1; then
        STRIP_EXTRA() { "$bindir/rust-objcopy" --strip-sections "$1" 2>/dev/null || true; }
    elif command -v llvm-objcopy >/dev/null 2>&1; then
        STRIP_EXTRA() { llvm-objcopy --strip-sections "$1" 2>/dev/null || true; }
    elif [ -x /opt/homebrew/opt/llvm/bin/llvm-objcopy ]; then
        STRIP_EXTRA() { /opt/homebrew/opt/llvm/bin/llvm-objcopy --strip-sections "$1" 2>/dev/null || true; }
    else
        echo "link-and-strip.sh: no strip/objcopy (brew install $gnu-binutils or llvm)" >&2
        exit 1
    fi
    tmp="${TMPDIR:-/tmp}/ltop-lld-$$.args"
    : > "$tmp"
    for arg in "$@"; do
        case "$arg" in
            -Wl,*) printf '%s\n' "${arg#-Wl,}" | tr ',' '\n' >> "$tmp" ;;
            -no-pie|-nostartfiles|-nodefaultlibs|-m64) ;;
            *) printf '%s\n' "$arg" >> "$tmp" ;;
        esac
    done
    # `--build-id`: the Linux gcc driver adds it implicitly (Debian
    # default), and rodata-first.ld INSERTs after .note.gnu.build-id,
    # so the section must exist. lld doesn't emit it unasked.
    LLD -m "$emu" --build-id @"$tmp"
    rm -f "$tmp"
    case "$out" in
        */min-stack/*) exit 0 ;;   # keep .stack_sizes for tools/stack-svg
    esac
    if [ -n "$out" ] && [ -f "$out" ]; then
        STRIP_EXTRA "$out"
    fi
    exit 0
fi

host_arch=$(uname -m)
CC=cc
STRIP=strip
case "$out" in
    */static-linux-aarch64/*)
        if [ "$host_arch" != "aarch64" ]; then
            CC=aarch64-linux-gnu-gcc
            STRIP=aarch64-linux-gnu-strip
        fi
        ;;
    */static-linux-x86_64/*)
        if [ "$host_arch" != "x86_64" ]; then
            CC=x86_64-linux-gnu-gcc
            STRIP=x86_64-linux-gnu-strip
        fi
        ;;
esac

"$CC" "$@"

# Skip probing invocations that don't produce an output file. For
# the `min-stack` profile, also skip — we deliberately keep
# `.stack_sizes` there for tools/stack-svg.
case "$out" in
    */min-stack/*) exit 0 ;;
esac
if [ -n "$out" ] && [ -f "$out" ]; then
    "$STRIP" \
        --strip-section-headers \
        --remove-section=.stack_sizes \
        --remove-section=.comment \
        "$out" 2>/dev/null || true
fi
