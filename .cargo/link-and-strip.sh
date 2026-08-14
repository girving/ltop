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
    # Prefer the active toolchain's bundled tools (zero setup), but
    # some nightlies ship a rust-lld/rust-objcopy with a broken
    # libLLVM @rpath — fall back to Homebrew's (`brew install lld llvm`).
    if ( "$bindir/rust-lld" -flavor gnu --version ) >/dev/null 2>&1; then
        LLD() { "$bindir/rust-lld" -flavor gnu "$@"; }
    elif command -v ld.lld >/dev/null 2>&1; then
        LLD() { ld.lld "$@"; }
    else
        echo "link-and-strip.sh: no working GNU-flavor lld (brew install lld)" >&2
        exit 1
    fi
    if ( "$bindir/rust-objcopy" --version ) >/dev/null 2>&1; then
        OBJCOPY() { "$bindir/rust-objcopy" "$@"; }
    elif command -v llvm-objcopy >/dev/null 2>&1; then
        OBJCOPY() { llvm-objcopy "$@"; }
    elif [ -x /opt/homebrew/opt/llvm/bin/llvm-objcopy ]; then
        OBJCOPY() { /opt/homebrew/opt/llvm/bin/llvm-objcopy "$@"; }
    else
        echo "link-and-strip.sh: no llvm-objcopy (brew install llvm)" >&2
        exit 1
    fi
    case "$out" in
        */static-linux-aarch64/*) emu=aarch64linux ;;
        *)                        emu=elf_x86_64 ;;
    esac
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
        # --strip-sections: drop all section headers plus any content
        # not inside a LOAD segment — covers what the GNU branch's
        # --strip-section-headers + --remove-section pair does.
        OBJCOPY --strip-sections "$out" 2>/dev/null || true
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
