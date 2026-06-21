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
