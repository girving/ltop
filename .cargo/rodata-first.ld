/* Augment the default linker script: place .rodata (and friends)
 * before .text so they share the first R-only LOAD segment with
 * the ELF header + program headers, instead of spilling into a
 * third LOAD segment after .text.
 *
 * Without this, the layout is three R-flag LOAD segments separated
 * by page-alignment padding; the 3.6 KB of padding after the ELF
 * header page goes unused, and a separate page is consumed at the
 * far end of the file just to hold ~2.7 KB of strings and tables.
 *
 * Measured effect: file size drops ~4.3 KB (39580 → 35240 B) and
 * LOAD segment count drops from 3 to 2 in the R/RX region.
 * VmRSS is unchanged — those padding bytes were never mapped.
 *
 * Using `INSERT BEFORE .text` augments the default script rather
 * than replacing it, so rustc's static-pie scaffolding (got, dynamic
 * bits, etc.) keeps working. `tests/stack_frames.rs` guards the
 * resulting layout so this doesn't silently regress.
 */

SECTIONS {
    .rodata       : { *(.rodata .rodata.*) }
    .eh_frame_hdr : { *(.eh_frame_hdr) }
}
INSERT AFTER .note.gnu.build-id;
