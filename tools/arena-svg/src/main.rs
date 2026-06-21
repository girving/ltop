//! arena-svg: render an arena trace log as a space-time diagram.
//!
//! Reads `./arena-trace.log` (or `argv[1]`) produced by ltop built with
//! `--features arena-trace`. Writes SVG to stdout (or `argv[2]`).
//!
//! The log grammar is three line shapes:
//!
//! ```text
//! A {new_offset} {label}\n   alloc: bump pointer moved to new_offset
//! R {new_offset}\n            rewind: bump pointer fell back to new_offset
//! U {new_offset}\n            uncommit_tail: madvise event (no rectangle)
//! ```
//!
//! Rendering: x axis is event ordinal; y axis is arena offset in bytes.
//! Each allocation becomes a filled rectangle `[start_evt..end_evt) ×
//! [offset_lo..offset_hi)`, coloured by a deterministic hash of its
//! label. U events are drawn as vertical ticks along the top strip.
//! A legend on the right lists every label with a swatch and its total
//! byte·event area (a crude "time-integrated footprint" number).
//!
//! Implementation choices:
//!   * No external deps. SVG is just text; we build it with a `String`.
//!   * X axis is event ordinal, not wall time — an idle gap between
//!     ticks shouldn't make the interesting parts invisible.
//!   * Colours are HSL(hash(label) mod 360, 65%, 55%) so re-runs produce
//!     stable, distinguishable hues.

use std::collections::HashMap;
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::io::{self, Write};
use std::process::ExitCode;

// ── Event model ──────────────────────────────────────────────────────────────

enum Event<'a> {
    Alloc { new_offset: u32, label: &'a str },
    Rewind { new_offset: u32 },
    Uncommit { new_offset: u32 },
}

fn parse(line: &str) -> Option<Event<'_>> {
    let line = line.trim_end();
    if line.is_empty() {
        return None;
    }
    let (kind, rest) = line.split_once(' ')?;
    match kind {
        "A" => {
            let (off, label) = rest.split_once(' ').unwrap_or((rest, ""));
            Some(Event::Alloc {
                new_offset: off.parse().ok()?,
                label,
            })
        }
        "R" => Some(Event::Rewind {
            new_offset: rest.parse().ok()?,
        }),
        "U" => Some(Event::Uncommit {
            new_offset: rest.parse().ok()?,
        }),
        _ => None,
    }
}

// ── Rectangle extraction ─────────────────────────────────────────────────────

struct Rect<'a> {
    start: u32,      // event index at which this alloc appears
    end: u32,        // event index at which it's reclaimed (exclusive)
    offset_lo: u32,  // low byte offset in the arena
    offset_hi: u32,  // high byte offset (exclusive)
    label: &'a str,
}

struct Live<'a> {
    start: u32,
    offset_lo: u32,
    offset_hi: u32,
    label: &'a str,
}

/// Map every event index to an x position on the plot.
///
/// Every unique label gets the SAME total x-axis width, independent of
/// its event count or allocation size. An alloc event of label L
/// advances x by `BUDGET_PER_LABEL / count[L]`, so a label used once
/// (e.g. `prev`, init-scope) takes the same x-space as a label cycling
/// 2000 times (`proc/stat_raw`, per-pid). This keeps tight per-pid
/// loops from eating the whole plot, while still giving rare allocs
/// the prominence their event count implies.
///
/// Rewinds and uncommits share a MIN_STEP — they mark scope boundaries
/// but don't contribute their own width.
///
/// Returns (event_x, total_x); `event_x[i]` is x BEFORE event i,
/// `event_x[events.len()]` is the final x.
const EVENT_X_MIN_STEP: u64 = 16;
const BUDGET_PER_LABEL: u64 = 32 * 1024;

fn compute_event_x(events: &[Event<'_>]) -> (Vec<u64>, u64) {
    // Count allocs per label (only Allocs have labels).
    let mut counts: HashMap<&str, u32> = HashMap::new();
    for ev in events {
        if let Event::Alloc { label, .. } = ev {
            *counts.entry(label).or_insert(0) += 1;
        }
    }

    let mut xs = Vec::with_capacity(events.len() + 1);
    let mut x: u64 = 0;
    xs.push(x);
    for ev in events {
        let delta = match *ev {
            Event::Alloc { label, .. } => {
                let c = counts.get(label).copied().unwrap_or(1).max(1) as u64;
                (BUDGET_PER_LABEL / c).max(EVENT_X_MIN_STEP)
            }
            Event::Rewind { .. } | Event::Uncommit { .. } => EVENT_X_MIN_STEP,
        };
        x += delta;
        xs.push(x);
    }
    (xs, x)
}

fn build_rects<'a>(events: &'a [Event<'a>]) -> (Vec<Rect<'a>>, Vec<(u32, u32)>, u32) {
    let mut rects = Vec::with_capacity(events.len());
    // Invariant: `live` is monotone in offset_lo. Allocations advance the
    // bump pointer, so each push has offset_lo >= the previous element's
    // offset_hi. Rewinds always shrink the stack from the top.
    let mut live: Vec<Live<'a>> = Vec::new();
    let mut current: u32 = 0;
    let mut peak: u32 = 0;
    // (event_index, new_offset) for each uncommit — rendered as a tick
    // mark at x=event_index, annotated with the offset.
    let mut uncommits: Vec<(u32, u32)> = Vec::new();

    for (i, ev) in events.iter().enumerate() {
        let i = i as u32;
        match *ev {
            Event::Alloc { new_offset, label } => {
                let lo = current;
                current = new_offset;
                if new_offset > peak {
                    peak = new_offset;
                }
                // A zero-size alloc (alignment-only bump) still consumed
                // a row in the trace — but it has no rectangle to draw.
                if new_offset > lo {
                    live.push(Live {
                        start: i,
                        offset_lo: lo,
                        offset_hi: new_offset,
                        label,
                    });
                }
            }
            Event::Rewind { new_offset } => {
                while let Some(top) = live.last() {
                    if top.offset_lo >= new_offset {
                        // Fully reclaimed — close at this event.
                        let l = live.pop().unwrap();
                        rects.push(Rect {
                            start: l.start,
                            end: i,
                            offset_lo: l.offset_lo,
                            offset_hi: l.offset_hi,
                            label: l.label,
                        });
                    } else if top.offset_hi <= new_offset {
                        // Fully below the rewind target; stays live.
                        break;
                    } else {
                        // Genuine partial overlap — shouldn't happen under
                        // arena discipline (rewinds land on scope
                        // boundaries), but don't crash if a log is weird.
                        eprintln!(
                            "warn: rewind to {new_offset} partially overlaps live alloc \
                             [{}..{}) (label {:?}); keeping alloc intact",
                            top.offset_lo, top.offset_hi, top.label,
                        );
                        break;
                    }
                }
                current = new_offset;
            }
            Event::Uncommit { new_offset } => {
                uncommits.push((i, new_offset));
                current = new_offset;
            }
        }
    }
    // Close anything still live at EOF.
    let end = events.len() as u32;
    for l in live.drain(..) {
        rects.push(Rect {
            start: l.start,
            end,
            offset_lo: l.offset_lo,
            offset_hi: l.offset_hi,
            label: l.label,
        });
    }
    (rects, uncommits, peak)
}

// ── Colour hashing ───────────────────────────────────────────────────────────

/// FNV-1a over the label bytes — small, deterministic, dep-free.
/// We only care about even hue distribution across the set of labels, so
/// any reasonable hash does.
fn hash_label(label: &str) -> u32 {
    let mut h: u32 = 0x811c9dc5;
    for &b in label.as_bytes() {
        h ^= b as u32;
        h = h.wrapping_mul(0x01000193);
    }
    h
}

fn hsl_css(label: &str) -> String {
    let hue = hash_label(label) % 360;
    format!("hsl({hue}, 65%, 55%)")
}

/// Escape `<`, `>`, `&` for SVG text content. Our labels can contain
/// bracket characters (`<nest:cursors>`) which XML parses as tag markup
/// if left raw.
fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            _ => out.push(c),
        }
    }
    out
}

// ── SVG emission ─────────────────────────────────────────────────────────────

const MARGIN_L: u32 = 60;
const MARGIN_R: u32 = 240;  // room for legend
const MARGIN_T: u32 = 30;   // uncommit ticks + title
const MARGIN_B: u32 = 40;   // x-axis label

const PLOT_W: u32 = 1600;
const PLOT_H: u32 = 700;

fn format_bytes(b: u32) -> String {
    if b >= 1024 * 1024 {
        format!("{:.1} MB", b as f64 / (1024.0 * 1024.0))
    } else if b >= 1024 {
        format!("{} KB", b / 1024)
    } else {
        format!("{b} B")
    }
}

fn render_svg(
    rects: &[Rect<'_>],
    uncommits: &[(u32, u32)],
    event_x: &[u64],
    total_x: u64,
    peak: u32,
) -> String {
    let w = MARGIN_L + PLOT_W + MARGIN_R;
    let h = MARGIN_T + PLOT_H + MARGIN_B;
    let total_x = total_x.max(1);
    let peak = peak.max(1);
    // Extend y-axis to ~arena ceiling (512 KB) if peak is close enough that
    // the ceiling line would fit on the chart, so the diagram shows how much
    // headroom we have. Otherwise clip to peak for detail.
    let y_max = if peak < 512 * 1024 && peak > 256 * 1024 {
        512 * 1024
    } else {
        peak
    };

    // x mapping: per-label-share coordinate → SVG x.
    let x = |evt: u32| -> f64 {
        let vx = event_x[evt as usize] as f64;
        MARGIN_L as f64 + (vx / total_x as f64) * PLOT_W as f64
    };
    let y = |off: u32| -> f64 {
        // Flip: y=0 at top of SVG → offset=peak at top of plot.
        MARGIN_T as f64 + PLOT_H as f64 * (1.0 - off as f64 / y_max as f64)
    };

    let mut s = String::with_capacity(rects.len() * 120);
    writeln!(s, r##"<?xml version="1.0" encoding="UTF-8"?>"##).unwrap();
    writeln!(
        s,
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="monospace" font-size="11">"##
    )
    .unwrap();
    writeln!(
        s,
        r##"<rect x="0" y="0" width="{w}" height="{h}" fill="white"/>"##
    )
    .unwrap();

    // Plot frame + y-axis gridlines.
    writeln!(
        s,
        r##"<rect x="{}" y="{}" width="{}" height="{}" fill="none" stroke="#888"/>"##,
        MARGIN_L, MARGIN_T, PLOT_W, PLOT_H
    )
    .unwrap();
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let off = (y_max as f64 * frac) as u32;
        let yy = y(off);
        writeln!(
            s,
            r##"<line x1="{}" x2="{}" y1="{:.1}" y2="{:.1}" stroke="#eee"/>"##,
            MARGIN_L,
            MARGIN_L + PLOT_W,
            yy,
            yy,
        )
        .unwrap();
        writeln!(
            s,
            r##"<text x="{}" y="{:.1}" text-anchor="end" dominant-baseline="middle" fill="#555">{}</text>"##,
            MARGIN_L - 6,
            yy,
            format_bytes(off),
        )
        .unwrap();
    }

    // Title.
    let n_events = (event_x.len().saturating_sub(1)) as u32;
    writeln!(
        s,
        r##"<text x="{}" y="16" font-size="13" fill="#222">arena timeline — {} events, peak {}, each label gets equal x-width ({} rects)</text>"##,
        MARGIN_L,
        n_events,
        format_bytes(peak),
        rects.len(),
    )
    .unwrap();

    // Alloc rectangles.
    for r in rects {
        let x0 = x(r.start);
        let x1 = x(r.end).max(x0 + 0.4);  // enforce a minimum visible width
        let y0 = y(r.offset_hi);          // hi offset → upper edge in SVG
        let y1 = y(r.offset_lo);
        let w = (x1 - x0).max(0.4);
        let h = (y1 - y0).max(0.4);
        writeln!(
            s,
            r##"<rect x="{x0:.2}" y="{y0:.2}" width="{w:.2}" height="{h:.2}" fill="{}" fill-opacity="0.85"><title>{}: [{}..{}) {} @ evt {}..{}</title></rect>"##,
            hsl_css(r.label),
            xml_escape(r.label),
            r.offset_lo,
            r.offset_hi,
            format_bytes(r.offset_hi - r.offset_lo),
            r.start,
            r.end,
        )
        .unwrap();
    }

    // Uncommit tick marks: thin dashed vertical line at the event, with
    // a "↓{new_offset}" annotation above.
    for &(evt, new_off) in uncommits {
        let xx = x(evt);
        writeln!(
            s,
            r##"<line x1="{xx:.1}" x2="{xx:.1}" y1="{}" y2="{}" stroke="#c22" stroke-width="0.8" stroke-dasharray="2,2"><title>uncommit_tail → {}</title></line>"##,
            MARGIN_T,
            MARGIN_T + PLOT_H,
            format_bytes(new_off),
        )
        .unwrap();
    }

    // Legend: aggregate by label. For each label track:
    //  * max rect size — what the bar's height visually shows; this is
    //    the number we display in the legend (plain bytes, so it matches
    //    the y-axis scale, no confusing byte·event units).
    //  * area-weighted vertical centroid — used only for ordering rows
    //    to match the top-to-bottom pattern of the plot.
    struct Agg {
        max_size: u64,
        total_area: u64,
        centroid_num: u128,  // sum(mid * area); centroid = num / area.
    }
    let mut agg: HashMap<&str, Agg> = HashMap::new();
    for r in rects {
        let size = (r.offset_hi - r.offset_lo) as u64;
        let span = r.end.saturating_sub(r.start) as u64;
        // Tiny floor so a zero-span rect still contributes to the
        // centroid (it has a real vertical position even if width=0).
        let area = size * span.max(1);
        let mid = (r.offset_lo as u64 + r.offset_hi as u64) / 2;
        let e = agg.entry(r.label).or_insert(Agg {
            max_size: 0,
            total_area: 0,
            centroid_num: 0,
        });
        if size > e.max_size { e.max_size = size; }
        e.total_area += area;
        e.centroid_num += (mid as u128) * (area as u128);
    }
    // (label, max_size_bytes, centroid_offset_bytes)
    let mut entries: Vec<(&str, u64, u64)> = agg
        .into_iter()
        .map(|(label, a)| {
            let centroid = if a.total_area > 0 {
                (a.centroid_num / a.total_area as u128) as u64
            } else {
                0
            };
            (label, a.max_size, centroid)
        })
        .collect();
    // Descending by centroid → high-offset labels at the top of the
    // legend (matching their visual position in the plot). Ties fall
    // through to larger-size-first to keep ordering stable.
    entries.sort_by(|a, b| b.2.cmp(&a.2).then(b.1.cmp(&a.1)));

    // Legend: vertically aligned with the plot — each row sits at (or
    // near) the y of its allocation block's area-weighted centroid, so
    // the eye moves horizontally from a rect to its label. Collisions
    // between nearby centroids are resolved by greedy top-down packing
    // with a minimum line height.
    let legend_x = (MARGIN_L + PLOT_W + 16) as f64;
    let line_h = 14.0;
    // Header.
    writeln!(
        s,
        r##"<text x="{:.1}" y="{:.1}" font-size="12" fill="#222">label (max size)</text>"##,
        legend_x, (MARGIN_T as f64) - 10.0,
    )
    .unwrap();

    // entries is sorted by centroid descending (top-of-plot first).
    // Translate centroid bytes → desired y in SVG coords, then greedy-
    // pack downward to avoid overlap.
    let mut ly = (MARGIN_T as f64) + line_h * 0.5;  // first row's baseline
    for (label, max_size, centroid) in entries.iter().take(40) {
        let desired_y = y(*centroid as u32);
        if desired_y > ly { ly = desired_y; }
        // Connector: thin line from the rect's row in the plot to the
        // legend swatch, so tightly-packed rows don't get ambiguous.
        writeln!(
            s,
            r##"<line x1="{:.1}" x2="{:.1}" y1="{:.1}" y2="{:.1}" stroke="#ccc" stroke-width="0.4"/>"##,
            MARGIN_L as f64 + PLOT_W as f64,
            legend_x - 2.0,
            desired_y,
            ly,
        )
        .unwrap();
        writeln!(
            s,
            r##"<rect x="{:.1}" y="{:.1}" width="12" height="10" fill="{}"/>"##,
            legend_x,
            ly - 5.0,
            hsl_css(label),
        )
        .unwrap();
        writeln!(
            s,
            r##"<text x="{:.1}" y="{:.1}" dominant-baseline="middle" fill="#222">{} ({})</text>"##,
            legend_x + 16.0,
            ly,
            xml_escape(label),
            format_bytes(*max_size as u32),
        )
        .unwrap();
        ly += line_h;
    }

    // X-axis: event-index tick marks. The x coordinate itself is in
    // arbitrary per-label-share units (labels occupy equal widths), so
    // a byte scale would be misleading; event indices let a reader
    // cross-reference against the `@ evt A..B` tooltips on each rect.
    let xl_y = (MARGIN_T + PLOT_H + 14) as f64;
    let n_events = event_x.len().saturating_sub(1) as u64;
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0] {
        // Walk event_x to find the event index at this x fraction.
        let target_x = (total_x as f64 * frac) as u64;
        let idx = event_x.partition_point(|&x| x <= target_x).saturating_sub(1);
        let svg_x = MARGIN_L as f64 + frac * PLOT_W as f64;
        writeln!(
            s,
            r##"<text x="{:.1}" y="{:.1}" text-anchor="middle" fill="#555">evt {}</text>"##,
            svg_x,
            xl_y,
            idx.min(n_events as usize),
        )
        .unwrap();
    }

    s.push_str("</svg>\n");
    s
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    let in_path = args.get(1).map(String::as_str).unwrap_or("./arena-trace.log");
    let out_path = args.get(2).map(String::as_str);

    let src = fs::read_to_string(in_path)
        .map_err(|e| format!("read {in_path}: {e}"))?;

    // Parse all events up-front so Rect's label can borrow from the source
    // string without any cloning.
    let events: Vec<Event<'_>> = src.lines().filter_map(parse).collect();
    let (rects, uncommits, peak) = build_rects(&events);
    let (event_x, total_x) = compute_event_x(&events);
    let svg = render_svg(&rects, &uncommits, &event_x, total_x, peak);

    match out_path {
        Some(p) => fs::write(p, &svg).map_err(|e| format!("write {p}: {e}"))?,
        None => io::stdout()
            .write_all(svg.as_bytes())
            .map_err(|e| format!("stdout: {e}"))?,
    }

    eprintln!(
        "arena-svg: {} events → {} rects (peak {}); {} unique labels",
        events.len(),
        rects.len(),
        format_bytes(peak),
        {
            let mut u = std::collections::HashSet::new();
            for r in &rects {
                u.insert(r.label);
            }
            u.len()
        },
    );

    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("arena-svg: {e}");
            ExitCode::FAILURE
        }
    }
}
