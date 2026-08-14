# io_registry_entry_get_property_bin (MIG 2879) — research notes

Motivation: the AGX root readout only needs one property
(`PerformanceStatistics`, a ~2–4 KB dict), but
`io_registry_entry_get_properties_bin` (2878) serialises the entry's
*entire* property table (~100 KB, dominated by `IOReportLegend`) into a
fresh OOL allocation every tick. Fetching the single property shrinks
the per-tick OOL traffic by ~25× and the parse work with it.

## Primary sources

- `xnu osfmk/device/device.defs` (apple-oss-distributions, main): the
  iokit subsystem starts at 2800; counting routines + skips gives
  **`io_registry_entry_get_property_bin` = 2879** (one after
  `get_properties_bin` = 2878; the caller-buffer variant
  `..._get_property_bin_buf` = 2889). Signature:

  ```
  routine io_registry_entry_get_property_bin(
              registry_entry : io_object_t;
          in  plane          : io_name_t;
          in  property_name  : io_name_t;
          in  options        : uint32_t;
          out properties     : io_buf_ptr_t, physicalcopy );
  ```

- `xnu iokit/Kernel/IOUserClient.cpp`
  `is_io_registry_entry_get_property_bin`: forwards to the `_buf`
  handler with no caller buffer. **Empty `plane` ⇒ direct
  `IOCopyPropertyCompatible(entry, property_name)`** — no iteration;
  `options` only matters with a plane (`kIORegistryIterateRecursively`
  etc). Gating is MACF `mac_iokit_check_get_property` only — the same
  check the 2878 path already passes for this property (unprivileged
  `ioreg` reads it). Reply built via `OSSerialize::binaryWithCapacity`
  ⇒ the property object serialised as the **root** of a
  kOSSerializeBinary blob, `copyoutkdata` ⇒ OOL descriptor, exactly
  the 2878 reply shape.

- `IOKitUser IOKitLib.c`: `IORegistryEntryCreateCFProperty(entry, key,
  alloc, options)` ⇒ `IORegistryEntrySearchCFProperty(entry, NULL, key,
  alloc, kNilOptions)` ⇒ (modern serialize mode) the `_bin` family
  with **plane = "" and options = 0**. OOL reply released with
  `vm_deallocate`.

- Local ground truth (macOS 26, dyld shared cache, `dyld_info
  -arch arm64e -disassemble .../IOKit`): `_IORegistryEntrySearchCFProperty`
  calls `_io_registry_entry_get_property_bin` (a global flag selects it
  vs `_bin_buf`), substituting a static `""` when plane is NULL. The
  MIG stub pins the wire layout below (msgh_id `#0x1513` bits,
  `mov x4, #0xb3f_00000000` ⇒ id 0xB3F = 2879, reply id checked
  against 0xBA3 = 2979).

## Request layout (from the local stub; matches the in-repo
`iokit_get_child_iter` io_name_t convention)

```
 0..24   mach header — bits 0x1513 (COPY_SEND | MAKE_SEND_ONCE<<8),
         size, remote = entry, local = reply port, id = 2879
24..32   NDR record (zeroed works, as elsewhere in mac_sys.rs)
32..36   0            (MIG string "offset" field, unused)
36..40   planeCnt     (strlen+1; 1 for "")
40..     plane bytes, NUL-terminated, padded to 4  (4 zero bytes for "")
 +0..4   0            (second string's offset field)
 +4..8   nameCnt      (strlen+1, ≤ 128)
 +8..    name bytes, NUL-terminated, padded to 4
 +0..4   options (u32) = 0
```

Total = 52 + padded_plane + padded_name; with plane `""` and
`"PerformanceStatistics\0"` (22 → 24 padded): 80 bytes.

`mig_strncpy` counts include the trailing NUL — same convention the
2813 sender already uses (`b"IOService\0"`.len() = 10).

## Reply

Identical to 2878: complex bit, descriptor count 1, OOL descriptor
(addr 8 B, flags 4 B with type byte = 1, size 4 B), NDR, count,
trailer. The blob's root object IS the property (an OSDictionary for
PerformanceStatistics). Parse with `osbinary`, look up
`"In use system memory"` directly in the root dict.

## Non-goals / rejected

- `_bin_buf` (2889) would avoid the OOL allocation entirely (kernel
  copyout into a caller buffer, OOL only on overflow), but needs two
  more request words, an in/out size, and a dual-path reply parse.
  The property blob is ~1 page; one small OOL + `vm_deallocate` per
  tick is already 25× less Mach VM traffic than today. Revisit only if
  vmmap shows the small OOL churn mattering.
- AGXDeviceUserClient children keep 2878: their blobs are ~1 KB and we
  read two properties from each — two 2879 round-trips would double
  the message count for no size win.
