/*
 * Stub `libnvidia-ml.so.1` — the tiny set of symbols our test build
 * references from the reference NVML backend (`src/gpu/nvml.rs`,
 * compiled only under cfg(all(test, has_nvml))). CI has no NVIDIA
 * driver, so the shared library has to be fakeable by us — we only
 * need the *names* to resolve at link time; nothing here is ever
 * called at runtime because the test that would call it is gated
 * `#[cfg_attr(not(has_gpu), ignore)]` and /dev/nvidiactl isn't
 * present on the runner.
 *
 * Build invocations that consume this file:
 *
 *   # CI (x86_64, Linux multi-arch path):
 *   sudo gcc -shared -fPIC -Wl,-soname,libnvidia-ml.so.1 \
 *        -o /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 \
 *        tools/nvml_stub.c
 *
 *   # Local cross-compile (aarch64, gcc-cross layout):
 *   sudo aarch64-linux-gnu-gcc -shared -fPIC -Wl,-soname,libnvidia-ml.so.1 \
 *        -o /usr/aarch64-linux-gnu/lib/libnvidia-ml.so.1 \
 *        tools/nvml_stub.c
 *
 * `build.rs` finds whatever lands in one of its candidate paths and
 * emits `-l:libnvidia-ml.so.1`; the production `cargo ltop` /
 * `cargo stack` builds skip the link entirely (libc_free path), so
 * this stub is only ever linked by `cargo build --release` and
 * `cargo test --release`.
 */

int nvmlInit_v2(void) { return 0; }
int nvmlDeviceGetCount_v2(unsigned int *c) { return 0; }
int nvmlDeviceGetHandleByIndex_v2(unsigned int i, void **d) { return 0; }
int nvmlDeviceGetUtilizationRates(void *d, void *u) { return 0; }
int nvmlDeviceGetMemoryInfo(void *d, void *m) { return 0; }
int nvmlDeviceGetComputeRunningProcesses(void *d, unsigned int *c, void *p) { return 0; }
int nvmlDeviceGetProcessUtilization(void *d, void *s, unsigned int *c, unsigned long long t) { return 0; }
