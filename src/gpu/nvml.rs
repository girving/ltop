//! NVML-backed GPU info, linked via build.rs when libnvidia-ml.so.1 is found.
//! Used as the reference implementation against which the `rm` backend is
//! eventually tested.

use std::ffi::c_uint;
use std::sync::OnceLock;

use super::GpuProc;
use crate::arena::FVec;

mod ffi {
    use std::ffi::{c_int, c_uint, c_void};

    pub type NvmlReturn = c_int;
    pub const NVML_SUCCESS: NvmlReturn = 0;

    #[repr(transparent)]
    #[derive(Clone, Copy)]
    pub struct Device(pub *mut c_void);

    #[repr(C)] pub struct Utilization { pub gpu: c_uint, pub memory: c_uint }
    #[repr(C)] pub struct Memory { pub total: u64, pub free: u64, pub used: u64 }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct ProcessInfo {
        pub pid: c_uint,
        pub used_gpu_memory: u64,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct ProcessUtilizationSample {
        pub pid: c_uint,
        pub time_stamp: u64,
        pub sm_util: c_uint,
        pub mem_util: c_uint,
        pub enc_util: c_uint,
        pub dec_util: c_uint,
    }

    unsafe extern "C" {
        pub fn nvmlInit_v2() -> NvmlReturn;
        pub fn nvmlDeviceGetCount_v2(count: *mut c_uint) -> NvmlReturn;
        pub fn nvmlDeviceGetHandleByIndex_v2(index: c_uint, device: *mut Device) -> NvmlReturn;
        pub fn nvmlDeviceGetUtilizationRates(device: Device, util: *mut Utilization) -> NvmlReturn;
        pub fn nvmlDeviceGetMemoryInfo(device: Device, memory: *mut Memory) -> NvmlReturn;
        pub fn nvmlDeviceGetComputeRunningProcesses(
            device: Device, count: *mut c_uint, infos: *mut ProcessInfo,
        ) -> NvmlReturn;
        pub fn nvmlDeviceGetProcessUtilization(
            device: Device,
            samples: *mut ProcessUtilizationSample,
            count: *mut c_uint,
            last_seen_timestamp: u64,
        ) -> NvmlReturn;
    }
}

pub fn has_gpu() -> bool {
    static INIT: OnceLock<bool> = OnceLock::new();
    *INIT.get_or_init(|| unsafe { ffi::nvmlInit_v2() == ffi::NVML_SUCCESS })
}

/// Mirror of `rm::populate`: push (pid, GpuProc) entries for this tick and
/// return (n_gpus, total_util, total_used_mib, total_mib). The caller freezes
/// `procs` afterward to merge duplicate PIDs.
pub fn populate(procs: &mut FVec<'_, (u32, GpuProc)>) -> (u32, u32, u64, u64) {
    if !has_gpu() { return (0, 0, 0, 0); }
    let (mut n_gpus, mut total_util, mut total_used_mib, mut total_mib) = (0u32, 0u32, 0u64, 0u64);
    unsafe {
        let mut count: c_uint = 0;
        if ffi::nvmlDeviceGetCount_v2(&mut count) != ffi::NVML_SUCCESS { return (0, 0, 0, 0); }

        for i in 0..count {
            let mut dev = ffi::Device(std::ptr::null_mut());
            if ffi::nvmlDeviceGetHandleByIndex_v2(i, &mut dev) != ffi::NVML_SUCCESS { continue; }

            let mut util = ffi::Utilization { gpu: 0, memory: 0 };
            if ffi::nvmlDeviceGetUtilizationRates(dev, &mut util) == ffi::NVML_SUCCESS {
                total_util += util.gpu;
            }

            let mut mem = ffi::Memory { total: 0, free: 0, used: 0 };
            if ffi::nvmlDeviceGetMemoryInfo(dev, &mut mem) == ffi::NVML_SUCCESS {
                total_used_mib += mem.used >> 20;
                total_mib += mem.total >> 20;
            }

            n_gpus += 1;

            // Per-process GPU memory. First call with null buffer queries count;
            // second call fills it.
            let mut pcount: c_uint = 0;
            ffi::nvmlDeviceGetComputeRunningProcesses(dev, &mut pcount, std::ptr::null_mut());
            if pcount > 0 {
                let mut infos: Vec<ffi::ProcessInfo> =
                    vec![ffi::ProcessInfo { pid: 0, used_gpu_memory: 0 }; pcount as usize];
                if ffi::nvmlDeviceGetComputeRunningProcesses(dev, &mut pcount, infos.as_mut_ptr())
                    == ffi::NVML_SUCCESS
                {
                    for p in &infos[..pcount as usize] {
                        let _ = procs.push((p.pid, GpuProc {
                            mem_mib: (p.used_gpu_memory >> 20) as u32, sm_pct: 0 }));
                    }
                }
            }

            // Per-process SM utilisation. Pass lastSeenTimeStamp=0 for the last
            // ~1s of samples. First call with null buffer queries the count;
            // second call fills it (with a bit of slack in case more arrive).
            let mut ucount: c_uint = 0;
            ffi::nvmlDeviceGetProcessUtilization(dev, std::ptr::null_mut(), &mut ucount, 0);
            if ucount > 0 {
                ucount += 8;
                let mut samples: Vec<ffi::ProcessUtilizationSample> = vec![
                    ffi::ProcessUtilizationSample {
                        pid: 0, time_stamp: 0, sm_util: 0, mem_util: 0, enc_util: 0, dec_util: 0,
                    };
                    ucount as usize
                ];
                if ffi::nvmlDeviceGetProcessUtilization(dev, samples.as_mut_ptr(), &mut ucount, 0)
                    == ffi::NVML_SUCCESS
                {
                    for s in &samples[..ucount as usize] {
                        let _ = procs.push((s.pid, GpuProc { sm_pct: s.sm_util, mem_mib: 0 }));
                    }
                }
            }
        }
    }
    (n_gpus, total_util, total_used_mib, total_mib)
}
