//! ltop internals exposed as a library. The binary (`src/main.rs`) does not
//! depend on this — it compiles its own copy of each module. The library
//! exists so we can run doc-tests (including `compile_fail` ones that prove
//! arena escapes don't compile) via `cargo test`, which only runs doc-tests
//! for library crates.

#![cfg_attr(not(test), no_std)]
#![cfg_attr(test, allow(dead_code))]

extern crate alloc;

pub mod arena;
#[cfg(target_os = "linux")]
pub mod sandbox;
pub mod syscall;
#[macro_use]
pub mod twrite;
pub mod zeroable;
#[cfg(target_os = "macos")]
pub mod mac_sys;
#[cfg(target_os = "macos")]
pub mod osbinary;
