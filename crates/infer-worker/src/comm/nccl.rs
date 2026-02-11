#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

mod inner {
    include!(concat!(env!("OUT_DIR"), "/nccl_bindings.rs"));
}

pub use inner::*;
