pub mod quant;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::{OnceLock, RwLock};

use half::{bf16, f16};

use crate::domain::types::{DataType, Dtype as StorageDtype};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DTypeId(pub u16);

impl DTypeId {
    pub const F32: DTypeId = DTypeId(0);
    pub const F16: DTypeId = DTypeId(1);
    pub const BF16: DTypeId = DTypeId(2);
    pub const I32: DTypeId = DTypeId(3);
    pub const I8: DTypeId = DTypeId(4);
    pub const F8E4M3: DTypeId = DTypeId(5);
    pub const F8E5M2: DTypeId = DTypeId(6);
    pub const U8: DTypeId = DTypeId(7);
    pub const U32: DTypeId = DTypeId(8);

    pub fn register(spec: DTypeSpec) -> DTypeId {
        static NEXT_ID: AtomicU16 = AtomicU16::new(1024);
        let id = DTypeId(NEXT_ID.fetch_add(1, Ordering::Relaxed));
        registry()
            .write()
            .expect("dtype registry poisoned")
            .insert(id.0, spec);
        id
    }

    pub fn size_bytes(self) -> usize {
        match self {
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I8 | Self::F8E4M3 | Self::F8E5M2 | Self::U8 => 1,
            id => registry()
                .read()
                .expect("dtype registry poisoned")
                .get(&id.0)
                .map(|spec| spec.size_bytes)
                .unwrap_or(0),
        }
    }

    pub fn is_float(self) -> bool {
        match self {
            Self::F32 | Self::F16 | Self::BF16 | Self::F8E4M3 | Self::F8E5M2 => true,
            Self::I32 | Self::I8 | Self::U8 | Self::U32 => false,
            id => registry()
                .read()
                .expect("dtype registry poisoned")
                .get(&id.0)
                .map(|spec| spec.is_float)
                .unwrap_or(false),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DTypeSpec {
    pub size_bytes: usize,
    pub is_float: bool,
    pub name: &'static str,
}

pub trait Dtype: StorageDtype {
    const ID: DTypeId;
    fn read_f64(raw: &Self) -> f64;
    fn write_f64(v: f64) -> Self;
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Fp8E4m3(pub u8);

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Fp8E5m2(pub u8);

pub trait Float: Dtype {}

impl Dtype for f32 {
    const ID: DTypeId = DTypeId::F32;
    fn read_f64(raw: &Self) -> f64 {
        f64::from(*raw)
    }
    fn write_f64(v: f64) -> Self {
        v as f32
    }
}

impl Dtype for f16 {
    const ID: DTypeId = DTypeId::F16;
    fn read_f64(raw: &Self) -> f64 {
        f64::from(raw.to_f32())
    }
    fn write_f64(v: f64) -> Self {
        f16::from_f32(v as f32)
    }
}

impl Dtype for bf16 {
    const ID: DTypeId = DTypeId::BF16;
    fn read_f64(raw: &Self) -> f64 {
        f64::from(raw.to_f32())
    }
    fn write_f64(v: f64) -> Self {
        bf16::from_f32(v as f32)
    }
}

impl Dtype for i32 {
    const ID: DTypeId = DTypeId::I32;
    fn read_f64(raw: &Self) -> f64 {
        f64::from(*raw)
    }
    fn write_f64(v: f64) -> Self {
        v as i32
    }
}

impl Dtype for i8 {
    const ID: DTypeId = DTypeId::I8;
    fn read_f64(raw: &Self) -> f64 {
        f64::from(*raw)
    }
    fn write_f64(v: f64) -> Self {
        v as i8
    }
}

impl StorageDtype for u8 {
    const DATA_TYPE: DataType = DataType::I8;
    const SIZE_BYTES: usize = 1;
}

impl Dtype for u8 {
    const ID: DTypeId = DTypeId::U8;
    fn read_f64(raw: &Self) -> f64 {
        f64::from(*raw)
    }
    fn write_f64(v: f64) -> Self {
        v.clamp(0.0, u8::MAX as f64) as u8
    }
}

impl StorageDtype for u32 {
    const DATA_TYPE: DataType = DataType::I32;
    const SIZE_BYTES: usize = 4;
}

impl Dtype for u32 {
    const ID: DTypeId = DTypeId::U32;
    fn read_f64(raw: &Self) -> f64 {
        f64::from(*raw)
    }
    fn write_f64(v: f64) -> Self {
        v.clamp(0.0, u32::MAX as f64) as u32
    }
}

impl StorageDtype for Fp8E4m3 {
    const DATA_TYPE: DataType = DataType::I8;
    const SIZE_BYTES: usize = 1;
}

impl Dtype for Fp8E4m3 {
    const ID: DTypeId = DTypeId::F8E4M3;
    fn read_f64(raw: &Self) -> f64 {
        f64::from(raw.0)
    }
    fn write_f64(v: f64) -> Self {
        Self(v.clamp(0.0, u8::MAX as f64) as u8)
    }
}

impl StorageDtype for Fp8E5m2 {
    const DATA_TYPE: DataType = DataType::I8;
    const SIZE_BYTES: usize = 1;
}

impl Dtype for Fp8E5m2 {
    const ID: DTypeId = DTypeId::F8E5M2;
    fn read_f64(raw: &Self) -> f64 {
        f64::from(raw.0)
    }
    fn write_f64(v: f64) -> Self {
        Self(v.clamp(0.0, u8::MAX as f64) as u8)
    }
}

impl Float for f32 {}
impl Float for f16 {}
impl Float for bf16 {}
impl Float for Fp8E4m3 {}
impl Float for Fp8E5m2 {}

fn registry() -> &'static RwLock<HashMap<u16, DTypeSpec>> {
    static REGISTRY: OnceLock<RwLock<HashMap<u16, DTypeSpec>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}
