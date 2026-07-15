pub mod quant;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::{OnceLock, RwLock};

use half::{bf16, f16};

use crate::types::{DataType, Dtype as StorageDtype};

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
    const DATA_TYPE: DataType = DataType::F8E4M3;
    const SIZE_BYTES: usize = 1;
}

impl Dtype for Fp8E4m3 {
    const ID: DTypeId = DTypeId::F8E4M3;
    fn read_f64(raw: &Self) -> f64 {
        let bits = raw.0;
        let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
        let exponent = (bits >> 3) & 0x0f;
        let mantissa = bits & 0x07;

        // safetensors F8_E4M3 is the finite-only E4M3FN format used by
        // torch.float8_e4m3fn.  It has bias 7, subnormal step 2^-9, and uses
        // only the all-ones exponent/mantissa pattern for NaN (0x7f/0xff).
        if exponent == 0 {
            sign * f64::from(mantissa) * 2f64.powi(-9)
        } else if exponent == 0x0f && mantissa == 0x07 {
            f64::NAN
        } else {
            sign * (1.0 + f64::from(mantissa) / 8.0) * 2f64.powi(i32::from(exponent) - 7)
        }
    }
    fn write_f64(v: f64) -> Self {
        if v.is_nan() {
            return Self(0x7f);
        }

        let sign = if v.is_sign_negative() { 0x80 } else { 0 };
        let magnitude = v.abs();
        if magnitude == 0.0 {
            return Self(sign);
        }
        if !magnitude.is_finite() || magnitude >= 448.0 {
            return Self(sign | 0x7e);
        }

        // Subnormals are integer multiples of 2^-9.  A rounded mantissa of 8
        // naturally carries into the smallest normal value (exponent field 1).
        if magnitude < 2f64.powi(-6) {
            let mantissa = (magnitude * 512.0).round_ties_even() as u8;
            return if mantissa >= 8 {
                Self(sign | 0x08)
            } else {
                Self(sign | mantissa)
            };
        }

        let unbiased = magnitude.log2().floor() as i32;
        let mut exponent = unbiased + 7;
        let base = 2f64.powi(unbiased);
        let mut mantissa = ((magnitude / base - 1.0) * 8.0).round_ties_even() as i32;
        if mantissa == 8 {
            exponent += 1;
            mantissa = 0;
        }

        // Rounding near the upper edge can otherwise produce the reserved NaN
        // code.  E4M3FN conversion saturates finite overflow to +/-448.
        if exponent > 15 || (exponent == 15 && mantissa > 6) {
            return Self(sign | 0x7e);
        }
        Self(sign | ((exponent as u8) << 3) | mantissa as u8)
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

#[cfg(test)]
mod tests {
    use super::{Dtype, Fp8E4m3};
    use crate::types::{DataType, Dtype as StorageDtype};

    #[test]
    fn fp8_e4m3_has_distinct_storage_dtype() {
        assert_eq!(<Fp8E4m3 as StorageDtype>::DATA_TYPE, DataType::F8E4M3);
        assert_eq!(DataType::F8E4M3.size_in_bytes(), 1);
    }

    #[test]
    fn fp8_e4m3fn_decodes_known_values() {
        let cases = [
            (0x00, 0.0),
            (0x01, 2f64.powi(-9)),
            (0x08, 2f64.powi(-6)),
            (0x38, 1.0),
            (0x3c, 1.5),
            (0x7e, 448.0),
            (0xb8, -1.0),
            (0xfe, -448.0),
        ];
        for (bits, expected) in cases {
            assert_eq!(Fp8E4m3::read_f64(&Fp8E4m3(bits)), expected);
        }
        assert!(Fp8E4m3::read_f64(&Fp8E4m3(0x7f)).is_nan());
        assert!(Fp8E4m3::read_f64(&Fp8E4m3(0xff)).is_nan());
    }

    #[test]
    fn fp8_e4m3fn_encodes_and_saturates() {
        let cases = [
            (0.0, 0x00),
            (-0.0, 0x80),
            (2f64.powi(-9), 0x01),
            (2f64.powi(-6), 0x08),
            (1.0, 0x38),
            (1.5, 0x3c),
            (448.0, 0x7e),
            (-448.0, 0xfe),
            (f64::INFINITY, 0x7e),
            (f64::NEG_INFINITY, 0xfe),
        ];
        for (value, expected) in cases {
            assert_eq!(Fp8E4m3::write_f64(value).0, expected);
        }
        assert_eq!(Fp8E4m3::write_f64(f64::NAN).0, 0x7f);
    }
}
