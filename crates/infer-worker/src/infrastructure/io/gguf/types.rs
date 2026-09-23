//! On-disk GGML IDs; shared quantized layouts come from infer-core.
//! Pinned to llama.cpp c550d2f60bde72df19fcef1fef627895095b8ba8.
use infer_core::dtype::quant::BlockQuantFormat;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockLayout {
    pub elements: u64,
    pub bytes: u64,
}

macro_rules! types {
    (block { $(($q:ident, $qid:literal)),+ $(,)? }
     plain { $(($p:ident, $pid:literal, $elements:literal, $bytes:literal)),+ $(,)? }) => {
        #[allow(non_camel_case_types)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        #[repr(u32)]
        pub enum GgmlType { $($q = $qid,)+ $($p = $pid),+ }
        impl GgmlType {
            pub const ALL: &'static [Self] = &[$(Self::$q,)+ $(Self::$p),+];
            pub fn from_id(id: u32) -> Option<Self> {
                match id { $($qid => Some(Self::$q),)+ $($pid => Some(Self::$p),)+ _ => None }
            }
            pub const fn id(self) -> u32 { self as u32 }
            pub const fn name(self) -> &'static str {
                match self { $(Self::$q => stringify!($q),)+ $(Self::$p => stringify!($p)),+ }
            }
            /// Encoding representation only, not a backend capability query.
            pub const fn block_quant_format(self) -> Option<BlockQuantFormat> {
                match self { $(Self::$q => Some(BlockQuantFormat::$q),)+ _ => None }
            }
            pub const fn layout(self) -> BlockLayout {
                match self {
                    $(Self::$q => {
                        let layout = BlockQuantFormat::$q.layout();
                        BlockLayout { elements: layout.elements as u64, bytes: layout.bytes as u64 }
                    },)+
                    $(Self::$p => BlockLayout { elements: $elements, bytes: $bytes }),+
                }
            }
        }
    }
}
types! {
    block {
        (Q2_K, 10), (Q3_K, 11), (Q4_K, 12), (Q5_K, 13), (Q6_K, 14), (Q8_0, 8),
        (IQ2_XXS, 16), (IQ2_XS, 17), (IQ2_S, 22), (IQ3_XXS, 18), (IQ3_S, 21),
        (IQ4_NL, 20), (IQ4_XS, 23),
    }
    plain {
        (F32, 0, 1, 4), (F16, 1, 1, 2),
        (Q4_0, 2, 32, 18), (Q4_1, 3, 32, 20),
        (Q5_0, 6, 32, 22), (Q5_1, 7, 32, 24), (Q8_1, 9, 32, 36),
        (Q8_K, 15, 256, 292), (IQ1_S, 19, 256, 50),
        (I8, 24, 1, 1), (I16, 25, 1, 2), (I32, 26, 1, 4),
        (I64, 27, 1, 8), (F64, 28, 1, 8), (IQ1_M, 29, 256, 56),
        (BF16, 30, 1, 2), (TQ1_0, 34, 256, 54), (TQ2_0, 35, 256, 66),
        (MXFP4, 39, 32, 17), (NVFP4, 40, 64, 36),
        (Q1_0, 41, 128, 18), (Q2_0, 42, 64, 18),
    }
}
