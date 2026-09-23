//! Little-endian GGML-compatible encodings, independent of GGUF type IDs.
//! Layouts pinned to llama.cpp c550d2f60bde72df19fcef1fef627895095b8ba8.
//! Representability does not imply that a backend implements arithmetic.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockLayout {
    pub elements: usize,
    pub bytes: usize,
}

macro_rules! formats {
    ($(($name:ident, $elements:literal, $bytes:literal)),+ $(,)?) => {
        #[allow(non_camel_case_types)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum BlockQuantFormat { $($name),+ }

        impl BlockQuantFormat {
            pub const ALL: &'static [Self] = &[$(Self::$name),+];
            pub const fn layout(self) -> BlockLayout {
                match self { $(Self::$name => BlockLayout { elements: $elements, bytes: $bytes }),+ }
            }
        }
    }
}

formats! {
    (Q2_K, 256, 84), (Q3_K, 256, 110), (Q4_K, 256, 144),
    (Q5_K, 256, 176), (Q6_K, 256, 210), (Q8_0, 32, 34),
    (IQ2_XXS, 256, 66), (IQ2_XS, 256, 74), (IQ2_S, 256, 82),
    (IQ3_XXS, 256, 98), (IQ3_S, 256, 110),
    (IQ4_NL, 32, 18), (IQ4_XS, 256, 136),
}
