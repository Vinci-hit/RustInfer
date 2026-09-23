use std::collections::HashMap;

macro_rules! values {
    ($(($variant:ident, $ty:ty, $getter:ident)),+ $(,)?) => {
        #[derive(Clone, Debug, PartialEq)]
        pub enum GgufValue {
            $($variant($ty),)+
            String(String),
            Array(GgufArray),
        }

        /// Homogeneous arrays retain their element type, including when empty.
        #[derive(Clone, Debug, PartialEq)]
        pub enum GgufArray {
            $($variant(Vec<$ty>),)+
            String(Vec<String>),
            Array(Vec<GgufArray>),
        }

        impl GgufValue {
            $(pub fn $getter(&self) -> Option<$ty> {
                match self { Self::$variant(v) => Some(*v), _ => None }
            })+
            pub fn as_str(&self) -> Option<&str> {
                match self { Self::String(v) => Some(v), _ => None }
            }
            pub fn as_array(&self) -> Option<&GgufArray> {
                match self { Self::Array(v) => Some(v), _ => None }
            }
        }

        impl GgufArray {
            pub fn len(&self) -> usize {
                match self { $(Self::$variant(v) => v.len(),)+ Self::String(v) => v.len(), Self::Array(v) => v.len() }
            }
            pub fn is_empty(&self) -> bool { self.len() == 0 }
        }
    }
}

values! {
    (U8, u8, as_u8), (I8, i8, as_i8), (U16, u16, as_u16),
    (I16, i16, as_i16), (U32, u32, as_u32), (I32, i32, as_i32),
    (F32, f32, as_f32), (Bool, bool, as_bool),
    (U64, u64, as_u64), (I64, i64, as_i64), (F64, f64, as_f64),
}

#[derive(Debug)]
pub struct GgufMetadata {
    pub(super) entries: Vec<(String, GgufValue)>,
    pub(super) by_name: HashMap<String, usize>,
}

impl GgufMetadata {
    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.by_name.get(key).map(|&i| &self.entries[i].1)
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&str, &GgufValue)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
