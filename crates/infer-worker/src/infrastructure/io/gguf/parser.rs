use std::collections::HashMap;
use std::mem::size_of;

use super::{
    ByteOrder, ErrorKind, GgmlType, GgufArray, GgufError, GgufHeader, GgufMetadata, GgufTensorInfo,
    GgufValue, ParseLimits,
};

pub(super) struct Index {
    pub header: GgufHeader,
    pub metadata: GgufMetadata,
    pub tensors: Vec<GgufTensorInfo>,
    pub by_name: HashMap<String, usize>,
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
    limits: &'a ParseLimits,
    allocated: u64,
    array_elements: u64,
}

macro_rules! numbers {
    ($(($name:ident, $ty:ty)),+ $(,)?) => {$(
        fn $name(&mut self) -> Result<$ty, GgufError> {
            let mut bytes = [0; size_of::<$ty>()];
            bytes.copy_from_slice(self.take(size_of::<$ty>())?);
            Ok(<$ty>::from_le_bytes(bytes))
        }
    )+}
}

impl<'a> Cursor<'a> {
    fn error(&self, kind: ErrorKind) -> GgufError {
        GgufError::at(self.pos, "field", kind)
    }

    fn check_available(&self, n: u64) -> Result<(), GgufError> {
        let end = (self.pos as u64)
            .checked_add(n)
            .ok_or_else(|| self.error(ErrorKind::Overflow))?;
        if end > self.bytes.len() as u64 {
            return Err(self.error(ErrorKind::Truncated {
                needed: n,
                remaining: (self.bytes.len() - self.pos) as u64,
            }));
        }
        if end > self.limits.max_header_bytes {
            return Err(self.error(ErrorKind::LimitExceeded("header bytes")));
        }
        Ok(())
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], GgufError> {
        self.check_available(n as u64)?;
        let start = self.pos;
        self.pos += n;
        Ok(&self.bytes[start..self.pos])
    }

    numbers! { (u8, u8), (i8, i8), (u16, u16), (i16, i16),
    (u32, u32), (i32, i32), (u64, u64), (i64, i64), (f32, f32), (f64, f64) }

    fn boolean(&mut self) -> Result<bool, GgufError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(GgufError::at(
                self.pos - 1,
                "bool",
                ErrorKind::InvalidField("bool must be 0 or 1"),
            )),
        }
    }

    fn charge(&mut self, n: u64) -> Result<(), GgufError> {
        self.allocated = self
            .allocated
            .checked_add(n)
            .ok_or_else(|| self.error(ErrorKind::Overflow))?;
        if self.allocated > self.limits.max_index_bytes {
            return Err(self.error(ErrorKind::LimitExceeded("index allocation budget")));
        }
        Ok(())
    }

    fn vector<T>(&mut self, count: u64) -> Result<Vec<T>, GgufError> {
        self.charge(
            count
                .checked_mul(size_of::<T>() as u64)
                .ok_or_else(|| self.error(ErrorKind::Overflow))?,
        )?;
        let count = usize::try_from(count).map_err(|_| self.error(ErrorKind::Overflow))?;
        let mut v = Vec::new();
        v.try_reserve_exact(count)
            .map_err(|_| self.error(ErrorKind::Allocation))?;
        Ok(v)
    }

    fn map(&mut self, count: u64) -> Result<HashMap<String, usize>, GgufError> {
        // Conservative upper bound for HashMap bucket/control/capacity overhead;
        // key bytes are charged separately, including each duplicated name.
        self.charge(
            count
                .checked_mul(4 * (size_of::<(String, usize)>() as u64 + 8))
                .ok_or_else(|| self.error(ErrorKind::Overflow))?,
        )?;
        let mut map = HashMap::new();
        map.try_reserve(usize::try_from(count).map_err(|_| self.error(ErrorKind::Overflow))?)
            .map_err(|_| self.error(ErrorKind::Allocation))?;
        Ok(map)
    }

    fn string(&mut self, maximum: u64, ascii: bool) -> Result<String, GgufError> {
        let len = self.u64()?;
        if len > maximum || len > self.limits.max_string_bytes {
            return Err(self.error(ErrorKind::LimitExceeded("string bytes")));
        }
        self.check_available(len)?;
        let len_usize = usize::try_from(len).map_err(|_| self.error(ErrorKind::Overflow))?;
        let start = self.pos;
        let data = self.take(len_usize)?;
        let s = std::str::from_utf8(data)
            .map_err(|_| GgufError::at(start, "string", ErrorKind::InvalidField("UTF-8")))?;
        if ascii && !s.is_ascii() {
            return Err(GgufError::at(
                start,
                "key",
                ErrorKind::InvalidField("metadata key must be ASCII"),
            ));
        }
        self.copy_string(s)
    }

    fn copy_string(&mut self, s: &str) -> Result<String, GgufError> {
        self.charge(s.len() as u64)?;
        let mut result = String::new();
        result
            .try_reserve_exact(s.len())
            .map_err(|_| self.error(ErrorKind::Allocation))?;
        result.push_str(s);
        Ok(result)
    }

    fn value(&mut self, ty: u32) -> Result<GgufValue, GgufError> {
        Ok(match ty {
            0 => GgufValue::U8(self.u8()?),
            1 => GgufValue::I8(self.i8()?),
            2 => GgufValue::U16(self.u16()?),
            3 => GgufValue::I16(self.i16()?),
            4 => GgufValue::U32(self.u32()?),
            5 => GgufValue::I32(self.i32()?),
            6 => GgufValue::F32(self.f32()?),
            7 => GgufValue::Bool(self.boolean()?),
            8 => GgufValue::String(self.string(u64::MAX, false)?),
            9 => GgufValue::Array(self.array(1)?),
            10 => GgufValue::U64(self.u64()?),
            11 => GgufValue::I64(self.i64()?),
            12 => GgufValue::F64(self.f64()?),
            id => return Err(self.error(ErrorKind::UnsupportedMetadataType(id))),
        })
    }

    fn array(&mut self, depth: usize) -> Result<GgufArray, GgufError> {
        // A hard ceiling protects the call stack even if callers raise limits.
        if depth > self.limits.max_array_depth.min(64) {
            return Err(self.error(ErrorKind::LimitExceeded("array depth")));
        }
        let ty = self.u32()?;
        let len = self.u64()?;
        self.array_elements = self
            .array_elements
            .checked_add(len)
            .ok_or_else(|| self.error(ErrorKind::Overflow))?;
        if self.array_elements > self.limits.max_array_elements {
            return Err(self.error(ErrorKind::LimitExceeded("array elements")));
        }
        // Even variable-sized items have a minimum encoded size. Check before
        // reserving their decoded container, including empty typed arrays.
        let minimum: u64 = match ty {
            0 | 1 | 7 => 1,
            2 | 3 => 2,
            4..=6 => 4,
            8 | 10..=12 => 8,
            9 => 12,
            id => return Err(self.error(ErrorKind::UnsupportedMetadataType(id))),
        };
        self.check_available(
            len.checked_mul(minimum)
                .ok_or_else(|| self.error(ErrorKind::Overflow))?,
        )?;
        macro_rules! read_array {
            ($variant:ident, $read:expr) => {{
                let mut items = self.vector(len)?;
                for _ in 0..len {
                    items.push($read);
                }
                GgufArray::$variant(items)
            }};
        }
        Ok(match ty {
            0 => read_array!(U8, self.u8()?),
            1 => read_array!(I8, self.i8()?),
            2 => read_array!(U16, self.u16()?),
            3 => read_array!(I16, self.i16()?),
            4 => read_array!(U32, self.u32()?),
            5 => read_array!(I32, self.i32()?),
            6 => read_array!(F32, self.f32()?),
            7 => read_array!(Bool, self.boolean()?),
            8 => read_array!(String, self.string(u64::MAX, false)?),
            9 => read_array!(Array, self.array(depth + 1)?),
            10 => read_array!(U64, self.u64()?),
            11 => read_array!(I64, self.i64()?),
            12 => read_array!(F64, self.f64()?),
            _ => unreachable!("type checked above"),
        })
    }
}

pub(super) fn parse(bytes: &[u8], limits: &ParseLimits) -> Result<Index, GgufError> {
    let mut c = Cursor {
        bytes,
        pos: 0,
        limits,
        allocated: 0,
        array_elements: 0,
    };
    if c.take(4)? != b"GGUF" {
        return Err(GgufError::at(0, "magic", ErrorKind::InvalidMagic));
    }
    let version = c.u32()?;
    if version.swap_bytes() == 2 || version.swap_bytes() == 3 {
        return Err(GgufError::at(4, "version", ErrorKind::UnsupportedEndian));
    }
    if version != 2 && version != 3 {
        return Err(GgufError::at(
            4,
            "version",
            ErrorKind::UnsupportedVersion(version),
        ));
    }
    let tensor_count = c.u64()?;
    let metadata_count = c.u64()?;
    if tensor_count > limits.max_tensors {
        return Err(c.error(ErrorKind::LimitExceeded("tensor count")));
    }
    if metadata_count > limits.max_metadata_entries {
        return Err(c.error(ErrorKind::LimitExceeded("metadata count")));
    }
    // Minimum 13 bytes per KV, 32 per tensor (one dimension, empty name).
    let minimum = metadata_count
        .checked_mul(13)
        .and_then(|n| tensor_count.checked_mul(32).and_then(|t| n.checked_add(t)))
        .ok_or_else(|| c.error(ErrorKind::Overflow))?;
    c.check_available(minimum)?;
    let mut metadata = GgufMetadata {
        entries: c.vector(metadata_count)?,
        by_name: c.map(metadata_count)?,
    };
    for _ in 0..metadata_count {
        let start = c.pos;
        let key = c.string(65535, true)?;
        if key.is_empty() {
            return Err(GgufError::at(
                start,
                "key",
                ErrorKind::InvalidField("empty metadata key"),
            ));
        }
        if metadata.by_name.contains_key(&key) {
            return Err(GgufError::at(
                start,
                &key,
                ErrorKind::Duplicate("metadata key"),
            ));
        }
        let ty = c.u32()?;
        let value = c.value(ty).map_err(|e| e.context(&key))?;
        metadata
            .by_name
            .insert(c.copy_string(&key)?, metadata.entries.len());
        metadata.entries.push((key, value));
    }
    let alignment = match metadata.get("general.alignment") {
        None => 32,
        Some(GgufValue::U32(v)) if *v != 0 && v % 8 == 0 => *v,
        Some(_) => {
            return Err(c.error(ErrorKind::InvalidField(
                "general.alignment must be nonzero U32 multiple of 8",
            )));
        }
    };
    let mut tensors: Vec<GgufTensorInfo> = c.vector(tensor_count)?;
    let mut by_name = c.map(tensor_count)?;
    // Keep directory positions for diagnostics until interval validation ends.
    let mut positions: Vec<usize> = c.vector(tensor_count)?;
    for _ in 0..tensor_count {
        let start = c.pos;
        let name = c.string(64, false)?;
        if name.is_empty() {
            return Err(GgufError::at(
                start,
                "tensor",
                ErrorKind::InvalidField("empty tensor name"),
            ));
        }
        if by_name.contains_key(&name) {
            return Err(GgufError::at(
                start,
                &name,
                ErrorKind::Duplicate("tensor name"),
            ));
        }
        let info = (|| {
            let rank = c.u32()?;
            if !(1..=4).contains(&rank) {
                return Err(c.error(ErrorKind::UnsupportedTensorShape));
            }
            let mut dimensions = c.vector(rank as u64)?;
            for _ in 0..rank {
                dimensions.push(c.u64()?);
            }
            let type_offset = c.pos;
            let id = c.u32()?;
            let ggml_type = GgmlType::from_id(id).ok_or_else(|| {
                GgufError::at(type_offset, "type", ErrorKind::UnsupportedTensorType(id))
            })?;
            let relative_offset = c.u64()?;
            if relative_offset % u64::from(alignment) != 0 {
                return Err(c.error(ErrorKind::InvalidField("unaligned tensor offset")));
            }
            let layout = ggml_type.layout();
            if dimensions.contains(&0) || dimensions[0] % layout.elements != 0 {
                return Err(c.error(ErrorKind::UnsupportedTensorShape));
            }
            let mut byte_len = (dimensions[0] / layout.elements)
                .checked_mul(layout.bytes)
                .ok_or_else(|| c.error(ErrorKind::Overflow))?;
            for d in &dimensions[1..] {
                byte_len = byte_len
                    .checked_mul(*d)
                    .ok_or_else(|| c.error(ErrorKind::Overflow))?;
            }
            Ok((ggml_type, dimensions, relative_offset, byte_len))
        })()
        .map_err(|e: GgufError| e.context(&name))?;
        by_name.insert(c.copy_string(&name)?, tensors.len());
        tensors.push(GgufTensorInfo {
            name,
            ggml_type: info.0,
            dimensions: info.1,
            relative_offset: info.2,
            file_offset: 0,
            byte_len: info.3,
        });
        positions.push(start);
    }
    let a = u64::from(alignment);
    let padding = (a - c.pos as u64 % a) % a;
    let data_offset = (c.pos as u64)
        .checked_add(padding)
        .ok_or_else(|| c.error(ErrorKind::Overflow))?;
    // Metadata-only files need not physically contain unused tensor padding.
    if tensor_count > 0 && data_offset > bytes.len() as u64 {
        return Err(c.error(ErrorKind::OutOfBounds));
    }
    let mut intervals: Vec<(u64, u64, usize)> = c.vector(tensor_count)?;
    for (i, t) in tensors.iter_mut().enumerate() {
        let err = |kind| GgufError::at(positions[i], &t.name, kind);
        let start = data_offset
            .checked_add(t.relative_offset)
            .ok_or_else(|| err(ErrorKind::Overflow))?;
        let end = start
            .checked_add(t.byte_len)
            .ok_or_else(|| err(ErrorKind::Overflow))?;
        if end > bytes.len() as u64 {
            return Err(err(ErrorKind::OutOfBounds));
        }
        usize::try_from(end).map_err(|_| err(ErrorKind::Overflow))?;
        t.file_offset = start;
        intervals.push((start, end, i));
    }
    intervals.sort_unstable_by_key(|&(start, _, _)| start);
    for pair in intervals.windows(2) {
        if pair[0].1 > pair[1].0 {
            let i = pair[1].2;
            return Err(GgufError::at(
                positions[i],
                &tensors[i].name,
                ErrorKind::Overlap,
            ));
        }
    }
    Ok(Index {
        header: GgufHeader {
            version,
            byte_order: ByteOrder::LittleEndian,
            tensor_count,
            metadata_count,
            alignment,
            data_offset,
        },
        metadata,
        tensors,
        by_name,
    })
}
