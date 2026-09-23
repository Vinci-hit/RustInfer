use super::parser::parse;
use super::*;

fn string(out: &mut Vec<u8>, s: &[u8]) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s);
}

type TensorSpec<'a> = (&'a str, &'a [u64], u32, u64);

/// Tiny hand-encoded cases supplement the independently generated upstream fixture.
fn file(
    version: u32,
    metadata: &[(&str, u32, Vec<u8>)],
    tensors: &[TensorSpec<'_>],
    alignment: usize,
    payload: usize,
) -> Vec<u8> {
    let mut b = b"GGUF".to_vec();
    b.extend_from_slice(&version.to_le_bytes());
    b.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    b.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    for (key, ty, value) in metadata {
        string(&mut b, key.as_bytes());
        b.extend_from_slice(&ty.to_le_bytes());
        b.extend_from_slice(value);
    }
    for (name, dims, ty, offset) in tensors {
        string(&mut b, name.as_bytes());
        b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in *dims {
            b.extend_from_slice(&d.to_le_bytes());
        }
        b.extend_from_slice(&ty.to_le_bytes());
        b.extend_from_slice(&offset.to_le_bytes());
    }
    b.resize(b.len().div_ceil(alignment) * alignment + payload, 0);
    b
}

fn error(bytes: &[u8]) -> GgufError {
    match parse(bytes, &ParseLimits::default()) {
        Ok(_) => panic!("expected malformed file to fail"),
        Err(e) => e,
    }
}

#[test]
fn versions_alignment_and_unsorted_offsets() {
    for version in [2, 3] {
        for a in [8, 24, 32, 64] {
            let b = file(
                version,
                &[("general.alignment", 4, (a as u32).to_le_bytes().to_vec())],
                &[("later", &[2, 3], 0, a as u64), ("first", &[1], 0, 0)],
                a,
                a + 24,
            );
            let i = parse(&b, &ParseLimits::default()).unwrap();
            assert_eq!(i.header.alignment, a as u32);
            assert_eq!(i.header.version, version);
            assert_eq!(i.header.data_offset % a as u64, 0);
            assert_eq!(i.tensors[0].dimensions(), &[2, 3]);
            assert_eq!(i.tensors[0].byte_len(), 24);
            assert_eq!(i.tensors[1].file_offset(), i.header.data_offset);
        }
    }
}

#[test]
fn every_truncation_and_deterministic_mutations_are_panic_free() {
    let b = file(
        3,
        &[("flag", 7, vec![1])],
        &[("weight", &[2, 3], 0, 0)],
        32,
        24,
    );
    for n in 0..b.len() {
        assert!(
            parse(&b[..n], &ParseLimits::default()).is_err(),
            "prefix {n}"
        );
    }
    // Include integer boundary values without relying on a random seed/fuzzer.
    for n in 0..b.len() {
        for replacement in [0, 1, 127, 128, 255] {
            let mut mutated = b.clone();
            mutated[n] = replacement;
            let _ = parse(&mutated, &ParseLimits::default());
        }
    }
}

#[test]
fn invalid_headers_and_counts() {
    let mut b = file(3, &[], &[], 32, 0);
    b[0] = 0;
    assert!(matches!(error(&b).kind(), Some(ErrorKind::InvalidMagic)));
    for v in [1, 4, u32::MAX] {
        assert!(matches!(
            error(&file(v, &[], &[], 32, 0)).kind(),
            Some(ErrorKind::UnsupportedVersion(_))
        ));
    }
    assert!(matches!(
        error(&file(3_u32.swap_bytes(), &[], &[], 32, 0)).kind(),
        Some(ErrorKind::UnsupportedEndian)
    ));
    let mut b = file(3, &[], &[], 32, 0);
    b[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(matches!(
        error(&b).kind(),
        Some(ErrorKind::LimitExceeded(_))
    ));
    let limits = ParseLimits {
        max_tensors: u64::MAX,
        ..Default::default()
    };
    assert!(matches!(
        parse(&b, &limits).err().unwrap().kind(),
        Some(ErrorKind::Overflow)
    ));
}

#[test]
fn malformed_metadata() {
    let bad = [
        vec![("flag", 7, vec![2])],
        vec![("utf8", 8, {
            let mut v = Vec::new();
            string(&mut v, &[255]);
            v
        })],
        vec![("future", 99, vec![0])],
        vec![("dup", 0, vec![1]), ("dup", 0, vec![2])],
        vec![("非ASCII", 0, vec![1])],
        vec![("", 0, vec![1])],
    ];
    for entries in bad {
        assert!(parse(&file(3, &entries, &[], 32, 0), &ParseLimits::default()).is_err());
    }
    for a in [0u32, 1, 7, 12] {
        assert!(matches!(
            error(&file(
                3,
                &[("general.alignment", 4, a.to_le_bytes().to_vec())],
                &[],
                32,
                0
            ))
            .kind(),
            Some(ErrorKind::InvalidField(_))
        ));
    }
    assert!(
        error(&file(
            3,
            &[("general.alignment", 10, 32_u64.to_le_bytes().to_vec())],
            &[],
            32,
            0
        ))
        .to_string()
        .contains("alignment")
    );
}

fn array(ty: u32, len: u64, bytes: &[u8]) -> Vec<u8> {
    let mut b = ty.to_le_bytes().to_vec();
    b.extend_from_slice(&len.to_le_bytes());
    b.extend_from_slice(bytes);
    b
}

#[test]
fn nested_empty_arrays_and_limits() {
    let nested = array(9, 1, &array(4, 2, &[1, 0, 0, 0, 2, 0, 0, 0]));
    let b = file(
        3,
        &[("nested", 9, nested), ("empty", 9, array(8, 0, &[]))],
        &[],
        32,
        0,
    );
    let i = parse(&b, &ParseLimits::default()).unwrap();
    assert_eq!(
        i.metadata.get("nested"),
        Some(&GgufValue::Array(GgufArray::Array(vec![GgufArray::U32(
            vec![1, 2]
        )])))
    );
    assert_eq!(
        i.metadata.get("empty"),
        Some(&GgufValue::Array(GgufArray::String(vec![])))
    );
    for limits in [
        ParseLimits {
            max_array_depth: 1,
            ..Default::default()
        },
        ParseLimits {
            max_array_elements: 2,
            ..Default::default()
        },
        ParseLimits {
            max_header_bytes: 24,
            ..Default::default()
        },
        ParseLimits {
            max_index_bytes: 64,
            ..Default::default()
        },
        ParseLimits {
            max_string_bytes: 2,
            ..Default::default()
        },
        ParseLimits {
            max_metadata_entries: 1,
            ..Default::default()
        },
    ] {
        assert!(matches!(
            parse(&b, &limits).err().unwrap().kind(),
            Some(ErrorKind::LimitExceeded(_))
        ));
    }
    let enormous = file(3, &[("a", 9, array(8, u64::MAX, &[]))], &[], 32, 0);
    assert!(matches!(
        error(&enormous).kind(),
        Some(ErrorKind::LimitExceeded(_))
    ));
    assert!(matches!(
        error(&file(3, &[("a", 9, array(99, 0, &[]))], &[], 32, 0)).kind(),
        Some(ErrorKind::UnsupportedMetadataType(99))
    ));
}

#[test]
fn index_budget_counts_decoded_strings_not_just_disk_bytes() {
    let b = file(3, &[("a", 9, array(8, 100, &vec![0; 800]))], &[], 32, 0);
    let limits = ParseLimits {
        max_index_bytes: 1024,
        ..Default::default()
    };
    assert!(b.len() < 1024);
    assert!(matches!(
        parse(&b, &limits).err().unwrap().kind(),
        Some(ErrorKind::LimitExceeded("index allocation budget"))
    ));
}

#[test]
fn invalid_tensor_descriptors() {
    for dims in [&[][..], &[0], &[1, 1, 1, 1, 1], &[128, 2]] {
        assert!(matches!(
            error(&file(3, &[], &[("bad", dims, 11, 0)], 32, 512)).kind(),
            Some(ErrorKind::UnsupportedTensorShape)
        ));
    }
    for ty in [4, 5, 31, 32, 33, 36, 37, 38, u32::MAX] {
        assert!(matches!(
            error(&file(3, &[], &[("unknown", &[256], ty, 0)], 32, 512)).kind(),
            Some(ErrorKind::UnsupportedTensorType(_))
        ));
    }
    assert!(matches!(
        error(&file(3, &[], &[("w", &[u64::MAX, 2], 0, 0)], 32, 0)).kind(),
        Some(ErrorKind::Overflow)
    ));
    assert!(matches!(
        error(&file(3, &[], &[("w", &[1], 0, u64::MAX - 31)], 32, 0)).kind(),
        Some(ErrorKind::Overflow)
    ));
    assert!(matches!(
        error(&file(3, &[], &[("w", &[1], 0, 1)], 32, 4)).kind(),
        Some(ErrorKind::InvalidField(_))
    ));
    assert!(matches!(
        error(&file(3, &[], &[("w", &[1], 0, 32)], 32, 4)).kind(),
        Some(ErrorKind::OutOfBounds)
    ));
    assert!(matches!(
        error(&file(
            3,
            &[],
            &[("w", &[1], 0, 0), ("w", &[1], 0, 32)],
            32,
            36
        ))
        .kind(),
        Some(ErrorKind::Duplicate(_))
    ));
    assert!(matches!(
        error(&file(
            3,
            &[],
            &[("a", &[16], 0, 0), ("b", &[1], 0, 32)],
            32,
            64
        ))
        .kind(),
        Some(ErrorKind::Overlap)
    ));
}

#[test]
fn metadata_only_and_unknown_keys() {
    let b = file(
        3,
        &[("Future.Custom", 10, u64::MAX.to_le_bytes().to_vec())],
        &[],
        32,
        0,
    );
    let i = parse(&b, &ParseLimits::default()).unwrap();
    assert_eq!(
        i.metadata.get("Future.Custom").unwrap().as_u64(),
        Some(u64::MAX)
    );
    assert_eq!(i.metadata.get("Future.Custom").unwrap().as_u32(), None);
    assert!(i.tensors.is_empty());
    // Header-only vocab container may omit unused alignment padding.
    assert!(parse(&file(3, &[], &[], 32, 0)[..24], &ParseLimits::default()).is_ok());
}

#[test]
fn mapped_view_borrows_exact_payload() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gguf/reference.gguf");
    let reader = GgufReader::open(&path).unwrap();
    let t = reader.read_view("test.q3_k").unwrap();
    assert_eq!(t.bytes.len() as u64, t.info.byte_len());
    assert_eq!(
        t.bytes.as_ptr(),
        reader.mmap[t.info.file_offset() as usize..].as_ptr()
    );
    assert!(reader.contains("test.q3_k"));
    assert!(matches!(
        reader.read_view("missing").unwrap_err().kind(),
        Some(ErrorKind::TensorNotFound)
    ));
    assert!(GgufReader::open(path.with_file_name("does-not-exist.gguf")).is_err());
}
