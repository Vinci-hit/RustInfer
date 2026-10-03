use super::*;

fn metadata() -> HashMap<&'static str, GgufValue> {
    let mut alphabet: Vec<_> = ByteLevel::alphabet().into_iter().collect();
    alphabet.sort_unstable();
    let mut tokens: Vec<_> = alphabet.into_iter().map(|c| c.to_string()).collect();
    tokens.extend(
        [
            "ab",
            "abc",
            "12",
            "<|endoftext|>",
            "<|im_end|>",
            "<think>",
            "</think>",
            "<unused>",
        ]
        .map(String::from),
    );
    let mut types = vec![1; 259];
    types.extend([3, 3, 4, 4, 5]);
    HashMap::from([
        ("tokenizer.ggml.model", GgufValue::String("gpt2".into())),
        ("tokenizer.ggml.pre", GgufValue::String("qwen35".into())),
        (
            "tokenizer.ggml.tokens",
            GgufValue::Array(GgufArray::String(tokens)),
        ),
        (
            "tokenizer.ggml.token_type",
            GgufValue::Array(GgufArray::I32(types)),
        ),
        (
            "tokenizer.ggml.merges",
            GgufValue::Array(GgufArray::String(
                ["a b", "ab c", "1 2"].map(String::from).to_vec(),
            )),
        ),
        ("tokenizer.ggml.bos_token_id", GgufValue::U32(259)),
        ("tokenizer.ggml.eos_token_id", GgufValue::U32(260)),
    ])
}

fn build(m: &HashMap<&str, GgufValue>) -> Result<GgufText> {
    GgufText::build(&m.iter().map(|(&k, v)| (k, v)).collect())
}

#[test]
fn bpe_merges_digits_unicode_and_control_tokens() {
    let t = build(&metadata()).unwrap();
    assert_eq!(t.encode("abcabc", false).unwrap(), [257, 257]);
    // Even though "12" is a legal merge, qwen35 splits digits before BPE.
    let digits = t.encode("12", false).unwrap();
    assert_eq!(digits.len(), 2);
    assert!(!digits.contains(&258));
    for text in [
        "你好🙂",
        "\r\n\t  x  ",
        "a\u{035c}b",
        "العربية हिन्दी",
        "can't I'M 123.4",
    ] {
        assert_eq!(
            t.decode(&t.encode(text, false).unwrap(), false).unwrap(),
            text
        );
    }
    assert_eq!(
        t.encode("cafe\u{301}", false).unwrap(),
        t.encode("café", false).unwrap()
    );
    let ids = t
        .encode("<|endoftext|><think>你好</think><|im_end|>", false)
        .unwrap();
    assert_eq!(&ids[..2], &[259, 261]);
    assert_eq!(t.decode(&ids, true).unwrap(), "<think>你好</think>");
    assert!(t.decode(&[-1], true).is_err());
    assert!(t.decode(&[264], true).is_err());
    assert!(t.is_end_token(259) && t.is_end_token(260));
    assert!(!t.is_end_token(261) && !t.is_end_token(-1));
}

#[test]
fn explicit_bos_eos_and_missing_template() {
    let mut m = metadata();
    m.insert("tokenizer.ggml.add_bos_token", GgufValue::Bool(true));
    m.insert("tokenizer.ggml.add_eos_token", GgufValue::Bool(true));
    let t = build(&m).unwrap();
    assert_eq!(t.encode("abc", true).unwrap(), [259, 257, 260]);
    assert_eq!(t.encode("abc", false).unwrap(), [257]);
    assert!(
        t.render_chat(&[message(ChatRole::User, "hello")], false, None)
            .is_err()
    );
}

#[test]
fn malformed_metadata_fails_without_silent_fallback() {
    for (key, value) in [
        ("tokenizer.ggml.pre", GgufValue::String("qwen2".into())),
        ("tokenizer.ggml.model", GgufValue::String("llama".into())),
        ("tokenizer.ggml.eos_token_id", GgufValue::U32(264)),
        ("tokenizer.ggml.eos_token_id", GgufValue::U32(1)),
        ("tokenizer.ggml.eos_token_id", GgufValue::I32(260)),
        (
            "tokenizer.ggml.add_bos_token",
            GgufValue::String("true".into()),
        ),
        (
            "tokenizer.ggml.token_type",
            GgufValue::Array(GgufArray::I32(vec![1])),
        ),
        (
            "tokenizer.ggml.merges",
            GgufValue::Array(GgufArray::String(vec!["a".into()])),
        ),
        (
            "tokenizer.ggml.merges",
            GgufValue::Array(GgufArray::String(vec!["a b".into(), "a b".into()])),
        ),
        (
            "tokenizer.ggml.merges",
            GgufValue::Array(GgufArray::String(vec!["b z".into()])),
        ),
    ] {
        let mut m = metadata();
        m.insert(key, value);
        assert!(build(&m).is_err(), "{key}");
    }
    let mut m = metadata();
    if let GgufValue::Array(GgufArray::String(v)) = m.get_mut("tokenizer.ggml.tokens").unwrap() {
        v[0] = v[1].clone();
    }
    assert!(build(&m).is_err());
}

fn message(role: ChatRole, content: &str) -> ChatMessage {
    ChatMessage {
        role,
        content: content.into(),
        reasoning_content: None,
    }
}

#[test]
fn executes_embedded_jinja_and_propagates_template_errors() {
    let mut m = metadata();
    // Exercise the features used by the real template, including pycompat.
    m.insert("tokenizer.chat_template", GgufValue::String(
        "{%- set ns = namespace(n=0) -%}{%- for m in messages[::-1] -%}{%- set ns.n = ns.n + 1 -%}{{ m.role }}:{{ m.content.strip() }};{%- endfor -%}{{ ns.n }}:{{ enable_thinking }}:{{ add_generation_prompt }}:{{ messages[0].content.startswith('hi') }}".into()));
    let t = build(&m).unwrap();
    let out = t
        .render_chat(
            &[
                message(ChatRole::User, "hi "),
                message(ChatRole::Assistant, " ok"),
            ],
            false,
            None,
        )
        .unwrap();
    assert_eq!(out, "assistant:ok;user:hi;2:False:True:True");
    m.insert(
        "tokenizer.chat_template",
        GgufValue::String("{{ raise_exception('bad message') }}".into()),
    );
    assert!(
        format!(
            "{:#}",
            build(&m)
                .unwrap()
                .render_chat(&[message(ChatRole::User, "x")], true, None)
                .unwrap_err()
        )
        .contains("bad message")
    );
    m.insert(
        "tokenizer.chat_template",
        GgufValue::String("{% invalid %}".into()),
    );
    assert!(
        build(&m)
            .unwrap()
            .render_chat(&[message(ChatRole::User, "x")], true, None)
            .is_err()
    );
}

#[test]
#[ignore = "requires RUSTINFER_GGUF_MODEL pointing to the local Qwen3.8 27B checkpoint"]
fn real_tokenizer_and_template_match_python_reference() {
    let reader = GgufReader::open(std::env::var("RUSTINFER_GGUF_MODEL").unwrap()).unwrap();
    let t = GgufText::from_gguf(&reader).unwrap();
    drop(reader);
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/gguf/qwen35_text_reference.json"
    ))
    .unwrap();
    for case in fixture["encodings"].as_array().unwrap() {
        let input = case["text"].as_str().unwrap();
        let expected: Vec<i32> = serde_json::from_value(case["ids"].clone()).unwrap();
        assert_eq!(t.encode(input, false).unwrap(), expected, "{input:?}");
        assert_eq!(
            t.decode(&expected, false).unwrap(),
            case["decoded"].as_str().unwrap()
        );
    }
    for case in fixture["chats"].as_array().unwrap() {
        let messages: Vec<ChatMessage> = serde_json::from_value(case["messages"].clone()).unwrap();
        let rendered = t
            .render_chat(
                &messages,
                case["thinking"].as_bool().unwrap(),
                case["effort"].as_str(),
            )
            .unwrap();
        assert_eq!(
            rendered,
            case["rendered"].as_str().unwrap(),
            "{}",
            case["name"]
        );
        let expected: Vec<i32> = serde_json::from_value(case["ids"].clone()).unwrap();
        assert_eq!(
            t.encode(&rendered, false).unwrap(),
            expected,
            "{}",
            case["name"]
        );
    }
    assert!(
        t.render_chat(
            &[
                message(ChatRole::User, "x"),
                message(ChatRole::System, "late")
            ],
            false,
            None
        )
        .is_err()
    );
    assert!(
        t.render_chat(&[message(ChatRole::User, "x")], true, Some("invalid"))
            .is_err()
    );
}
