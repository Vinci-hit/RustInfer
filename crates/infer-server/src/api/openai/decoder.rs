//! 渐进式 UTF-8 安全解码
//!
//! 大模型按 token 输出，单个 Unicode 字符（中文 3 字节、emoji 4 字节）可能被
//! 拆分到相邻 token。如果直接把每个 token 解码后当作合法 String 下发，HuggingFace
//! tokenizer 会在边界处填充 `U+FFFD`（REPLACEMENT CHARACTER）——客户端会看到
//! 短暂的乱码闪现。
//!
//! 本模块提供 [`IncrementalDecoder`]：
//!
//! - 维护已收到的 token id 缓冲。
//! - 每次 [`push`](IncrementalDecoder::push) 后用 tokenizer 重 decode 整个缓冲，
//!   把末尾「未确认」的字符（即 `U+FFFD`）暂留，等下一个 token 到来再判定是
//!   真正乱码还是只是被拆开的合法字符。
//! - 通过相邻两次「已确认输出」求 prefix delta，输出对客户端永远合法的增量。
//!
//! 解码失败会向上传播为 [`anyhow::Error`]，由调用方决定如何处理（典型做法是
//! yield Error chunk 并终止流）。

use anyhow::{Context, Result};
use tokenizers::Tokenizer;

const REPLACEMENT_CHAR: char = '\u{FFFD}';

/// 渐进式 UTF-8 解码器。
///
/// 仅向客户端 yield 「已确认合法」的文本：末尾若出现 `U+FFFD`，会被暂留到下一轮
/// 重新 decode 时再裁决（被替换为真实字符则补发，仍为 `U+FFFD` 则视为真乱码 yield）。
///
/// **不是** `Sync` 安全的：每个流持有自己的实例，无并发需求。
pub struct IncrementalDecoder {
    tokenizer: Tokenizer,
    /// 累积的 token id 缓冲；tokenizer.decode 需要看到完整序列才能正确处理 BPE 等。
    token_buffer: Vec<u32>,
    /// 上次已经 yield 给客户端的「已确认」文本前缀。
    confirmed: String,
    /// 是否跳过特殊 token（与 tokenizer.decode 的 skip_special_tokens 对应）。
    skip_special_tokens: bool,
}

impl IncrementalDecoder {
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self {
            tokenizer,
            token_buffer: Vec::new(),
            confirmed: String::new(),
            skip_special_tokens: true,
        }
    }

    /// 推入一个 token，返回**自上次 push 以来**新确认的合法 UTF-8 增量。
    ///
    /// 返回 `Ok(None)` 表示本轮没有可下发的文本（被未确认尾部吞了）。
    /// 返回 `Err` 表示 tokenizer decode 失败——上层应当 yield Error chunk 并终止流，
    /// 因为 tokenizer 状态可能已不一致。
    pub fn push(&mut self, token_id: u32) -> Result<Option<String>> {
        self.token_buffer.push(token_id);

        let full_text = self
            .tokenizer
            .decode(&self.token_buffer, self.skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("{}", e))
            .context("tokenizer.decode failed")?;

        // 取已确认部分：末尾 U+FFFD 之前。
        let confirmed_now = trim_trailing_replacement(&full_text);

        // 与上次确认结果求 char-level 公共前缀，输出 delta。
        // 用 char prefix 而非 byte prefix：tokenizer 在累积过程中可能调整空白
        // 标准化等，char 级求 delta 更稳健。
        let delta = char_prefix_delta(&self.confirmed, confirmed_now);
        self.confirmed = confirmed_now.to_string();

        Ok(if delta.is_empty() { None } else { Some(delta) })
    }

    /// 流结束时调用，返回任何被滞留在「未确认尾部」的文本。
    ///
    /// 含义：如果到流结束都还没补全的 `U+FFFD`，那它就是真正的乱码 / 不可解码字节，
    /// 此时应当 yield 出去而不是丢弃，否则用户会丢字。
    pub fn flush(&mut self) -> Result<Option<String>> {
        let full_text = self
            .tokenizer
            .decode(&self.token_buffer, self.skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("{}", e))
            .context("tokenizer.decode failed in flush")?;

        let delta = char_prefix_delta(&self.confirmed, &full_text);
        self.confirmed = full_text;

        Ok(if delta.is_empty() { None } else { Some(delta) })
    }
}

/// 截掉 `s` 末尾连续的 `U+FFFD`。
///
/// 仅末尾的 `U+FFFD` 视为「未确认」；中间出现的 `U+FFFD` 是真实的不可解码字节
/// （例如模型确实输出了非法序列），保留并下发，避免静默丢字。
fn trim_trailing_replacement(s: &str) -> &str {
    s.trim_end_matches(REPLACEMENT_CHAR)
}

/// 求 `prev` 与 `current` 的 char-level 公共前缀，返回 `current` 中超出的部分。
///
/// 当 `current` 比 `prev` 短或发生分歧时，返回空串（不输出「回退」给客户端，
/// 因为 SSE 是只增协议；这种情况通常是 tokenizer 在累积过程中调整了已输出
/// 字符的形态，我们保守地等下一轮再说）。
fn char_prefix_delta(prev: &str, current: &str) -> String {
    let mut prev_iter = prev.char_indices();
    let mut cur_iter = current.char_indices();
    let mut common_byte_len = 0usize;

    loop {
        match (prev_iter.next(), cur_iter.next()) {
            (Some((_, p)), Some((ci, c))) if p == c => {
                common_byte_len = ci + c.len_utf8();
            }
            (None, _) => break,             // prev 耗尽，current 剩余即为新增
            _ => return String::new(),      // 出现分歧，保守不下发回退
        }
    }

    current[common_byte_len..].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------- 纯函数测试：覆盖 trim_trailing_replacement 与 char_prefix_delta --------

    #[test]
    fn trim_trailing_replacement_only_strips_tail() {
        assert_eq!(trim_trailing_replacement("abc"), "abc");
        assert_eq!(trim_trailing_replacement("abc\u{FFFD}"), "abc");
        assert_eq!(trim_trailing_replacement("abc\u{FFFD}\u{FFFD}"), "abc");
        // 中间的 U+FFFD 必须保留（视为真实乱码）
        assert_eq!(trim_trailing_replacement("a\u{FFFD}b"), "a\u{FFFD}b");
        assert_eq!(trim_trailing_replacement("a\u{FFFD}b\u{FFFD}"), "a\u{FFFD}b");
        // 只有 U+FFFD 的情况
        assert_eq!(trim_trailing_replacement("\u{FFFD}"), "");
        assert_eq!(trim_trailing_replacement("\u{FFFD}\u{FFFD}"), "");
    }

    #[test]
    fn char_prefix_delta_basic() {
        assert_eq!(char_prefix_delta("", ""), "");
        assert_eq!(char_prefix_delta("", "abc"), "abc");
        assert_eq!(char_prefix_delta("ab", "abc"), "c");
        assert_eq!(char_prefix_delta("ab", "abcd"), "cd");
        // current 是 prev 的真前缀（理论上不该发生）→ 不输出回退
        assert_eq!(char_prefix_delta("abc", "ab"), "");
        // 分歧 → 保守返回空
        assert_eq!(char_prefix_delta("abc", "abd"), "");
    }

    #[test]
    fn char_prefix_delta_multibyte() {
        // 中文多字节字符
        assert_eq!(char_prefix_delta("中", "中文"), "文");
        assert_eq!(char_prefix_delta("", "中"), "中");
        assert_eq!(char_prefix_delta("中文", "中文你好"), "你好");
        // emoji 4 字节
        assert_eq!(char_prefix_delta("hi ", "hi 🦀"), "🦀");
        // 不在 char 边界上的字节级前缀也能正确处理（不会切到字符内部）
        assert_eq!(char_prefix_delta("中", "中a"), "a");
    }

    // -------- 端到端测试：用一个真实 tokenizer 验证跨 token UTF-8 行为 --------
    //
    // 测试依赖一个实际的 tokenizer.json 文件。如果环境里没有，测试自动跳过——
    // 这样 CI 在无网/无模型环境也能跑通纯函数测试。

    fn try_load_test_tokenizer() -> Option<Tokenizer> {
        // 优先级：环境变量 > 几个常见相对路径
        let candidates: Vec<std::path::PathBuf> = std::env::var("RUSTINFER_TEST_TOKENIZER")
            .ok()
            .map(std::path::PathBuf::from)
            .into_iter()
            .chain(
                [
                    "tests/data/tokenizer.json",
                    "../../tests/data/tokenizer.json",
                    "/root/RustInfer/tests/data/tokenizer.json",
                ]
                .iter()
                .map(std::path::PathBuf::from),
            )
            .collect();

        for path in candidates {
            if path.exists() {
                if let Ok(tk) = Tokenizer::from_file(&path) {
                    return Some(tk);
                }
            }
        }
        None
    }

    /// 把字节序列编码为 token 序列：先 decode 一个最简短文本拿到该 tokenizer 对应字节的 token id。
    /// 这里不通用——只对 byte-level BPE 的 tokenizer 有效。如果不是，跳过。
    fn encode_bytes_via_tokenizer(tk: &Tokenizer, text: &str) -> Option<Vec<u32>> {
        let enc = tk.encode(text, false).ok()?;
        Some(enc.get_ids().to_vec())
    }

    #[test]
    fn end_to_end_chinese_streaming() {
        let Some(tk) = try_load_test_tokenizer() else {
            eprintln!("skipping: no test tokenizer available");
            return;
        };
        let Some(token_ids) = encode_bytes_via_tokenizer(&tk, "你好世界") else {
            eprintln!("skipping: encode failed");
            return;
        };

        let mut dec = IncrementalDecoder::new(tk);
        let mut out = String::new();
        for id in token_ids {
            if let Some(s) = dec.push(id).expect("decode") {
                // 每次 yield 的内容必须是合法 UTF-8（String 类型已保证），
                // 且不应以 U+FFFD 结尾（除非 flush 阶段）。
                assert!(!s.ends_with(REPLACEMENT_CHAR));
                out.push_str(&s);
            }
        }
        if let Some(tail) = dec.flush().expect("flush") {
            out.push_str(&tail);
        }
        assert_eq!(out, "你好世界");
    }

    #[test]
    fn end_to_end_emoji_streaming() {
        let Some(tk) = try_load_test_tokenizer() else {
            return;
        };
        let Some(token_ids) = encode_bytes_via_tokenizer(&tk, "hi 🦀") else {
            return;
        };

        let mut dec = IncrementalDecoder::new(tk);
        let mut out = String::new();
        for id in token_ids {
            if let Some(s) = dec.push(id).expect("decode") {
                assert!(!s.ends_with(REPLACEMENT_CHAR));
                out.push_str(&s);
            }
        }
        if let Some(tail) = dec.flush().expect("flush") {
            out.push_str(&tail);
        }
        assert_eq!(out, "hi 🦀");
    }
}
