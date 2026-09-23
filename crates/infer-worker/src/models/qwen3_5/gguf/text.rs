//! Qwen35 byte-level BPE and the GGUF's own Jinja chat template.
//! No external tokenizer files, model-name guessing, or llama.cpp runtime.
use crate::infrastructure::io::gguf::{GgufArray, GgufReader, GgufValue};
use anyhow::{Context, Result, anyhow, bail, ensure};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tokenizers::{
    AddedToken, SplitDelimiterBehavior, Tokenizer,
    models::bpe::{BPE, Vocab},
    normalizers::unicode::NFC,
    pre_tokenizers::{
        byte_level::ByteLevel,
        sequence::Sequence,
        split::{Split, SplitPattern},
    },
};

// Qwen3_5Tokenizer's PRETOKENIZE_REGEX: combining marks belong with letters,
// numbers split one digit at a time. NFC precedes byte-level BPE.
const QWEN35_SPLIT: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    Developer,
    User,
    Assistant,
}

/// Text messages only. Image/video/tool payloads are not part of this runner.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

pub struct GgufText {
    tokenizer: Tokenizer,
    template: Option<String>,
    vocab_size: usize,
    bos: Option<u32>,
    eos: u32,
    add_bos: bool,
    add_eos: bool,
    end_tokens: HashSet<u32>,
}

impl GgufText {
    pub fn from_gguf(reader: &GgufReader) -> Result<Self> {
        Self::build(&reader.metadata().iter().collect())
    }

    fn build(m: &HashMap<&str, &GgufValue>) -> Result<Self> {
        let string = |key| m.get(key).and_then(|v| v.as_str());
        ensure!(
            string("tokenizer.ggml.model") == Some("gpt2"),
            "GGUF text requires tokenizer.ggml.model=gpt2"
        );
        ensure!(
            string("tokenizer.ggml.pre") == Some("qwen35"),
            "unsupported GGUF pre-tokenizer (expected qwen35)"
        );
        let strings = |key| -> Result<&Vec<String>> {
            match m.get(key) {
                Some(GgufValue::Array(GgufArray::String(v))) => Ok(v),
                _ => bail!("missing or invalid GGUF string array {key}"),
            }
        };
        let tokens = strings("tokenizer.ggml.tokens")?;
        ensure!(
            !tokens.is_empty() && tokens.len() <= i32::MAX as usize,
            "invalid tokenizer vocabulary size"
        );
        let types = match m.get("tokenizer.ggml.token_type") {
            Some(GgufValue::Array(GgufArray::I32(v))) if v.len() == tokens.len() => v,
            _ => bail!("missing or invalid tokenizer.ggml.token_type"),
        };
        ensure!(
            types.iter().all(|t| matches!(t, 1 | 3 | 4 | 5)),
            "unsupported Qwen35 token type"
        );
        let mut vocab = Vocab::default();
        for (id, token) in tokens.iter().enumerate() {
            ensure!(
                !token.is_empty() && vocab.insert(token.clone(), id as u32).is_none(),
                "empty or duplicate GGUF token at ID {id}"
            );
        }
        for byte in ByteLevel::alphabet() {
            ensure!(
                vocab
                    .get(&byte.to_string())
                    .is_some_and(|&id| types[id as usize] == 1),
                "missing byte alphabet token {byte:?}"
            );
        }
        let mut seen = HashSet::new();
        let mut merges = Vec::new();
        for merge in strings("tokenizer.ggml.merges")? {
            let (left, right) = merge.split_once(' ').context("malformed GGUF BPE merge")?;
            ensure!(
                !left.is_empty() && !right.is_empty() && !right.contains(' '),
                "malformed GGUF BPE merge"
            );
            ensure!(seen.insert(merge), "duplicate GGUF BPE merge");
            for piece in [left.to_owned(), right.to_owned(), format!("{left}{right}")] {
                ensure!(
                    vocab.get(&piece).is_some_and(|&id| types[id as usize] == 1),
                    "merge references missing/non-normal token {piece:?}"
                );
            }
            merges.push((left.to_owned(), right.to_owned()));
        }
        let model = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .build()
            .map_err(tok_error)?;
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_normalizer(Some(NFC)).map_err(tok_error)?;
        tokenizer.with_pre_tokenizer(Some(Sequence::new(vec![
            Split::new(
                SplitPattern::Regex(QWEN35_SPLIT.into()),
                SplitDelimiterBehavior::Isolated,
                false,
            )
            .map_err(tok_error)?
            .into(),
            ByteLevel::new(false, true, false).into(),
        ])));
        tokenizer.with_decoder(Some(ByteLevel::default()));
        // CONTROL is removed by skip_control; USER_DEFINED (e.g. <think>) stays.
        tokenizer
            .add_tokens(
                tokens
                    .iter()
                    .zip(types)
                    .filter(|(_, kind)| matches!(**kind, 3 | 4))
                    .map(|(token, &kind)| {
                        AddedToken::from(token.clone(), kind == 3).normalized(false)
                    }),
            )
            .map_err(tok_error)?;
        ensure!(
            tokenizer.get_vocab_size(true) == tokens.len(),
            "added tokens changed GGUF IDs"
        );
        let id = |key: &str| -> Result<Option<u32>> {
            let Some(value) = m.get(key) else {
                return Ok(None);
            };
            let n = value.as_u32().with_context(|| format!("invalid {key}"))?;
            ensure!(
                (n as usize) < tokens.len() && matches!(types[n as usize], 3 | 4),
                "invalid special token ID {key}"
            );
            Ok(Some(n))
        };
        let boolean = |key: &str| -> Result<bool> {
            m.get(key).map_or(Ok(false), |v| {
                v.as_bool().with_context(|| format!("invalid {key}"))
            })
        };
        let bos = id("tokenizer.ggml.bos_token_id")?;
        let eos =
            id("tokenizer.ggml.eos_token_id")?.context("missing tokenizer.ggml.eos_token_id")?;
        let add_bos = boolean("tokenizer.ggml.add_bos_token")?;
        let add_eos = boolean("tokenizer.ggml.add_eos_token")?;
        ensure!(
            !add_bos || bos.is_some(),
            "add_bos_token requires bos_token_id"
        );
        let mut end_tokens = HashSet::from([eos]);
        for key in ["tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"] {
            if let Some(id) = id(key)? {
                end_tokens.insert(id);
            }
        }
        // Qwen may also terminate with endoftext rather than the turn-ending EOS.
        if let Some(id) = tokenizer.token_to_id("<|endoftext|>") {
            ensure!(types[id as usize] == 3, "endoftext must be a control token");
            end_tokens.insert(id);
        }
        let template = match m.get("tokenizer.chat_template") {
            Some(GgufValue::String(v)) => Some(v.clone()),
            None => None,
            _ => bail!("unsupported tokenizer.chat_template type"),
        };
        Ok(Self {
            tokenizer,
            template,
            vocab_size: tokens.len(),
            bos,
            eos,
            add_bos,
            add_eos,
            end_tokens,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn is_end_token(&self, id: i32) -> bool {
        id >= 0 && self.end_tokens.contains(&(id as u32))
    }

    /// Raw completion encoding honors the GGUF's add_bos/add_eos flags when
    /// requested. Rendered chat must pass false: its template owns boundaries.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i32>> {
        let encoded = self.tokenizer.encode(text, false).map_err(tok_error)?;
        let mut ids = Vec::with_capacity(encoded.len() + 2);
        if add_special_tokens && self.add_bos {
            ids.push(self.bos.unwrap() as i32);
        }
        ids.extend(encoded.get_ids().iter().map(|&id| id as i32));
        if add_special_tokens && self.add_eos {
            ids.push(self.eos as i32);
        }
        Ok(ids)
    }

    /// Decode the complete sequence together: individual IDs can end inside a
    /// UTF-8 character. USER_DEFINED tags such as <think> are preserved.
    pub fn decode(&self, ids: &[i32], skip_control: bool) -> Result<String> {
        ensure!(
            ids.iter()
                .all(|&id| id >= 0 && (id as usize) < self.vocab_size),
            "decode token ID outside vocabulary"
        );
        self.tokenizer
            .decode(
                &ids.iter().map(|&id| id as u32).collect::<Vec<_>>(),
                skip_control,
            )
            .map_err(tok_error)
    }

    pub fn render_chat(
        &self,
        messages: &[ChatMessage],
        thinking: bool,
        reasoning_effort: Option<&str>,
    ) -> Result<String> {
        ensure!(!messages.is_empty(), "chat messages must not be empty");
        let source = self
            .template
            .as_deref()
            .context("GGUF is missing tokenizer.chat_template; use raw completion")?;
        let mut env = minijinja::Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function(
            "raise_exception",
            |msg: String| -> std::result::Result<String, minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    msg,
                ))
            },
        );
        env.add_template("chat", source)
            .context("invalid GGUF chat template")?;
        let mut context = serde_json::json!({
            "messages": messages, "add_generation_prompt": true, "enable_thinking": thinking,
            "bos_token": self.bos.and_then(|id| self.tokenizer.id_to_token(id)),
            "eos_token": self.tokenizer.id_to_token(self.eos)
        });
        if let Some(effort) = reasoning_effort {
            context["reasoning_effort"] = effort.into();
        }
        env.get_template("chat")?
            .render(context)
            .context("GGUF chat rendering failed")
    }
}

fn tok_error(err: tokenizers::Error) -> anyhow::Error {
    anyhow!("GGUF tokenizer: {err}")
}

#[cfg(test)]
mod tests;
