/// A static extent or a named symbolic extent bound at compilation/execution.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Dim {
    Static(usize),
    Symbol(String),
}

impl Dim {
    /// 辅助构造：直接从数字转成静态维度
    pub fn from_static(val: usize) -> Self {
        Dim::Static(val)
    }

    /// 辅助构造：直接创建符号维度
    pub fn from_sym(name: &str) -> Self {
        Dim::Symbol(name.to_string())
    }
}

/// 顺便 permitem 各种极爽的语法糖：直接把 usize 或 &str 转成 Dim
impl From<usize> for Dim {
    fn from(v: usize) -> Self { Dim::Static(v) }
}
impl From<&str> for Dim {
    fn from(s: &str) -> Self { Dim::Symbol(s.to_string()) }
}
