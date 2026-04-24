use half::bf16;
use rayon::prelude::*;

// --- 定义用于转换的 Trait ---
// 这是一个通用的 trait，可以表示任何可以从类型 F 转换到类型 T 的操作
pub trait CastFrom<F> {
    fn cast_from(v: F) -> Self;
}

// --- 为我们需要的转换实现 Trait ---

// a) BF16 -> F32
impl CastFrom<bf16> for f32 {
    #[inline(always)]
    fn cast_from(v: bf16) -> f32 {
        v.to_f32()
    }
}

// b) F32 -> BF16
impl CastFrom<f32> for bf16 {
    #[inline(always)]
    fn cast_from(v: f32) -> bf16 {
        bf16::from_f32(v)
    }
}

impl CastFrom<f32> for f32 {
    #[inline(always)]
    fn cast_from(v: f32) -> f32 {
        v // 直接返回自身
    }
}

// d) BF16 -> BF16 (恒等转换)
impl CastFrom<bf16> for bf16 {
    #[inline(always)]
    fn cast_from(v: bf16) -> bf16 {
        v // 直接返回自身
    }
}
// (未来可以添加 i8 -> f32 等其他转换)


/// 将一个 `from` 切片的内容转换为 `to` 切片。
/// `F` 是 From (源) 类型, `T` 是 To (目标) 类型。
pub fn cast_kernel<F, T>(from_slice: &[F], to_slice: &mut [T])
where
    T: CastFrom<F> + Send,
    F: Send + Sync + Copy,
{
    to_slice
        .par_iter_mut()
        .zip(from_slice.par_iter())
        .for_each(|(to_val, from_val)| {
            *to_val = T::cast_from(*from_val);
        });
}