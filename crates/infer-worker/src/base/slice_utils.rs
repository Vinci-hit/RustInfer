//! 小工具集合。

use crate::base::error::{Error, Result};

/// 从一个 `&mut [T]` 里一次性拿出 N 个 **互不重叠** 的 `&mut T`，N 为运行期动态长度。
///
/// 这是对 `slice::get_disjoint_mut` 的动态长度补丁：标准库只支持 `[usize; N]`（编译期 N）
/// 以及 Range 变体，动态 N 的场景需要自己做 runtime 去重校验 + 裸指针构造。
///
/// # Errors
/// - `indices` 里任何 index 越界 → `Error::InvalidArgument`
/// - `indices` 里有重复 → `Error::InvalidArgument`
pub fn disjoint_mut<'a, T>(slice: &'a mut [T], indices: &[usize]) -> Result<Vec<&'a mut T>> {
    let len = slice.len();
    // 运行时校验：排序 → 相邻去重 → 比长度
    let mut sorted = indices.to_vec();
    sorted.sort_unstable();
    for pair in sorted.windows(2) {
        if pair[0] == pair[1] {
            return Err(Error::InvalidArgument(format!(
                "disjoint_mut: duplicate index {}", pair[0]
            )).into());
        }
    }
    if let Some(&max) = sorted.last()
        && max >= len {
            return Err(Error::InvalidArgument(format!(
                "disjoint_mut: index {} out of range (len = {})", max, len
            )).into());
        }

    // SAFETY: 上面校验了 indices 互不重复、且都 < len。
    //         slice 的生命周期 'a 会自然约束返回的每个 &mut T：编译器把它们视作
    //         对不同 T 的互斥借用，且不会超出 slice 的存活期。
    let base = slice.as_mut_ptr();
    let refs = indices
        .iter()
        .map(|&i| unsafe { &mut *base.add(i) })
        .collect();
    Ok(refs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disjoint_basic() {
        let mut v = vec![10, 20, 30, 40, 50];
        let refs = disjoint_mut(&mut v, &[0, 2, 4]).unwrap();
        for r in refs {
            *r *= 2;
        }
        assert_eq!(v, vec![20, 20, 60, 40, 100]);
    }

    #[test]
    fn disjoint_rejects_duplicates() {
        let mut v = vec![1, 2, 3];
        let err = disjoint_mut(&mut v, &[0, 0]);
        assert!(err.is_err());
    }

    #[test]
    fn disjoint_rejects_oob() {
        let mut v = vec![1, 2, 3];
        let err = disjoint_mut(&mut v, &[3]);
        assert!(err.is_err());
    }

    #[test]
    fn disjoint_preserves_input_order() {
        let mut v = vec![1, 2, 3, 4, 5];
        let refs = disjoint_mut(&mut v, &[4, 1, 2]).unwrap();
        assert_eq!(*refs[0], 5);
        assert_eq!(*refs[1], 2);
        assert_eq!(*refs[2], 3);
    }
}
