# 第 6 章共写记录：KV Cache 的物理布局与所有权

正文见[第 6 章：KV Cache 的物理布局与所有权](../chapters/worker/06-kv-layout-and-ownership.md)。

## 解释一个 slot 存了什么

A 已有五个 token 的 KV，本步输入 token 17，获得 slot 10。用自己的话说明：slot 10 在一层 K/V 张量中对应什么，在整个模型中又对应什么？本步生成的下一个 token 是否已经拥有自己的 KV？

<!-- 保留作者自己的解释，不代填回答。 -->

## 从形状推导容量

假设模型有 32 个完整 Attention 层，8 个 KV head，head dimension 为 128，KV 使用 BF16。解释每 token 字节数中的每一个因子。再把条件改成两 rank、KV heads 均匀切分，说明每个 rank 的缓存宽度和每 token 字节数如何变化。

继续区分“GPU 上的 pool 已分配容量”和“当前空闲 slot 数”。为什么序列结束后可用 slot 增加，整份 KV pool 的显存占用可以保持不变？

<!-- 作者的容量推导与解释。 -->

## 亲手推进分配器的四个字段

画出 `GlobalKvAllocator` 的 `total`、`free`、`head` 和 `released`，用八个 slot 依次执行下面的操作。这里单独使用分配器保留的懒回收接口，并假设归还前已经满足设备复用条件。

```text
new(8)
alloc_indices(6)
release([2, 4])
alloc_indices(1)
alloc_indices(3)
```

每一步写出返回的编号、完整 `free` 数组、`head`、`released`、`available` 和 `total_free`。说明哪一次申请触发整理，以及为什么已经归还了 2、4，下一次申请仍可能拿到更大的编号。

然后将第三步换成 `free([2, 4])`，重新推演后两步。最后解释 `allocator.release(indices)` 和 `lease.release(&mut allocator)` 各自调用了什么。

<!-- 作者自己的分配与回收账本。 -->

## 沿一个编号走到显存

假设本次拿到 slot 7，解释这个数字依次出现在 lease、`SeqStep`、主机暂存区和设备索引的什么位置。kernel 还需要哪些信息，才能从 7 算出具体 K/V 地址？

再沿归还方向解释：执行 `free([7])` 时，CPU 数组、GPU block table、K/V 张量内容和 Storage 引用分别会发生什么变化？哪些内容需要后续步骤更新？

<!-- 作者对编号、索引与显存联动的解释。 -->

## 把预留计入账本

关闭前缀缓存，只考虑一个有 12 个可分配 slot 的小池，忽略池外的额外预留 block。A 已提交五个 slot，本步新借一个 slot，下一步还预留一个 slot。此时 B 要处理四个新输入 token。

先判断 B 能否一次借到四个 slot，再解释为什么只看 A 的 `block_table.len()` 会算错。最后说明本步取消 A 时，旧表、本步 lease 和下一步预留分别由哪里持有，回收还需要考虑什么设备依赖。

<!-- 作者的账本与取消推演。 -->

## 共享编号怎样安全归还

B 的表为 `[21, 22, 13, 14]`，C 的表为 `[21, 22, 15]`，其中 21、22 是共享前缀。解释 B 结束后哪些引用消失，为什么不能立即归还整张 B 的表，以及谁决定缓存中的 slot 最终可以重新分配。

再解释两个问题：复制 `Vec<u32>` 会不会自动给这些 slot 增加引用计数？`KvLease::commit()` 会不会等待 GPU 并自动更新序列长度？

<!-- 作者对跨层所有权与提交的解释。 -->

返回[书籍入口](../00-README.md)。
