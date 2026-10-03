# 第 10 章共写记录：CUDA Graph 与动态批次

正文见[第 10 章：CUDA Graph 与动态批次](../chapters/worker/10-cuda-graph-and-dynamic-batching.md)。本页保留作者的提问与后续解释。

## 五条请求使用八行 Graph

作者提问：

> 实际只有 5 个请求，向上取整用了 batch size 为 8 的 Graph，多出来的 3 个位置怎么处理，才能避免它们往真实请求的 KV Cache 里写数据？

结合正文中的代码，用自己的话把后面三行走完：输入 token、query 长度、KV 长度、block table、scatter、attention、argmax 和结果提交分别如何处理。再解释为什么只把 token ID 改成 0，或者只在最后丢弃输出，都不足以保护真实 KV。

<!-- 作者的复述与执行图。 -->

## 地址固定，内容为什么还能变化

设某个节点记录了设备指针 `p`。解释下面两种修改的区别：在 `p` 指向的原缓冲里写入新 token；重新分配一个缓冲 `q`，然后让 Rust 中的 Tensor 变量改为引用 `q`。

再区分：改写设备长度数组、改变捕获时传入的 batch 标量、改变 Rust 端算法分支，哪些会直接影响旧 Graph 的下一次执行，哪些需要额外机制。

<!-- 作者的指针、数据与节点参数示意。 -->

## 八行缩成五行，哪些旧值必须失效

上一轮八条请求全部有效，其中三条结束；本轮只剩五条。画出控制数组只覆盖前五行时，尾部可能保留的内容。假设旧 slot 已被另一请求借走，沿索引解释错误写入如何发生。

再解释为什么 Worker mixed 的一 token Pad 行采用临时 KV lease，而普通 Decode 的 Graph 容量尾部不需要借出三份同样的 lease。

<!-- 作者的尾部状态与 slot 归属推演。 -->

## mixed Graph 为什么有多个形状维度

设 Decode 前缀包含 3 行，后面一条 Prefill 本步输入 17 个 token；capture sizes 为 `[1, 2, 4, 8]`，容量足够，采用默认 mixed token/tile 分桶且没有调优覆盖。先直接从这个四行实际计划推导 `rows`、`tokens`、`tiles` 与 `decode_prefix`，不额外加入服务层 Pad。

写出实际 `q_lens`、两个阶段对应的 token 区间、每行的末 token 索引，再解释捕获用计划的 query 上界为何可以大于真实长度，而不会让 Prefill 凭空多出 token。

<!-- 作者的 bucket、有效元数据与输出位置推导。 -->

返回[书籍入口](../00-README.md)。
