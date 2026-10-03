# 专题 A 共写记录：推测解码

正文见[专题 A：从草稿提案到状态提交](../topics/01-speculative-decoding.md)。本页保留作者自己的流程复述、位置推导与设计解释。

## 从 Prefill 结束开始，讲完一轮

假设 target 已有 6 个 KV 位置，刚刚返回 token 17。请用自己的话解释 pending 是什么，然后依次说出服务循环、SpeculativeServing、proposer、draft head、Runtime、GreedyVerifier 与 commit_decode 的工作。

重点解释谁持有请求状态，谁拥有两份 KV，谁决定接受数量，谁最后把资源变成正式历史。

<!-- 作者的口述或组件时序图。 -->

## 亲手填写拒绝、全接受与 EOS 三张账本

草稿为 `[23,24,25]`，target 的四行预测为 `[23,99,26,27]`，临时 slot 为 `[40,41,42,43]`。写出各行输入、它预测的位置、最终输出、保留输入、归还 slot、下一轮 pending 与 KV 长度。

再分别把预测改成 `[23,24,25,26]`，以及把第二个草稿改成 EOS 并令全部草稿匹配。解释 accepted_drafts、输出长度与 materialized_tokens 的关系。

<!-- 作者的三组位置与资源账本。 -->

## 草稿缓存为什么还要 catch-up

把 prompt `[x0,x1,x2,x3,x4,x5]` 分成两块，画出 MTP/EAGLE3 的 shifted pairs 与末尾 carry。再沿“只接受一个草稿”更新到下一轮，说明 target 长度 L 与草稿已确认 pair 数为什么差一。

解释即使三个草稿全部接受，为什么仍要用真实 target 特征更新 ConditionedProposer；再对照 DFlash，说明它的 context 为什么没有这一位错位。

<!-- 作者的对齐图、状态变化与两种 proposer 的解释。 -->

## 拒绝后，哪些内容必须重算

分别考虑 Full Attention target、带 conv/SSM state 的混合 target，以及 DFlash 的临时块。说明哪些旧字节可以保留但失去可见性，哪些状态需要 snapshot/restore/replay，哪些缓存需要被真实特征覆盖。

最后解释：把每轮草稿数从 3 加到 7，为什么可能降低每 token 速度；如果要扩展到多请求或 TP，需要增加哪些状态与协调机制？

<!-- 作者的恢复流程、成本解释与扩展设计。 -->

返回[书籍入口](../00-README.md)。
