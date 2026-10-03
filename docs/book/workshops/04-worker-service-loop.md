# 第 4 章共写记录：Worker 全貌与服务循环

正文见[第 4 章：Worker 全貌与服务循环](../chapters/worker/04-worker-service-loop.md)。

## 作者提出的讲解重点

作者原话：

> 解释一下控制面，数据面，WorkGroup内部怎么协作，server收到信息后怎么组装好变成plan丢给runner

本章将这些问题与 A 正在 Decode、B 的 Prefill 到达的场景结合，展开 Worker Server、Model Runner 和组内 TP 的协作。原第二篇大纲中的详细职责说明已经融入正文。

## 从一个问题开始复述

A 当前已经有 5 个输入 token 的 KV，最近生成的 token 是 17；B 的输入为四个 token。先凭理解说明：Worker Server 为 A、B 分别准备什么，交给 Runtime 的 `StepRequest` 包含什么，Runtime 还需要完成哪些准备才能执行？

<!-- 保留作者之后的解释，再围绕具体疑问讨论。 -->

## 继续连接通信与 Group

沿同一场景解释 Prefill、Cancel、StepOutput 分别走哪条链路，谁读取这些消息。再将单卡改为两卡 TP，说明哪些状态留在服务层、哪些工作由两个 Runtime 一起推进，以及最终由谁向 Scheduler 返回结果。

<!-- 作者的通信与 Group 协作解释。 -->

## 用时间顺序串起一轮执行

解释上一轮结果完成、序列状态更新、下一轮提交和结果回传的先后关系。再让 A 结束或让 B 在已提交计算期间被取消，说明哪些状态需要清理、哪些完成处理仍需要继续。

<!-- 作者的时间线与追问。 -->

返回[书籍入口](../00-README.md)。
