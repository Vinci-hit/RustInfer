# 第 3 章共写记录：Scheduler 与 Worker Group

正文见[第 3 章：Scheduler——面向 Worker Group 的通信与资源调度](../chapters/requests/03-scheduler-and-worker-group.md)。本页保留作者的设计动机、口述与推演。

## 最初的设计意图

作者原话：

> 其实在我的最初设想里，这是承担着多机之间通信和资源调度的作用，然后worker要视为worker group。

关于将 Scheduler 独立出来的首要目的，作者确认：

> 统一管理多机器上的 Worker Group，并分配请求与资源

这一定义作为本章开头：Group 提供一个模型实例的推理能力，Scheduler 面向 Group 管理请求与资源。当前单 Group 的机制与后续多机器、多 Group 的设计分别展开。

## 职责如何划分

可以沿一个具体部署继续解释：两台机器各部署一个完整模型副本，每个副本在组内使用两张 GPU 做 TP。新请求到达后，哪些决定属于 Scheduler，哪些由 Worker Group 内部完成？

<!-- 作者口述或图示；收到后保留原稿，再整理进正文。 -->

## 从连接到就绪

正文见 [3.2：建立连接、报告容量与确认就绪](../chapters/requests/03-scheduler-and-worker-group.md#connection-and-readiness)。

可以先凭理解串起三个时刻：Worker 已经能交换 Hello，Worker 已经发送 Ready，Scheduler Engine 已经开始调度。分别解释这时完成了哪些准备，以及哪些工作还需要继续。

<!-- 作者对启动过程的复述。 -->

继续沿容量报告推演：一个 TP Group 的两个 rank 分别能容纳 3,000 和 2,400 个逻辑 token 的 KV，为什么整组不能按 5,400 接纳？把这个判断与一条序列在各 rank 上的计算联系起来。

<!-- 作者的容量解释。 -->

## 从请求接纳到事件推进

正文见 [3.3：请求接纳与 Engine 事件循环](../chapters/requests/03-scheduler-and-worker-group.md#request-admission)。

沿正文中的 A、B 复述一次：A 正在生成，B 已经通过校验进入等待队列，当前资源不足。为什么 B 已被接纳却还不能派发？A 又返回一个普通 token 后，Engine 会做什么，B 是否一定能开始？

<!-- 作者对接纳、派发与事件触发的解释。 -->

再从并发角度解释：Engine 只有一个任务依次更新请求表，为什么系统仍然可以同时服务多个请求？如果这个任务等待发送队列腾出空间，通信线程、Worker 和 Engine 下一次事件处理分别处于什么状态？

<!-- 作者对状态所有权与任务协作的复述。 -->

## 后续的预算与选批推演

让 A 正在生成、B 在等待、C 的 Prefill 已派发但尚未确认。围绕同一组请求，逐步补充资源数字，解释下一次调度允许承诺多少工作、收到结果后哪些状态需要更新。

<!-- 作者的预算推导与选批过程。 -->

## 用自己的话串起协作过程

从请求进入 Scheduler 开始，串起建档、预算、选批、派发、Worker 本地推进、结果返回与下一轮调度。多机器扩展时，再加入 Group 选择与按组维护的状态。

<!-- 作者复述。 -->

返回[书籍入口](../00-README.md)。
