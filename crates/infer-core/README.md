# 优化方向
1、加入更多的后端，比如tilelang和triton。
2、支持云原生，动态扩展后端，多用户。
3、实现PageAttenion。


# 注意事项
1、flashattention采用异步读取实现，使用了cp.async指令，要求Ampere以上的架构。
2、cublas采用了v2后缀，要求新版本cuda

# 优化记录
1、支持了bf16后，prefill阶段得到了极大地提升，大于200%，但是decoding阶段提升不大，还是访存密集型。
2、通过Nsight观察发现，decoding阶段中launch占用了2ms，memcpy（包括内核计算时间）占用了5ms，因此下一步将采用CudaGraph进行优化，预计能减少一半用时。
