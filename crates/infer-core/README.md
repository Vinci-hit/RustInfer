# 优化方向
1、加入更多的后端，比如tilelang和triton。
2、支持云原生，动态扩展后端，多用户。
3、实现PageAttenion。


# 注意事项
1、flashattention采用异步读取实现，使用了cp.async指令，要求Ampere以上的架构。
2、cublas采用了v2后缀，要求新版本cuda