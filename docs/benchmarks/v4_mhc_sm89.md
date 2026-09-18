# mHC sm_89 本地观测

2026-09-18，RTX 4070 Ti SUPER 16GB，CUDA sm_89。
源程序：`crates/infer-backend-cuda/examples/v4_mhc_bench.rs`。

非独占桌面 GPU；运行前/后桌面显存约 2.8–3.0GB，功耗约 25–28W。
初次受高负载干扰的数据未收录。下面是负载下降后的两次完整运行，
不代表独占 GPU 的稳定吞吐或跨框架加速倍数。CUDA 编译始终为 `-O3`；
Rust 使用 dev profile，计时使用 CUDA events 包围已捕获的图重放。
方法、数值检查和接口边界见 [mHC 文档](../DEEPSEEK_V4_MHC.md)。

## 第 1 轮

```jsonl
{"tokens":1,"dim":128,"pre_us":6.970,"post_us":1.024,"head_us":3.249,"pipeline_us":11.168,"serial_pre_us":10.854,"scratch_bytes":200,"batch_serial_equal":true}
{"tokens":128,"dim":128,"pre_us":8.877,"post_us":1.354,"head_us":3.891,"pipeline_us":13.746,"serial_pre_us":886.093,"scratch_bytes":25600,"batch_serial_equal":true}
{"tokens":1,"dim":4096,"pre_us":12.063,"post_us":1.169,"head_us":8.056,"pipeline_us":19.738,"serial_pre_us":12.493,"scratch_bytes":6400,"batch_serial_equal":true}
{"tokens":4,"dim":4096,"pre_us":12.800,"post_us":1.163,"head_us":7.941,"pipeline_us":22.899,"serial_pre_us":46.490,"scratch_bytes":25600,"batch_serial_equal":true}
{"tokens":128,"dim":4096,"pre_us":56.115,"post_us":6.861,"head_us":20.986,"pipeline_us":84.889,"serial_pre_us":1447.059,"scratch_bytes":819200,"batch_serial_equal":true}
{"tokens":1024,"dim":4096,"pre_us":416.450,"post_us":130.918,"head_us":128.845,"pipeline_us":928.536,"serial_pre_us":13116.614,"scratch_bytes":6553600,"batch_serial_equal":true}
```

## 第 2 轮

```jsonl
{"tokens":1,"dim":128,"pre_us":6.989,"post_us":1.043,"head_us":3.275,"pipeline_us":11.315,"serial_pre_us":9.965,"scratch_bytes":200,"batch_serial_equal":true}
{"tokens":128,"dim":128,"pre_us":8.883,"post_us":1.869,"head_us":3.866,"pipeline_us":13.797,"serial_pre_us":890.061,"scratch_bytes":25600,"batch_serial_equal":true}
{"tokens":1,"dim":4096,"pre_us":12.114,"post_us":1.165,"head_us":8.070,"pipeline_us":21.329,"serial_pre_us":16.998,"scratch_bytes":6400,"batch_serial_equal":true}
{"tokens":4,"dim":4096,"pre_us":12.890,"post_us":1.163,"head_us":7.947,"pipeline_us":21.970,"serial_pre_us":47.104,"scratch_bytes":25600,"batch_serial_equal":true}
{"tokens":128,"dim":4096,"pre_us":56.237,"post_us":6.784,"head_us":21.018,"pipeline_us":83.711,"serial_pre_us":1451.213,"scratch_bytes":819200,"batch_serial_equal":true}
{"tokens":1024,"dim":4096,"pre_us":404.141,"post_us":130.048,"head_us":124.054,"pipeline_us":936.155,"serial_pre_us":13110.631,"scratch_bytes":6553600,"batch_serial_equal":true}
```
