# V4 Indexer capacity and Beam baseline

Date: 2026-09-17. GPU: RTX 4070 Ti SUPER, 16 GiB, sm_89. CUDA kernels: nvcc -O3.
Command: `cargo run -p infer-backend-cuda --example v4_indexer_topk_bench`.
Seven CUDA-event samples, each five graph replays; median per call. Decode graphs
contain 32 calls, prefill graphs 8. Q/K/weights are prepared before timing.
`legacy_beam_us` includes the unchanged Beam path's sort, exp/CDF and logprobs.
All CPU-sort, bucket/full-capacity and applicable Beam ID comparisons passed.

```jsonl
{"tokens":128,"start":0,"capacity":32,"score_capacity":32,"k":512,"score_us":6.067,"topk_us":7.194,"pipeline_us":13.082,"workspace_bytes":4,"score_bytes":16384,"legacy_beam_us":null,"legacy_beam_workspace_bytes":0,"exact_cpu_sort_match":true}
{"tokens":1024,"start":0,"capacity":256,"score_capacity":256,"k":512,"score_us":84.372,"topk_us":48.398,"pipeline_us":132.582,"workspace_bytes":4,"score_bytes":1048576,"legacy_beam_us":null,"legacy_beam_workspace_bytes":0,"exact_cpu_sort_match":true}
{"tokens":128,"start":32768,"capacity":8224,"score_capacity":8224,"k":512,"score_us":494.387,"topk_us":68.787,"pipeline_us":566.118,"workspace_bytes":5242880,"score_bytes":4210688,"legacy_beam_us":null,"legacy_beam_workspace_bytes":0,"exact_cpu_sort_match":true}
{"tokens":1,"start":127,"capacity":32,"score_capacity":32,"k":512,"score_us":4.299,"topk_us":4.582,"pipeline_us":8.851,"workspace_bytes":4,"score_bytes":128,"legacy_beam_us":null,"legacy_beam_workspace_bytes":0,"exact_cpu_sort_match":true}
{"tokens":1,"start":1023,"capacity":256,"score_capacity":256,"k":512,"score_us":4.794,"topk_us":4.602,"pipeline_us":9.355,"workspace_bytes":4,"score_bytes":1024,"legacy_beam_us":null,"legacy_beam_workspace_bytes":0,"exact_cpu_sort_match":true}
{"tokens":1,"start":32767,"capacity":8192,"score_capacity":8192,"k":512,"score_us":5.888,"topk_us":13.670,"pipeline_us":19.621,"workspace_bytes":32768,"score_bytes":32768,"legacy_beam_us":81.242,"legacy_beam_workspace_bytes":268800,"exact_cpu_sort_match":true}
{"tokens":1,"start":131071,"capacity":32768,"score_capacity":32768,"k":512,"score_us":18.214,"topk_us":20.179,"pipeline_us":38.502,"workspace_bytes":131072,"score_bytes":131072,"legacy_beam_us":47.782,"legacy_beam_workspace_bytes":1058304,"exact_cpu_sort_match":true}
{"tokens":1,"start":1048575,"capacity":262144,"score_capacity":262144,"k":512,"score_us":167.110,"topk_us":34.393,"pipeline_us":200.698,"workspace_bytes":1048576,"score_bytes":1048576,"legacy_beam_us":55.051,"legacy_beam_workspace_bytes":8427008,"exact_cpu_sort_match":true}
{"tokens":1,"start":127,"capacity":8192,"score_capacity":8192,"k":512,"score_us":4.343,"topk_us":13.178,"pipeline_us":17.451,"workspace_bytes":32768,"score_bytes":32768,"legacy_beam_us":null,"legacy_beam_workspace_bytes":0,"exact_cpu_sort_match":true}
{"tokens":1,"start":127,"capacity":8192,"score_capacity":32,"k":512,"score_us":4.326,"topk_us":4.608,"pipeline_us":8.876,"workspace_bytes":4,"score_bytes":128,"legacy_beam_us":null,"legacy_beam_workspace_bytes":0,"exact_cpu_sort_match":true}
{"tokens":1,"start":127,"capacity":262144,"score_capacity":262144,"k":512,"score_us":5.843,"topk_us":31.360,"pipeline_us":36.113,"workspace_bytes":1048576,"score_bytes":1048576,"legacy_beam_us":null,"legacy_beam_workspace_bytes":0,"exact_cpu_sort_match":true}
{"tokens":1,"start":127,"capacity":262144,"score_capacity":32,"k":512,"score_us":4.326,"topk_us":4.608,"pipeline_us":8.940,"workspace_bytes":4,"score_bytes":128,"legacy_beam_us":null,"legacy_beam_workspace_bytes":0,"exact_cpu_sort_match":true}
{"tokens":1,"start":32767,"capacity":262144,"score_capacity":262144,"k":512,"score_us":9.510,"topk_us":29.651,"pipeline_us":39.110,"workspace_bytes":1048576,"score_bytes":1048576,"legacy_beam_us":50.354,"legacy_beam_workspace_bytes":8427008,"exact_cpu_sort_match":true}
{"tokens":1,"start":32767,"capacity":262144,"score_capacity":8192,"k":512,"score_us":5.901,"topk_us":13.779,"pipeline_us":19.736,"workspace_bytes":32768,"score_bytes":32768,"legacy_beam_us":77.738,"legacy_beam_workspace_bytes":268800,"exact_cpu_sort_match":true}
{"tokens":128,"start":0,"capacity":262144,"score_capacity":262144,"k":512,"score_us":671.795,"topk_us":406.140,"pipeline_us":1102.105,"workspace_bytes":134217728,"score_bytes":134217728,"legacy_beam_us":null,"legacy_beam_workspace_bytes":0,"exact_cpu_sort_match":true}
{"tokens":128,"start":0,"capacity":262144,"score_capacity":32,"k":512,"score_us":5.683,"topk_us":6.758,"pipeline_us":12.288,"workspace_bytes":4,"score_bytes":16384,"legacy_beam_us":null,"legacy_beam_workspace_bytes":0,"exact_cpu_sort_match":true}
```
