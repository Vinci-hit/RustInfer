"""GPU correctness check for the standalone CuTe DSL attention kernel.

Run with RustInfer's .venv Python (CuTe DSL 4.7.1); requires an SM89 GPU.
Checks full FP32 reference attention, ragged/page/tail boundaries, empty rows,
strided output canaries, and the baseline input lengths. Does not time kernels.
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch,cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
from flash_attention2 import flash_attention2_bf16_b1_hq32_hkv8_d128_q64_kv64_bs1 as kernel
SCHEDULER_Q_TILE = int(os.environ.get("ATTENTION_SCHEDULER_Q_TILE", "64"))
@cute.jit
def launch(q:cute.Tensor,k:cute.Tensor,v:cute.Tensor,out:cute.Tensor,pages:cute.Tensor,lens:cute.Tensor,cu:cute.Tensor,req:cute.Tensor,idx:cute.Tensor,active:cute.Tensor,bs:cutlass.Int32,causal:cutlass.Boolean,stream:cuda.CUstream):
    kernel(q.iterator,k.iterator,v.iterator,out.iterator,pages.iterator,lens.iterator,cu.iterator,req.iterator,idx.iterator,active.iterator,
        cutlass.Int64(q.stride[0]),cutlass.Int64(q.stride[1]),cutlass.Int64(out.stride[0]),cutlass.Int64(out.stride[1]),
        cutlass.Int32(pages.shape[1]),bs,cutlass.Int32(req.shape[0]),cutlass.Int32(lens.shape[0]),cutlass.Int32(q.shape[0]),
        cutlass.Int32(32),cutlass.Int32(8),cutlass.Int32(128),cutlass.Float32(128**-0.5),causal,stream,scheduler_q_tile=SCHEDULER_Q_TILE
    ).launch(grid=(req.shape[0]*(SCHEDULER_Q_TILE//64),32,1),block=(128,1,1),stream=stream)
torch.manual_seed(35)
torch.backends.cuda.matmul.allow_tf32=False
cases=[([63],[63],1,True),([64],[64],1,True),([129],[17],1,True),([65],[49],1,True),([129],[17],1,False),([47],[47],1,True),([48],[48],1,True),([49],[49],1,True),([97],[97],1,True),([1],[1],1,True),([65],[65],1,True),([129],[129],1,True),([128],[128],1,True),([512],[512],1,True),([896],[896],1,True),([2048],[2048],1,True),([1,17,65],[0,63,257],1,True),([65,3],[129,513],1,True),([65],[17],1,True),([17,65],[129,257],1,False),([0,0],[0,0],1,False)]
# Stress the approximate exponential with uniform, tiny and sharp logits.
cases += [([65],[129],1,True,0.0),([65],[129],1,True,0.001),([65],[129],1,True,8.0)]
cases += [([n],[n],1,True) for n in (15,16,17,31,32,33)]
for case in cases:
    qlens,kvlens,bs,causal = case[:4]
    qk_gain = case[4] if len(case) > 4 else 1.0
    prefix=[0]; req=[]; idx=[]
    for r,n in enumerate(qlens):
        prefix.append(prefix[-1]+n)
        for j in range((n+SCHEDULER_Q_TILE-1)//SCHEDULER_Q_TILE): req.append(r);idx.append(j)
    count=len(req);capacity=count+2
    npages=sum((n+bs-1)//bs for n in kvlens);width=max(1,max((n+bs-1)//bs for n in kvlens))
    pages=torch.full((len(qlens),width),npages,device='cuda',dtype=torch.int32)
    perm=torch.randperm(npages,device='cuda');pos=0
    for r,n in enumerate(kvlens):
        nn=(n+bs-1)//bs;pages[r,:nn]=perm[pos:pos+nn].int();pos+=nn
    qb=torch.randn((max(1,prefix[-1]),6144),device='cuda',dtype=torch.bfloat16)
    q=qb[:,:4096].view(-1,32,128)
    k=torch.randn((npages+1,bs,8,128),device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
    q.mul_(qk_gain);k.mul_(qk_gain)
    k[-1]=float('nan');v[-1]=float('nan')
    storage=torch.full((max(1,prefix[-1])+2,32,136),123,device='cuda',dtype=torch.bfloat16)
    out=storage[1:-1,:,:128]
    ts=[torch.tensor(x,device='cuda',dtype=torch.int32) for x in (kvlens,prefix,req+[-1]*2,idx+[-1]*2,[count])]
    args=tuple(from_dlpack(t,assumed_align=16) for t in (q,k,v,out,pages.to(torch.uint32),*ts))+(cutlass.Int32(bs),cutlass.Boolean(causal),cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    fn=cute.compile(launch,*args,options='--gpu-arch=sm_89 --keep-ptx')
    fn(*args);torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    capture_args = args[:-1] + (cuda.CUstream(capture_stream.cuda_stream),)
    with torch.cuda.graph(graph, stream=capture_stream):
        fn(*capture_args)
    out.fill_(123)
    graph.replay()
    torch.cuda.synchronize()
    ref=torch.zeros((prefix[-1],32,128),device='cuda',dtype=torch.float32)
    for r,(n,nkv) in enumerate(zip(qlens,kvlens)):
        if n==0 or nkv==0:continue
        tokens=torch.arange(nkv,device='cuda');pg=pages[r,tokens//bs].long()
        kk=k[pg,tokens%bs].float().repeat_interleave(4,dim=1).permute(1,2,0)
        vv=v[pg,tokens%bs].float().repeat_interleave(4,dim=1).permute(1,0,2)
        qq=q[prefix[r]:prefix[r+1]].float().permute(1,0,2)
        scores=qq@kk*(128**-0.5)
        if causal:
            mask=tokens[None,:] <= nkv-n+torch.arange(n,device='cuda')[:,None]
            scores=scores.masked_fill(~mask[None],float('-inf'))
        pp=torch.nan_to_num(scores.softmax(-1),nan=0.)
        ref[prefix[r]:prefix[r+1]]=(pp@vv).permute(1,0,2)
    actual=out[:prefix[-1]].float()
    assert torch.isfinite(actual).all()
    if actual.numel():
        err=(actual-ref).abs();nrmse=((actual-ref).square().mean().sqrt()/ref.square().mean().sqrt().clamp_min(1e-12)).item()
        assert (err <= .02+.02*ref.abs()).all(),err.max().item()
        assert nrmse<=.01,nrmse
        print('PASS',qlens,kvlens,bs,causal,'qk_gain',qk_gain,'maxabs',err.max().item(),'NRMSE',nrmse,flush=True)
    else:print('PASS empty batch',flush=True)
    assert (storage[0]==123).all() and (storage[-1]==123).all() and (storage[:,:,128:]==123).all()
    if count==0:assert (out==123).all()
