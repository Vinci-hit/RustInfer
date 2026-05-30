"""
Element-wise diff Rust dump vs Python dump for `noise_refiner.0`.
Walks through each saved tensor in lockstep, prints first divergence.
"""
import sys
from pathlib import Path
import numpy as np

PY_DIR = Path("/tmp/zimage_dump_python")
RS_DIR = Path("/tmp/zimage_dump_rust")

# Order matters: walk from input → outward.
NAMES = [
    "step0_latent_5d_in",       # only Rust dumps this (Python uses x after embedder)
    "step0_patches",            # only Rust
    "step0_x_emb",              # only Rust
    "step0_x_padded_in",        # both
    "step0_adaln_input",
    "step0_freqs_cis",
    "step0_nr0_mod_out",
    "step0_nr0_scale_msa",
    "step0_nr0_gate_msa",
    "step0_nr0_scale_mlp",
    "step0_nr0_gate_mlp",
    "step0_nr0_norm1_x",
    "step0_nr0_qkv_q",
    "step0_nr0_qkv_k",
    "step0_nr0_qkv_v",
    "step0_nr0_q_normed",
    "step0_nr0_k_normed",
    "step0_nr0_q_roped",
    "step0_nr0_k_roped",
    "step0_nr0_attn_out_premerge",
    "step0_nr0_attn_out_post",
    "step0_nr0_norm2_attn",
    "step0_nr0_after_attn",
    "step0_nr0_norm1_ffn",
    "step0_nr0_w1_out",
    "step0_nr0_w3_out",
    "step0_nr0_silu_w1_x_w3",
    "step0_nr0_w2_out",
    "step0_nr0_norm2_ffn",
    "step0_nr0_block_out",
]


def stat(name: str, a: np.ndarray) -> str:
    flat = a.ravel()
    n = flat.size
    nan = int(np.isnan(flat).sum())
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return f"{name}: shape={a.shape} ALL non-finite (nan={nan})"
    return (f"{name}: shape={a.shape} nan={nan} "
            f"mean={finite.mean():.4f} std={finite.std():.4f} "
            f"min={finite.min():.4f} max={finite.max():.4f}")


def diff(py: np.ndarray, rs: np.ndarray, name: str, max_print: int = 5):
    if py.shape != rs.shape:
        # Try to squeeze leading batch dim 1 from Python.
        if py.ndim == rs.ndim + 1 and py.shape[0] == 1:
            py = py[0]
        elif rs.ndim == py.ndim + 1 and rs.shape[0] == 1:
            rs = rs[0]
        if py.shape != rs.shape:
            print(f"  SHAPE MISMATCH: py={py.shape} rs={rs.shape}")
            return False

    py_f = py.ravel().astype(np.float64)
    rs_f = rs.ravel().astype(np.float64)

    # Index of first divergence beyond bf16 noise (~1%).
    py_finite = np.isfinite(py_f)
    rs_finite = np.isfinite(rs_f)
    both_finite = py_finite & rs_finite

    # Where Rust is non-finite but Python is finite — that's a wreck.
    bad_idx = np.where(py_finite & ~rs_finite)[0]
    if bad_idx.size:
        i = int(bad_idx[0])
        print(f"  RUST NON-FINITE: first at flat idx {i}, py={py_f[i]} rs={rs_f[i]}")
        print(f"  rust non-finite count: {bad_idx.size} / {py_f.size}")
        # Where do non-finites start (first run)?
        # Print 4 elements around it.
        lo, hi = max(0, i - 2), min(py_f.size, i + 3)
        print(f"  py[{lo}..{hi}]: {py_f[lo:hi]}")
        print(f"  rs[{lo}..{hi}]: {rs_f[lo:hi]}")
        return False

    if not both_finite.all():
        print(f"  py non-finite count: {(~py_finite).sum()}")

    abs_err = np.abs(py_f - rs_f)
    rel_err = abs_err / np.maximum(np.abs(py_f), 1e-3)
    abs_err = np.where(both_finite, abs_err, 0)
    rel_err = np.where(both_finite, rel_err, 0)

    max_abs = float(abs_err.max())
    max_rel = float(rel_err.max())
    # Allow some bf16 noise: rel < 12% or abs < 0.1.
    threshold = (abs_err < 0.1) | (rel_err < 0.12)
    bad = ~threshold & both_finite
    bad_idx = np.where(bad)[0]
    bad_count = int(bad.sum())

    print(f"  max_abs={max_abs:.6f} max_rel={max_rel:.6f} "
          f"divergent_count={bad_count}/{py_f.size}")
    if bad_count and max_print:
        for i in bad_idx[:max_print]:
            print(f"    [{i}] py={py_f[i]:.6f} rs={rs_f[i]:.6f} "
                  f"abs={abs_err[i]:.6f} rel={rel_err[i]:.6f}")
    return bad_count == 0


def main():
    for name in NAMES:
        py_path = PY_DIR / f"{name}.npy"
        rs_path = RS_DIR / f"{name}.npy"
        py_exists = py_path.exists()
        rs_exists = rs_path.exists()
        if not py_exists and not rs_exists:
            continue
        print(f"\n=== {name} ===")
        if py_exists:
            py = np.load(py_path)
            print(f"  py: {stat('python', py)}")
        else:
            print("  py: <missing>")
            py = None
        if rs_exists:
            rs = np.load(rs_path)
            print(f"  rs: {stat('rust  ', rs)}")
        else:
            print("  rs: <missing>")
            rs = None
        if py is not None and rs is not None:
            ok = diff(py, rs, name)
            if not ok:
                print(f"  ⚠️  {name} diverged — continuing to find downstream issues.")
            else:
                print(f"  ✓ {name} matches within bf16 tolerance")


if __name__ == "__main__":
    main()
