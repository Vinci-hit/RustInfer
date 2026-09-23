#!/usr/bin/env python3
"""Deterministic encode-side vectors. Python stdlib only; no inference engine.

Choose logical integers/scales/signs, pack them, and compute expected values
from those logical values BEFORE packing. This does not invoke any decoder.
"""
import hashlib
import pathlib
import re
import struct

ROOT = pathlib.Path(__file__).resolve().parent
TABLES = ROOT.parents[3] / "infer-core/src/dtype/quant/codebooks.rs"
FORMATS = [
    ("Q2_K", 256, 84), ("Q3_K", 256, 110), ("Q4_K", 256, 144),
    ("Q5_K", 256, 176), ("Q6_K", 256, 210), ("Q8_0", 32, 34),
    ("IQ2_XXS", 256, 66), ("IQ2_XS", 256, 74), ("IQ2_S", 256, 82),
    ("IQ3_XXS", 256, 98), ("IQ3_S", 256, 110),
    ("IQ4_NL", 32, 18), ("IQ4_XS", 256, 136),
]
GRIDS = {
    name: [int(x) for x in re.findall(r"\d+", body)]
    for name, body in re.findall(r"const (IQ\w+): \[u8; \d+\] = \[(.*?)\];", TABLES.read_text(), re.S)
}
NL = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def mul(a, b):
    return f32(a * b)


def make(name, size, case):
    data = bytearray(size)
    bits = [0, 0x8000, 1, 0x3FF, 0x400, 0x2400, 0x3800, 0xBC00, 0x3555, 0x7BFF][case % 10]
    d = struct.unpack("<e", struct.pack("<H", bits))[0]
    dm = 0.125
    offset = {"Q2_K": 80, "Q3_K": 108, "Q6_K": 208}.get(name, 0)
    struct.pack_into("<H", data, offset, bits)
    out = []
    if name == "Q8_0":
        qs = [((case * 32 + i) % 256) - 128 for i in range(32)]
        data[2:] = bytes(q & 255 for q in qs)
        out = [mul(d, q) for q in qs]
    elif name == "Q2_K":
        struct.pack_into("<e", data, 82, dm)
        for group in range(16):
            s, m = (case + group) % 16, (case * 3 + group * 5) % 16
            data[group] = s | (m << 4)
            for j in range(16):
                idx = group * 16 + j
                q = (case + group + j) % 4
                # Two 128-element slabs; each slab has four 32-element planes.
                slab, plane = divmod(idx, 128)
                plane, lane = divmod(plane, 32)
                data[16 + slab * 32 + lane] |= q << (plane * 2)
                out.append(f32(mul(mul(d, s), q) - mul(dm, m)))
    elif name == "Q3_K":
        for group in range(16):
            s = (case * 7 + group * 11) % 64 - 32
            encoded = s + 32
            data[96 + group % 8] |= (encoded & 15) << (4 * (group // 8))
            data[104 + group % 4] |= (encoded >> 4) << (2 * (group // 4))
            for j in range(16):
                idx = group * 16 + j
                q = (case * 3 + group + j) % 8 - 4
                slab, rem = divmod(idx, 128)
                plane, lane = divmod(rem, 32)
                data[32 + slab * 32 + lane] |= (q & 3) << (2 * plane)
                if q >= 0:
                    data[idx % 32] |= 1 << (idx // 32)
                out.append(mul(mul(d, s), q))
    elif name in ("Q4_K", "Q5_K"):
        struct.pack_into("<e", data, 2, dm)
        scales = [(case * 7 + g * 13) % 64 for g in range(8)]
        mins = [(case * 11 + g * 3) % 64 for g in range(8)]
        for g in range(4):
            data[4 + g] = scales[g] | ((scales[g + 4] >> 4) << 6)
            data[8 + g] = mins[g] | ((mins[g + 4] >> 4) << 6)
            data[12 + g] = (scales[g + 4] & 15) | ((mins[g + 4] & 15) << 4)
        qbits = 5 if name == "Q5_K" else 4
        start = 48 if qbits == 5 else 16
        for g in range(8):
            for lane in range(32):
                q = (case * 7 + g * 3 + lane) % (1 << qbits)
                data[start + (g // 2) * 32 + lane] |= (q & 15) << (4 * (g % 2))
                if qbits == 5:
                    data[16 + lane] |= (q >> 4) << g
                out.append(f32(mul(mul(d, scales[g]), q) - mul(dm, mins[g])))
    elif name == "Q6_K":
        for g in range(16):
            s = (case * 17 + g * 7) % 256 - 128
            data[192 + g] = s & 255
            for j in range(16):
                idx = g * 16 + j
                q = (case * 3 + g * 5 + j) % 64 - 32
                packed = q + 32
                slab, rem = divmod(idx, 128)
                data[slab * 64 + rem % 64] |= (packed & 15) << (4 * (rem // 64))
                data[128 + slab * 32 + rem % 32] |= (packed >> 4) << (2 * (rem // 32))
                out.append(mul(mul(d, s), q))
    elif name in ("IQ4_NL", "IQ4_XS"):
        groups = 1 if name == "IQ4_NL" else 8
        for g in range(groups):
            s = (case * 7 + g * 13) % 64 - 32
            if groups == 8:
                data[4 + g // 2] |= ((s + 32) & 15) << (4 * (g % 2))
                high = ((s + 32) >> 4) << (2 * g)
                data[2] |= high & 255
                data[3] |= high >> 8
            scale = d if groups == 1 else mul(d, s)
            start = 2 if groups == 1 else 8 + g * 16
            for j in range(32):
                q = (case + g * 7 + j) % 16
                data[start + j % 16] |= q << (4 * (j // 16))
                out.append(mul(scale, NL[q]))
    else:
        is_two = name.startswith("IQ2")
        step = 8 if is_two else 4
        grid = GRIDS[name]
        # Covers every codebook entry (including 9/10-bit high indices).
        indices = [(case * (256 // step) + j) % (len(grid) // step) for j in range(256 // step)]
        masks = []
        for j in range(32):
            mask = (case * 17 + j * 11) % (256 if name.endswith("_S") else 128)
            if not name.endswith("_S"):
                mask |= (mask.bit_count() % 2) << 7
            masks.append(mask)
        subsize = 16 if name in ("IQ2_XS", "IQ2_S") else 32
        scales = [(case + j * 3) % 16 for j in range(256 // subsize)]
        for idx in range(256):
            s = scales[idx // subsize]
            if name == "IQ3_S":
                dl = mul(d, 1 + 2 * s)
            else:
                dl = mul(mul(d, 0.5 + s), 0.25 if is_two else 0.5)
            val = mul(dl, grid[indices[idx // step] * step + idx % step])
            out.append(mul(val, -1 if masks[idx // 8] & (1 << (idx % 8)) else 1))
        if name == "IQ2_XXS":
            for g in range(8):
                data[2 + g * 8:6 + g * 8] = bytes(indices[g * 4:g * 4 + 4])
                aux = scales[g] << 28
                for j in range(4):
                    aux |= (masks[g * 4 + j] & 127) << (7 * j)
                struct.pack_into("<I", data, 6 + g * 8, aux)
        elif name == "IQ2_XS":
            for j, index in enumerate(indices):
                struct.pack_into("<H", data, 2 + j * 2, index | ((masks[j] & 127) << 9))
            for j, s in enumerate(scales):
                data[66 + j // 2] |= s << (4 * (j % 2))
        elif name == "IQ2_S":
            for j, index in enumerate(indices):
                data[2 + j] = index & 255
                data[66 + j // 4] |= (index >> 8) << (2 * (j % 4))
            data[34:66] = bytes(masks)
            for j, s in enumerate(scales):
                data[74 + j // 2] |= s << (4 * (j % 2))
        elif name == "IQ3_XXS":
            data[2:66] = bytes(indices)
            for g in range(8):
                aux = scales[g] << 28
                for j in range(4):
                    aux |= (masks[g * 4 + j] & 127) << (7 * j)
                struct.pack_into("<I", data, 66 + g * 4, aux)
        elif name == "IQ3_S":
            for j, index in enumerate(indices):
                data[2 + j] = index & 255
                data[66 + j // 8] |= (index >> 8) << (j % 8)
            data[74:106] = bytes(masks)
            for j, s in enumerate(scales):
                data[106 + j // 2] |= s << (4 * (j % 2))
    return data, out


def main():
    hashes = []
    for name, count, size in FORMATS:
        result = bytearray(struct.pack("<I", 64))
        for case in range(64):
            encoded, expected = make(name, size, case)
            assert len(encoded) == size and len(expected) == count
            result += encoded + struct.pack("<" + "f" * count, *expected)
        (ROOT / (name + ".bin")).write_bytes(result)
        hashes.append(hashlib.sha256(result).hexdigest() + "  " + name + ".bin")
    (ROOT / "SHA256SUMS").write_text("\n".join(hashes) + "\n")


if __name__ == "__main__":
    main()
