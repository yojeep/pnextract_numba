# import numpy as np
# from numba import njit, uint64

# # ================= 1. 核心算法实现 (完全对应 C++ libmorton) =================

# # 64位 Magic Bits 掩码常量
# MASK_0 = uint64(0x1fffff)
# MASK_1 = uint64(0x1f00000000ffff)
# MASK_2 = uint64(0x1f0000ff0000ff)
# MASK_3 = uint64(0x100f00f00f00f00f)
# MASK_4 = uint64(0x10c30c30c30c30c3)
# MASK_5 = uint64(0x1249249249249249)

# # 64位 Magic Bits 解码掩码常量 (对应 libmorton 中的 magicbit3D_masks64_decode)
# DECODE_MASK_0 = uint64(0x9249249249249249)
# DECODE_MASK_1 = uint64(0x10c30c30c30c30c3)
# DECODE_MASK_2 = uint64(0x100f00f00f00f00f)
# DECODE_MASK_3 = uint64(0x1f0000ff0000ff)
# DECODE_MASK_4 = uint64(0x1f00000000ffff)
# DECODE_MASK_5 = uint64(0x1fffff)

# @njit(cache=True)
# def expand_bits_3d(x):
#     """将 21 位坐标扩展为 63 位 (每两位之间插入两个 0)"""
#     x = x & MASK_0
#     x = (x | (x << 32)) & MASK_1
#     x = (x | (x << 16)) & MASK_2
#     x = (x | (x << 8))  & MASK_3
#     x = (x | (x << 4))  & MASK_4
#     x = (x | (x << 2))  & MASK_5
#     return x

# @njit(cache=True)
# def compact_bits_3d(x):
#     """将 63 位 Morton 码压缩回 21 位坐标 (提取 x/y/z 对应的位)"""
#     x = x & DECODE_MASK_0
#     x = (x ^ (x >> 2))  & DECODE_MASK_1
#     x = (x ^ (x >> 4))  & DECODE_MASK_2
#     x = (x ^ (x >> 8))  & DECODE_MASK_3
#     x = (x ^ (x >> 16)) & DECODE_MASK_4
#     x = (x ^ (x >> 32)) & DECODE_MASK_5
#     return x

# @njit(cache=True)
# def encode_morton_3d(x_arr, y_arr, z_arr):
#     """批量编码 3D Morton 码"""
#     n = len(x_arr)
#     result = np.empty(n, dtype=np.uint64)
#     for i in range(n):
#         result[i] = expand_bits_3d(x_arr[i]) | \
#                     (expand_bits_3d(y_arr[i]) << 1) | \
#                     (expand_bits_3d(z_arr[i]) << 2)
#     return result

# @njit(cache=True)
# def decode_morton_3d(morton_arr):
#     """批量解码 3D Morton 码"""
#     n = len(morton_arr)
#     x_out = np.empty(n, dtype=np.uint64)
#     y_out = np.empty(n, dtype=np.uint64)
#     z_out = np.empty(n, dtype=np.uint64)
#     for i in range(n):
#         m = morton_arr[i]
#         x_out[i] = compact_bits_3d(m)
#         y_out[i] = compact_bits_3d(m >> 1)
#         z_out[i] = compact_bits_3d(m >> 2)
#     return x_out, y_out, z_out


# # ================= 2. 算例与验证 =================

# if __name__ == "__main__":
#     # 构造算例：中心点 (10, 10, 10) 在 X, Y, Z 方向正负 1 的相邻点
#     # 总共 6 个点
#     base = 10
#     coords = np.array([
#         [base - 1, base, base],  # X - 1
#         [base + 1, base, base],  # X + 1
#         [base, base - 1, base],  # Y - 1
#         [base, base + 1, base],  # Y + 1
#         [base, base, base - 1],  # Z - 1
#         [base, base, base + 1],  # Z + 1
#     ], dtype=np.uint64)

#     x = coords[:, 0]
#     y = coords[:, 1]
#     z = coords[:, 2]

#     print("原始坐标 (X, Y, Z):")
#     for i in range(len(x)):
#         print(f"  {x[i], y[i], z[i]}")

#     # 编码
#     morton_codes = encode_morton_3d(x, y, z)
#     print("\n计算出的 Morton 码 (uint64):")
#     for code in morton_codes:
#         print(f"  {code} (Hex: {hex(code)})")

#     # 解码
#     decoded_x, decoded_y, decoded_z = decode_morton_3d(morton_codes)
#     print("\n解码后的坐标 (X, Y, Z):")
#     for i in range(len(decoded_x)):
#         print(f"  {decoded_x[i], decoded_y[i], decoded_z[i]}")

#     # 验证是否完全一致
#     is_match = np.all(x == decoded_x) and np.all(y == decoded_y) and np.all(z == decoded_z)
#     print(f"\n✅ 编解码结果是否完全一致: {is_match}")


import numpy as np
import numba as nb


@nb.njit(cache=True)
def test(x):
    p = np.float32(20.0)  # ← 关键：显式 float32 常量

    def add(a, b):
        return a + b + p

    return add(x, np.float32(1.0))


print(test(np.float32(1.0)))


# 1) 类型推导 + Numba IR（最常用，看哪里退化为 pyobject）
test.inspect_types()

# 2) LLVM IR（未优化版，很冗长）
print(test.inspect_llvm())

# 3) 优化后的原生汇编（x86_64 / AVX 等，性能调优看这个）
print(test.inspect_asm())

# 控制流图（需要 llvmlite）
# cfg = test.inspect_cfg()
# cfg.display()  # Jupyter 里直接渲染
