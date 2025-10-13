import numpy as np

bool_arr = np.array([True, False], dtype=bool)
int32_arr = np.array([1, 0], dtype=np.int32)

print(bool_arr.itemsize)  # 输出：1（字节）
print(int32_arr.itemsize) # 输出：4（字节）