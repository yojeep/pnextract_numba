from pnextract import extract
from pathlib import Path
import numpy as np
import time

Path_img = Path(r"./image_500_500_500.npz")
img = np.load(Path_img)["arr_0"]

# img = np.fromfile(
#     r"C:\Users\yjp\Desktop\fsdownload\0.raw",
#     dtype=np.uint8,
# ).reshape((6000, 960, 560))
img_bool = img == 0
del img

t0 = time.time()
VElems = extract(img_bool)
print(f"time_cost:{time.time() - t0}")
num_pore = np.unique(VElems).size - 1
print("*" * 20)
print(f"num_pores:{num_pore}")
print("*" * 20)

# plt.imshow(VElems[1])
# plt.show()
VElems = VElems[1:-1, 1:-1, 1:-1]
np.savez_compressed(
    Path_img.with_name(Path_img.stem + "_python_pne.npz"), VElems.astype(np.int32)
)
