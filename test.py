import numpy as np
from pathlib import Path

Path_img = Path(r"./image_500_500_500.raw")
img = np.fromfile(
    Path_img,
    dtype=np.uint8,
).reshape((500, 500, 500))

np.savez_compressed(Path_img.with_suffix(".npz"), img)
