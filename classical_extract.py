import pypne
from pathlib import Path
import numpy as np
import time

t0 = time.time()
Path_img = Path(r"./Data/image_500_500_500.npz")
img = np.load(Path_img)["arr_0"]
img = img == 1

VElems_pypne, pn = pypne.pnextract(img, verbose=True, n_workers=64)

print(f"Time cost: {time.time() - t0:.4f} s")
VElems_pypne = VElems_pypne[1:-1, 1:-1, 1:-1].astype(np.int32)
np.savez_compressed(
    Path_img.with_name(Path_img.stem + "_classical_pne.npz"), VElems_pypne
)
