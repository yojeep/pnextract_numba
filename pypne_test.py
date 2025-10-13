from encodings.punycode import T
import pypne
from pathlib import Path
import numpy as np


image = np.fromfile(
    r"D:\yjp\Workdir\Code\ZJU\Study\Python\multi-physic-network-model\Papers\0\Data\_N2.500_sample0\pne\image_125_125_125.raw",
    dtype=np.uint8,
).reshape((125, 125, 125))
image = image == 1

VElems_pypne, pn = pypne.pnextract(image,verbose=True)

VElems_pypne = VElems_pypne[1:126, 1:126, 1:126].astype(np.int32)
VElems_pypne.tofile("VElems_pypne.raw")
