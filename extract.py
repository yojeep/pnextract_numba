from jaraco.classes.ancestry import all_bases
from dataclasses import dataclass
import os

os.environ["NUMBA_OPT"] = "max"
os.environ["NUMBA_SLP_VECTORIZE"] = "1"
os.environ["NUMBA_ENABLE_AVX"] = "1"
import numpy as np
from _edt_numba import nb_classic_edt, nb_edt as edt
import numba as nb
import numpy.linalg as LA
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from _extraction_functions_numba import (
    _mp5,
    nb_parallel_sum,
    nb_where,
    smooth_radius,
    paradox_pre_removeincludedballI,
    paradox_removeincludedballI,
    moveUphill,
    moveUphillp1,
    findBoss,
)

from _blocknet_numba import (
    grow_pores,
    grow_pores_med_eqs,
    grow_pores_med_eqs_loose,
    grow_pores_med_strict,
    grow_pores_median,
    grow_pores_X2,
    CreateVElem,
    median_elem,
    retreat_pores_median,
    refine_with_master_ball,
)


class Defaults:
    def __init__(self, avgR):
        self.avgR = avgR  # 存储 avgR 以便后续计算

    @property
    def minRp(self):
        return min(1.25, self.avgR * 0.25) + 0.5

    @property
    def clipROutx(self):
        return 0.05

    @property
    def clipROutyz(self):
        return 0.98

    @property
    def midRf(self):
        return 0.7

    @property
    def MSNoise(self):
        return 1.0 * abs(self.minRp) + 1.0

    @property
    def lenNf(self):
        return 0.6

    @property
    def vmvRadRelNf(self):
        return 1.1

    @property
    def nRSmoothing(self):
        return 3

    @property
    def RCorsnf(self):
        return 0.15

    @property
    def RCorsn(self):
        return abs(self.minRp)


class Balls_cls:
    def __init__(self, isball, dt):
        self.update(isball, dt)

    def update(self, isball=None, dt=None):
        if isball is not None:
            self.indices = nb_where(isball, isball.sum())
            self.findices = self.indices - _mp5
        if dt is not None:
            self.R = dt[self.indices[:, 0], self.indices[:, 1], self.indices[:, 2]]
        self.sort()

    def sort(self):
        sorted_indices = np.argsort(-self.R, kind="stable")
        self.indices = self.indices[sorted_indices]
        self.findices = self.findices[sorted_indices]
        self.R = self.R[sorted_indices]


_mp5 = np.float32(-0.5)


img = np.fromfile(
    r"D:\yjp\Workdir\Code\ZJU\Study\Python\multi-physic-network-model\Papers\0\Data\_N2.500_sample0\pne\image_125_125_125.raw",
    dtype=np.uint8,
).reshape((125, 125, 125))


# img = np.fromfile(
#     r"C:\Users\yjp\Desktop\fsdownload\0.raw",
#     dtype=np.uint8,
# ).reshape((6000, 960, 560))
img_bool = img == 0


nVxls = nb_parallel_sum(img_bool)
print("num_voxels:", nVxls)

zsysxs_v = nb_where(img_bool, nVxls)


# print(nb_where.inspect_types())
dt = nb_classic_edt(img_bool, _clipROutyz=0.98, _clipROutx=0.05)
# dt = edt(img_bool, black_border=True)
# plt.imshow(dt[3])
# plt.show()
avgR = np.mean(dt[img_bool])
print("avgR:", avgR)
defaults = Defaults(avgR)

# dt = gaussian_filter(dt, 0.3)
for _ in range(defaults.nRSmoothing):
    smooth_radius(img_bool, dt, zsysxs_v)

isball = np.zeros_like(img_bool, dtype=bool)
paradox_pre_removeincludedballI(img_bool, dt, isball, defaults.minRp)
Balls = Balls_cls(isball, dt)
print("num_balls_init:", Balls.indices.shape[0])

# Balls = Balls_cls(ball_indices, ball_R)
paradox_removeincludedballI(
    Balls.indices,
    Balls.R,
    img_bool,
    dt,
    isball,
    defaults.RCorsnf,
    defaults.RCorsn,
    defaults.MSNoise,
)

Balls.update(isball, dt)
print("num_balls:", Balls.indices.shape[0])

moveUphill(Balls.indices, Balls.findices, Balls.R, img_bool, dt)
moveUphillp1(Balls.indices, Balls.findices, Balls.R, img_bool, dt, isball)
Balls.update(isball, dt)
moveUphill(Balls.indices, Balls.findices, Balls.R, img_bool, dt)
Balls.sort()
Balls.boss = np.arange(Balls.indices.shape[0], dtype=np.int32)
findBoss(
    Balls.indices,
    Balls.findices,
    Balls.R,
    Balls.boss,
    img_bool,
    dt,
    isball,
    defaults.MSNoise,
    defaults.midRf,
    defaults.vmvRadRelNf,
    defaults.lenNf,
)
image_VElems = np.pad(img_bool, 1, mode="constant", constant_values=False)
dt_VElems = np.pad(dt, 1, mode="constant", constant_values=0)

isball_VElems = np.pad(isball, 1, mode="constant", constant_values=False)
del img_bool, dt, img
VElems, poreIs = CreateVElem(
    image_VElems, dt_VElems, isball_VElems, Balls.findices, Balls.R, Balls.boss
)


last_pore = len(poreIs) - 1

VElems = grow_pores_med_strict(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_strict(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_strict(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_median(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_strict(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_median(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_strict(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_median(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_strict(image_VElems, dt_VElems, VElems, 0, last_pore, -1)

VElems = grow_pores_median(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_strict(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_median(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_strict(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_median(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_strict(image_VElems, dt_VElems, VElems, 0, last_pore, -1)

VElems = grow_pores_med_eqs(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_median(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_strict(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_median(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_median(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_strict(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_eqs(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_median(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_median(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_strict(image_VElems, dt_VElems, VElems, 0, last_pore, -1)

VElems = grow_pores_med_eqs(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_eqs(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_eqs(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_eqs(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_eqs(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_eqs_loose(image_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_eqs(image_VElems, dt_VElems, VElems, 0, last_pore, -1)

VElems = grow_pores_med_eqs(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_eqs_loose(image_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_eqs(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_eqs_loose(image_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_eqs(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_eqs_loose(image_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_eqs(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_eqs_loose(image_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_eqs(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_eqs_loose(image_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_eqs(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_eqs_loose(image_VElems, VElems, 0, last_pore, -1)

VElems = grow_pores(image_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_median(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_eqs(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores(image_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_median(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores(image_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores(image_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores(image_VElems, VElems, 0, last_pore, -1)

VElems = grow_pores(image_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores(image_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores(image_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_median(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores(image_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores(image_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores(image_VElems, VElems, 0, last_pore, -1)

VElems = grow_pores(image_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores(image_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores(image_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores(image_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores(image_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores(image_VElems, VElems, 0, last_pore, -1)

VElems = grow_pores(image_VElems, VElems, 0, last_pore, -1)
VElems, _ = grow_pores_X2(image_VElems, VElems, 0, last_pore, -1)
VElems, _ = grow_pores_X2(image_VElems, VElems, 0, last_pore, -1)
VElems, _ = grow_pores_X2(image_VElems, VElems, 0, last_pore, -1)
VElems, _ = grow_pores_X2(image_VElems, VElems, 0, last_pore, -1)

VElems = median_elem(VElems, 0, last_pore)
VElems = median_elem(VElems, 0, last_pore)
VElems = median_elem(VElems, 0, last_pore)
VElems = median_elem(VElems, 0, last_pore)
VElems = median_elem(VElems, 0, last_pore)

VElems = grow_pores(image_VElems, VElems, 0, last_pore, -1)
while grow_pores_X2(image_VElems, VElems, 0, last_pore, -1)[1] != 0:
    VElems = grow_pores_X2(image_VElems, VElems, 0, last_pore, -1)[0]
VElems = grow_pores(image_VElems, VElems, 0, last_pore, -1)
VElems = retreat_pores_median(image_VElems, VElems, 0, last_pore, -1)

VElems = refine_with_master_ball(VElems, Balls.boss, Balls.findices)

VElems = grow_pores_median(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_eqs(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_eqs(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_eqs(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_median(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_eqs(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_eqs(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_eqs_loose(image_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_eqs(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_eqs_loose(image_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_eqs(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_eqs_loose(image_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_eqs(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores(image_VElems, VElems, 0, last_pore, -1)
VElems, _ = grow_pores_X2(image_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_eqs(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores(image_VElems, VElems, 0, last_pore, -1)
VElems, _ = grow_pores_X2(image_VElems, VElems, 0, last_pore, -1)

VElems = median_elem(VElems, 0, last_pore)
VElems = grow_pores_med_eqs_loose(image_VElems, VElems, 0, last_pore, -1)
VElems = refine_with_master_ball(VElems, Balls.boss, Balls.findices)


# res = np.argwhere(VElems != -1)
# plt.imshow(dt_VElems[1])
# plt.show()
print("*" * 20)
print(f"num_pores:{len(poreIs)}")
print("*" * 20)
plt.imshow(VElems[1])
plt.show()

# VElems.astype(np.int32).tofile("VElems.raw")
