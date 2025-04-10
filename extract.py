import numpy as np
from edt import edt
import numba as nb
import numpy.linalg as LA
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

image = np.fromfile(
    r"./image_100_300_300.raw", dtype=np.uint8
).reshape(300, 300, 100)
image = image == 0


# cube1 = footprint_rectangle(shape=(6, 6, 6))
# image = np.zeros((20, 20, 20))
# image[7:13, 7:13, 7:13] = cube1
# image = ~image.astype(bool)


dt = edt(image)

avgR = np.mean(dt[image])
print("avgR:", avgR)

from extraction_functions import *

_minRp = min(1.25, avgR * 0.25) + 0.5
print("_minRp:", _minRp)
_clipROutx = 0.05
_clipROutyz = 0.98
_midRf = 0.7
_MSNoise = 1.0 * abs(_minRp) + 1.0
_lenNf = 0.6
_vmvRadRelNf = 1.1
_nRSmoothing = 3
_RCorsnf = 0.15
_RCorsn = abs(_minRp)
_mp5 = -0.5

isball = np.zeros_like(image, dtype=bool)

# @nb.njit(cache=True)
# def get_masterball(ball_master, ball_index):
#     if ball_master[ball_index] == ball_index:
#         return ball_index
#     else:
#         return get_masterball(ball_master, ball_master[ball_index])


dt = gaussian_filter(dt, 0.3)
# dt = smooth_radius(dt, *image.shape)
dt[~image] = 0

paradox_pre_removeincludedballI(image, dt, isball,_minRp)
print(np.count_nonzero(isball))
ball_indices, ball_R = get_sorted_ball_indices_ball_R(dt, isball)
paradox_removeincludedballI(ball_indices, ball_R, image, dt, isball, _RCorsnf, _RCorsn, _MSNoise)
dt[ball_indices[:, 0], ball_indices[:, 1], ball_indices[:, 2]] = ball_R
ball_indices, ball_R = get_sorted_ball_indices_ball_R(dt, isball)
print(len(ball_indices))
ball_findices = ball_indices - _mp5
moveUphill(ball_indices, ball_findices, ball_R, image, dt)
moveUphillp1(ball_indices, ball_findices, ball_R, image, dt, isball)
moveUphill(ball_indices, ball_findices, ball_R, image, dt)
sorted_indices = np.argsort(ball_R)[::-1]
ball_indices = ball_indices[sorted_indices]
ball_findices = ball_findices[sorted_indices]
ball_R = ball_R[sorted_indices]
ball_boss = np.arange(ball_indices.shape[0],dtype=np.int32)
whichball = np.full_like(isball, -1, dtype=int)
whichball[ball_indices[:, 0], ball_indices[:, 1], ball_indices[:, 2]] = ball_boss
findBoss(ball_indices, ball_findices, ball_R, ball_boss, image, dt, isball, whichball, _MSNoise, _midRf, _vmvRadRelNf, _lenNf)
image_VElems = np.pad(image, 1, mode="constant", constant_values=False)
dt_VElems = np.pad(dt, 1, mode="constant", constant_values=0)
isball_VElems = np.pad(isball, 1, mode="constant", constant_values=False)
del image,dt
VElems, poreIs = CreateVElem(image_VElems, dt_VElems, isball_VElems, ball_findices, ball_R, ball_boss)

print(len(poreIs))
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
VElems,_ = grow_pores_X2(image_VElems, VElems, 0, last_pore, -1)
VElems,_ = grow_pores_X2(image_VElems, VElems, 0, last_pore, -1)
VElems,_ = grow_pores_X2(image_VElems, VElems, 0, last_pore, -1)
VElems,_ = grow_pores_X2(image_VElems, VElems, 0, last_pore, -1)

VElems = median_elem(VElems, 0, last_pore)
VElems = median_elem(VElems, 0, last_pore)
VElems = median_elem(VElems, 0, last_pore)
VElems = median_elem(VElems, 0, last_pore)
VElems = median_elem(VElems, 0, last_pore)

VElems = grow_pores(image_VElems, VElems, 0, last_pore, -1)
while grow_pores_X2(image_VElems, VElems, 0, last_pore, -1)[1]!= 0:
    VElems = grow_pores_X2(image_VElems, VElems, 0, last_pore, -1)[0]
VElems = grow_pores(image_VElems, VElems, 0, last_pore, -1)
VElems = retreat_pores_median(image_VElems, VElems, 0, last_pore,-1)

VElems = refine_with_master_ball(VElems,ball_boss,ball_findices)

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
VElems,_ = grow_pores_X2(image_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores_med_eqs(image_VElems, dt_VElems, VElems, 0, last_pore, -1)
VElems = grow_pores(image_VElems, VElems, 0, last_pore, -1)
VElems,_ = grow_pores_X2(image_VElems, VElems, 0, last_pore, -1)

VElems = median_elem(VElems, 0, last_pore)

VElems = refine_with_master_ball(VElems,ball_boss,ball_findices)

VElems = grow_pores_med_eqs_loose(image_VElems, VElems, 0, last_pore, -1)
# res = np.argwhere(VElems != -1)
plt.imshow(dt_VElems[3])
plt.show()

plt.imshow(VElems[3])
plt.show()

VElems.astype(np.int32).tofile("VElems.raw")