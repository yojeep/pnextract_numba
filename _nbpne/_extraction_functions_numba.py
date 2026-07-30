import numpy as np
import numba as nb

# from numpy import linalg as LA
from numba.core import types
from numba.typed import Dict

_mp5 = np.float32(-0.5)


@nb.njit(
    parallel=True,
    cache=True,
    fastmath=True,
    nogil=True,
    error_model="numpy",
    inline="always",
    forceinline=True,
)
def nb_parallel_sum(arr):
    arr = arr.reshape(-1)
    _sum = 0
    for i in nb.prange(arr.size):
        _sum += arr[i]
    return _sum


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True, error_model="numpy")
def nb_where(arr, nVxls):
    indices = np.empty(nVxls, dtype=np.int64)
    _, ny, nx = arr.shape
    arr = arr.reshape(-1)
    nzyx = arr.size
    index = 0
    for i in range(nzyx):
        if arr[i]:
            indices[index] = i
            index += 1

    xs = indices % nx
    indices = indices // nx
    zs = indices // ny
    ys = indices % ny
    zsysxs = np.empty((zs.size, 3), dtype=np.int64)
    for ipar in nb.prange(zs.size):
        zsysxs_i = zsysxs[ipar]
        zsysxs_i[0] = zs[ipar]
        zsysxs_i[1] = ys[ipar]
        zsysxs_i[2] = xs[ipar]
    return zsysxs


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True, error_model="numpy")
def smooth_radius(image, dt, zsysxs_v):
    nz, ny, nx = dt.shape
    print("smoothing R")
    delR = np.empty(image.shape, dtype=np.float32)
    for ipar in nb.prange(zsysxs_v.shape[0]):
        zo, yo, xo = zsysxs_v[ipar]
        sum_r = 0.0
        counter = 0
        for zi in range(-1, 2):
            for yi in range(-1, 2):
                for xi in range(-1, 2):
                    z = zo + zi
                    y = yo + yi
                    x = xo + xi
                    if (
                        z < 0
                        or z >= nz
                        or y < 0
                        or y >= ny
                        or x < 0
                        or x >= nx
                        or ~image[z, y, x]
                    ):
                        continue
                    sum_r += dt[z, y, x]
                    counter += 1
        delR[zo, yo, xo] = 4.0 * sum_r / (3 * counter + 27) - dt[zo, yo, xo]

    for ipar in nb.prange(zsysxs_v.shape[0]):
        zo, yo, xo = zsysxs_v[ipar]
        sum_del_r = 0.0
        counter = 0
        for zi in range(-1, 2):
            for yi in range(-1, 2):
                for xi in range(-1, 2):
                    z = zo + zi
                    y = yo + yi
                    x = xo + xi
                    if (
                        z < 0
                        or z >= nz
                        or y < 0
                        or y >= ny
                        or x < 0
                        or x >= nx
                        or ~image[z, y, x]
                    ):
                        continue
                    sum_del_r += delR[z, y, x]
                    counter += 1

        dt[zo, yo, xo] += min(
            max(
                0.02 * (delR[zo, yo, xo] - 0.99 * 2.0 * sum_del_r / (counter + 27)),
                -0.005,
            ),
            0.01,
        )
    # 计算最大半径
    print("maxrrr", np.max(dt))
    return dt


@nb.njit(
    parallel=False,
    cache=True,
    fastmath=True,
    nogil=True,
    error_model="numpy",
    inline="always",
    forceinline=True,
)
def get_masterball(ball_boss, ball_index):
    while ball_boss[ball_index] != ball_index:
        ball_index = ball_boss[ball_index]
    return ball_index


@nb.njit(
    parallel=False,
    cache=True,
    fastmath=True,
    nogil=True,
    error_model="numpy",
    inline="always",
    forceinline=True,
)
def get_ball_level(ball_boss, ball_index):
    level = 1
    current = ball_index
    while ball_boss[current] != current:
        current = ball_boss[current]
        level += 1
    return level


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True, error_model="numpy")
def inParents(ball_boss, ball_index_i, ball_index_j):
    """判断 ball_j 是否是 ball_i 的祖先（父节点或更高层级父节点）"""
    current = ball_boss[ball_index_i]  # 当前检查的节点
    while True:
        if current == ball_index_j:
            return True  # 找到目标父节点
        if current == ball_boss[current]:
            return False  # 到达根节点（boss指向自己）
        current = ball_boss[current]  # 继续检查父节点


# def paradox_pre_removeincludedballI(image, dt, isball, _minRp): Avoid too long name, due to numba cache limit
@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True, error_model="numpy")
def paradox_pre_rmincludedballI(image, dt, isball, _minRp):
    nz, ny, nx = image.shape
    nz_half = (nz + 1) // 2
    ny_half = (ny + 1) // 2
    nx_half = (nx + 1) // 2
    for _z in nb.prange(0, nz_half):
        zo = _z * 2
        for _y in nb.prange(0, ny_half):
            yo = _y * 2
            for _x in nb.prange(0, nx_half):
                xo = _x * 2
                max_val = -np.inf
                max_z = -1
                max_y = -1
                max_x = -1
                for zi in range(2):
                    for yi in range(2):
                        for xi in range(2):
                            z = zo + zi
                            y = yo + yi
                            x = xo + xi
                            if (
                                z < 0
                                or z >= nz
                                or y < 0
                                or y >= ny
                                or x < 0
                                or x >= nx
                                or ~image[z, y, x]
                            ):
                                continue
                            value = dt[z, y, x]
                            if value > max_val and value > _minRp:
                                max_val = value
                                max_z = z
                                max_y = y
                                max_x = x
                if max_z >= 0:
                    isball[max_z, max_y, max_x] = True


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True, error_model="numpy")
def paradox_removeincludedballI(
    ball_indices, ball_R, image, dt, isball, _RCorsnf, _RCorsn, _MSNoise
):
    removed_ball = 0
    for i in range(ball_indices.shape[0]):
        zo, yo, xo = ball_indices[i]
        if ~isball[zo, yo, xo]:
            continue
        nz, ny, nx = image.shape
        ri = ball_R[i]
        mbmbDist = _RCorsnf * ri + _RCorsn
        ripinc = ri + 0.55
        ripinc2 = ripinc**2
        rz = np.int32(ripinc)
        for zi in range(-rz, rz + 1):
            ry2 = ripinc2 - zi**2
            if ry2 <= 0:
                continue
            ry = np.int32(np.sqrt(ry2))
            for yi in range(-ry, ry + 1):
                rx2 = ry2 - yi**2
                if rx2 <= 0:
                    continue
                rx = np.int32(np.sqrt(rx2))
                for xi in range(-rx, rx + 1):
                    z = zo + zi
                    y = yo + yi
                    x = xo + xi
                    if (
                        (zi == 0 and yi == 0 and xi == 0)
                        or z < 0
                        or z >= nz
                        or y < 0
                        or y >= ny
                        or x < 0
                        or x >= nx
                        or ~isball[z, y, x]
                    ):
                        continue

                    rj = dt[z, y, x]
                    if rj <= ri:
                        D = np.sqrt(zi**2 + yi**2 + xi**2)
                        if D < mbmbDist or D + rj < ripinc + _MSNoise:
                            isball[z, y, x] = False
                            removed_ball += 1
    print(f"removed ball {removed_ball}")


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True, error_model="numpy")
def moveUphill(ball_indices, ball_findices, ball_R, image, dt):
    nz, ny, nx = image.shape
    for i in nb.prange(ball_indices.shape[0]):
        disp = np.array([0, 0, 0], dtype=np.float32)
        iz, iy, ix = ball_indices[i]
        vi_r = dt[iz, iy, ix]
        vjm_z = iz - 1
        vjm_y = iy
        vjm_x = ix
        vjp_z = iz + 1
        vjp_y = iy
        vjp_x = ix
        if 0 <= vjm_z and vjp_z < nz:
            if image[vjm_z, vjm_y, vjm_x] and image[vjp_z, vjp_y, vjp_x]:
                vjm_r = dt[vjm_z, vjm_y, vjm_x]
                vjp_r = dt[vjp_z, vjp_y, vjp_x]
                gp = vjp_r - vi_r
                gm = vi_r - vjm_r
                if abs(gp - gm) > 0.01:
                    disp[0] = max(-0.49, min(0.49, -0.5 * (gp + gm) / (gp - gm)))

        vjm_z = iz
        vjm_y = iy - 1
        vjm_x = ix
        vjp_z = iz
        vjp_y = iy + 1
        vjp_x = ix
        if 0 <= vjm_y and vjp_y < ny:
            if image[vjm_z, vjm_y, vjm_x] and image[vjp_z, vjp_y, vjp_x]:
                vjm_r = dt[vjm_z, vjm_y, vjm_x]
                vjp_r = dt[vjp_z, vjp_y, vjp_x]
                gp = vjp_r - vi_r
                gm = vi_r - vjm_r
                if abs(gp - gm) > 0.01:
                    disp[1] = max(-0.49, min(0.49, -0.5 * (gp + gm) / (gp - gm)))

        vjm_z = iz
        vjm_y = iy
        vjm_x = ix - 1
        vjp_z = iz
        vjp_y = iy
        vjp_x = ix + 1
        if 0 <= vjm_x and vjp_x < nx:
            if image[vjm_z, vjm_y, vjm_x] and image[vjp_z, vjp_y, vjp_x]:
                vjm_r = dt[vjm_z, vjm_y, vjm_x]
                vjp_r = dt[vjp_z, vjp_y, vjp_x]
                gp = vjp_r - vi_r
                gm = vi_r - vjm_r
                if abs(gp - gm) > 0.01:
                    disp[2] = max(-0.49, min(0.49, -0.5 * (gp + gm) / (gp - gm)))

        ball_findices[i] = ball_indices[i] + disp - _mp5
        ball_R[i] = vi_r + 0.95 * np.sqrt(disp[0] ** 2 + disp[1] ** 2 + disp[2] ** 2)
        # dt[iz, iy, ix] = R_modified


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True, error_model="numpy")
def moveUphillp1(ball_indices, ball_findices, ball_R, image, dt, isball):
    nz, ny, nx = image.shape
    for i in range(ball_indices.shape[0]):
        disp_z = 0.0
        disp_y = 0.0
        disp_x = 0.0
        grad_z = 0.0
        grad_y = 0.0
        grad_x = 0.0
        iz, iy, ix = ball_indices[i]
        vi_r = dt[iz, iy, ix]
        vjm_z = iz - 1
        vjm_y = iy
        vjm_x = ix
        vjp_z = iz + 1
        vjp_y = iy
        vjp_x = ix
        if 0 <= vjm_z and vjp_z < nz:
            if image[vjm_z, vjm_y, vjm_x] and image[vjp_z, vjp_y, vjp_x]:
                vjm_r = dt[vjm_z, vjm_y, vjm_x]
                vjp_r = dt[vjp_z, vjp_y, vjp_x]
                gp = vjp_r - vi_r
                gm = vi_r - vjm_r
                grad_z = 0.5 * (gp + gm)
                if abs(gp - gm) > 0.01:
                    disp_z = max(-0.59, min(0.59, -0.5 * (gp + gm) / (gp - gm)))
        vjm_z = iz
        vjm_y = iy - 1
        vjm_x = ix
        vjp_z = iz
        vjp_y = iy + 1
        vjp_x = ix
        if 0 <= vjm_y and vjp_y < ny:
            if image[vjm_z, vjm_y, vjm_x] and image[vjp_z, vjp_y, vjp_x]:
                vjm_r = dt[vjm_z, vjm_y, vjm_x]
                vjp_r = dt[vjp_z, vjp_y, vjp_x]
                gp = vjp_r - vi_r
                gm = vi_r - vjm_r
                grad_y = 0.5 * (gp + gm)
                if abs(gp - gm) > 0.01:
                    disp_y = max(-0.59, min(0.59, -0.5 * (gp + gm) / (gp - gm)))
        vjm_z = iz
        vjm_y = iy
        vjm_x = ix - 1
        vjp_z = iz
        vjp_y = iy
        vjp_x = ix + 1
        if 0 <= vjm_x and vjp_x < nx:
            if image[vjm_z, vjm_y, vjm_x] and image[vjp_z, vjp_y, vjp_x]:
                vjm_r = dt[vjm_z, vjm_y, vjm_x]
                vjp_r = dt[vjp_z, vjp_y, vjp_x]
                gp = vjp_r - vi_r
                gm = vi_r - vjm_r
                grad_x = 0.5 * (gp + gm)
                if abs(gp - gm) > 0.01:
                    disp_x = max(-0.59, min(0.59, -0.5 * (gp + gm) / (gp - gm)))
        disp_z += 1.4 * grad_z
        disp_y += 1.4 * grad_y
        disp_x += 1.4 * grad_x
        disp_norm = 0.55 * np.sqrt(disp_z**2 + disp_y**2 + disp_x**2) + 0.05
        disp_z /= disp_norm
        disp_y /= disp_norm
        disp_x /= disp_norm
        vxlj_z = np.int32(iz + disp_z)
        vxlj_y = np.int32(iy + disp_y)
        vxlj_x = np.int32(ix + disp_x)
        if (
            0 <= vxlj_z < nz
            and 0 <= vxlj_y < ny
            and 0 <= vxlj_x < nx
            and (vxlj_z != iz or vxlj_y != iy or vxlj_x != ix)
        ):
            if (
                ~isball[vxlj_z, vxlj_y, vxlj_x]
                and dt[vxlj_z, vxlj_y, vxlj_x] > ball_R[i]
            ):
                isball[iz, iy, ix] = False
                isball[vxlj_z, vxlj_y, vxlj_x] = True
                ball_indices[i, 0] = vxlj_z
                ball_indices[i, 1] = vxlj_y
                ball_indices[i, 2] = vxlj_x
                ball_findices[i, 0] = vxlj_z - _mp5
                ball_findices[i, 1] = vxlj_y - _mp5
                ball_findices[i, 2] = vxlj_x - _mp5
                ball_R[i] = dt[vxlj_z, vxlj_y, vxlj_x]


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True, error_model="numpy")
def makeFriend(ball_R, ball_boss, ball_indices, ball_findices, vi, vj):
    if ball_R[vj] > ball_R[vi]:
        vi, vj = vj, vi


@nb.njit(
    parallel=False,
    cache=True,
    fastmath=True,
    nogil=True,
    error_model="numpy",
    inline="always",
    forceinline=True,
)
def competeForParent(
    vi,
    vj,
    ball_findices,
    ball_R,
    ball_boss,
    image,
    dt,
    _MSNoise,
    _midRf,
    _vmvRadRelNf,
    _lenNf,
):
    nz, ny, nx = image.shape
    noise = _MSNoise
    ri = ball_R[vi]
    rj = ball_R[vj]
    riSqr = ri * ri
    rjSqr = rj * rj
    fiz, fiy, fix = ball_findices[vi]
    fjz, fjy, fjx = ball_findices[vj]
    dSqr = np.sum((ball_findices[vi] - ball_findices[vj]) ** 2)
    wsinv = 1.0 / (riSqr + rjSqr)
    middlevxlz = np.int32((fiz * rjSqr + fjz * riSqr) * wsinv)
    middlevxly = np.int32((fiy * rjSqr + fjy * riSqr) * wsinv)
    middlevxlx = np.int32((fix * rjSqr + fjx * riSqr) * wsinv)
    if (
        middlevxlz < 0
        or middlevxlz >= nz
        or middlevxly < 0
        or middlevxly >= ny
        or middlevxlx < 0
        or middlevxlx >= nx
        or ~image[middlevxlz, middlevxly, middlevxlx]
    ):
        return

    if (
        dt[middlevxlz, middlevxly, middlevxlx] > min(ri, rj) * _midRf - 0.5
        and 1.01 * np.sqrt(dSqr) < ri + rj + 1.0 + 1.0 * noise
    ):
        if ball_boss[vj] == vj:
            if get_masterball(ball_boss, vi) != vj:
                if ri >= rj:
                    ball_boss[vj] = vi
                elif ball_R[ball_boss[vi]] <= rj:
                    ball_boss[vi] = vj
                elif ri >= rj - noise and ri * _vmvRadRelNf + 1.0 * noise >= rj:
                    ball_boss[vj] = vi
        elif ball_boss[vi] == vi:
            if get_masterball(ball_boss, vj) != vi:
                if rj >= ri:
                    ball_boss[vi] = vj
                elif ball_R[ball_boss[vj]] <= ri:
                    ball_boss[vj] = vi
                elif rj >= ri - noise and rj * _vmvRadRelNf + 1.0 * noise >= ri:
                    ball_boss[vi] = vj

        mvi = get_masterball(ball_boss, vi)
        mvj = get_masterball(ball_boss, vj)

        if mvi != vj and mvj != vi:
            if mvi == mvj:
                leveli = get_ball_level(ball_boss, vi)
                levelj = get_ball_level(ball_boss, vj)
                bvi = ball_boss[vi]
                bvj = ball_boss[vj]
                bvi_R = ball_R[bvi]
                bvj_R = ball_R[bvj]
                dist_vivj = np.sqrt(
                    np.sum((ball_findices[vi] - ball_findices[vj]) ** 2)
                )
                dist_bvi_vi = np.sqrt(
                    np.sum((ball_findices[bvi] - ball_findices[vi]) ** 2)
                )
                dist_bvj_vj = np.sqrt(
                    np.sum((ball_findices[bvj] - ball_findices[vj]) ** 2)
                )
                if leveli + 1 < levelj and (bvj_R - rj + 2.0 * noise) / (
                    dist_bvj_vj + 0.25
                ) < (ri - rj + 2.0 * noise + 0.01) / (dist_vivj + 0.2):
                    ball_boss[vj] = vi
                elif leveli > levelj + 1 and (bvi_R - ri + 2.0 * noise) / (
                    dist_bvi_vi + 0.25
                ) < (rj - ri + 2.0 * noise + 0.01) / (dist_vivj + 0.2):
                    ball_boss[vi] = vj
                elif (
                    leveli > levelj
                    and (bvi_R - ri + 2.0 * noise) / (dist_bvi_vi + 1.2)
                    < (rj - ri + 2.0 * noise) / (dist_vivj + 1.3)
                    and not inParents(ball_boss, vj, vi)
                ):
                    ball_boss[vi] = vj
                elif (
                    leveli < levelj
                    and (bvj_R - rj + 2.0 * noise) / (dist_bvj_vj + 1.2)
                    < (ri - rj + 2.0 * noise) / (dist_vivj + 1.3)
                    and not inParents(ball_boss, vi, vj)
                ):
                    ball_boss[vj] = vi
                    # elif (
                    #     dt[middlevxlz, middlevxly, middlevxlx]
                    #     >= 0.45 * (ri + rj) - 1.0
                    #     and np.sqrt(dSqr) < (ri + rj) * 0.5 + 2
                    # ):
                    #     # makeFriend(ball_R, ball_boss, ball_indices, ball_findices, vi,vj)
                    #     vi, vj = vj, vi
                    """
                    def makeFriend(ball_R, ball_boss, ball_indices, ball_findices, vi, vj):
                        if ball_R[vj] > ball_R[vi]:
                        ball_R[vi], ball_R[vj] = ball_R[vj], ball_R[vi]
                        ball_boss[vi], ball_boss[vj] = ball_boss[vj], ball_boss[vi]
                        ball_indices[vi], ball_indices[vj] = ball_indices[vj], ball_indices[vi]
                        ball_findices[vi], ball_findices[vj] = ball_findices[vj], ball_findices[vi]
                    """
                # elif ... make friends
                # if get_masterball(ball_boss, vi) != get_masterball(
                #     ball_boss, vj
                # ):
                #     print("Warning: paradox")
            else:  # mvi != mvj:
                mvi_R = ball_R[mvi]
                mvj_R = ball_R[mvj]
                if np.sum((ball_findices[mvi] - ball_findices[mvj]) ** 2) <= _lenNf * (
                    0.5 * (mvi_R + mvj_R) + 2.0 * noise
                ) * (0.5 * (mvi_R + mvj_R) + 2.0 * noise):
                    if mvi_R < mvj_R:
                        vi, vj, mvi, mvj = vj, vi, mvj, mvi

                    mvj_R = ball_R[mvj]
                    if (
                        mvj_R < _vmvRadRelNf * ball_R[vj] + noise
                        and mvj_R < _vmvRadRelNf * ball_R[vi] + noise
                        and mvj_R < _vmvRadRelNf * ball_R[ball_boss[vi]] + noise
                    ):
                        while (
                            vj != ball_boss[vj]
                            and mvj_R < _vmvRadRelNf * ball_R[ball_boss[vj]] + noise
                        ):
                            pvj = ball_boss[vj]
                            ball_boss[vj] = vi
                            vi = vj
                            vj = pvj
                        if ball_boss[vj] == vj and get_masterball(ball_boss, vi) != vj:
                            ball_boss[vj] = vi
                if vi != ball_boss[vj]:
                    mvi = get_masterball(ball_boss, vi)
                    mvj = get_masterball(ball_boss, vj)
                    leveli = get_ball_level(ball_boss, vi)
                    levelj = get_ball_level(ball_boss, vj)
                    distAvg = (
                        np.sqrt(np.sum((ball_findices[mvj] - ball_findices[mvi]) ** 2))
                        + 0.5 * noise
                    )
                    while leveli >= levelj and (
                        ball_R[ball_boss[vi]] - ball_R[vi] + 0.55 * noise
                    ) / (
                        np.sqrt(np.sum((ball_findices[mvi] - ball_findices[vi]) ** 2))
                        + distAvg
                    ) < (ball_R[vj] - ball_R[vi] + 0.5 * noise) / (
                        np.sqrt(np.sum((ball_findices[mvj] - ball_findices[vi]) ** 2))
                        + distAvg
                    ):
                        pvi = ball_boss[vi]
                        ball_boss[vi] = vj
                        vj = vi
                        vi = pvi
                        levelj += 1
                        leveli -= 1
                    while levelj >= leveli and (
                        ball_R[ball_boss[vj]] - ball_R[vj] + 0.55 * noise
                    ) / (
                        np.sqrt(np.sum((ball_findices[mvj] - ball_findices[vj]) ** 2))
                        + distAvg
                    ) < (ball_R[vi] - ball_R[vj] + 0.5 * noise) / (
                        np.sqrt(np.sum((ball_findices[mvi] - ball_findices[vj]) ** 2))
                        + distAvg
                    ):
                        pvj = ball_boss[vj]
                        ball_boss[vj] = vi
                        vi = vj
                        vj = pvj
                        leveli += 1
                        levelj -= 1
                    # vi, vj = vj, vi
                    # makeFriend(ball_R, ball_boss, ball_indices, ball_findices, vi, vj)

                    # make friends


@nb.njit(
    parallel=True,
    cache=True,
    fastmath=True,
    nogil=True,
    error_model="numpy",
)
def float_range(start, stop, step):
    x = start
    n = 0
    while x < stop:
        yield x
        n += 1
        x = start + n * step


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True, error_model="numpy")
def findBoss(
    ball_indices,
    ball_findices,
    ball_R,
    ball_master,
    image,
    dt,
    isball,
    _MSNoise,
    _midRf,
    _vmvRadRelNf,
    _lenNf,
):
    nz, ny, nx = image.shape
    nBalls = ball_indices.shape[0]
    whichball = Dict.empty(
        key_type=types.int64,  # 或 types.int32，取决于你的索引范围
        value_type=types.int64,
        n_keys=nBalls,
    )
    keys = (
        ball_indices[:, 0] * ny * nx + ball_indices[:, 1] * nx + ball_indices[:, 2]
    ).astype(np.int64)
    for i in range(nBalls):
        whichball[keys[i]] = np.int64(i)

    for i in range(nBalls):
        zo, yo, xo = (
            ball_findices[i, 0],
            ball_findices[i, 1],
            ball_findices[i, 2],
        )

        ripp = ball_R[i] * 0.6 + 2.0 * _MSNoise + 2.0
        ripp2 = ripp**2
        rz = ripp
        zis = np.arange(-rz, rz + 1e-6, 1.0, dtype=np.float32)
        for zi in zis:
            ry2 = ripp2 - zi**2
            if ry2 <= 0:
                continue
            ry = np.sqrt(ry2)
            yis = np.arange(-ry, ry + 1e-6, 1.0, dtype=np.float32)
            for yi in yis:
                rx2 = ry2 - yi**2
                if rx2 <= 0:
                    continue
                rx = np.sqrt(rx2)
                xis = np.arange(-rx, rx + 1e-6, 1.0, dtype=np.float32)
                for xi in xis:
                    z = np.int64(zo + zi)
                    y = np.int64(yo + yi)
                    x = np.int64(xo + xi)
                    if (
                        (zi == 0 and yi == 0 and xi == 0)
                        or z < 0
                        or z >= nz
                        or y < 0
                        or y >= ny
                        or x < 0
                        or x >= nx
                        or ~isball[z, y, x]
                    ):
                        continue
                    vj = whichball[z * ny * nx + y * nx + x]
                    competeForParent(
                        np.int64(i),
                        vj,
                        ball_findices,
                        ball_R,
                        ball_master,
                        image,
                        dt,
                        _MSNoise,
                        _midRf,
                        _vmvRadRelNf,
                        _lenNf,
                    )
