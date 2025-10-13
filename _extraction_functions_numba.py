from xml.dom import WRONG_DOCUMENT_ERR
import numpy as np
import numba as nb
from numpy import linalg as LA
from numba.core import types
from numba.typed import Dict

_mp5 = np.float32(-0.5)


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def nb_parallel_sum(arr):
    arr = arr.reshape(-1)
    _sum = 0
    for i in nb.prange(arr.size):
        _sum += arr[i]
    return _sum


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
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
    zsysxs = np.empty((zs.size, 3), dtype=np.int32)
    for ipar in nb.prange(zs.size):
        zsysxs_i = zsysxs[ipar]
        zsysxs_i[0] = zs[ipar]
        zsysxs_i[1] = ys[ipar]
        zsysxs_i[2] = xs[ipar]
    return zsysxs


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def smooth_radius(img_bool, dt, zsysxs_v):
    nz, ny, nx = dt.shape
    print("smoothing R")
    delR = np.zeros_like(dt)
    for ipar in nb.prange(zsysxs_v.shape[0]):
        k, j, i = zsysxs_v[ipar]
        sum_r = 0.0
        counter = 0

        # 3x3x3邻域遍历
        for kk in range(max(k - 1, 0), min(k + 2, nz)):
            for jj in range(max(j - 1, 0), min(j + 2, ny)):
                # 简化segment处理，直接取i-1到i+1范围
                for ii in range(max(i - 1, 0), min(i + 2, nx)):
                    if img_bool[kk, jj, ii]:
                        sum_r += dt[kk, jj, ii]
                        counter += 1

        delR[k, j, i] = 4.0 * sum_r / (3 * counter + 27) - dt[k, j, i]

    #
    # 第二部分：应用平滑
    for ipar in nb.prange(zsysxs_v.shape[0]):
        k, j, i = zsysxs_v[ipar]
        sum_del_r = 0.0
        counter = 0
        # 再次遍历邻域
        for kk in range(max(k - 1, 0), min(k + 2, nz)):
            for jj in range(max(j - 1, 0), min(j + 2, ny)):
                for ii in range(max(i - 1, 0), min(i + 2, nx)):
                    if img_bool[kk, jj, ii]:
                        sum_del_r += delR[kk, jj, ii]
                        counter += 1

        dt[k, j, i] += min(
            max(
                0.02 * (delR[k, j, i] - 0.99 * 2.0 * sum_del_r / (counter + 27)), -0.005
            ),
            0.01,
        )
    # 计算最大半径
    print("maxrrr", np.max(dt))
    return dt


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
def get_masterball(ball_boss, ball_index):
    while ball_boss[ball_index] != ball_index:
        ball_index = ball_boss[ball_index]
    return ball_index


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
def get_ball_level(ball_boss, ball_index):
    level = 1
    current = ball_index
    while ball_boss[current] != current:
        current = ball_boss[current]
        level += 1
    return level


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
def inParents(ball_boss, ball_index_i, ball_index_j):
    """判断 ball_j 是否是 ball_i 的祖先（父节点或更高层级父节点）"""
    current = ball_boss[ball_index_i]  # 当前检查的节点
    while True:
        if current == ball_index_j:
            return True  # 找到目标父节点
        if current == ball_boss[current]:
            return False  # 到达根节点（boss指向自己）
        current = ball_boss[current]  # 继续检查父节点


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def paradox_pre_removeincludedballI(image, dt, isball, _minRp):
    nz, ny, nx = image.shape
    zs = np.arange(0, nz - 1, 2)
    ys = np.arange(0, ny - 1, 2)
    xs = np.arange(0, nx - 1, 2)
    for z_id in nb.prange(len(zs)):
        z = zs[z_id]
        for y_id in nb.prange(len(ys)):
            y = ys[y_id]
            for x_id in nb.prange(len(xs)):
                x = xs[x_id]
                if image[z, y, x]:
                    max_val = -np.inf
                    max_z = 0
                    max_y = 0
                    max_x = 0
                    for i in range(2):
                        for j in range(2):
                            for k in range(2):
                                value = dt[z + i, y + j, x + k]
                                if value > max_val:
                                    max_val = value
                                    max_z = z + i
                                    max_y = y + j
                                    max_x = x + k
                    if max_val > _minRp:
                        isball[max_z, max_y, max_x] = True


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
def paradox_removeincludedballI(
    ball_indices, ball_R, image, dt, isball, _RCorsnf, _RCorsn, _MSNoise
):
    removed_ball = 0
    for i in range(ball_indices.shape[0]):
        c_ball_indices = ball_indices[i]
        if ~isball[c_ball_indices[0], c_ball_indices[1], c_ball_indices[2]]:
            continue
        nz, ny, nx = image.shape
        z, y, x = ball_indices[i]
        ri = ball_R[i]
        ripinc = ri + 0.55
        mbmbDist = _RCorsnf * ri + _RCorsn
        ripinc_sq = ripinc * ripinc
        ez = int(ripinc)
        for c in range(-ez, ez + 1):
            c_sq = c * c
            temp = ripinc_sq - c_sq
            if temp <= 0:
                continue
            ey = int(np.sqrt(temp))
            for b in range(-ey, ey + 1):
                b_sq = b * b
                temp = ripinc_sq - c_sq - b_sq
                if temp <= 0:
                    continue
                ex = int(np.sqrt(temp))
                for a in range(-ex, ex + 1):
                    a_sq = a * a
                    vn_z = z + c
                    vn_y = y + b
                    vn_x = x + a
                    if (
                        0 <= vn_z < nz
                        and 0 <= vn_y < ny
                        and 0 <= vn_x < nx
                        and (a != 0 or b != 0 or c != 0)
                    ):
                        if isball[vn_z, vn_y, vn_x]:
                            rj = dt[vn_z, vn_y, vn_x]
                            if rj <= ri:
                                D = np.sqrt(a_sq + b_sq + c_sq)
                                if D < mbmbDist or D + rj < ripinc + _MSNoise:
                                    isball[vn_z, vn_y, vn_x] = False
                                    removed_ball += 1
    print(f"removed ball {removed_ball}")


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def moveUphill(ball_indices, ball_findices, ball_R, image, dt):
    nz, ny, nx = image.shape
    for i in nb.prange(ball_indices.shape[0]):
        disp = np.array([0, 0, 0])
        fz, fy, fx = ball_findices[i]
        fz_int = int(fz)
        fy_int = int(fy)
        fx_int = int(fx)
        vi_r = dt[fz_int, fy_int, fx_int]
        vjm_z = fz_int - 1
        vjm_y = fy_int
        vjm_x = fx_int
        vjp_z = fz_int + 1
        vjp_y = fy_int
        vjp_x = fx_int
        if (
            0 <= vjm_z < nz
            and 0 <= vjm_y < ny
            and 0 <= vjm_x < nx
            and 0 <= vjp_z < nz
            and 0 <= vjp_y < ny
            and 0 <= vjp_x < nx
        ):
            if image[vjm_z, vjm_y, vjm_x] and image[vjp_z, vjp_y, vjp_x]:
                vjm_r = dt[vjm_z, vjm_y, vjm_x]
                vjp_r = dt[vjp_z, vjp_y, vjp_x]
                gp = vjp_r - vi_r
                gm = vi_r - vjm_r
                if abs(gp - gm) > 0.01:
                    disp[0] = max(-0.49, min(0.49, -0.5 * (gp + gm) / (gp - gm)))

        vjm_z = fz_int
        vjm_y = fy_int - 1
        vjm_x = fx_int
        vjp_z = fz_int
        vjp_y = fy_int + 1
        vjp_x = fx_int
        if (
            0 <= vjm_z < nz
            and 0 <= vjm_y < ny
            and 0 <= vjm_x < nx
            and 0 <= vjp_z < nz
            and 0 <= vjp_y < ny
            and 0 <= vjp_x < nx
        ):
            if image[vjm_z, vjm_y, vjm_x] and image[vjp_z, vjp_y, vjp_x]:
                vjm_r = dt[vjm_z, vjm_y, vjm_x]
                vjp_r = dt[vjp_z, vjp_y, vjp_x]
                gp = vjp_r - vi_r
                gm = vi_r - vjm_r
                if abs(gp - gm) > 0.01:
                    disp[1] = max(-0.49, min(0.49, -0.5 * (gp + gm) / (gp - gm)))

        vjm_z = fz_int
        vjm_y = fy_int
        vjm_x = fx_int - 1
        vjp_z = fz_int
        vjp_y = fy_int
        vjp_x = fx_int + 1
        if (
            0 <= vjm_z < nz
            and 0 <= vjm_y < ny
            and 0 <= vjm_x < nx
            and 0 <= vjp_z < nz
            and 0 <= vjp_y < ny
            and 0 <= vjp_x < nx
        ):
            if image[vjm_z, vjm_y, vjm_x] and image[vjp_z, vjp_y, vjp_x]:
                vjm_r = dt[vjm_z, vjm_y, vjm_x]
                vjp_r = dt[vjp_z, vjp_y, vjp_x]
                gp = vjp_r - vi_r
                gm = vi_r - vjm_r
                if abs(gp - gm) > 0.01:
                    disp[2] = max(-0.49, min(0.49, -0.5 * (gp + gm) / (gp - gm)))

        ball_findices[i] = ball_indices[i] + disp - _mp5
        R_modified = ball_R[i] + 0.95 * np.sqrt(
            disp[0] ** 2 + disp[1] ** 2 + disp[2] ** 2
        )
        ball_R[i] = R_modified
        dt[ball_indices[i, 0], ball_indices[i, 1], ball_indices[i, 2]] = R_modified


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
def moveUphillp1(ball_indices, ball_findices, ball_R, image, dt, isball):
    nz, ny, nx = image.shape
    for i in range(ball_indices.shape[0]):
        disp = np.array([0.0, 0.0, 0.0])
        grad = np.array([0.0, 0.0, 0.0])
        fz, fy, fx = ball_findices[i]
        fz_int = int(fz)
        fy_int = int(fy)
        fx_int = int(fx)
        vi_r = dt[fz_int, fy_int, fx_int]
        vjm_z = fz_int - 1
        vjm_y = fy_int
        vjm_x = fx_int
        vjp_z = fz_int + 1
        vjp_y = fy_int
        vjp_x = fx_int
        if (
            0 <= vjm_z < nz
            and 0 <= vjm_y < ny
            and 0 <= vjm_x < nx
            and 0 <= vjp_z < nz
            and 0 <= vjp_y < ny
            and 0 <= vjp_x < nx
        ):
            if image[vjm_z, vjm_y, vjm_x] and image[vjp_z, vjp_y, vjp_x]:
                vjm_r = dt[vjm_z, vjm_y, vjm_x]
                vjp_r = dt[vjp_z, vjp_y, vjp_x]
                gp = vjp_r - vi_r
                gm = vi_r - vjm_r
                grad[0] = 0.5 * (gp + gm)
                if abs(gp - gm) > 0.01:
                    disp[0] = max(-0.59, min(0.59, -0.5 * (gp + gm) / (gp - gm)))
        vjm_z = fz_int
        vjm_y = fy_int - 1
        vjm_x = fx_int
        vjp_z = fz_int
        vjp_y = fy_int + 1
        vjp_x = fx_int
        if (
            0 <= vjm_z < nz
            and 0 <= vjm_y < ny
            and 0 <= vjm_x < nx
            and 0 <= vjp_z < nz
            and 0 <= vjp_y < ny
            and 0 <= vjp_x < nx
        ):
            if image[vjm_z, vjm_y, vjm_x] and image[vjp_z, vjp_y, vjp_x]:
                vjm_r = dt[vjm_z, vjm_y, vjm_x]
                vjp_r = dt[vjp_z, vjp_y, vjp_x]
                gp = vjp_r - vi_r
                gm = vi_r - vjm_r
                grad[1] = 0.5 * (gp + gm)
                if abs(gp - gm) > 0.01:
                    disp[1] = max(-0.59, min(0.59, -0.5 * (gp + gm) / (gp - gm)))
        vjm_z = fz_int
        vjm_y = fy_int
        vjm_x = fx_int - 1
        vjp_z = fz_int
        vjp_y = fy_int
        vjp_x = fx_int + 1
        if (
            0 <= vjm_z < nz
            and 0 <= vjm_y < ny
            and 0 <= vjm_x < nx
            and 0 <= vjp_z < nz
            and 0 <= vjp_y < ny
            and 0 <= vjp_x < nx
        ):
            if image[vjm_z, vjm_y, vjm_x] and image[vjp_z, vjp_y, vjp_x]:
                vjm_r = dt[vjm_z, vjm_y, vjm_x]
                vjp_r = dt[vjp_z, vjp_y, vjp_x]
                gp = vjp_r - vi_r
                gm = vi_r - vjm_r
                grad[2] = 0.5 * (gp + gm)
                if abs(gp - gm) > 0.01:
                    disp[2] = max(-0.59, min(0.59, -0.5 * (gp + gm) / (gp - gm)))
        disp += 1.4 * grad
        disp /= 0.55 * LA.norm(disp) + 0.05
        vxlj_z = int(fz_int + disp[0])
        vxlj_y = int(fy_int + disp[1])
        vxlj_x = int(fx_int + disp[2])
        if (
            0 <= vxlj_z < nz
            and 0 <= vxlj_y < ny
            and 0 <= vxlj_x < nx
            and (vxlj_z != fz_int or vxlj_y != fy_int or vxlj_x != fx_int)
        ):
            if (
                ~isball[vxlj_z, vxlj_y, vxlj_x]
                and dt[vxlj_z, vxlj_y, vxlj_x] > ball_R[i]
            ):
                z, y, x = ball_indices[i]
                isball[z, y, x] = False
                isball[vxlj_z, vxlj_y, vxlj_x] = True
                ball_findices[i, 0] = vxlj_z - _mp5
                ball_findices[i, 1] = vxlj_y - _mp5
                ball_findices[i, 2] = vxlj_x - _mp5
                ball_R[i] = dt[vxlj_z, vxlj_y, vxlj_x]
                ball_indices[i, 0] = vxlj_z
                ball_indices[i, 1] = vxlj_y
                ball_indices[i, 2] = vxlj_x


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
def makeFriend(ball_R, ball_boss, ball_indices, ball_findices, vi, vj):
    if ball_R[vj] > ball_R[vi]:
        vi, vj = vj, vi


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
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
    middlevxlz = int((fiz * rjSqr + fjz * riSqr) * wsinv)
    middlevxly = int((fiy * rjSqr + fjy * riSqr) * wsinv)
    middlevxlx = int((fix * rjSqr + fjx * riSqr) * wsinv)
    if 0 <= middlevxlz < nz and 0 <= middlevxly < ny and 0 <= middlevxlx < nx:
        if image[middlevxlz, middlevxly, middlevxlx]:
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
                        dist_vivj = LA.norm(ball_findices[vi] - ball_findices[vj])
                        dist_bvi_vi = LA.norm(ball_findices[bvi] - ball_findices[vi])
                        dist_bvj_vj = LA.norm(ball_findices[bvj] - ball_findices[vj])
                        if leveli + 1 < levelj and (bvj_R - rj + 2.0 * noise) / (
                            dist_bvj_vj + 0.25
                        ) < (ri - rj + 2.0 * noise + 0.01) / (dist_vivj + 0.2):
                            ball_boss[vj] = vi
                        elif leveli > levelj + 1 and (bvi_R - ri + 2.0 * noise) / (
                            dist_bvi_vi + 0.25
                        ) < (rj - ri + 2.0 * noise + 0.01) / (dist_vivj + 0.2):
                            ball_boss[vi] = vj
                        else:
                            if (
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
                            elif (
                                dt[middlevxlz, middlevxly, middlevxlx]
                                >= 0.45 * (ri + rj) - 1.0
                                and np.sqrt(dSqr) < (ri + rj) * 0.5 + 2
                            ):
                                # makeFriend(ball_R, ball_boss, ball_indices, ball_findices, vi,vj)
                                vi, vj = vj, vi
                                """
                                def makeFriend(ball_R, ball_boss, ball_indices, ball_findices, vi, vj):
                                    if ball_R[vj] > ball_R[vi]:
                                    ball_R[vi], ball_R[vj] = ball_R[vj], ball_R[vi]
                                    ball_boss[vi], ball_boss[vj] = ball_boss[vj], ball_boss[vi]
                                    ball_indices[vi], ball_indices[vj] = ball_indices[vj], ball_indices[vi]
                                    ball_findices[vi], ball_findices[vj] = ball_findices[vj], ball_findices[vi]
                                """
                        # elif ... make friends
                        if get_masterball(ball_boss, vi) != get_masterball(
                            ball_boss, vj
                        ):
                            print("Warning: paradox")
                    else:  # mvi != mvj:
                        mvi_R = ball_R[mvi]
                        mvj_R = ball_R[mvj]
                        if np.sum(
                            (ball_findices[mvi] - ball_findices[mvj]) ** 2
                        ) <= _lenNf * (0.5 * (mvi_R + mvj_R) + 2.0 * noise) * (
                            0.5 * (mvi_R + mvj_R) + 2.0 * noise
                        ):
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
                                    and mvj_R
                                    < _vmvRadRelNf * ball_R[ball_boss[vj]] + noise
                                ):
                                    pvj = ball_boss[vj]
                                    ball_boss[vj] = vi
                                    vi = vj
                                    vj = pvj
                                if (
                                    ball_boss[vj] == vj
                                    and get_masterball(ball_boss, vi) != vj
                                ):
                                    ball_boss[vj] = vi
                        if vi != ball_boss[vj]:
                            mvi = get_masterball(ball_boss, vi)
                            mvj = get_masterball(ball_boss, vj)
                            leveli = get_ball_level(ball_boss, vi)
                            levelj = get_ball_level(ball_boss, vj)
                            distAvg = (
                                LA.norm(ball_findices[mvj] - ball_findices[mvi])
                                + 0.5 * noise
                            )
                            while leveli >= levelj and (
                                ball_R[ball_boss[vi]] - ball_R[vi] + 0.55 * noise
                            ) / (
                                LA.norm(ball_findices[mvi] - ball_findices[vi])
                                + distAvg
                            ) < (
                                ball_R[vj] - ball_R[vi] + 0.5 * noise
                            ) / (
                                LA.norm(ball_findices[mvj] - ball_findices[vi])
                                + distAvg
                            ):
                                pvi = ball_boss[vi]
                                ball_boss[vi] = vj
                                vj = vi
                                vi = pvi
                                leveli += 1
                                levelj -= 1
                            while levelj >= leveli and (
                                ball_R[ball_boss[vj]] - ball_R[vj] + 0.55 * noise
                            ) / (
                                LA.norm(ball_findices[mvj] - ball_findices[vj])
                                + distAvg
                            ) < (
                                ball_R[vi] - ball_R[vj] + 0.5 * noise
                            ) / (
                                LA.norm(ball_findices[mvi] - ball_findices[vj])
                                + distAvg
                            ):
                                pvj = ball_boss[vj]
                                ball_boss[vj] = vi
                                vi = vj
                                vj = pvj
                                leveli += 1
                                levelj -= 1
                            vi, vj = vj, vi
                            # makeFriend(ball_R, ball_boss, ball_indices, ball_findices, vi, vj)

                            # make friends


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
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
    nz = np.int64(nz)
    ny = np.int64(ny)
    nx = np.int64(nx)
    nBalls = ball_indices.shape[0]
    whichball = Dict.empty(
        key_type=types.int64,  # 或 types.int32，取决于你的索引范围
        value_type=types.int32,
        n_keys=nBalls,
    )
    keys = ball_indices[:, 0] * ny * nx + ball_indices[:, 1] * nx + ball_indices[:, 2]
    for i in range(nBalls):
        whichball[keys[i]] = np.int32(i)

    for i in range(nBalls):
        z, y, x = ball_findices[i]
        ripp = ball_R[i] * 0.6 + 2.0 * _MSNoise + 2.0
        ez = z + ripp
        zpcs = np.arange(2.0 * z - ez, ez + 0.001, 1.0)
        for zpc in zpcs:
            temp = ripp * ripp - (zpc - z) * (zpc - z)
            if temp <= 0:
                continue
            ey = y + np.sqrt(temp)
            ypbs = np.arange(2.0 * y - ey, ey + 0.001, 1.0)
            for ypb in ypbs:
                temp = ripp * ripp - (zpc - z) * (zpc - z) - (ypb - y) * (ypb - y)
                if temp <= 0:
                    continue
                ex = x + np.sqrt(temp)
                xpas = np.arange(2.0 * x - ex, ex + 0.001, 1.0)
                for xpa in xpas:
                    zpc = np.int32(zpc)
                    ypb = np.int32(ypb)
                    xpa = np.int32(xpa)
                    if (
                        0 <= zpc < nz
                        and 0 <= ypb < ny
                        and 0 <= xpa < nx
                        and not (zpc == 0 and ypb == 0 and xpa == 0)
                    ):
                        if isball[zpc, ypb, xpa]:
                            vj = whichball[zpc * ny * nx + ypb * nx + xpa]
                            competeForParent(
                                np.int32(i),
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
