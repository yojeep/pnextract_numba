import numpy as np
import numba as nb
from numpy import linalg as LA


# avgR = 4.225585
# _minRp = min(1.25, avgR * 0.25) + 0.5
# _clipROutx = 0.05
# _clipROutyz = 0.98
# _midRf = 0.7
# _MSNoise = 1.0 * abs(_minRp) + 1.0
# _lenNf = 0.6
# _vmvRadRelNf = 1.1
# _nRSmoothing = 3
# _RCorsnf = 0.15
# _RCorsn = abs(_minRp)
_mp5 = -0.5


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
    zs = np.arange(0, nz, 2)
    ys = np.arange(0, ny, 2)
    xs = np.arange(0, nx, 2)
    for z_id in nb.prange(len(zs)):
        z = zs[z_id]
        for y_id in nb.prange(len(ys)):
            y = ys[y_id]
            for x_id in nb.prange(len(xs)):
                x = xs[x_id]
                if image[z, y, x]:
                    max_val = -np.inf
                    max_index = np.array([0, 0, 0])
                    for i in range(2):
                        for j in range(2):
                            for k in range(2):
                                value = dt[z + i, y + j, x + k]
                                if value > max_val:
                                    max_val = value
                                    max_index[0] = z + i
                                    max_index[1] = y + j
                                    max_index[2] = x + k
                    if max_val > _minRp:
                        isball[max_index[0], max_index[1], max_index[2]] = True


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
def paradox_removeincludedballI(
    ball_indices, ball_R, image, dt, isball, _RCorsnf, _RCorsn, _MSNoise
):
    removed_ball = 0
    for i in range(ball_indices.shape[0]):
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
            if temp >= 0:
                ey = int(np.sqrt(temp))
                for b in range(-ey, ey + 1):
                    b_sq = b * b
                    temp = ripinc_sq - c_sq - b_sq
                    if temp >= 0:
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


def get_sorted_ball_indices_ball_R(dt, isball):
    balls_indices = np.column_stack(np.where(isball))
    R = dt[balls_indices[:, 0], balls_indices[:, 1], balls_indices[:, 2]]
    sorted_indices = np.argsort(R)[::-1]
    balls_indices = balls_indices[sorted_indices]
    R = R[sorted_indices]
    return balls_indices, R


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
def moveUphill(ball_indices, ball_findices, ball_R, image, dt):
    nz, ny, nx = image.shape
    for i in range(ball_indices.shape[0]):
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
    z,
    y,
    x,
    zpc,
    ypb,
    xpa,
    ball_indices,
    ball_findices,
    ball_R,
    ball_boss,
    image,
    dt,
    isball,
    whichball,
    _MSNoise,
    _midRf,
    _vmvRadRelNf,
    _lenNf,
):
    nz, ny, nx = image.shape
    noise = _MSNoise
    vi = whichball[z, y, x]
    vj = whichball[zpc, ypb, xpa]
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
    whichball,
    _MSNoise,
    _midRf,
    _vmvRadRelNf,
    _lenNf,
):
    nz, ny, nx = image.shape
    for i in range(ball_indices.shape[0]):
        z, y, x = ball_findices[i]
        ripp = ball_R[i] * 0.6 + 2.0 * _MSNoise + 2.0
        ez = z + ripp
        zpcs = np.arange(2.0 * z - ez, ez + 0.001, 1.0)
        for zpc in zpcs:
            temp = ripp * ripp - (zpc - z) * (zpc - z)
            if temp > 0:
                ey = y + np.sqrt(temp)
                ypbs = np.arange(2.0 * y - ey, ey + 0.001, 1.0)
                for ypb in ypbs:
                    temp = ripp * ripp - (zpc - z) * (zpc - z) - (ypb - y) * (ypb - y)
                    if temp >= 0:
                        ex = x + np.sqrt(temp)
                        xpas = np.arange(2.0 * x - ex, ex + 0.001, 1.0)
                        for xpa in xpas:
                            zpc = int(zpc)
                            ypb = int(ypb)
                            xpa = int(xpa)
                            z_int = int(z)
                            y_int = int(y)
                            x_int = int(x)
                            if 0 <= zpc < nz and 0 <= ypb < ny and 0 <= xpa < nx:
                                if isball[zpc, ypb, xpa] and (
                                    zpc != z_int or ypb != y_int or xpa != x_int
                                ):
                                    competeForParent(
                                        z_int,
                                        y_int,
                                        x_int,
                                        zpc,
                                        ypb,
                                        xpa,
                                        ball_indices,
                                        ball_findices,
                                        ball_R,
                                        ball_master,
                                        image,
                                        dt,
                                        isball,
                                        whichball,
                                        _MSNoise,
                                        _midRf,
                                        _vmvRadRelNf,
                                        _lenNf,
                                    )


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
def CreateVElem(
    image_VElems, dt_VElems, isball_VElems, ball_findices, ball_R, ball_boss
):
    uasyned = -1
    firstPores = 0
    nz, ny, nx = image_VElems.shape
    VElems = np.full((nz, ny, nx), -1, dtype=np.int32)
    poreIs = np.where(ball_boss == np.arange(ball_findices.shape[0]))[0]
    # print(poreIs)
    lastPores = len(poreIs) - 1
    firstPoreInd = firstPores
    lastPoreInd = lastPores
    for ind in nb.prange(len(poreIs)):
        pore_ind = poreIs[ind]
        z = int(ball_findices[pore_ind, 0] + 1)
        y = int(ball_findices[pore_ind, 1] + 1)
        x = int(ball_findices[pore_ind, 2] + 1)
        VElems[z, y, x] = ind

    for ball_index in range(ball_findices.shape[0]):
        # if ball_boss[ball_index] == ball_index:
        #     continue
        masterball = get_masterball(ball_boss, ball_index)
        cpmi = ball_findices[masterball, 0]
        bpmi = ball_findices[masterball, 1]
        apmi = ball_findices[masterball, 2]
        VElemV = VElems[int(cpmi + 1), int(bpmi + 1), int(apmi + 1)]
        assert 0 <= VElemV < len(poreIs), f"Invalid VElemV: {VElemV}"
        z = ball_findices[ball_index, 0]
        y = ball_findices[ball_index, 1]
        x = ball_findices[ball_index, 2]
        R = ball_R[ball_index]
        r2 = int((max(R * 0.25 - 1.0, 1.001)) ** 2)
        ez = np.sqrt(r2)
        z_start = max(z - ez, 0.5)
        z_end = min(z + ez, nz - 0.5) + 0.001
        zpcss = np.arange(z_start, z_end, 1.0)
        for zpc in zpcss:
            temp = r2 - (zpc - z) * (zpc - z)
            temp = max(temp, 0.0)
            ey = np.sqrt(temp)
            y_start = max(y - ey, 0.5)
            y_end = min(y + ey, ny - 0.5) + 0.001
            ypbs = np.arange(y_start, y_end, 1.0)
            for ypb in ypbs:
                temp = r2 - (zpc - z) * (zpc - z) - (ypb - y) * (ypb - y)
                temp = max(temp, 0.0)
                ex = np.sqrt(temp)
                x_start = max(x - ex, 0.5)
                x_end = min(x + ex, nx - 0.5) + 0.001
                xpas = np.arange(x_start, x_end, 1.0)
                for xpa in xpas:
                    zpc_VE = int(zpc + 1)
                    ypb_VE = int(ypb + 1)
                    xpa_VE = int(xpa + 1)
                    if 0 <= zpc_VE < nz and 0 <= ypb_VE < ny and 0 <= xpa_VE < nx:
                        if image_VElems[zpc_VE, ypb_VE, xpa_VE]:
                            idj = VElems[zpc_VE, ypb_VE, xpa_VE]
                            if idj == -1:
                                VElems[zpc_VE, ypb_VE, xpa_VE] = VElemV
                            elif VElemV != idj and (firstPoreInd <= idj <= lastPoreInd):

                                if (
                                    ~isball_VElems[zpc_VE, ypb_VE, xpa_VE]
                                    and dt_VElems[zpc_VE, ypb_VE, xpa_VE] < R
                                ):
                                    mvj = poreIs[idj]
                                    cmi = zpc - ball_findices[masterball, 0]
                                    bmi = ypb - ball_findices[masterball, 1]
                                    ami = xpa - ball_findices[masterball, 2]
                                    cmj = zpc - ball_findices[mvj, 0]
                                    bmj = ypb - ball_findices[mvj, 1]
                                    amj = xpa - ball_findices[mvj, 2]
                                    if (
                                        cmi * cmi + bmi * bmi + ami * ami
                                        < cmj * cmj + bmj * bmj + amj * amj
                                    ):
                                        VElems[zpc_VE, ypb_VE, xpa_VE] = VElemV
    return VElems, poreIs


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def grow_pores(image_VElems, VElems, bgn, lst, por_value):
    voxls = VElems.copy()
    n_changes = 0
    nz, ny, nx = image_VElems.shape
    # 遍历内部体素（排除边界）
    for k in nb.prange(1, nx - 1):
        for j in nb.prange(1, ny - 1):
            for i in nb.prange(1, nz - 1):
                if VElems[i, j, k] == por_value and image_VElems[i, j, k]:
                    if bgn <= voxls[i + 1, j, k] <= lst:
                        VElems[i, j, k] = voxls[i + 1, j, k]
                        n_changes += 1
                    elif bgn <= voxls[i - 1, j, k] <= lst:
                        VElems[i, j, k] = voxls[i - 1, j, k]
                        n_changes += 1
                    elif bgn <= voxls[i, j + 1, k] <= lst:
                        VElems[i, j, k] = voxls[i, j + 1, k]
                        n_changes += 1
                    elif bgn <= voxls[i, j - 1, k] <= lst:
                        VElems[i, j, k] = voxls[i, j - 1, k]
                        n_changes += 1
                    elif bgn <= voxls[i, j, k + 1] <= lst:
                        VElems[i, j, k] = voxls[i, j, k + 1]
                        n_changes += 1
                    elif bgn <= voxls[i, j, k - 1] <= lst:
                        VElems[i, j, k] = voxls[i, j, k - 1]
                        n_changes += 1
    print(f"ngrowPors changes: {n_changes}")
    return VElems


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def grow_pores_X2(image_VElems, VElems, bgn, lst, por_value):
    n_changes_total = 0
    nz, ny, nx = image_VElems.shape
    # 第一次遍历：正向扫描
    voxls = VElems.copy()
    n_changes = 0
    for k in nb.prange(1, nx - 1):
        for j in nb.prange(1, ny - 1):
            for i in nb.prange(1, nz - 1):
                if VElems[i, j, k] == por_value and image_VElems[i, j, k]:
                    if bgn <= voxls[i + 1, j, k] <= lst:
                        VElems[i, j, k] = voxls[i + 1, j, k]
                        n_changes += 1
                    elif bgn <= voxls[i - 1, j, k] <= lst:
                        VElems[i, j, k] = voxls[i - 1, j, k]
                        n_changes += 1
                    elif bgn <= voxls[i, j + 1, k] <= lst:
                        VElems[i, j, k] = voxls[i, j + 1, k]
                        n_changes += 1
                    elif bgn <= voxls[i, j - 1, k] <= lst:
                        VElems[i, j, k] = voxls[i, j - 1, k]
                        n_changes += 1
                    elif bgn <= voxls[i, j, k + 1] <= lst:
                        VElems[i, j, k] = voxls[i, j, k + 1]
                        n_changes += 1
                    elif bgn <= voxls[i, j, k - 1] <= lst:
                        VElems[i, j, k] = voxls[i, j, k - 1]
                        n_changes += 1
    print(f"  ngrowX3:{n_changes},")
    n_changes_total += n_changes
    n_changes = 0
    voxls = VElems.copy()
    for k in nb.prange(1, nx - 1):
        for j in nb.prange(1, ny - 1):
            for i in nb.prange(1, nz - 1):
                if VElems[i, j, k] == por_value and image_VElems[i, j, k]:
                    if bgn <= voxls[i + 1, j, k] <= lst:
                        VElems[i, j, k] = voxls[i + 1, j, k]
                        n_changes += 1
                    elif bgn <= voxls[i - 1, j, k] <= lst:
                        VElems[i, j, k] = voxls[i - 1, j, k]
                        n_changes += 1
                    elif bgn <= voxls[i, j + 1, k] <= lst:
                        VElems[i, j, k] = voxls[i, j + 1, k]
                        n_changes += 1
                    elif bgn <= voxls[i, j - 1, k] <= lst:
                        VElems[i, j, k] = voxls[i, j - 1, k]
                        n_changes += 1
                    elif bgn <= voxls[i, j, k + 1] <= lst:
                        VElems[i, j, k] = voxls[i, j, k + 1]
                        n_changes += 1
                    elif bgn <= voxls[i, j, k - 1] <= lst:
                        VElems[i, j, k] = voxls[i, j, k - 1]
                        n_changes += 1
    print(f"{n_changes},")
    n_changes_total += n_changes
    n_changes = 0
    voxls = VElems.copy()
    k_s = np.arange(nx - 2, 1, -1)
    j_s = np.arange(ny - 2, 1, -1)
    i_s = np.arange(nz - 2, 1, -1)
    for k_ind in nb.prange(len(k_s)):
        k = k_s[k_ind]
        for j_ind in nb.prange(len(j_s)):
            j = j_s[j_ind]
            for i_ind in nb.prange(len(i_s)):
                i = i_s[i_ind]
                if image_VElems[i, j, k]:
                    if VElems[i, j, k] == por_value:
                        if bgn <= voxls[i + 1, j, k] <= lst:
                            VElems[i, j, k] = voxls[i + 1, j, k]
                            n_changes += 1
                        elif bgn <= voxls[i - 1, j, k] <= lst:
                            VElems[i, j, k] = voxls[i - 1, j, k]
                            n_changes += 1
                        elif bgn <= voxls[i, j + 1, k] <= lst:
                            VElems[i, j, k] = voxls[i, j + 1, k]
                            n_changes += 1
                        elif bgn <= voxls[i, j - 1, k] <= lst:
                            VElems[i, j, k] = voxls[i, j - 1, k]
                            n_changes += 1
                        elif bgn <= voxls[i, j, k + 1] <= lst:
                            VElems[i, j, k] = voxls[i, j, k + 1]
                            n_changes += 1
                        elif bgn <= voxls[i, j, k - 1] <= lst:
                            VElems[i, j, k] = voxls[i, j, k - 1]
                            n_changes += 1
    print(f"  ngrowX2:{n_changes}  ")
    n_changes_total += n_changes

    return VElems, n_changes


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def grow_pores_med_strict(image_VElems, dt_VElems, VElems, bgn, lst, raw_value):
    voxls = VElems.copy()
    n_changes = 0
    nz, ny, nx = image_VElems.shape
    for k in nb.prange(1, nx - 1):
        for j in nb.prange(1, ny - 1):
            for i in nb.prange(1, nz - 1):
                if image_VElems[i, j, k]:
                    pID = voxls[k, j, i]
                    if pID == raw_value:
                        R = dt_VElems[i, j, k]
                        neIs = np.full(6, -1, dtype=np.int32)
                        nDifferentID = 0
                        V_current = voxls[i - 1, j, k]
                        R_current = dt_VElems[i - 1, j, k]
                        if bgn <= V_current <= lst:
                            if R_current >= R:
                                nDifferentID += 1
                            if R_current > R:
                                neIs[0] = V_current
                        V_current = voxls[i + 1, j, k]
                        R_current = dt_VElems[i + 1, j, k]
                        if bgn <= V_current <= lst:
                            if R_current >= R:
                                nDifferentID += 1
                            if R_current > R:
                                neIs[1] = V_current
                        V_current = voxls[i, j - 1, k]
                        R_current = dt_VElems[i, j - 1, k]
                        if bgn <= V_current <= lst:
                            if R_current >= R:
                                nDifferentID += 1
                            if R_current > R:
                                neIs[2] = V_current
                        V_current = voxls[i, j + 1, k]
                        R_current = dt_VElems[i, j + 1, k]
                        if bgn <= V_current <= lst:
                            if R_current >= R:
                                nDifferentID += 1
                            if R_current > R:
                                neIs[3] = V_current
                        V_current = voxls[i, j, k - 1]
                        R_current = dt_VElems[i, j, k - 1]
                        if bgn <= V_current <= lst:
                            if R_current >= R:
                                nDifferentID += 1
                            if R_current > R:
                                neIs[4] = V_current
                        V_current = voxls[i, j, k + 1]
                        R_current = dt_VElems[i, j, k + 1]
                        if bgn <= V_current <= lst:
                            if R_current >= R:
                                nDifferentID += 1
                            if R_current > R:
                                neIs[5] = V_current

                        if nDifferentID >= 3:
                            neI_pos = neIs[neIs >= 0]
                            if neI_pos.size > 0:
                                max_count_nei, max_count = get_max_count_nei(neI_pos)
                                if max_count >= 3:
                                    VElems[i, j, k] = max_count_nei
                                    n_changes += 1

    print(f"ngMedStrict changes: {n_changes}")
    return voxls


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def grow_pores_median(image_VElems, dt_VElems, VElems, bgn, lst, raw_value):
    voxls = VElems.copy()
    n_changes = 0
    nz, ny, nx = image_VElems.shape
    for k in nb.prange(1, nx - 1):
        for j in nb.prange(1, ny - 1):
            for i in nb.prange(1, nz - 1):
                if image_VElems[i, j, k]:
                    pID = voxls[k, j, i]
                    if pID == raw_value:
                        R = dt_VElems[i, j, k]
                        neIs = np.full(6, -1, dtype=np.int32)
                        nDifferentID = 0
                        V_current = voxls[i - 1, j, k]
                        R_current = dt_VElems[i - 1, j, k]
                        if bgn <= V_current <= lst:
                            if R_current > R:
                                nDifferentID += 1
                                neIs[0] = V_current
                        V_current = voxls[i + 1, j, k]
                        R_current = dt_VElems[i + 1, j, k]
                        if bgn <= V_current <= lst:
                            if R_current > R:
                                nDifferentID += 1
                                neIs[1] = V_current
                        V_current = voxls[i, j - 1, k]
                        R_current = dt_VElems[i, j - 1, k]
                        if bgn <= V_current <= lst:
                            if R_current > R:
                                nDifferentID += 1
                                neIs[2] = V_current
                        V_current = voxls[i, j + 1, k]
                        R_current = dt_VElems[i, j + 1, k]
                        if bgn <= V_current <= lst:
                            if R_current > R:
                                nDifferentID += 1
                                neIs[3] = V_current
                        V_current = voxls[i, j, k - 1]
                        R_current = dt_VElems[i, j, k - 1]
                        if bgn <= V_current <= lst:
                            if R_current > R:
                                nDifferentID += 1
                                neIs[4] = V_current
                        V_current = voxls[i, j, k + 1]
                        R_current = dt_VElems[i, j, k + 1]
                        if bgn <= V_current <= lst:
                            if R_current > R:
                                nDifferentID += 1
                                neIs[5] = V_current

                        if nDifferentID >= 2:
                            neI_pos = neIs[neIs >= 0]
                            if neI_pos.size > 0:
                                max_count_nei, max_count = get_max_count_nei(neI_pos)
                                if max_count >= 2:
                                    VElems[i, j, k] = max_count_nei
                                    n_changes += 1

    print(f"ngMedian changes: {n_changes}")
    return VElems


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def grow_pores_med_eqs(image_VElems, dt_VElems, VElems, bgn, lst, raw_value):
    voxls = VElems.copy()
    n_changes = 0
    nz, ny, nx = image_VElems.shape
    for k in nb.prange(1, nx - 1):
        for j in nb.prange(1, ny - 1):
            for i in nb.prange(1, nz - 1):
                if image_VElems[i, j, k]:
                    pID = voxls[k, j, i]
                    if pID == raw_value:
                        R = dt_VElems[i, j, k]
                        neIs = np.full(6, -1, dtype=np.int32)
                        nDifferentID = 0
                        V_current = voxls[i - 1, j, k]
                        R_current = dt_VElems[i - 1, j, k]
                        if bgn <= V_current <= lst:
                            if R_current >= R:
                                nDifferentID += 1
                                neIs[0] = V_current
                        V_current = voxls[i + 1, j, k]
                        R_current = dt_VElems[i + 1, j, k]
                        if bgn <= V_current <= lst:
                            if R_current >= R:
                                nDifferentID += 1
                                neIs[1] = V_current
                        V_current = voxls[i, j - 1, k]
                        R_current = dt_VElems[i, j - 1, k]
                        if bgn <= V_current <= lst:
                            if R_current >= R:
                                nDifferentID += 1
                                neIs[2] = V_current
                        V_current = voxls[i, j + 1, k]
                        R_current = dt_VElems[i, j + 1, k]
                        if bgn <= V_current <= lst:
                            if R_current >= R:
                                nDifferentID += 1
                                neIs[3] = V_current
                        V_current = voxls[i, j, k - 1]
                        R_current = dt_VElems[i, j, k - 1]
                        if bgn <= V_current <= lst:
                            if R_current >= R:
                                nDifferentID += 1
                                neIs[4] = V_current
                        V_current = voxls[i, j, k + 1]
                        R_current = dt_VElems[i, j, k + 1]
                        if bgn <= V_current <= lst:
                            if R_current >= R:
                                nDifferentID += 1
                                neIs[5] = V_current

                        if nDifferentID >= 2:
                            neI_pos = neIs[neIs >= 0]
                            if neI_pos.size > 0:
                                max_count_nei, max_count = get_max_count_nei(neI_pos)
                                if max_count >= 2:
                                    VElems[i, j, k] = max_count_nei
                                    n_changes += 1

    print(f"ngMedEqs changes: {n_changes}")
    return VElems


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def grow_pores_med_eqs_loose(image_VElems, VElems, bgn, lst, raw_value):
    voxls = VElems.copy()
    n_changes = 0
    nz, ny, nx = image_VElems.shape
    for k in nb.prange(1, nx - 1):
        for j in nb.prange(1, ny - 1):
            for i in nb.prange(1, nz - 1):
                if image_VElems[i, j, k]:
                    pID = voxls[k, j, i]
                    if pID == raw_value:
                        neIs = np.full(6, -1, dtype=np.int32)
                        nDifferentID = 0
                        V_current = voxls[i - 1, j, k]
                        if bgn <= V_current <= lst:
                            nDifferentID += 1
                            neIs[0] = V_current
                        V_current = voxls[i + 1, j, k]
                        if bgn <= V_current <= lst:
                            nDifferentID += 1
                            neIs[1] = V_current
                        V_current = voxls[i, j - 1, k]
                        if bgn <= V_current <= lst:
                            nDifferentID += 1
                            neIs[2] = V_current
                        V_current = voxls[i, j + 1, k]
                        if bgn <= V_current <= lst:
                            nDifferentID += 1
                            neIs[3] = V_current
                        V_current = voxls[i, j, k - 1]
                        if bgn <= V_current <= lst:
                            nDifferentID += 1
                            neIs[4] = V_current
                        V_current = voxls[i, j, k + 1]
                        if bgn <= V_current <= lst:
                            nDifferentID += 1
                            neIs[5] = V_current

                        if nDifferentID >= 2:
                            neI_pos = neIs[neIs >= 0]
                            if neI_pos.size > 0:
                                max_count_nei, max_count = get_max_count_nei(neI_pos)
                                if max_count >= 2:
                                    VElems[i, j, k] = max_count_nei
                                    n_changes += 1

    print(f"ngMedLoose changes: {n_changes}")
    return VElems


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def median_elem(VElems, bgn, lst):
    voxls = VElems.copy()
    n_changes = 0
    nz, ny, nx = VElems.shape
    for k in nb.prange(1, nx - 1):
        for j in nb.prange(1, ny - 1):
            for i in nb.prange(1, nz - 1):
                pID = voxls[i, j, k]
                if bgn <= pID <= lst:
                    n_same = 0
                    n_diff = 0
                    neIs = np.full(6, -1, dtype=np.int32)
                    V_current = voxls[i - 1, j, k]
                    if V_current == pID:
                        n_same += 1
                    elif bgn <= V_current <= lst:
                        n_diff += 1
                        neIs[0] = V_current
                    V_current = voxls[i + 1, j, k]
                    if V_current == pID:
                        n_same += 1
                    elif bgn <= V_current <= lst:
                        n_diff += 1
                        neIs[1] = V_current
                    V_current = voxls[i, j - 1, k]
                    if V_current == pID:
                        n_same += 1
                    elif bgn <= V_current <= lst:
                        n_diff += 1
                        neIs[2] = V_current
                    V_current = voxls[i, j + 1, k]
                    if V_current == pID:
                        n_same += 1
                    elif bgn <= V_current <= lst:
                        n_diff += 1
                        neIs[3] = V_current
                    V_current = voxls[i, j, k - 1]
                    if V_current == pID:
                        n_same += 1
                    elif bgn <= V_current <= lst:
                        n_diff += 1
                        neIs[4] = V_current
                    V_current = voxls[i, j, k + 1]
                    if V_current == pID:
                        n_same += 1
                    elif bgn <= V_current <= lst:
                        n_diff += 1
                        neIs[5] = V_current

                    neis_pos = neIs[neIs >= 0]
                    if n_diff > n_same and neis_pos.size > 0:
                        max_count_nei, max_count = get_max_count_nei(neis_pos)
                        if max_count > n_same:
                            VElems[i, j, k] = max_count_nei
                            n_changes += 1

    print(f"nMedian: {n_changes}")
    return VElems


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def retreat_pores_median(image_VElems, VElems, bgn, lst, unassigned):
    voxls = VElems.copy()
    n_changes = 0
    nz, ny, nx = image_VElems.shape

    for k in nb.prange(1, nx - 1):
        for j in nb.prange(1, ny - 1):
            for i in nb.prange(1, nz - 1):
                pID = voxls[i, j, k]
                if bgn <= pID <= lst:
                    nSameID = 0
                    nDiffereID = 0
                    V_current = voxls[i - 1, j, k]
                    if V_current == pID:
                        nSameID += 1
                    elif bgn <= V_current <= lst:
                        nDiffereID += 1
                    V_current = voxls[i + 1, j, k]
                    if V_current == pID:
                        nSameID += 1
                    elif bgn <= V_current <= lst:
                        nDiffereID += 1
                    V_current = voxls[i, j - 1, k]
                    if V_current == pID:
                        nSameID += 1
                    elif bgn <= V_current <= lst:
                        nDiffereID += 1
                    V_current = voxls[i, j + 1, k]
                    if V_current == pID:
                        nSameID += 1
                    elif bgn <= V_current <= lst:
                        nDiffereID += 1
                    V_current = voxls[i, j, k - 1]
                    if V_current == pID:
                        nSameID += 1
                    elif bgn <= V_current <= lst:
                        nDiffereID += 1
                    V_current = voxls[i, j, k + 1]
                    if V_current == pID:
                        nSameID += 1
                    elif bgn <= V_current <= lst:
                        nDiffereID += 1

                    if nDiffereID > 0 and nSameID > 0:
                        VElems[i, j, k] = unassigned
                        n_changes += 1

    print(f"nRetreat: {n_changes}")
    return VElems


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
def refine_with_master_ball(VElems, ball_boss, ball_findices):
    voxls = VElems.copy()
    for i in nb.prange(ball_boss.size):
        ball_master = get_masterball(ball_boss, i)
        fzi, fyi, fxi = ball_findices[i]
        fzm, fym, fxm = ball_findices[ball_master]
        VElems[int(fzi + 1), int(fyi + 1), int(fxi + 1)] = voxls[
            int(fzm + 1), int(fym + 1), int(fxm + 1)
        ]
    return VElems


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
def get_max_count_nei(neis):
    # 降序排序（大的数在前）
    sorted_neis = np.sort(neis)[::-1]
    if len(sorted_neis) == 0:
        return -1, 0  # 处理空输入

    max_value = sorted_neis[0]  # 当前候选最大值
    max_count = 1  # 当前最大出现次数

    current_value = sorted_neis[0]  # 当前遍历的值
    current_count = 1  # 当前值的出现次数

    # 单次遍历统计
    for i in range(1, len(sorted_neis)):
        if sorted_neis[i] == current_value:
            current_count += 1
        else:
            # 当遇到新值时，比较并更新最大值
            if (current_count > max_count) or (
                current_count == max_count and current_value > max_value
            ):
                max_count = current_count
                max_value = current_value
            # 重置当前统计
            current_value = sorted_neis[i]
            current_count = 1

    # 处理最后一个元素的统计
    if (current_count > max_count) or (
        current_count == max_count and current_value > max_value
    ):
        max_count = current_count
        max_value = current_value

    return max_value, max_count


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def smooth_radius(dt, nz, ny, nx):
    # print("smoothing R", end='', flush=True)

    # 初始化delta R数组
    del_rrr = np.zeros_like(dt)
    # print("*", end='', flush=True)

    # 第一部分：计算delta R
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if dt[k, j, i] == 0:
                    continue  # 简化原代码中seg.value==0的判断

                sum_r = 0.0
                counter = 0

                # 3x3x3邻域遍历
                for kk in range(max(k - 1, 0), min(k + 2, nz)):
                    for jj in range(max(j - 1, 0), min(j + 2, ny)):
                        # 简化segment处理，直接取i-1到i+1范围
                        for ii in range(max(i - 1, 0), min(i + 2, nx)):
                            if dt[kk, jj, ii] == 0:
                                sum_r += dt[kk, jj, ii]
                                counter += 1

                # 计算delta R
                denominator = 3 * counter + 27
                if denominator == 0:
                    del_rrr[k, j, i] = 0
                else:
                    del_rrr[k, j, i] = 4 * sum_r / denominator - dt[k, j, i]
    # print("*", end='', flush=True)
    #
    # 第二部分：应用平滑
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if dt[k, j, i] == 0:
                    continue

                sum_del_r = 0.0
                counter = 0

                # 再次遍历邻域
                for kk in range(max(k - 1, 0), min(k + 2, nz)):
                    for jj in range(max(j - 1, 0), min(j + 2, ny)):
                        for ii in range(max(i - 1, 0), min(i + 2, nx)):
                            if dt[kk, jj, ii] == 0:
                                sum_del_r += del_rrr[kk, jj, ii]
                                counter += 1

                # 计算调整量
                if counter == 0:
                    adj = 0
                else:
                    term = 0.02 * (
                        del_rrr[k, j, i] - 0.99 * 2 * sum_del_r / (counter + 27)
                    )
                    adj = max(min(term, 0.01), -0.005)

                dt[k, j, i] += adj
    # print("*", end='', flush=True)

    # 计算最大半径
    max_r = np.max(dt)
    print(f" maxrrr {max_r}")

    return dt
