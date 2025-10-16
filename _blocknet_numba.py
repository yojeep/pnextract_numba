import numba as nb
import numpy as np
from _extraction_functions_numba import get_masterball


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
def get_max_count_nei(neis):
    n = neis.size
    max_val = -1
    max_count = 0

    # 对每个唯一值统计频次（暴力但高效，因为 n <= 6）
    for i in range(n):
        val = neis[i]
        if val == -1:
            continue
        count = 0
        for j in range(n):
            count += neis[j] == val
        # 更新规则：频次更高，或频次相同但值更大
        if count > max_count or (count == max_count and val > max_val):
            max_count = count
            max_val = val

    return max_val, max_count


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
def CreateVElem(img_bool, dt, isball, ball_indices, ball_findices, ball_R, ball_boss):
    raw_value = -1
    firstPores = 0
    nz, ny, nx = img_bool.shape
    VElems = np.full((nz + 2, ny + 2, nx + 2), raw_value, dtype=np.int32)
    poreIs = np.where(ball_boss == np.arange(ball_boss.shape[0]))[0]
    for ind in nb.prange(poreIs.size):
        pore_ind = poreIs[ind]
        z, y, x = ball_indices[pore_ind]
        z += 1
        y += 1
        x += 1
        VElems[z, y, x] = ind

    for ball_index in range(ball_findices.shape[0]):
        # if ball_boss[ball_index] == ball_index:
        #     continue
        masterball = get_masterball(ball_boss, ball_index)
        cpmi, bpmi, apmi = ball_indices[masterball]
        VElemV = VElems[cpmi + 1, bpmi + 1, apmi + 1]
        # assert 0 <= VElemV < len(poreIs), f"Invalid VElemV: {VElemV}"
        z, y, x = ball_findices[ball_index]
        R = ball_R[ball_index]
        r2 = int((max(R * 0.25 - 1.0, 1.001)) ** 2)
        ez = np.sqrt(r2)
        z_start = max(z - ez, 0.5)
        z_end = min(z + ez, nz - 0.5) + 0.001
        zpcss = np.arange(z_start, z_end, 1.0)
        for zpc in zpcss:
            temp = r2 - (zpc - z) * (zpc - z)
            if temp < 0:
                continue
            ey = np.sqrt(temp)
            y_start = max(y - ey, 0.5)
            y_end = min(y + ey, ny - 0.5) + 0.001
            ypbs = np.arange(y_start, y_end, 1.0)
            for ypb in ypbs:
                temp = r2 - (zpc - z) * (zpc - z) - (ypb - y) * (ypb - y)
                if temp < 0:
                    continue
                ex = np.sqrt(temp)
                x_start = max(x - ex, 0.5)
                x_end = min(x + ex, nx - 0.5) + 0.001
                xpas = np.arange(x_start, x_end, 1.0)
                for xpa in xpas:
                    zpci = int(zpc)
                    ypbi = int(ypb)
                    xpai = int(xpa)
                    zpci_VE = zpci + 1
                    ypbi_VE = ypbi + 1
                    xpai_VE = xpai + 1
                    if 0 <= zpci < nz and 0 <= ypbi < ny and 0 <= xpai < nx:
                        if img_bool[zpci, ypbi, xpai]:
                            idj = VElems[zpci_VE, ypbi_VE, xpai_VE]
                            if idj == raw_value:
                                VElems[zpci_VE, ypbi_VE, xpai_VE] = VElemV
                            elif VElemV != idj:
                                if (
                                    ~isball[zpci, ypbi, xpai]
                                    and dt[zpci, ypbi, xpai] < R
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
                                        VElems[zpci_VE, ypbi_VE, xpai_VE] = VElemV
    return VElems, poreIs


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def grow_pores(zsysxs_v, VElems, bgn, raw_value):
    nVxls = zsysxs_v.shape[0]
    voxls = VElems.copy()
    n_changes = 0
    for ipar in nb.prange(nVxls):
        i, j, k = zsysxs_v[ipar]
        if voxls[i, j, k] == raw_value:
            if bgn <= voxls[i, j, k + 1]:
                VElems[i, j, k] = voxls[i, j, k + 1]
                n_changes += 1
            elif bgn <= voxls[i, j, k - 1]:
                VElems[i, j, k] = voxls[i, j, k - 1]
                n_changes += 1
            elif bgn <= voxls[i, j + 1, k]:
                VElems[i, j, k] = voxls[i, j + 1, k]
                n_changes += 1
            elif bgn <= voxls[i, j - 1, k]:
                VElems[i, j, k] = voxls[i, j - 1, k]
                n_changes += 1
            elif bgn <= voxls[i + 1, j, k]:
                VElems[i, j, k] = voxls[i + 1, j, k]
                n_changes += 1
            elif bgn <= voxls[i - 1, j, k]:
                VElems[i, j, k] = voxls[i - 1, j, k]
                n_changes += 1

    print(f"ngrowPors changes: {n_changes}")
    return VElems


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def grow_pores_X2(zsysxs_v, VElems, bgn, raw_value):
    nVxls = zsysxs_v.shape[0]
    voxls = VElems.copy()
    # 第一次遍历：正向扫描
    n_changes = 0
    for ipar in nb.prange(nVxls):
        i, j, k = zsysxs_v[ipar]
        if voxls[i, j, k] == raw_value:
            if bgn <= voxls[i, j, k + 1]:
                VElems[i, j, k] = voxls[i, j, k + 1]
                n_changes += 1
            elif bgn <= voxls[i, j, k - 1]:
                VElems[i, j, k] = voxls[i, j, k - 1]
                n_changes += 1
            elif bgn <= voxls[i, j + 1, k]:
                VElems[i, j, k] = voxls[i, j + 1, k]
                n_changes += 1
            elif bgn <= voxls[i, j - 1, k]:
                VElems[i, j, k] = voxls[i, j - 1, k]
                n_changes += 1
            elif bgn <= voxls[i + 1, j, k]:
                VElems[i, j, k] = voxls[i + 1, j, k]
                n_changes += 1
            elif bgn <= voxls[i - 1, j, k]:
                VElems[i, j, k] = voxls[i - 1, j, k]
                n_changes += 1

    print(f"  ngrowX3:{n_changes},")

    n_changes = 0
    voxls = VElems.copy()
    for ipar in nb.prange(nVxls):
        i, j, k = zsysxs_v[ipar]
        if voxls[i, j, k] == raw_value:
            if bgn <= voxls[i, j, k + 1]:
                VElems[i, j, k] = voxls[i, j, k + 1]
                n_changes += 1
            elif bgn <= voxls[i, j, k - 1]:
                VElems[i, j, k] = voxls[i, j, k - 1]
                n_changes += 1
            elif bgn <= voxls[i, j + 1, k]:
                VElems[i, j, k] = voxls[i, j + 1, k]
                n_changes += 1
            elif bgn <= voxls[i, j - 1, k]:
                VElems[i, j, k] = voxls[i, j - 1, k]
                n_changes += 1
            elif bgn <= voxls[i + 1, j, k]:
                VElems[i, j, k] = voxls[i + 1, j, k]
                n_changes += 1
            elif bgn <= voxls[i - 1, j, k]:
                VElems[i, j, k] = voxls[i - 1, j, k]
                n_changes += 1

    print(f"{n_changes},")

    n_changes = 0
    voxls = VElems.copy()
    for ipar in nb.prange(nVxls):
        i, j, k = zsysxs_v[ipar]
        if voxls[i, j, k] == raw_value:
            if bgn <= voxls[i, j, k + 1]:
                VElems[i, j, k] = voxls[i, j, k + 1]
                n_changes += 1
            elif bgn <= voxls[i, j, k - 1]:
                VElems[i, j, k] = voxls[i, j, k - 1]
                n_changes += 1
            elif bgn <= voxls[i, j + 1, k]:
                VElems[i, j, k] = voxls[i, j + 1, k]
                n_changes += 1
            elif bgn <= voxls[i, j - 1, k]:
                VElems[i, j, k] = voxls[i, j - 1, k]
                n_changes += 1
            elif bgn <= voxls[i + 1, j, k]:
                VElems[i, j, k] = voxls[i + 1, j, k]
                n_changes += 1
            elif bgn <= voxls[i - 1, j, k]:
                VElems[i, j, k] = voxls[i - 1, j, k]
                n_changes += 1

    print(f"  ngrowX2:{n_changes}  ")

    return VElems, n_changes


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def grow_pores_med_strict(zsysxs_v, dt_p1, VElems, bgn, raw_value):
    nVxls = zsysxs_v.shape[0]
    voxls = VElems.copy()
    n_changes = 0
    for ipar in nb.prange(nVxls):
        i, j, k = zsysxs_v[ipar]
        pID = voxls[i, j, k]
        if pID == raw_value:
            R = dt_p1[i, j, k]
            neIs = np.array((-1, -1, -1, -1, -1, -1), dtype=np.int32)
            nDifferentID = 0
            V_current = voxls[i, j, k + 1]
            R_current = dt_p1[i, j, k + 1]
            if bgn <= V_current and R_current >= R:
                nDifferentID += 1
                if R_current > R:
                    neIs[0] = V_current
            V_current = voxls[i, j, k - 1]
            R_current = dt_p1[i, j, k - 1]
            if bgn <= V_current and R_current >= R:
                nDifferentID += 1
                if R_current > R:
                    neIs[1] = V_current
            V_current = voxls[i, j + 1, k]
            R_current = dt_p1[i, j + 1, k]
            if bgn <= V_current and R_current >= R:
                nDifferentID += 1
                if R_current > R:
                    neIs[2] = V_current
            V_current = voxls[i, j - 1, k]
            R_current = dt_p1[i, j - 1, k]
            if bgn <= V_current and R_current >= R:
                nDifferentID += 1
                if R_current > R:
                    neIs[3] = V_current
            V_current = voxls[i + 1, j, k]
            R_current = dt_p1[i + 1, j, k]
            if bgn <= V_current and R_current >= R:
                nDifferentID += 1
                if R_current > R:
                    neIs[4] = V_current
            V_current = voxls[i - 1, j, k]
            R_current = dt_p1[i - 1, j, k]
            if bgn <= V_current and R_current >= R:
                nDifferentID += 1
                if R_current > R:
                    neIs[5] = V_current

            if nDifferentID >= 3:
                max_count_nei, max_count = get_max_count_nei(neIs)
                if max_count >= 3:
                    VElems[i, j, k] = max_count_nei
                    n_changes += 1

    print(f"ngMedStrict changes: {n_changes}")
    return voxls


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def grow_pores_median(zsysxs_v, dt_p1, VElems, bgn, raw_value):
    nVxls = zsysxs_v.shape[0]
    voxls = VElems.copy()
    n_changes = 0
    for ipar in nb.prange(nVxls):
        i, j, k = zsysxs_v[ipar]
        pID = voxls[i, j, k]
        if pID == raw_value:
            R = dt_p1[i, j, k]
            neIs = np.array((-1, -1, -1, -1, -1, -1), dtype=np.int32)
            nDifferentID = 0
            V_current = voxls[i, j, k + 1]
            R_current = dt_p1[i, j, k + 1]
            if bgn <= V_current and R_current > R:
                nDifferentID += 1
                neIs[0] = V_current
            V_current = voxls[i, j, k - 1]
            R_current = dt_p1[i, j, k - 1]
            if bgn <= V_current and R_current > R:
                nDifferentID += 1
                neIs[1] = V_current
            V_current = voxls[i, j + 1, k]
            R_current = dt_p1[i, j + 1, k]
            if bgn <= V_current and R_current > R:
                nDifferentID += 1
                neIs[2] = V_current
            V_current = voxls[i, j - 1, k]
            R_current = dt_p1[i, j - 1, k]
            if bgn <= V_current and R_current > R:
                nDifferentID += 1
                neIs[3] = V_current
            V_current = voxls[i + 1, j, k]
            R_current = dt_p1[i + 1, j, k]
            if bgn <= V_current and R_current > R:
                nDifferentID += 1
                neIs[4] = V_current
            V_current = voxls[i - 1, j, k]
            R_current = dt_p1[i - 1, j, k]
            if bgn <= V_current and R_current > R:
                nDifferentID += 1
                neIs[5] = V_current

            if nDifferentID >= 2:
                max_count_nei, max_count = get_max_count_nei(neIs)
                if max_count >= 2:
                    VElems[i, j, k] = max_count_nei
                    n_changes += 1

    print(f"ngMedian changes: {n_changes}")
    return VElems


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def grow_pores_med_eqs(zsysxs_v, dt_p1, VElems, bgn, raw_value):
    nVxls = zsysxs_v.shape[0]
    voxls = VElems.copy()
    n_changes = 0
    for ipar in nb.prange(nVxls):
        i, j, k = zsysxs_v[ipar]
        pID = voxls[i, j, k]
        if pID == raw_value:
            R = dt_p1[i, j, k]
            neIs = np.array((-1, -1, -1, -1, -1, -1), dtype=np.int32)
            nDifferentID = 0
            V_current = voxls[i, j, k + 1]
            R_current = dt_p1[i, j, k + 1]
            if bgn <= V_current and R_current >= R:
                nDifferentID += 1
                neIs[0] = V_current
            V_current = voxls[i, j, k - 1]
            R_current = dt_p1[i, j, k - 1]
            if bgn <= V_current and R_current >= R:
                nDifferentID += 1
                neIs[1] = V_current
            V_current = voxls[i, j + 1, k]
            R_current = dt_p1[i, j + 1, k]
            if bgn <= V_current and R_current >= R:
                nDifferentID += 1
                neIs[2] = V_current
            V_current = voxls[i, j - 1, k]
            R_current = dt_p1[i, j - 1, k]
            if bgn <= V_current and R_current >= R:
                nDifferentID += 1
                neIs[3] = V_current
            V_current = voxls[i + 1, j, k]
            R_current = dt_p1[i + 1, j, k]
            if bgn <= V_current and R_current >= R:
                nDifferentID += 1
                neIs[4] = V_current
            V_current = voxls[i - 1, j, k]
            R_current = dt_p1[i - 1, j, k]
            if bgn <= V_current and R_current >= R:
                nDifferentID += 1
                neIs[5] = V_current

            if nDifferentID >= 2:
                max_count_nei, max_count = get_max_count_nei(neIs)
                if max_count >= 2:
                    VElems[i, j, k] = max_count_nei
                    n_changes += 1

    print(f"ngMedEqs changes: {n_changes}")
    return VElems


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def grow_pores_med_eqs_loose(zsysxs_v, VElems, bgn, raw_value):
    nVxls = zsysxs_v.shape[0]
    voxls = VElems.copy()
    n_changes = 0
    for ipar in nb.prange(nVxls):
        i, j, k = zsysxs_v[ipar]
        pID = voxls[i, j, k]
        if pID == raw_value:
            nDifferentID = 0
            neIs = np.array((-1, -1, -1, -1, -1, -1), dtype=np.int32)
            V_current = voxls[i, j, k + 1]
            if bgn <= V_current:
                nDifferentID += 1
                neIs[0] = V_current
            V_current = voxls[i, j, k - 1]
            if bgn <= V_current:
                nDifferentID += 1
                neIs[1] = V_current
            V_current = voxls[i, j - 1, k]
            if bgn <= V_current:
                nDifferentID += 1
                neIs[2] = V_current
            V_current = voxls[i, j + 1, k]
            if bgn <= V_current:
                nDifferentID += 1
                neIs[3] = V_current
            V_current = voxls[i + 1, j, k]
            if bgn <= V_current:
                nDifferentID += 1
                neIs[4] = V_current
            V_current = voxls[i - 1, j, k]
            if bgn <= V_current:
                nDifferentID += 1
                neIs[5] = V_current

            if nDifferentID >= 2:
                max_count_nei, max_count = get_max_count_nei(neIs)
                if max_count >= 2:
                    VElems[i, j, k] = max_count_nei
                    n_changes += 1

    print(f"ngMedLoose changes: {n_changes}")
    return VElems


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def median_elem(zsysxs_v, VElems, bgn):
    nVxls = zsysxs_v.shape[0]
    voxls = VElems.copy()
    n_changes = 0

    for ipar in nb.prange(nVxls):
        i, j, k = zsysxs_v[ipar]
        pID = voxls[i, j, k]
        if bgn <= pID:
            n_same = 0
            n_diff = 0
            neIs = np.array((-1, -1, -1, -1, -1, -1), dtype=np.int32)
            V_current = voxls[i, j, k + 1]
            if V_current == pID:
                n_same += 1
            elif bgn <= V_current:
                n_diff += 1
                neIs[0] = V_current
            V_current = voxls[i, j, k - 1]
            if V_current == pID:
                n_same += 1
            elif bgn <= V_current:
                n_diff += 1
                neIs[1] = V_current
            V_current = voxls[i, j + 1, k]
            if V_current == pID:
                n_same += 1
            elif bgn <= V_current:
                n_diff += 1
                neIs[2] = V_current
            V_current = voxls[i, j - 1, k]
            if V_current == pID:
                n_same += 1
            elif bgn <= V_current:
                n_diff += 1
                neIs[3] = V_current
            V_current = voxls[i + 1, j, k]
            if V_current == pID:
                n_same += 1
            elif bgn <= V_current:
                n_diff += 1
                neIs[4] = V_current
            V_current = voxls[i - 1, j, k]
            if V_current == pID:
                n_same += 1
            elif bgn <= V_current:
                n_diff += 1
                neIs[5] = V_current
            if n_diff > n_same:
                max_count_nei, max_count = get_max_count_nei(neIs)
                if max_count > n_same:
                    VElems[i, j, k] = max_count_nei
                    n_changes += 1

    print(f"nMedian: {n_changes}")
    return VElems


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def retreat_pores_median(zsysxs_v, VElems, bgn, raw_value):
    nVxls = zsysxs_v.shape[0]
    voxls = VElems.copy()
    n_changes = 0
    for ipar in nb.prange(nVxls):
        i, j, k = zsysxs_v[ipar]
        pID = voxls[i, j, k]
        if bgn <= pID:
            nSameID = 0
            nDiffereID = 0
            V_current = voxls[i, j, k + 1]
            if V_current == pID:
                nSameID += 1
            elif bgn <= V_current:
                nDiffereID += 1
            V_current = voxls[i, j, k - 1]
            if V_current == pID:
                nSameID += 1
            elif bgn <= V_current:
                nDiffereID += 1
            V_current = voxls[i, j + 1, k]
            if V_current == pID:
                nSameID += 1
            elif bgn <= V_current:
                nDiffereID += 1
            V_current = voxls[i, j - 1, k]
            if V_current == pID:
                nSameID += 1
            elif bgn <= V_current:
                nDiffereID += 1
            V_current = voxls[i + 1, j, k]
            if V_current == pID:
                nSameID += 1
            elif bgn <= V_current:
                nDiffereID += 1
            V_current = voxls[i - 1, j, k]
            if V_current == pID:
                nSameID += 1
            elif bgn <= V_current:
                nDiffereID += 1

            if nDiffereID > 0 and nSameID > 0:
                VElems[i, j, k] = raw_value
                n_changes += 1

    print(f"nRetreat: {n_changes}")
    return VElems


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def refine_with_master_ball(VElems, ball_boss, ball_indices):
    nBalls = ball_indices.shape[0]
    for ipar in nb.prange(nBalls):
        ball_master = np.int32(ipar)
        while ball_boss[ball_master] != ball_master:
            ball_master = ball_boss[ball_master]
        zi, yi, xi = ball_indices[ipar]
        zm, ym, xm = ball_indices[ball_master]
        VElems[zi + 1, yi + 1, xi + 1] = VElems[zm + 1, ym + 1, xm + 1]
    return VElems
