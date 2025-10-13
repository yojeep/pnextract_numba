import numba as nb
import numpy as np
from _extraction_functions_numba import get_masterball


@nb.njit(cache=True, fastmath=True, nogil=True)
def get_max_count_nei(neis):
    n = len(neis)
    if n == 0:
        return -1, 0

    max_val = -1
    max_count = 0

    # 对每个唯一值统计频次（暴力但高效，因为 n <= 6）
    for i in range(n):
        val = neis[i]
        count = 0
        for j in range(n):
            if neis[j] == val:
                count += 1

        # 更新规则：频次更高，或频次相同但值更大
        if count > max_count or (count == max_count and val > max_val):
            max_count = count
            max_val = val

    return max_val, max_count


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
def CreateVElem(
    image_VElems, dt_VElems, isball_VElems, ball_findices, ball_R, ball_boss
):
    uasyned = -1
    firstPores = 0
    nz, ny, nx = image_VElems.shape
    VElems = np.full(nz * ny * nx, uasyned, dtype=np.int32).reshape((nz, ny, nx))
    poreIs = np.where(ball_boss == np.arange(ball_findices.shape[0]))[0]
    # print(poreIs)
    firstPoreInd = firstPores
    lastPoreInd = poreIs.size - 1
    for ind in nb.prange(poreIs.size):
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
                    pID = voxls[i, j, k]
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
                    pID = voxls[i, j, k]
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
                    pID = voxls[i, j, k]
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
                    pID = voxls[i, j, k]
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
