import numba as nb
import numpy as np
from ._extraction_functions_numba import get_masterball


@nb.njit(
    parallel=False,
    cache=True,
    fastmath=True,
    nogil=True,
    inline="always",
    forceinline=True,
)
def get_max_count_nei(neis, n_neIs):
    max_count = 0
    max_val = -1

    for i in range(n_neIs):
        count = 0
        val = neis[i]

        for j in range(i, n_neIs):
            count += neis[j] == val

        if count > max_count or (count == max_count and val < max_val):
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
        z, y, x = ball_indices[pore_ind] + 1
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


@nb.njit(
    parallel=True,
    cache=True,
    fastmath=True,
    nogil=True,
    inline="always",
    forceinline=True,
)
def assign_array(src, dst):
    src_flat = src.reshape(-1)
    dst_flat = dst.reshape(-1)
    for i in nb.prange(dst.size):
        dst_flat[i] = src_flat[i]


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def grow_pores(zsysxs_v, VElems, voxls, bgn, raw_value):
    nVxls = zsysxs_v.shape[0]
    assign_array(VElems, voxls)
    n_changes = 0
    for ipar in nb.prange(nVxls):
        z, y, x = zsysxs_v[ipar]
        if VElems[z, y, x] != raw_value:
            continue
        if bgn <= voxls[z, y, x - 1]:
            VElems[z, y, x] = voxls[z, y, x - 1]
            n_changes += 1
        elif bgn <= voxls[z, y, x + 1]:
            VElems[z, y, x] = voxls[z, y, x + 1]
            n_changes += 1
        elif bgn <= voxls[z, y - 1, x]:
            VElems[z, y, x] = voxls[z, y - 1, x]
            n_changes += 1
        elif bgn <= voxls[z, y + 1, x]:
            VElems[z, y, x] = voxls[z, y + 1, x]
            n_changes += 1
        elif bgn <= voxls[z - 1, y, x]:
            VElems[z, y, x] = voxls[z - 1, y, x]
            n_changes += 1
        elif bgn <= voxls[z + 1, y, x]:
            VElems[z, y, x] = voxls[z + 1, y, x]
            n_changes += 1

    print(f"ngrowPors changes: {n_changes}")
    return VElems


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def grow_pores_X2(zsysxs_v, VElems, voxls, bgn, raw_value):
    nVxls = zsysxs_v.shape[0]
    assign_array(VElems, voxls)
    n_changes = 0
    for ipar in nb.prange(nVxls):
        z, y, x = zsysxs_v[ipar]
        if VElems[z, y, x] != raw_value:
            continue
        if bgn <= voxls[z, y, x - 1]:
            VElems[z, y, x] = voxls[z, y, x - 1]
            n_changes += 1
        elif bgn <= voxls[z, y, x + 1]:
            VElems[z, y, x] = voxls[z, y, x + 1]
            n_changes += 1
        elif bgn <= voxls[z, y - 1, x]:
            VElems[z, y, x] = voxls[z, y - 1, x]
            n_changes += 1
        elif bgn <= voxls[z, y + 1, x]:
            VElems[z, y, x] = voxls[z, y + 1, x]
            n_changes += 1
        elif bgn <= voxls[z - 1, y, x]:
            VElems[z, y, x] = voxls[z - 1, y, x]
            n_changes += 1
        elif bgn <= voxls[z + 1, y, x]:
            VElems[z, y, x] = voxls[z + 1, y, x]
            n_changes += 1

    print(f"  ngrowX3:{n_changes},")

    n_changes = 0
    assign_array(VElems, voxls)
    for ipar in nb.prange(nVxls):
        z, y, x = zsysxs_v[ipar]
        if VElems[z, y, x] != raw_value:
            continue
        if bgn <= voxls[z, y, x - 1]:
            VElems[z, y, x] = voxls[z, y, x - 1]
            n_changes += 1
        elif bgn <= voxls[z, y, x + 1]:
            VElems[z, y, x] = voxls[z, y, x + 1]
            n_changes += 1
        elif bgn <= voxls[z, y - 1, x]:
            VElems[z, y, x] = voxls[z, y - 1, x]
            n_changes += 1
        elif bgn <= voxls[z, y + 1, x]:
            VElems[z, y, x] = voxls[z, y + 1, x]
            n_changes += 1
        elif bgn <= voxls[z - 1, y, x]:
            VElems[z, y, x] = voxls[z - 1, y, x]
            n_changes += 1
        elif bgn <= voxls[z + 1, y, x]:
            VElems[z, y, x] = voxls[z + 1, y, x]
            n_changes += 1

    print(f"{n_changes},")

    n_changes = 0
    assign_array(VElems, voxls)
    for ipar in nb.prange(nVxls):
        z, y, x = zsysxs_v[ipar]
        if VElems[z, y, x] != raw_value:
            continue
        if bgn <= voxls[z, y, x - 1]:
            VElems[z, y, x] = voxls[z, y, x - 1]
            n_changes += 1
        elif bgn <= voxls[z, y, x + 1]:
            VElems[z, y, x] = voxls[z, y, x + 1]
            n_changes += 1
        elif bgn <= voxls[z, y - 1, x]:
            VElems[z, y, x] = voxls[z, y - 1, x]
            n_changes += 1
        elif bgn <= voxls[z, y + 1, x]:
            VElems[z, y, x] = voxls[z, y + 1, x]
            n_changes += 1
        elif bgn <= voxls[z - 1, y, x]:
            VElems[z, y, x] = voxls[z - 1, y, x]
            n_changes += 1
        elif bgn <= voxls[z + 1, y, x]:
            VElems[z, y, x] = voxls[z + 1, y, x]
            n_changes += 1

    print(f"  ngrowX2:{n_changes}  ")

    return VElems, n_changes


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def grow_pores_med_strict(zsysxs_v, dt_p1, VElems, voxls, bgn, raw_value):
    nVxls = zsysxs_v.shape[0]
    assign_array(VElems, voxls)
    n_changes = 0
    for ipar in nb.prange(nVxls):
        z, y, x = zsysxs_v[ipar]
        if VElems[z, y, x] != raw_value:
            continue
        R = dt_p1[z, y, x]
        nDifferentID = 0
        neIs = np.empty(6, dtype=np.int32)
        n_neIs = 0
        V_current = voxls[z, y, x - 1]
        R_current = dt_p1[z, y, x - 1]
        if bgn <= V_current and R_current >= R:
            nDifferentID += 1
            if R_current > R:
                neIs[n_neIs] = V_current
                n_neIs += 1
        V_current = voxls[z, y, x + 1]
        R_current = dt_p1[z, y, x + 1]
        if bgn <= V_current and R_current >= R:
            nDifferentID += 1
            if R_current > R:
                neIs[n_neIs] = V_current
                n_neIs += 1
        V_current = voxls[z, y - 1, x]
        R_current = dt_p1[z, y - 1, x]
        if bgn <= V_current and R_current >= R:
            nDifferentID += 1
            if R_current > R:
                neIs[n_neIs] = V_current
                n_neIs += 1
        V_current = voxls[z, y + 1, x]
        R_current = dt_p1[z, y + 1, x]
        if bgn <= V_current and R_current >= R:
            nDifferentID += 1
            if R_current > R:
                neIs[n_neIs] = V_current
                n_neIs += 1
        V_current = voxls[z - 1, y, x]
        R_current = dt_p1[z - 1, y, x]
        if bgn <= V_current and R_current >= R:
            nDifferentID += 1
            if R_current > R:
                neIs[n_neIs] = V_current
                n_neIs += 1
        V_current = voxls[z + 1, y, x]
        R_current = dt_p1[z + 1, y, x]
        if bgn <= V_current and R_current >= R:
            nDifferentID += 1
            if R_current > R:
                neIs[n_neIs] = V_current
                n_neIs += 1

        if nDifferentID >= 3:
            max_count_nei, max_count = get_max_count_nei(neIs, n_neIs)
            if max_count >= 3:
                VElems[z, y, x] = max_count_nei
                n_changes += 1

    print(f"ngMedStrict changes: {n_changes}")
    return VElems


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def grow_pores_median(zsysxs_v, dt_p1, VElems, voxls, bgn, raw_value):
    nVxls = zsysxs_v.shape[0]
    assign_array(VElems, voxls)
    n_changes = 0
    for ipar in nb.prange(nVxls):
        z, y, x = zsysxs_v[ipar]
        if VElems[z, y, x] != raw_value:
            continue
        R = dt_p1[z, y, x]
        nDifferentID = 0
        neIs = np.empty(6, dtype=np.int32)
        n_neIs = 0
        V_current = voxls[z, y, x - 1]
        R_current = dt_p1[z, y, x - 1]
        if bgn <= V_current and R_current > R:
            nDifferentID += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        V_current = voxls[z, y, x + 1]
        R_current = dt_p1[z, y, x + 1]
        if bgn <= V_current and R_current > R:
            nDifferentID += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        V_current = voxls[z, y - 1, x]
        R_current = dt_p1[z, y - 1, x]
        if bgn <= V_current and R_current > R:
            nDifferentID += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        V_current = voxls[z, y + 1, x]
        R_current = dt_p1[z, y + 1, x]
        if bgn <= V_current and R_current > R:
            nDifferentID += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        V_current = voxls[z - 1, y, x]
        R_current = dt_p1[z - 1, y, x]
        if bgn <= V_current and R_current > R:
            nDifferentID += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        V_current = voxls[z + 1, y, x]
        R_current = dt_p1[z + 1, y, x]
        if bgn <= V_current and R_current > R:
            nDifferentID += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        if nDifferentID >= 2:
            max_count_nei, max_count = get_max_count_nei(neIs, n_neIs)
            if max_count >= 2:
                VElems[z, y, x] = max_count_nei
                n_changes += 1

    print(f"ngMedian changes: {n_changes}")
    return VElems


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def grow_pores_med_eqs(zsysxs_v, dt_p1, VElems, voxls, bgn, raw_value):
    nVxls = zsysxs_v.shape[0]
    assign_array(VElems, voxls)
    n_changes = 0
    for ipar in nb.prange(nVxls):
        z, y, x = zsysxs_v[ipar]
        if VElems[z, y, x] != raw_value:
            continue
        R = dt_p1[z, y, x]
        nDifferentID = 0
        neIs = np.empty(6, dtype=np.int32)
        n_neIs = 0
        V_current = voxls[z, y, x - 1]
        R_current = dt_p1[z, y, x - 1]
        if bgn <= V_current and R_current >= R:
            nDifferentID += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        V_current = voxls[z, y, x + 1]
        R_current = dt_p1[z, y, x + 1]
        if bgn <= V_current and R_current >= R:
            nDifferentID += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        V_current = voxls[z, y - 1, x]
        R_current = dt_p1[z, y - 1, x]
        if bgn <= V_current and R_current >= R:
            nDifferentID += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        V_current = voxls[z, y + 1, x]
        R_current = dt_p1[z, y + 1, x]
        if bgn <= V_current and R_current >= R:
            nDifferentID += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        V_current = voxls[z - 1, y, x]
        R_current = dt_p1[z - 1, y, x]
        if bgn <= V_current and R_current >= R:
            nDifferentID += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        V_current = voxls[z + 1, y, x]
        R_current = dt_p1[z + 1, y, x]
        if bgn <= V_current and R_current >= R:
            nDifferentID += 1
            neIs[n_neIs] = V_current
            n_neIs += 1

        if nDifferentID >= 2:
            max_count_nei, max_count = get_max_count_nei(neIs, n_neIs)
            if max_count >= 2:
                VElems[z, y, x] = max_count_nei
                n_changes += 1

    print(f"ngMedEqs changes: {n_changes}")
    return VElems


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def grow_pores_med_eqs_loose(zsysxs_v, VElems, voxls, bgn, raw_value):
    nVxls = zsysxs_v.shape[0]
    assign_array(VElems, voxls)
    n_changes = 0
    for ipar in nb.prange(nVxls):
        z, y, x = zsysxs_v[ipar]
        if VElems[z, y, x] != raw_value:
            continue
        nDifferentID = 0
        neIs = np.empty(6, dtype=np.int32)
        n_neIs = 0
        V_current = voxls[z, y, x - 1]
        if bgn <= V_current:
            nDifferentID += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        V_current = voxls[z, y, x + 1]
        if bgn <= V_current:
            nDifferentID += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        V_current = voxls[z, y - 1, x]
        if bgn <= V_current:
            nDifferentID += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        V_current = voxls[z, y + 1, x]
        if bgn <= V_current:
            nDifferentID += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        V_current = voxls[z - 1, y, x]
        if bgn <= V_current:
            nDifferentID += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        V_current = voxls[z + 1, y, x]
        if bgn <= V_current:
            nDifferentID += 1
            neIs[n_neIs] = V_current
            n_neIs += 1

        if nDifferentID >= 2:
            max_count_nei, max_count = get_max_count_nei(neIs, n_neIs)
            if max_count >= 2:
                VElems[z, y, x] = max_count_nei
                n_changes += 1

    print(f"ngMedLoose changes: {n_changes}")
    return VElems


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def median_elem(zsysxs_v, VElems, voxls, bgn):
    nVxls = zsysxs_v.shape[0]
    assign_array(VElems, voxls)
    n_changes = 0

    for ipar in nb.prange(nVxls):
        z, y, x = zsysxs_v[ipar]
        pID = voxls[z, y, x]
        if pID < bgn:
            continue
        n_same = 0
        n_diff = 0
        neIs = np.empty(6, dtype=np.int32)
        n_neIs = 0
        V_current = VElems[z, y, x - 1]
        if V_current == pID:
            n_same += 1
        elif bgn <= V_current:
            n_diff += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        V_current = voxls[z, y, x + 1]
        if V_current == pID:
            n_same += 1
        elif bgn <= V_current:
            n_diff += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        V_current = voxls[z, y - 1, x]
        if V_current == pID:
            n_same += 1
        elif bgn <= V_current:
            n_diff += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        V_current = voxls[z, y + 1, x]
        if V_current == pID:
            n_same += 1
        elif bgn <= V_current:
            n_diff += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        V_current = voxls[z - 1, y, x]
        if V_current == pID:
            n_same += 1
        elif bgn <= V_current:
            n_diff += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        V_current = voxls[z + 1, y, x]
        if V_current == pID:
            n_same += 1
        elif bgn <= V_current:
            n_diff += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        if n_diff > n_same:
            max_count_nei, max_count = get_max_count_nei(neIs, n_neIs)
            if max_count > n_same:
                VElems[z, y, x] = max_count_nei
                n_changes += 1

    print(f"nMedian: {n_changes}")
    return VElems


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def retreat_pores_median(zsysxs_v, VElems, voxls, bgn, raw_value):
    nVxls = zsysxs_v.shape[0]
    assign_array(VElems, voxls)
    n_changes = 0
    for ipar in nb.prange(nVxls):
        z, y, x = zsysxs_v[ipar]
        pID = VElems[z, y, x]
        if pID < bgn:
            continue
        nSameID = 0
        nDiffereID = 0
        V_current = voxls[z, y, x - 1]
        if V_current == pID:
            nSameID += 1
        elif bgn <= V_current:
            nDiffereID += 1
        V_current = voxls[z, y, x + 1]
        if V_current == pID:
            nSameID += 1
        elif bgn <= V_current:
            nDiffereID += 1
        V_current = voxls[z, y - 1, x]
        if V_current == pID:
            nSameID += 1
        elif bgn <= V_current:
            nDiffereID += 1
        V_current = voxls[z, y + 1, x]
        if V_current == pID:
            nSameID += 1
        elif bgn <= V_current:
            nDiffereID += 1
        V_current = voxls[z - 1, y, x]
        if V_current == pID:
            nSameID += 1
        elif bgn <= V_current:
            nDiffereID += 1
        V_current = voxls[z + 1, y, x]
        if V_current == pID:
            nSameID += 1
        elif bgn <= V_current:
            nDiffereID += 1

        if nDiffereID > 0 and nSameID > 0:
            VElems[z, y, x] = raw_value
            n_changes += 1

    print(f"nRetreat: {n_changes}")
    return VElems


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def refine_with_master_ball(VElems, ball_boss, ball_indices):
    nBalls = ball_indices.shape[0]
    for ipar in nb.prange(nBalls):
        ball_master = np.int64(ipar)
        while ball_boss[ball_master] != ball_master:
            ball_master = ball_boss[ball_master]
        zi, yi, xi = ball_indices[ipar]
        zm, ym, xm = ball_indices[ball_master]
        VElems[zi + 1, yi + 1, xi + 1] = VElems[zm + 1, ym + 1, xm + 1]
    return VElems


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def refine_with_boss_ball(VElems, ball_indices, poreIs):
    for ipar in nb.prange(poreIs.size):
        ball_idx = poreIs[ipar]
        zi, yi, xi = ball_indices[ball_idx]
        VElems[zi + 1, yi + 1, xi + 1] = np.int32(ipar)
    return VElems
