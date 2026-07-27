import numba as nb


@nb.njit(
    parallel=False,
    cache=True,
    fastmath=True,
    nogil=True,
    error_model="numpy",
    # inline="always",
    # forceinline=True,
)
def float_range(start, stop, step):
    x = start
    n = 0
    while x < stop:
        yield x
        n += 1
        x = start + n * step


@nb.njit(
    parallel=True,
    cache=True,
    fastmath=True,
    nogil=True,
    error_model="numpy",
    inline="always",
    forceinline=True,
)
def demo():
    for x in float_range(-6.7, 6.7, 1.0):
        print(x)


demo()
