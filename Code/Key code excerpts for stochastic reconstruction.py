# ==========================================================
# (1) Conditional co-occurrence probability matrix P_ij(x)
#     based on the Moore neighborhood
# ==========================================================

import numpy as np

OFFSETS_N8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1)
]

    For i != j:
      numerator(i,j)   = sum_p sum_q I(x(p)=i) I(x(q)=j)
      denominator(i)   = sum_p sum_q I(x(p)=i) I(x(q)!=i)
      P_ij(x)          = numerator(i,j) / denominator(i)

    where q belongs to the Moore neighborhood of p.
    Boundary pixels are handled without periodic wrapping.
    """
    H, W = field.shape
    num = np.zeros((n_phases, n_phases), dtype=np.int64)
    den = np.zeros(n_phases, dtype=np.int64)

    for y in range(H):
        for x in range(W):
            i = int(field[y, x])
            for dy, dx in OFFSETS_N8:
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W:
                    j = int(field[yy, xx])
                    if j != i:
                        num[i, j] += 1
                        den[i] += 1

    P = np.zeros((n_phases, n_phases), dtype=np.float64)
    for i in range(n_phases):
        if den[i] > 0:
            P[i, :] = num[i, :] / den[i]

    np.fill_diagonal(P, 0.0)
    return P


# ==========================================================
# (2) Two-point autocorrelation S2 
# ==========================================================

def compute_S2_periodic_fft(field: np.ndarray, n_phases: int) -> np.ndarray:
    """
    Two-point autocorrelation S2^(i)(r; x) with periodic shifting, consistent with Eq. (4).

    S2_i(r) = (1/|Omega|) * sum_p B_i(p) * B_i(shifted_p)

    The FFT-based implementation is equivalent under periodic boundary conditions.
    """
    H, W = field.shape
    S2 = np.zeros((n_phases, H, W), dtype=np.float64)
    norm = float(H * W)

    for i in range(n_phases):
        B = (field == i).astype(np.float64)
        F = np.fft.fftn(B)
        C = np.fft.ifftn(F * np.conj(F)).real / norm
        S2[i, :, :] = C

    return S2