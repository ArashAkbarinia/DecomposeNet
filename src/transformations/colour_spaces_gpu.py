"""
Functions related to change or manipulation of colour spaces.
PyTorch CUDA version — tensors are (Batch, Channel, Height, Width).
All operations stay on the tensor's own device (CPU or CUDA).
"""

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Transformation matrices — registered as plain tensors; moved to device on use
# ---------------------------------------------------------------------------

_yog_from_rgb = torch.tensor([
    [+0.25, +0.50, +0.25],
    [+0.50, +0.00, -0.50],
    [-0.25, +0.50, -0.25],
], dtype=torch.float32)  # (3, 3)  out_ch × in_ch

_rgb_from_yog = torch.tensor([
    [+1.0, +1.0, -1.0],
    [+1.0, +0.0, +1.0],
    [+1.0, -1.0, -1.0],
], dtype=torch.float32)

_rgb_from_dkl = torch.tensor([
    [+0.49995000, +0.50001495, +0.49999914],
    [+0.99998394, -0.29898596, +0.01714922],
    [-0.17577361, +0.15319546, -0.99994349],
], dtype=torch.float32).T

_dkl_from_rgb = torch.tensor([
    [0.4251999971, +0.8273000025, +0.2267999991],
    [1.4303999955, -0.5912000011, +0.7050999939],
    [0.1444000069, -0.2360000005, -0.9318999983],
], dtype=torch.float32).T

_xyz_from_lms = torch.tensor([
    [+1.99831835e+00, -1.18730329e+00, +1.88189487e-01],
    [+7.07957782e-01, -2.92384281e-01, +5.61719491e-09],
    [-2.22739159e-08, -1.46290052e-08, +9.98233271e-01],
], dtype=torch.float32)

_lms_from_xyz = torch.tensor([
    [-1.1408616727, +4.6327689355, +0.2150781317],
    [-2.7623985003, +7.7972892807, +0.5207743801],
    [-0.0000000659, +0.0000002176, +1.0017698683],
], dtype=torch.float32)

_rgb_from_xyz = torch.tensor([
    [+3.2404542, -0.9692660, +0.0556434],
    [-1.5371385, +1.8760108, -0.2040259],
    [-0.4985314, +0.0415560, +1.0572252],
], dtype=torch.float32).T

_xyz_from_rgb = torch.tensor([
    [0.4124564323, 0.2126728463, 0.0193339041],
    [0.3575760763, 0.7151521672, 0.1191920282],
    [0.1804374803, 0.0721749996, 0.9503040737],
], dtype=torch.float32).T

_lms_max = torch.tensor([3.78259774, 5.73874728, 1.09075725], dtype=torch.float32)
_lms_min = torch.tensor([0., 0., 0.], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to(mat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Move matrix to the same device/dtype as x."""
    return mat.to(device=x.device, dtype=x.dtype)


def _apply_matrix(x: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
    """
    Apply a (3×3) colour matrix to a (B, 3, H, W) tensor.
    mat shape: (out_channels, in_channels)
    Returns: (B, 3, H, W)
    """
    mat = _to(mat, x)
    # einsum: b c h w, o c -> b o h w
    return torch.einsum("bchw,oc->bohw", x, mat)


def _rgb2double(x: torch.Tensor) -> torch.Tensor:
    """uint8 (B,C,H,W) → float32 [0,1].  No-op if already float."""
    if x.dtype == torch.uint8:
        return x.float() / 255.0
    else:
        x = x.float()
        x = x / x.max()
    return x


def _uint8im(x: torch.Tensor) -> torch.Tensor:
    """float32 [0,1] → uint8, clamped."""
    return (x.clamp(0.0, 1.0) * 255.0).to(torch.uint8)


# ---------------------------------------------------------------------------
# LMS
# ---------------------------------------------------------------------------

def rgb012lms(x: torch.Tensor) -> torch.Tensor:
    """float [0,1] RGB → LMS (linear)."""
    return xyz2lms(rgb012xyz(x))


def rgb2lms(x: torch.Tensor) -> torch.Tensor:
    return rgb012lms(_rgb2double(x))


def rgb012lms01(x: torch.Tensor) -> torch.Tensor:
    """float [0,1] RGB → LMS normalised to [0,1]."""
    out = rgb012lms(x)
    lms_max = _to(_lms_max, out)
    lms_min = _to(_lms_min, out)
    rng = lms_max - lms_min  # (3,)
    out = (out - lms_min[None, :, None, None]) / rng[None, :, None, None]
    return out.clamp(0.0, 1.0)


def rgb2lms01(x: torch.Tensor) -> torch.Tensor:
    return rgb012lms01(_rgb2double(x))


def lms2rgb01(x: torch.Tensor) -> torch.Tensor:
    return xyz2rgb01(lms2xyz(x))


def lms2rgb(x: torch.Tensor) -> torch.Tensor:
    return _uint8im(lms2rgb01(x))


def lms012rgb01(x: torch.Tensor) -> torch.Tensor:
    lms_max = _to(_lms_max, x)
    lms_min = _to(_lms_min, x)
    rng = lms_max - lms_min
    out = x * rng[None, :, None, None] + lms_min[None, :, None, None]
    return lms2rgb01(out).clamp(0.0, 1.0)


def lms012rgb(x: torch.Tensor) -> torch.Tensor:
    return _uint8im(lms012rgb01(x))


# ---------------------------------------------------------------------------
# XYZ
# ---------------------------------------------------------------------------

def lms2xyz(x: torch.Tensor) -> torch.Tensor:
    # original used np.dot(x, xyz_from_lms) where xyz_from_lms is already .T
    # matrix stored as (3,3) mapping lms→xyz: out_ch = xyz, in_ch = lms
    return _apply_matrix(x, _to(_xyz_from_lms, x))


def xyz2lms(x: torch.Tensor) -> torch.Tensor:
    return _apply_matrix(x, _to(_lms_from_xyz, x))


def rgb012xyz(x: torch.Tensor) -> torch.Tensor:
    return _apply_matrix(x, _to(_xyz_from_rgb, x))


def rgb2xyz(x: torch.Tensor) -> torch.Tensor:
    return rgb012xyz(_rgb2double(x))


def xyz2rgb01(x: torch.Tensor) -> torch.Tensor:
    return _apply_matrix(x, _to(_rgb_from_xyz, x))


def xyz2rgb(x: torch.Tensor) -> torch.Tensor:
    return _uint8im(xyz2rgb01(x))


# ---------------------------------------------------------------------------
# DKL
# ---------------------------------------------------------------------------

def rgb012dkl(x: torch.Tensor) -> torch.Tensor:
    return _apply_matrix(x, _to(_dkl_from_rgb, x))


def rgb2dkl(x: torch.Tensor) -> torch.Tensor:
    return rgb012dkl(_rgb2double(x))


def rgb2dkl01(x: torch.Tensor) -> torch.Tensor:
    out = rgb2dkl(x)
    out = out / 2.0
    out[:, 1] += 0.5
    out[:, 2] += 0.5
    return out


def dkl2rgb01(x: torch.Tensor) -> torch.Tensor:
    return _apply_matrix(x, _to(_rgb_from_dkl, x))


def dkl2rgb(x: torch.Tensor) -> torch.Tensor:
    return _uint8im(dkl2rgb01(x))


def dkl012rgb01(x: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    out[:, 1] -= 0.5
    out[:, 2] -= 0.5
    out = out * 2.0
    return dkl2rgb01(out)


def dkl012rgb(x: torch.Tensor) -> torch.Tensor:
    return _uint8im(dkl012rgb01(x))


# ---------------------------------------------------------------------------
# YOG
# ---------------------------------------------------------------------------

def rgb012yog(x: torch.Tensor) -> torch.Tensor:
    return _apply_matrix(x, _to(_yog_from_rgb, x))


def rgb2yog(x: torch.Tensor) -> torch.Tensor:
    return rgb012yog(_rgb2double(x))


def rgb2yog01(x: torch.Tensor) -> torch.Tensor:
    out = rgb2yog(x)
    out[:, 1] += 0.5
    out[:, 2] += 0.5
    return out


def yog2rgb01(x: torch.Tensor) -> torch.Tensor:
    return _apply_matrix(x, _to(_rgb_from_yog, x))


def yog2rgb(x: torch.Tensor) -> torch.Tensor:
    return _uint8im(yog2rgb01(x))


def yog012rgb01(x: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    out[:, 1] -= 0.5
    out[:, 2] -= 0.5
    return yog2rgb01(out)


def yog012rgb(x: torch.Tensor) -> torch.Tensor:
    return _uint8im(yog012rgb01(x))


# ---------------------------------------------------------------------------
# HSV  (pure PyTorch, no cv2)
# ---------------------------------------------------------------------------

def rgb2hsv01(x: torch.Tensor) -> torch.Tensor:
    """
    uint8 or float [0,1] RGB (B,3,H,W) → float HSV [0,1] (B,3,H,W).
    H∈[0,1], S∈[0,1], V∈[0,1].
    """
    x = _rgb2double(x)
    r, g, b = x[:, 0], x[:, 1], x[:, 2]

    v, _ = x.max(dim=1)          # value
    x_min, _ = x.min(dim=1)
    diff = v - x_min             # chroma

    s = torch.where(v > 0, diff / v, torch.zeros_like(v))

    # Hue
    eps = 1e-8
    h = torch.zeros_like(v)

    mask_r = (v == r) & (diff > 0)
    mask_g = (v == g) & (diff > 0)
    mask_b = (v == b) & (diff > 0)

    h[mask_r] = ((g[mask_r] - b[mask_r]) / (diff[mask_r] + eps)) % 6
    h[mask_g] = (b[mask_g] - r[mask_g]) / (diff[mask_g] + eps) + 2
    h[mask_b] = (r[mask_b] - g[mask_b]) / (diff[mask_b] + eps) + 4

    h = h / 6.0  # normalise to [0,1]

    return torch.stack([h, s, v], dim=1)


def hsv012rgb(x: torch.Tensor) -> torch.Tensor:
    """
    float HSV [0,1] (B,3,H,W) → uint8 RGB (B,3,H,W).
    """
    return _uint8im(hsv012rgb01(x))


def hsv012rgb01(x: torch.Tensor) -> torch.Tensor:
    """
    float HSV [0,1] (B,3,H,W) → float RGB [0,1] (B,3,H,W).
    """
    h, s, v = x[:, 0], x[:, 1], x[:, 2]
    h6 = h * 6.0
    i = h6.long() % 6
    f = h6 - h6.floor()

    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    rgb = torch.stack([v, v, v], dim=1).clone()  # placeholder shape

    def _ch(a, b, c):
        return torch.stack([a, b, c], dim=1)

    cases = [
        _ch(v, t, p),   # i == 0
        _ch(q, v, p),   # i == 1
        _ch(p, v, t),   # i == 2
        _ch(p, q, v),   # i == 3
        _ch(t, p, v),   # i == 4
        _ch(v, p, q),   # i == 5
    ]

    rgb = torch.zeros_like(x)
    for idx, case in enumerate(cases):
        mask = (i == idx).unsqueeze(1)  # (B,1,H,W)
        rgb = torch.where(mask, case, rgb)

    return rgb.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# LAB / LCH  (pure PyTorch, no cv2)
# D65 illuminant, sRGB primaries)
# ---------------------------------------------------------------------------

_LAB_EPS = 0.008856
_LAB_KAPPA = 903.3


def _rgb2lab_f(t: torch.Tensor) -> torch.Tensor:
    return torch.where(t > _LAB_EPS, t.clamp(min=1e-10).pow(1 / 3), (_LAB_KAPPA * t + 16) / 116)


def rgb012lab(x: torch.Tensor) -> torch.Tensor:
    """float [0,1] RGB (B,3,H,W) → LAB (B,3,H,W)."""
    # linear RGB → XYZ
    xyz = rgb012xyz(x)

    # D65 white point normalisation
    d65 = torch.tensor([0.95047, 1.00000, 1.08883], device=x.device, dtype=x.dtype)
    xyz = xyz / d65[None, :, None, None]

    fx, fy, fz = _rgb2lab_f(xyz[:, 0]), _rgb2lab_f(xyz[:, 1]), _rgb2lab_f(xyz[:, 2])

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    return torch.stack([L, a, b], dim=1)


def rgb2lab(x: torch.Tensor) -> torch.Tensor:
    return rgb012lab(_rgb2double(x))


def lab2rgb01(x: torch.Tensor) -> torch.Tensor:
    """LAB (B,3,H,W) → float [0,1] RGB."""
    L, a, b = x[:, 0], x[:, 1], x[:, 2]
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200

    def _inv_f(t):
        t3 = t ** 3
        return torch.where(t3 > _LAB_EPS, t3, (116 * t - 16) / _LAB_KAPPA)

    d65 = torch.tensor([0.95047, 1.00000, 1.08883], device=x.device, dtype=x.dtype)
    xyz = torch.stack([_inv_f(fx), _inv_f(fy), _inv_f(fz)], dim=1)
    xyz = xyz * d65[None, :, None, None]

    return xyz2rgb01(xyz).clamp(0.0, 1.0)


def lab2rgb(x: torch.Tensor) -> torch.Tensor:
    return _uint8im(lab2rgb01(x))


def lab2lch(x: torch.Tensor) -> torch.Tensor:
    """LAB (B,3,H,W) → LCH (B,3,H,W)."""
    L, a, b = x[:, 0], x[:, 1], x[:, 2]
    C = torch.sqrt(a ** 2 + b ** 2)
    H = torch.atan2(b, a)
    H = torch.where(H >= 0, torch.rad2deg(H), 360 - torch.rad2deg(H.abs()))
    return torch.stack([L, C, H], dim=1)


def lch2lab(x: torch.Tensor) -> torch.Tensor:
    """LCH (B,3,H,W) → LAB (B,3,H,W)."""
    L, C, H = x[:, 0], x[:, 1], x[:, 2]
    a = torch.cos(torch.deg2rad(H)) * C
    b = torch.sin(torch.deg2rad(H)) * C
    return torch.stack([L, a, b], dim=1)


def lab2lch01(x: torch.Tensor) -> torch.Tensor:
    lch = lab2lch(x)
    lch = lch.clone()
    lch[:, 0] /= 100
    lch[:, 1] /= 134
    lch[:, 2] /= 360
    return lch


def lch012lab(x: torch.Tensor) -> torch.Tensor:
    lch = x.clone()
    lch[:, 0] *= 100
    lch[:, 1] *= 134
    lch[:, 2] *= 360
    return lch2lab(lch)


# ---------------------------------------------------------------------------
# Generic opponency helpers
# ---------------------------------------------------------------------------

def rgb2opponency(image_rgb: torch.Tensor, opponent_space: str = 'lab') -> torch.Tensor:
    image_rgb = _rgb2double(image_rgb)
    if opponent_space is None:
        return image_rgb
    elif opponent_space == 'lab':
        return rgb012lab(image_rgb)
    elif opponent_space == 'dkl':
        return rgb012dkl(image_rgb)
    else:
        raise ValueError(f'Not supported colour space {opponent_space}')


def opponency2rgb(image_opponent: torch.Tensor, opponent_space: str = 'lab') -> torch.Tensor:
    if opponent_space is None:
        return image_opponent
    elif opponent_space == 'lab':
        return lab2rgb(image_opponent)
    elif opponent_space == 'dkl':
        return dkl2rgb(image_opponent)
    else:
        raise ValueError(f'Not supported colour space {opponent_space}')


def get_max_lightness(opponent_space: str = 'lab') -> float:
    if opponent_space is None:
        return 255.0
    elif opponent_space == 'lab':
        return 100.0
    elif opponent_space == 'dkl':
        return 2.0
    else:
        raise ValueError(f'Not supported colour space {opponent_space}')