"""SE(3) poses stored as ``[x, y, z, r0..r5]``: position plus the 6D rotation representation.

The 6D part is the first two COLUMNS of the rotation matrix, ``[R[:, 0], R[:, 1]]``
(Zhou et al., 2019) -- the convention of ``data_werkzeug.conversion.rotations`` that
the oopsie datasets are written with. Note that GR00T's helpers in
``lerobot.policies.groot.utils`` use the first two ROWS instead; the two are
transposes of each other and must not be mixed.

Relative poses express a target pose in a reference pose's frame:

* ``frame="ee"``    -- in the reference's own (end-effector) frame:
  ``p_rel = R_refᵀ (p - p_ref)``,  ``R_rel = R_refᵀ R``
* ``frame="world"`` -- as a world-frame displacement:
  ``p_rel = p - p_ref``,           ``R_rel = R R_refᵀ``

``pose_to_absolute`` inverts ``pose_to_relative`` exactly, up to float precision.
"""

from typing import Literal

import torch
from torch import Tensor

PoseFrame = Literal["ee", "world"]
POSE_FRAMES = ("ee", "world")
POSE_DIM = 9

_EPS = 1e-8


def rot6d_to_matrix(rot6d: Tensor) -> Tensor:
    """``(..., 6)`` column-convention 6D rotation -> ``(..., 3, 3)`` rotation matrix.

    Gram-Schmidt orthonormalises the input, so a network output that is not exactly
    orthonormal still maps to a valid rotation.
    """
    a1, a2 = rot6d[..., :3], rot6d[..., 3:6]
    b1 = a1 / a1.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    a2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = a2 / a2.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_rot6d(matrix: Tensor) -> Tensor:
    """``(..., 3, 3)`` rotation matrix -> ``(..., 6)`` column-convention 6D rotation."""
    return torch.cat([matrix[..., :, 0], matrix[..., :, 1]], dim=-1)


def _check_frame(frame: str) -> None:
    if frame not in POSE_FRAMES:
        raise ValueError(f"pose frame must be one of {POSE_FRAMES}, got {frame!r}")


def pose_to_relative(pose: Tensor, reference: Tensor, frame: PoseFrame = "ee") -> Tensor:
    """Express ``pose`` relative to ``reference``; both ``(..., 9)`` and broadcastable.

    Computed in float32 (or float64 if given) and returned in ``pose``'s dtype.
    """
    _check_frame(frame)
    out_dtype = pose.dtype
    dtype = torch.float64 if torch.float64 in (pose.dtype, reference.dtype) else torch.float32
    pose, reference = pose.to(dtype), reference.to(dtype)

    r_ref = rot6d_to_matrix(reference[..., 3:9])
    r = rot6d_to_matrix(pose[..., 3:9])
    dp = pose[..., :3] - reference[..., :3]
    if frame == "ee":
        r_ref_t = r_ref.transpose(-1, -2)
        p_rel = (r_ref_t @ dp.unsqueeze(-1)).squeeze(-1)
        r_rel = r_ref_t @ r
    else:
        p_rel = dp
        r_rel = r @ r_ref.transpose(-1, -2)
    return torch.cat([p_rel, matrix_to_rot6d(r_rel)], dim=-1).to(out_dtype)


def pose_to_absolute(relative: Tensor, reference: Tensor, frame: PoseFrame = "ee") -> Tensor:
    """Inverse of :func:`pose_to_relative`: compose ``relative`` onto ``reference``."""
    _check_frame(frame)
    out_dtype = relative.dtype
    dtype = torch.float64 if torch.float64 in (relative.dtype, reference.dtype) else torch.float32
    relative, reference = relative.to(dtype), reference.to(dtype)

    r_ref = rot6d_to_matrix(reference[..., 3:9])
    r_rel = rot6d_to_matrix(relative[..., 3:9])
    if frame == "ee":
        p = reference[..., :3] + (r_ref @ relative[..., :3].unsqueeze(-1)).squeeze(-1)
        r = r_ref @ r_rel
    else:
        p = reference[..., :3] + relative[..., :3]
        r = r_rel @ r_ref
    return torch.cat([p, matrix_to_rot6d(r)], dim=-1).to(out_dtype)


def pose_group_names(prefix: str) -> list[str]:
    """Feature names of one pose group, e.g. ``right_ee`` -> ``right_ee_x .. right_ee_rot6d_5``."""
    return [f"{prefix}_{axis}" for axis in ("x", "y", "z")] + [f"{prefix}_rot6d_{i}" for i in range(6)]
