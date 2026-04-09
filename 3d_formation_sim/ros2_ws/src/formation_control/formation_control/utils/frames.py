import numpy as np


def enu_to_ned(x_enu, y_enu, z_enu):
    """Convert ENU (East-North-Up) to NED (North-East-Down)."""
    return y_enu, x_enu, -z_enu


def ned_to_enu(x_ned, y_ned, z_ned):
    """Convert NED to ENU."""
    return y_ned, x_ned, -z_ned


def enu_to_ned_array(pos_enu: np.ndarray) -> np.ndarray:
    """Convert (3,) or (N,3) ENU array to NED.

    Args:
        pos_enu: Position(s) in ENU frame, shape (3,) or (N, 3).

    Returns:
        Position(s) in NED frame, same shape as input.
    """
    pos = np.atleast_2d(pos_enu).copy()
    result = np.empty_like(pos)
    result[..., 0] = pos[..., 1]
    result[..., 1] = pos[..., 0]
    result[..., 2] = -pos[..., 2]
    if pos_enu.ndim == 1:
        return result[0]
    return result


def ned_to_enu_array(pos_ned: np.ndarray) -> np.ndarray:
    """Convert (3,) or (N,3) NED array to ENU.

    Args:
        pos_ned: Position(s) in NED frame, shape (3,) or (N, 3).

    Returns:
        Position(s) in ENU frame, same shape as input.
    """
    pos = np.atleast_2d(pos_ned).copy()
    result = np.empty_like(pos)
    result[..., 0] = pos[..., 1]
    result[..., 1] = pos[..., 0]
    result[..., 2] = -pos[..., 2]
    if pos_ned.ndim == 1:
        return result[0]
    return result
