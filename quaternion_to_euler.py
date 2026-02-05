import math

import torch


def quaternion_to_euler(q):
    """Convert quaternions to Euler angles (roll, pitch, yaw).

    Args:
        q (torch.Tensor): Quaternions with shape (..., 4) where q = [w, x, y, z]

    Returns:
        torch.Tensor: Euler angles [roll, pitch, yaw] in degrees
    """
    # Extract quaternion components
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.sign(sinp) * torch.tensor(math.pi / 2, device=q.device),
        torch.asin(sinp)
    )

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    # Convert from radians to degrees
    roll_deg = roll * 180.0 / math.pi
    pitch_deg = pitch * 180.0 / math.pi
    yaw_deg = yaw * 180.0 / math.pi

    return torch.stack([roll_deg, pitch_deg, yaw_deg], dim=-1)


def single_quaternion_to_euler(q):
    """Convert a single quaternion to Euler angles (roll, pitch, yaw).

    Args:
        q (list or array): Single quaternion [w, x, y, z]

    Returns:
        tuple: Euler angles (roll, pitch, yaw) in degrees
    """
    # Extract quaternion components
    w, x, y, z = q[0], q[1], q[2], q[3]

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    # Check for gimbal lock, pitch = ±90°
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    # Convert from radians to degrees
    roll_deg = roll * 180.0 / math.pi
    pitch_deg = pitch * 180.0 / math.pi
    yaw_deg = yaw * 180.0 / math.pi

    return roll_deg, pitch_deg, yaw_deg