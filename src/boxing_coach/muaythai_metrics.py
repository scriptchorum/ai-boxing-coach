"""Muay Thai & Boxing biomechanical form assessment metrics.

Provides lightweight, interpretable indicators:
1. Rear-Hand Guard Defense: Detects dropped guard during punches/combinations.
2. Hip Rotation Angle: Quantifies core engagement during crosses and roundhouse kicks.
3. Arm Extension Ratio: Measures lockout completeness on straight punches (Jab / Cross).
"""
from __future__ import annotations

import math
from typing import Any, Sequence

# Key landmark indices in MediaPipe Pose
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24


def get_xy(point: dict | None) -> tuple[float, float] | None:
    """Extract (x, y) if point is observed and valid."""
    if not point or point.get('state') != 'observed' or not point.get('image'):
        return None
    img = point['image']
    x, y = img.get('x'), img.get('y')
    if x is not None and y is not None and math.isfinite(x) and math.isfinite(y):
        return (x, y)
    return None


def estimate_chin_y(points_by_idx: dict[int, dict]) -> float | None:
    """Estimate chin level y-coordinate (in normalized 0-1 image space, where top is 0)."""
    # If nose is visible, chin is slightly below nose
    nose_xy = get_xy(points_by_idx.get(NOSE))
    l_sh_xy = get_xy(points_by_idx.get(LEFT_SHOULDER))
    r_sh_xy = get_xy(points_by_idx.get(RIGHT_SHOULDER))

    if l_sh_xy and r_sh_xy:
        sh_mid_y = (l_sh_xy[1] + r_sh_xy[1]) / 2.0
        if nose_xy:
            # Chin is roughly 40% of the way from nose to shoulder midpoint
            return nose_xy[1] + 0.4 * (sh_mid_y - nose_xy[1])
        # If nose missing, chin is roughly 15% above shoulder line
        sh_dist = math.hypot(l_sh_xy[0] - r_sh_xy[0], l_sh_xy[1] - r_sh_xy[1])
        return sh_mid_y - (sh_dist * 0.35)

    if nose_xy:
        return nose_xy[1] + 0.05
    return None


def check_rear_guard(
    landmarks: Sequence[dict],
    stance: str = 'orthodox',
    chin_tolerance_ratio: float = 0.04,
) -> dict[str, Any]:
    """Check if the defensive rear hand drops below chin level.

    In Orthodox stance: Rear hand is Right Wrist (16).
    In Southpaw stance: Rear hand is Left Wrist (15).

    Returns:
        dict with:
            - 'rear_guard_safe': bool (True if wrist is above or at chin level)
            - 'wrist_y': float or None
            - 'chin_y': float or None
            - 'drop_distance': float (positive means dropped below chin)
    """
    points = {p['index']: p for p in landmarks if 'index' in p}
    rear_wrist_idx = RIGHT_WRIST if stance.lower() == 'orthodox' else LEFT_WRIST
    wrist_xy = get_xy(points.get(rear_wrist_idx))
    chin_y = estimate_chin_y(points)

    if not wrist_xy or chin_y is None:
        return {
            'rear_guard_safe': None,
            'wrist_y': wrist_xy[1] if wrist_xy else None,
            'chin_y': chin_y,
            'drop_distance': None,
            'status': 'unobservable',
        }

    # In normalized image coordinates, y increases downward.
    # Therefore, wrist_y > chin_y means wrist is lower than chin!
    drop_distance = wrist_xy[1] - chin_y
    # Safe if wrist is above chin or within acceptable tolerance
    safe = drop_distance <= chin_tolerance_ratio

    return {
        'rear_guard_safe': safe,
        'wrist_y': round(wrist_xy[1], 4),
        'chin_y': round(chin_y, 4),
        'drop_distance': round(drop_distance, 4),
        'status': 'observed',
    }


def compute_hip_rotation_angle(landmarks: Sequence[dict]) -> dict[str, Any]:
    """Compute the angle of the hip line (left_hip -> right_hip) in degrees.

    Returns:
        dict with:
            - 'angle_degrees': float (-180 to 180)
            - 'hip_width_normalized': float
            - 'status': 'observed' or 'unobservable'
    """
    points = {p['index']: p for p in landmarks if 'index' in p}
    l_hip = get_xy(points.get(LEFT_HIP))
    r_hip = get_xy(points.get(RIGHT_HIP))

    if not l_hip or not r_hip:
        return {'angle_degrees': None, 'hip_width_normalized': None, 'status': 'unobservable'}

    dx = r_hip[0] - l_hip[0]
    dy = r_hip[1] - l_hip[1]
    dist = math.hypot(dx, dy)
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)

    return {
        'angle_degrees': round(angle_deg, 2),
        'hip_width_normalized': round(dist, 4),
        'status': 'observed',
    }


def compute_arm_extension_ratio(
    landmarks: Sequence[dict],
    arm: str = 'lead',
    stance: str = 'orthodox',
) -> dict[str, Any]:
    """Compute punch extension ratio: dist(shoulder, wrist) / (upper_arm + forearm).

    Values close to 1.0 indicate full extension / lockout.
    Values < 0.6 indicate tucked guard or elbow flexion.
    """
    points = {p['index']: p for p in landmarks if 'index' in p}
    if stance.lower() == 'orthodox':
        sh_idx, el_idx, wr_idx = (
            (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST) if arm == 'lead'
            else (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
        )
    else:
        sh_idx, el_idx, wr_idx = (
            (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST) if arm == 'lead'
            else (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
        )

    sh = get_xy(points.get(sh_idx))
    el = get_xy(points.get(el_idx))
    wr = get_xy(points.get(wr_idx))

    if not sh or not el or not wr:
        return {'extension_ratio': None, 'status': 'unobservable'}

    upper_arm = math.hypot(el[0] - sh[0], el[1] - sh[1])
    forearm = math.hypot(wr[0] - el[0], wr[1] - el[1])
    total_arm = upper_arm + forearm

    if total_arm <= 1e-5:
        return {'extension_ratio': None, 'status': 'zero_length'}

    direct_reach = math.hypot(wr[0] - sh[0], wr[1] - sh[1])
    ratio = min(direct_reach / total_arm, 1.0)

    return {
        'extension_ratio': round(ratio, 3),
        'is_extended': ratio >= 0.85,
        'status': 'observed',
    }
