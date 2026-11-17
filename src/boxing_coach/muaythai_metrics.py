"""Muay Thai & Boxing biomechanical form assessment metrics.

Provides lightweight, interpretable indicators:
1. Rear-Hand Guard Defense: Detects dropped guard during punches/combinations.
2. Hip Rotation Angle: Quantifies core engagement during crosses and roundhouse kicks.
3. Arm Extension Ratio: Measures lockout completeness on straight punches (Jab / Cross).
"""
from __future__ import annotations

import math
from statistics import median
from typing import Any, Sequence

from .motion_windows import LEG_INTERVAL_MS, MIN_CONFIDENCE, excursions, supported_signal
from .rear_straight_review import straight_events

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
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


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


def check_muaythai_high_guard(
    landmarks: Sequence[dict],
    stance: str = 'orthodox',
    chin_tolerance_ratio: float = 0.045,
) -> dict[str, Any]:
    """Evaluate Muay Thai high/long guard (双拳护眉/护颌高抱架) on upper-body landmarks.

    Works on both full-body and upper-body (half-body) clips.
    """
    points = {p['index']: p for p in landmarks if 'index' in p}
    chin_y = estimate_chin_y(points)
    if stance.lower() == 'orthodox':
        lead_wr = get_xy(points.get(LEFT_WRIST))
        rear_wr = get_xy(points.get(RIGHT_WRIST))
    else:
        lead_wr = get_xy(points.get(RIGHT_WRIST))
        rear_wr = get_xy(points.get(LEFT_WRIST))

    if chin_y is None or (lead_wr is None and rear_wr is None):
        return {
            'high_guard_active': None,
            'both_hands_high': None,
            'lead_safe': None,
            'rear_safe': None,
            'status': 'unobservable',
        }

    lead_safe = (lead_wr[1] - chin_y <= chin_tolerance_ratio) if lead_wr else None
    rear_safe = (rear_wr[1] - chin_y <= chin_tolerance_ratio) if rear_wr else None
    both_high = bool(lead_safe and rear_safe)
    active = bool(lead_safe or rear_safe)
    return {
        'high_guard_active': active,
        'both_hands_high': both_high,
        'lead_safe': lead_safe,
        'rear_safe': rear_safe,
        'status': 'observed',
    }


def _reliable_landmark(point):
    xy = get_xy(point)
    return (xy is not None and all(0 <= v <= 1 for v in xy)
            and all(isinstance(point.get(key), (int, float)) and MIN_CONFIDENCE <= point[key] <= 1
                    for key in ("visibility", "presence")))


def _leg_channels(points, hip, knee, ankle, support, support_ankle, aspect):
    def image(index):
        return points[index]["image"] if index in points else None

    h, k, a, other, other_a = map(image, (hip, knee, ankle, support, support_ankle))
    if not all((h, k, a)):
        return {}
    def depth(point):
        z = point.get("z")
        return z if isinstance(z, (int, float)) and math.isfinite(z) else None

    az, hz = depth(a), depth(h)
    return {
        "kx": (k["x"] - h["x"]) * aspect, "ky": k["y"] - h["y"],
        "ax": (a["x"] - h["x"]) * aspect, "ay": a["y"] - h["y"],
        "az": (az - hz) * aspect if az is not None and hz is not None else None,
        "other_ay": (other_a["y"] - h["y"]) if other_a else None,
        "hip_width": (h["x"] - other["x"]) * aspect if other else None,
        "hip_tilt": h["y"] - other["y"] if other else None,
    }


def _classify_leg(ch, start, end, scale, times, guards):
    """Classify one completed excursion. Ambiguous extended legs stay unclassified."""
    kx, ky, ax, ay = (ch[k] for k in ("kx", "ky", "ax", "ay"))
    moving = range(start + 1, end)
    def shin_x(i):
        return abs(ax[i] - kx[i])

    zs = ch["az"][start:end + 1]
    depth_observed = None not in zs
    def foot_excursion(i):
        lateral = shin_x(i) - shin_x(start)
        depth = abs(ch["az"][i] - zs[0]) if depth_observed else 0.0
        return max(lateral, depth)

    foot = max(moving, key=foot_excursion)
    knee = min(moving, key=ky.__getitem__)
    knee_rise = ky[start] - ky[knee]
    # Footwork steps and stance switches keep knees low; require real knee lift (>= 22% of leg length).
    if knee_rise < 0.22 * scale:
        return None
    # Require the kicking/lifted ankle to clearly leave the floor (both vs start and vs standing ankle)
    # so stepping, pivoting, stance-switching, and upward-tilt depth noise never trigger false kicks.
    ankle_lift = ay[start] - ay[foot]
    other_ay_foot = ch["other_ay"][foot] if "other_ay" in ch else None
    standing_clearance = (other_ay_foot - ay[foot]) if other_ay_foot is not None else ankle_lift
    foot_clearly_airborne = ankle_lift >= 0.30 * scale and standing_clearance >= 0.28 * scale

    depth_sweep = depth_observed and abs(ch["az"][foot] - zs[0]) >= 0.20 * scale
    reaches_out = shin_x(foot) >= 0.28 * scale and shin_x(foot) - shin_x(start) >= 0.20 * scale
    if (reaches_out or depth_sweep) and foot_clearly_airborne:
        # Compare the pelvis with its own pre-lift pose. Side-on hip widths can approach zero.
        hip_windows = [ch[key][start:foot + 1] for key in ("hip_width", "hip_tilt")]
        if any(None in values for values in hip_windows):
            return None
        turn = max(max(abs(v - values[0]) for v in values) for values in hip_windows) / scale
        if turn >= 0.12 and depth_sweep:
            return "roundhouse_kick", foot
        chamber = next((i for i in range(start + 1, foot)
                        if ky[start] - ky[i] >= 0.18 * scale and shin_x(i) < 0.20 * scale), None)
        origin = chamber if chamber is not None else start
        dx, dy = ax[foot] - ax[origin], ay[foot] - ay[origin]
        chord = math.hypot(dx, dy)
        if chord < 0.22 * scale:
            return None
        path = range(origin, foot + 1)
        bend = max(abs(dx * (ay[i] - ay[origin]) - dy * (ax[i] - ax[origin]))
                   for i in path) / chord / scale
        horizontal_shin = shin_x(foot) >= abs(ay[foot] - ky[foot])
        if turn >= 0.12 and horizontal_shin and bend >= 0.06:
            return "roundhouse_kick", foot
        thigh = math.hypot(kx[foot], ky[foot])
        shin = math.hypot(ax[foot] - kx[foot], ay[foot] - ky[foot])
        extension = math.hypot(ax[foot], ay[foot]) / max(thigh + shin, 1e-6)
        path_length = sum(math.hypot(ax[i] - ax[i-1], ay[i] - ay[i-1]) for i in range(origin+1, foot+1))
        if (turn <= 0.08 and extension >= 0.84 and chord >= 0.92 * path_length
                and chamber is not None and times[foot] - times[chamber] >= 40
                and abs(dx) >= 0.22 * scale and bend <= 0.09 and not depth_sweep):
            return "teep_kick", foot
        return None

    other_ay_knee = ch["other_ay"][knee] if "other_ay" in ch else None
    knee_ankle_clearance = (other_ay_knee - ay[knee]) if other_ay_knee is not None else (ay[start] - ay[knee])
    if knee_ankle_clearance < 0.20 * scale:
        return None
    knee_drive = abs(kx[knee] - kx[start]) / scale
    if knee_drive >= 0.17 and knee_rise >= 0.22 * scale:
        return "knee_strike", knee
    if (knee_drive < 0.15 and shin_x(knee) <= 0.15 * scale
            and ay[knee] - ky[knee] > 0.12 * scale and guards[knee].get("both_hands_high")):
        return "shin_check", knee
    return None


def _leg_movements(rows, times, guards):
    keys = ("kx", "ky", "ax", "ay", "az", "other_ay", "hip_width", "hip_tilt")
    ch = {k: supported_signal([row.get(k) for row in rows], times) for k in keys}
    standing = sorted((v, i) for i, v in enumerate(ch["ay"]) if v is not None)
    if not standing:
        return
    # Use the most grounded quarter to estimate this leg's resting pose.
    grounded = [i for _, i in standing[3 * len(standing) // 4:]]
    base_k, base_a, base_x = (median(ch[k][i] for i in grounded) for k in ("ky", "ay", "ax"))
    scale = median(math.hypot(ch["ax"][i], ch["ay"][i]) for i in grounded)
    if scale <= 1e-4:
        return
    signal = [None if k is None else max(base_k - k, base_a - a, abs(x - base_x)) / scale
              for k, a, x in zip(ch["ky"], ch["ay"], ch["ax"])]
    last_peak = -math.inf
    for start, _, end in excursions(signal, times, 0.12, 0.18):
        event = _classify_leg(ch, start, end, scale, times, guards)
        if event is not None:
            kind, peak = event
            if times[peak] - last_peak >= LEG_INTERVAL_MS:
                yield kind, start, peak, end
                last_peak = times[peak]


def analyze_muaythai_arsenal(
    frames: Sequence[dict[str, Any]],
    *,
    stance: str = 'orthodox',
    single_person_confirmed: bool = True,
    width: int = 1280,
    height: int = 720,
    punch_events: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """B03: Analyze Muay Thai kicks, knee strikes, shin-check defense, high guard, and punch-to-kick/knee combos.

    Gracefully supports both full-body and half-body (upper-body only) videos.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    is_orthodox = stance.lower() == 'orthodox'
    lead_hip_idx, rear_hip_idx = (LEFT_HIP, RIGHT_HIP) if is_orthodox else (RIGHT_HIP, LEFT_HIP)
    lead_knee_idx, rear_knee_idx = (LEFT_KNEE, RIGHT_KNEE) if is_orthodox else (RIGHT_KNEE, LEFT_KNEE)
    lead_ankle_idx, rear_ankle_idx = (LEFT_ANKLE, RIGHT_ANKLE) if is_orthodox else (RIGHT_ANKLE, LEFT_ANKLE)

    high_guard_frames = 0
    upper_observed_frames = 0
    lower_observed_frames = 0

    times = [float(f["timestamp_ms"]) for f in frames]
    guards = []
    lead_rows = []
    rear_rows = []
    for f in frames:
        lms = f.get("landmarks", [])
        pts = {p["index"]: p for p in lms if _reliable_landmark(p)}
        hg = check_muaythai_high_guard(list(pts.values()), stance=stance)
        guards.append(hg)
        if hg["status"] == "observed":
            upper_observed_frames += 1
            high_guard_frames += int(hg["high_guard_active"])
        lead_rows.append(_leg_channels(pts, lead_hip_idx, lead_knee_idx, lead_ankle_idx, rear_hip_idx, rear_ankle_idx, width / height))
        rear_rows.append(_leg_channels(pts, rear_hip_idx, rear_knee_idx, rear_ankle_idx, lead_hip_idx, lead_ankle_idx, width / height))
        lower_observed_frames += int(bool(lead_rows[-1] or rear_rows[-1]))

    total_frames = len(frames)
    lower_coverage = (lower_observed_frames / total_frames) if total_frames > 0 else 0.0
    high_guard_rate = (
        round(high_guard_frames / upper_observed_frames, 3)
        if upper_observed_frames > 0 and single_person_confirmed
        else None
    )

    if lower_coverage < 0.20:
        body_framing = 'upper_body_only'
    else:
        body_framing = 'full_body'

    if punch_events is None:
        punch_events = straight_events(frames, stance.lower(), width=width, height=height) if single_person_confirmed else []
    leg_events = []
    shin_checks = []
    names = {"roundhouse_kick": "腿泰式扫踢", "teep_kick": "腿正蹬 / Stab (Teep)",
             "knee_strike": "膝撞 (冲膝)", "shin_check": "腿提膝格挡 (Shin Check)"}
    if body_framing == "full_body" and single_person_confirmed and stance.lower() in ("orthodox", "southpaw"):
        movements = [(peak, side, kind, start, end) for side, rows in (("前", lead_rows), ("后", rear_rows))
                     for kind, start, peak, end in _leg_movements(rows, times, guards)]
        for peak, side, kind, start, end in sorted(movements):
            event = {"category": kind, "technique_name": side + names[kind],
                     "frame_index": frames[peak]["source_frame_index"], "timestamp_ms": times[peak],
                     "start_ms": times[start], "return_ms": times[end]}
            if kind == "shin_check":
                shin_checks.append(dict(event, check_id=f"mt_check_{len(shin_checks)+1:02d}", high_guard_sealed=True))
            else:
                leg_events.append(dict(event, event_id=f"mt_leg_{len(leg_events)+1:02d}",
                                       return_time_ms=round(times[end] - times[peak], 2),
                                       guard_kept=bool(guards[peak].get("high_guard_active"))))

    # 3. Pair Boxing -> Muay Thai Kick/Knee Combos (拳腿/拳膝立体连击)
    punch_leg_combos: list[dict[str, Any]] = []
    used_legs: set[str] = set()
    for p in punch_events:
        best_leg = None
        for leg in leg_events:
            if leg['event_id'] in used_legs:
                continue
            dt = leg['timestamp_ms'] - p['timestamp_ms']
            if 60.0 <= dt <= 950.0:
                best_leg = (leg, round(dt, 1))
                break
        if best_leg is not None:
            leg, dt_ms = best_leg
            used_legs.add(leg['event_id'])
            punch_leg_combos.append({
                'combo_id': f'mt_combo_{len(punch_leg_combos) + 1:02d}',
                'combo_name': f"{p['punch_type']} → {leg['technique_name']}",
                'punch_frame_index': p['frame_index'],
                'leg_frame_index': leg['frame_index'],
                'transition_ms': dt_ms,
                'guard_kept_on_leg': leg['guard_kept'],
            })

    kick_count = sum(1 for e in leg_events if e['category'] == 'roundhouse_kick')
    teep_count = sum(1 for e in leg_events if e["category"] == "teep_kick")
    knee_count = sum(1 for e in leg_events if e['category'] == 'knee_strike')
    check_count = len(shin_checks)
    combo_count = len(punch_leg_combos)

    # 4. Build Coaching Feedback & B03 Drill Prescription
    if not single_person_confirmed:
        feedback = '待确认单人训练后启用泰拳专项（扫踢、正蹬、膝撞、提膝格挡与拳腿连击）统计。'
        drill = None
    elif body_framing == 'upper_body_only':
        hg_pct = f'{high_guard_rate * 100:.0f}%' if high_guard_rate is not None else '暂无'
        feedback = (
            f'当前为半身/近景画面（下肢关键点未完整入镜），已为您输出上肢泰拳高抱架保持率（{hg_pct}）；'
            '如需统计泰拳扫踢、正蹬、冲膝、提膝格挡与拳腿立体连击，请使用包含膝踝的全身镜头。'
        )
        drill = {
            'drill_id': 'B03_HIGH_GUARD_SHIELD',
            'title': '泰拳长抱架/高抱架（Long & High Guard）抗击打定型练习',
            'target_issue': '半身画面下的上肢泰拳抱架护眉与出拳回撤防迎击巩固',
            'dosage': '3 组 × 每组 90 秒空击定型',
            'steps': '双拳抬高至眉弓/额角高度，手肘内收护肋；每次刺拳或后手直拳出击后，异侧拳保持贴近眉弓与下颌。',
            'cue': '“拳高齐眉、肘收护肋、出拳不掉架”',
        }
    elif teep_count and not (kick_count or knee_count or check_count):
        feedback = (f"已回落动作中，扫踢 {kick_count} 次、正蹬 {teep_count} 次、冲膝 {knee_count} 次、"
                    f"提膝格挡 {check_count} 次。可按峰值源帧回看片段。")
        drill = None
    elif kick_count + knee_count + check_count == 0:
        hg_pct = f'{high_guard_rate * 100:.0f}%' if high_guard_rate is not None else '暂无'
        feedback = (
            f'全身画面已就绪（泰拳高抱架保持率 {hg_pct}），本段未识别到分类证据充分且完成回落的腿部动作。'
        )
        drill = {
            'drill_id': 'B03_DUTCH_COMBO_INTRO',
            'title': '1-2 直拳衔接后腿泰式扫踢（拳腿立体转换）练习',
            'target_issue': '从纯拳法过渡到拳腿立体进攻，缩短拳后起腿犹豫时间',
            'dosage': '3 组 × 10 次（1-2 直拳掩护 → 后腿中段扫踢 → 原路线落回抱架）',
            'steps': '前手刺拳与后手直拳吸引对手高位防守，后手回收至下巴瞬间前脚外展 45° 作轴，送髋挥出后腿扫踢。',
            'cue': '“拳收腿起、前脚开轴、踢完即回”',
        }
    else:
        unguarded_legs = [e for e in leg_events if not e['guard_kept']]
        if unguarded_legs:
            first_bad = unguarded_legs[0]
            feedback = (
                f"检出泰拳扫踢 {kick_count} 次、正蹬 {teep_count} 次、膝撞 {knee_count} 次、提膝格挡 {check_count} 次、拳腿/拳膝连击 {combo_count} 组；"
                f"注意在源帧 {first_bad['frame_index']}（{first_bad['technique_name']}）起腿瞬间护头手下掉，易遭迎击拳反击。"
            )
            drill = {
                'drill_id': 'B03_KICK_GUARD_SEAL',
                'title': '扫踢/冲膝单手甩臂与异侧贴腮护颌协同练习',
                'target_issue': '起腿或冲膝瞬间双臂同时下甩导致下巴暴露',
                'dosage': '3 组 × 12 次（慢速分解 4 次 + 正常速度 8 次）',
                'steps': '起后腿扫踢或冲膝时，前手紧贴眉弓/颧骨护头（或前伸封堵对手视线），踢击落回抱架后再松手。',
                'cue': '“一手甩平衡、一手贴眉弓”',
            }
        else:
            feedback = (
                f'泰拳立体攻防优秀：共检出扫踢 {kick_count} 次、正蹬 {teep_count} 次、膝撞 {knee_count} 次、'
                f'提膝格挡 {check_count} 次、拳腿/拳膝连击 {combo_count} 组，起腿期间高抱架护头稳定。'
            )
            drill = {
                'drill_id': 'B03_CHECK_AND_COUNTER',
                'title': '提膝格挡（Shin Check）秒接后腿反击扫踢/冲膝练习',
                'target_issue': '巩固立体攻防转换速度与格挡后重心的瞬时回弹',
                'dosage': '3 组 × 10 次',
                'steps': '前腿提膝 45° 外旋贴肘完成格挡，脚尖落地瞬间立即反弹起后腿扫踢或后膝撞，全程双拳高抱架。',
                'cue': '“提膝贴肘、落地即反”',
            }

    return {
        'stance': stance.lower(),
        'body_framing': body_framing,
        'lower_body_coverage_ratio': round(lower_coverage, 3),
        'high_guard_retention_rate': high_guard_rate,
        'kick_count': kick_count,
        'teep_count': teep_count,
        'knee_count': knee_count,
        'shin_check_count': check_count,
        'combo_count': combo_count,
        'leg_events': leg_events,
        'shin_checks': shin_checks,
        'punch_leg_combos': punch_leg_combos,
        'muaythai_feedback': feedback,
        'muaythai_drill': drill,
    }
