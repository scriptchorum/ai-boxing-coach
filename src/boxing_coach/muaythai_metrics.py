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


def _median_or_default(values: Sequence[float], default: float) -> float:
    if not values:
        return default
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 == 1 else (s[mid - 1] + s[mid]) / 2.0


def analyze_muaythai_arsenal(
    frames: Sequence[dict[str, Any]],
    *,
    stance: str = 'orthodox',
    single_person_confirmed: bool = True,
) -> dict[str, Any]:
    """B03: Analyze Muay Thai kicks, knee strikes, shin-check defense, high guard, and punch-to-kick/knee combos.

    Gracefully supports both full-body and half-body (upper-body only) videos.
    """
    is_orthodox = stance.lower() == 'orthodox'
    lead_hip_idx, rear_hip_idx = (LEFT_HIP, RIGHT_HIP) if is_orthodox else (RIGHT_HIP, LEFT_HIP)
    lead_knee_idx, rear_knee_idx = (LEFT_KNEE, RIGHT_KNEE) if is_orthodox else (RIGHT_KNEE, LEFT_KNEE)
    lead_ankle_idx, rear_ankle_idx = (LEFT_ANKLE, RIGHT_ANKLE) if is_orthodox else (RIGHT_ANKLE, LEFT_ANKLE)

    high_guard_frames = 0
    upper_observed_frames = 0
    lower_observed_frames = 0

    lead_knee_gaps: list[float] = []
    rear_knee_gaps: list[float] = []
    lead_ankle_ys: list[float] = []
    rear_ankle_ys: list[float] = []

    per_frame: list[dict[str, Any]] = []
    for idx, f in enumerate(frames):
        lms = f.get('landmarks', [])
        pts = {p['index']: p for p in lms if 'index' in p}
        hg = check_muaythai_high_guard(lms, stance=stance)
        if hg['status'] == 'observed':
            upper_observed_frames += 1
            if hg['high_guard_active']:
                high_guard_frames += 1

        lead_ext = compute_arm_extension_ratio(lms, arm='lead', stance=stance).get('extension_ratio')
        rear_ext = compute_arm_extension_ratio(lms, arm='rear', stance=stance).get('extension_ratio')

        l_hip = get_xy(pts.get(lead_hip_idx))
        r_hip = get_xy(pts.get(rear_hip_idx))
        l_knee = get_xy(pts.get(lead_knee_idx))
        r_knee = get_xy(pts.get(rear_knee_idx))
        l_ank = get_xy(pts.get(lead_ankle_idx))
        r_ank = get_xy(pts.get(rear_ankle_idx))

        has_lower = bool((l_hip and l_knee) or (r_hip and r_knee))
        if has_lower:
            lower_observed_frames += 1
        if l_hip and l_knee:
            lead_knee_gaps.append(l_knee[1] - l_hip[1])
        if r_hip and r_knee:
            rear_knee_gaps.append(r_knee[1] - r_hip[1])
        if l_ank:
            lead_ankle_ys.append(l_ank[1])
        if r_ank:
            rear_ankle_ys.append(r_ank[1])

        per_frame.append({
            'pos': idx,
            'frame_index': int(f.get('source_frame_index', idx)),
            'timestamp_ms': float(f.get('timestamp_ms', idx * 33.333)),
            'lead_ext': lead_ext,
            'rear_ext': rear_ext,
            'hg': hg,
            'l_hip': l_hip,
            'r_hip': r_hip,
            'l_knee': l_knee,
            'r_knee': r_knee,
            'l_ank': l_ank,
            'r_ank': r_ank,
        })

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

    base_l_knee_gap = _median_or_default(lead_knee_gaps, 0.18)
    base_r_knee_gap = _median_or_default(rear_knee_gaps, 0.18)
    base_l_ank_y = _median_or_default(lead_ankle_ys, 0.84)
    base_r_ank_y = _median_or_default(rear_ankle_ys, 0.84)

    punch_events: list[dict[str, Any]] = []
    leg_events: list[dict[str, Any]] = []
    shin_checks: list[dict[str, Any]] = []

    last_lead_punch_pos = -5
    last_rear_punch_pos = -5
    last_leg_pos = -6

    for i, pf in enumerate(per_frame):
        # 1. Detect straight punch peaks for combo pairing
        for arm_key, p_type, last_pos_ref in (
            ('lead_ext', '前手刺拳', 'lead'),
            ('rear_ext', '后手直拳', 'rear'),
        ):
            val = pf[arm_key]
            if val is not None and val >= 0.80:
                prev_v = per_frame[i - 1][arm_key] if i > 0 else None
                next_v = per_frame[i + 1][arm_key] if i + 1 < len(per_frame) else None
                if (prev_v is None or val >= prev_v) and (next_v is None or val >= next_v):
                    last_pos = last_lead_punch_pos if last_pos_ref == 'lead' else last_rear_punch_pos
                    if i - last_pos >= 4:
                        if last_pos_ref == 'lead':
                            last_lead_punch_pos = i
                        else:
                            last_rear_punch_pos = i
                        punch_events.append({
                            'pos': i,
                            'punch_type': p_type,
                            'frame_index': pf['frame_index'],
                            'timestamp_ms': pf['timestamp_ms'],
                        })

        if not single_person_confirmed or body_framing == 'upper_body_only':
            continue

        if i - last_leg_pos < 5:
            continue

        # 2. Check Lead / Rear Leg actions (Roundhouse Kick, Knee Strike, Shin Check)
        for side_label, hip_pt, knee_pt, ank_pt, base_gap, base_ank in (
            ('前', pf['l_hip'], pf['l_knee'], pf['l_ank'], base_l_knee_gap, base_l_ank_y),
            ('后', pf['r_hip'], pf['r_knee'], pf['r_ank'], base_r_knee_gap, base_r_ank_y),
        ):
            if not hip_pt or not knee_pt:
                continue
            knee_gap = knee_pt[1] - hip_pt[1]
            knee_rise = base_gap - knee_gap
            ank_rise = (base_ank - ank_pt[1]) if ank_pt is not None else 0.0

            # Check if local apex of knee/ankle elevation
            next_pf = per_frame[i + 1] if i + 1 < len(per_frame) else None
            prev_pf = per_frame[i - 1] if i > 0 else None
            knee_key = 'l_knee' if side_label == '前' else 'r_knee'
            ank_key = 'l_ank' if side_label == '前' else 'r_ank'
            if prev_pf and prev_pf[knee_key] and knee_pt[1] > prev_pf[knee_key][1] + 0.005:
                continue
            if next_pf and next_pf[knee_key] and knee_pt[1] > next_pf[knee_key][1] + 0.005:
                continue

            # Evaluate leg extension & horizontal reach for Roundhouse Kick
            if ank_pt is not None and (
                ank_pt[1] <= hip_pt[1] + 0.12
                or (ank_rise >= 0.12 and abs(ank_pt[0] - hip_pt[0]) >= 0.10)
            ):
                thigh = math.hypot(knee_pt[0] - hip_pt[0], knee_pt[1] - hip_pt[1])
                shin = math.hypot(ank_pt[0] - knee_pt[0], ank_pt[1] - knee_pt[1])
                leg_len = thigh + shin
                direct_leg = math.hypot(ank_pt[0] - hip_pt[0], ank_pt[1] - hip_pt[1])
                leg_ext_ratio = (direct_leg / leg_len) if leg_len > 1e-4 else 0.0
                horiz_swing = abs(ank_pt[0] - hip_pt[0])

                if leg_ext_ratio >= 0.78 or horiz_swing >= 0.12:
                    # Find kick return frame
                    ret_ms: float | None = None
                    for j in range(i + 1, len(per_frame)):
                        fut_ank = per_frame[j][ank_key]
                        if fut_ank is not None and abs(fut_ank[1] - base_ank) <= 0.045:
                            ret_ms = round(max(0.0, per_frame[j]['timestamp_ms'] - pf['timestamp_ms']), 2)
                            break
                    last_leg_pos = i
                    leg_events.append({
                        'event_id': f'mt_leg_{len(leg_events) + 1:02d}',
                        'category': 'roundhouse_kick',
                        'technique_name': f'{side_label}腿泰式扫踢',
                        'frame_index': pf['frame_index'],
                        'timestamp_ms': round(pf['timestamp_ms'], 2),
                        'return_time_ms': ret_ms,
                        'guard_kept': bool(pf['hg'].get('high_guard_active')),
                    })
                    break

            # Otherwise check Knee Strike vs Shin Check Defense when knee rises
            if knee_rise >= 0.055 or knee_gap <= 0.09:
                shin_vertical = (
                    ank_pt is not None
                    and abs(ank_pt[0] - knee_pt[0]) <= 0.06
                    and ank_pt[1] > knee_pt[1] + 0.05
                )
                both_arms_tucked = (
                    (pf['lead_ext'] is None or pf['lead_ext'] < 0.72)
                    and (pf['rear_ext'] is None or pf['rear_ext'] < 0.72)
                    and bool(pf['hg'].get('both_hands_high'))
                )
                knee_forward_thrust = abs(knee_pt[0] - hip_pt[0]) >= 0.065 or knee_gap <= 0.04
                if shin_vertical and both_arms_tucked and not knee_forward_thrust:
                    last_leg_pos = i
                    shin_checks.append({
                        'check_id': f'mt_check_{len(shin_checks) + 1:02d}',
                        'technique_name': f'{side_label}腿提膝格挡 (Shin Check)',
                        'frame_index': pf['frame_index'],
                        'timestamp_ms': round(pf['timestamp_ms'], 2),
                        'high_guard_sealed': True,
                    })
                    break
                else:
                    last_leg_pos = i
                    leg_events.append({
                        'event_id': f'mt_leg_{len(leg_events) + 1:02d}',
                        'category': 'knee_strike',
                        'technique_name': f'{side_label}膝撞 (冲膝)',
                        'frame_index': pf['frame_index'],
                        'timestamp_ms': round(pf['timestamp_ms'], 2),
                        'return_time_ms': None,
                        'guard_kept': bool(pf['hg'].get('high_guard_active')),
                    })
                    break

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
    knee_count = sum(1 for e in leg_events if e['category'] == 'knee_strike')
    check_count = len(shin_checks)
    combo_count = len(punch_leg_combos)

    # 4. Build Coaching Feedback & B03 Drill Prescription
    if not single_person_confirmed:
        feedback = '待确认单人训练后启用泰拳专项（扫踢、膝撞、提膝格挡与拳腿连击）统计。'
        drill = None
    elif body_framing == 'upper_body_only':
        hg_pct = f'{high_guard_rate * 100:.0f}%' if high_guard_rate is not None else '暂无'
        feedback = (
            f'当前为半身/近景画面（下肢关键点未完整入镜），已为您输出上肢泰拳高抱架保持率（{hg_pct}）；'
            '如需统计泰拳扫踢、冲膝、提膝格挡与拳腿立体连击，请使用包含膝踝的全身镜头。'
        )
        drill = {
            'drill_id': 'B03_HIGH_GUARD_SHIELD',
            'title': '泰拳长抱架/高抱架（Long & High Guard）抗击打定型练习',
            'target_issue': '半身画面下的上肢泰拳抱架护眉与出拳回撤防迎击巩固',
            'dosage': '3 组 × 每组 90 秒空击定型',
            'steps': '双拳抬高至眉弓/额角高度，手肘内收护肋；每次刺拳或后手直拳出击后，异侧拳保持贴近眉弓与下颌。',
            'cue': '“拳高齐眉、肘收护肋、出拳不掉架”',
        }
    elif kick_count + knee_count + check_count == 0:
        hg_pct = f'{high_guard_rate * 100:.0f}%' if high_guard_rate is not None else '暂无'
        feedback = (
            f'全身画面已就绪（泰拳高抱架保持率 {hg_pct}），本段以拳法动作为主，未检出明显扫踢、冲膝或提膝格挡。'
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
                f"检出泰拳扫踢 {kick_count} 次、膝撞 {knee_count} 次、提膝格挡 {check_count} 次、拳腿/拳膝连击 {combo_count} 组；"
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
                f'泰拳立体攻防优秀：共检出扫踢 {kick_count} 次、膝撞 {knee_count} 次、'
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
        'knee_count': knee_count,
        'shin_check_count': check_count,
        'combo_count': combo_count,
        'leg_events': leg_events,
        'shin_checks': shin_checks,
        'punch_leg_combos': punch_leg_combos,
        'muaythai_feedback': feedback,
        'muaythai_drill': drill,
    }
