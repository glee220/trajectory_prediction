from shapely.geometry import Polygon
import numpy as np

def angle_between(v1, v2):
    """计算两个向量间的夹角（角度）"""
    unit_v1 = v1 / (np.linalg.norm(v1) + 1e-6)
    unit_v2 = v2 / (np.linalg.norm(v2) + 1e-6)
    dot = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
    angle = np.arccos(dot)
    return np.degrees(angle)

def direction_change(traj):
    """起始方向和终止方向的夹角"""
    v1 = traj[1] - traj[0]
    v2 = traj[-1] - traj[-2]
    return angle_between(v1, v2)


def crosses_multiple_lanes(traj, map_api):
    lane_ids = set()
    for pt in traj:
        lanes = map_api.get_records_in_radius(pt[0], pt[1], 1.0, ['lane'])
        for lane in lanes.get("lane", []):
            lane_ids.add(lane)
    return len(lane_ids) > 1


def within_same_road(traj, map_api):
    road_ids = set()
    for pt in traj:
        recs = map_api.get_records_in_radius(pt[0], pt[1], 1.0, ['road_segment'])
        for road in recs.get('road_segment', []):
            road_ids.add(road)
    return len(road_ids) == 1


def crosses_intersection(traj, map_api):
    """判断轨迹是否穿过路口区域（交叉口）"""
    intersection_hits = 0
    for pt in traj:
        roads = map_api.get_records_in_radius(pt[0], pt[1], 1.0, ['road_segment'])
        for token in roads.get('road_segment', []):
            road_info = map_api.get('road_segment', token)
            if road_info.get('is_intersection', False):
                intersection_hits += 1
                break
    return intersection_hits >= 2  # 至少2个点在路口上


def cumulative_turn_angle(traj):
    """计算轨迹中每一段的转角总和（累积转向角）"""
    angles = 0
    for i in range(1, len(traj) - 1):
        v1 = traj[i] - traj[i - 1]
        v2 = traj[i + 1] - traj[i]
        angle = angle_between(v1, v2)
        angles += angle
    return angles

def direction_variability(traj):
    angles = []
    for i in range(1, len(traj) - 1):
        v1 = traj[i] - traj[i - 1]
        v2 = traj[i + 1] - traj[i]
        angles.append(angle_between(v1, v2))
    return np.std(angles), np.mean(angles)

def is_turn(trajectory, angle_threshold=20):
    """trajectory: list of (x, y), 判断是否有显著转弯"""
    angles = []
    for i in range(1, len(trajectory)-1):
        v1 = np.array(trajectory[i]) - np.array(trajectory[i-1])
        v2 = np.array(trajectory[i+1]) - np.array(trajectory[i])
        if np.linalg.norm(v1) < 1e-2 or np.linalg.norm(v2) < 1e-2:
            continue
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        angles.append(angle)
    turn_count = sum(1 for a in angles if a > angle_threshold)
    return turn_count >= 2


def classify_motion_type(observe_traj, future_traj, map_api):
    """
    输入：
        observe_traj: numpy array, [T1, 2]
        future_traj: numpy array, [T2, 2]
        map_api: NuScenesMap 对象
    输出：
        str: motion_type 类型
        分类类型: 'straight', 'turn', 'lane_change', 'intersection_pass', 'other'
    """
    traj_full = np.concatenate([observe_traj, future_traj], axis=0)
    
    total_length = np.sum(np.linalg.norm(np.diff(traj_full, axis=0), axis=1))

    heading_change = angle_between(observe_traj[-1] - observe_traj[-2], future_traj[-1] - future_traj[-2])

    curve_total = cumulative_turn_angle(traj_full)

    std_angle, mean_angle = direction_variability(traj_full)
    
    dir_change = direction_change(traj_full)

    # 1. 非法/短轨迹：排除过短轨迹
    if total_length < 3.0:
        return 'other'

    # if crosses_multiple_lanes(future_traj, map_api) and within_same_road(traj_full, map_api):
    #     if dir_change < 20 and curve_total < 40:
    #         return 'lane_change'

    # 然后再判断是否直行
    if abs(heading_change) < 10 and std_angle < 5 and curve_total < 30:
        return 'straight'


    # 4. 转弯：明显转角，曲率高，方向突变
    if is_turn(traj_full) or curve_total > 60 or dir_change > 45:
        return 'turn'

    # 5. 其他：无法归类的情况
    return 'other'                                                                                                                      


