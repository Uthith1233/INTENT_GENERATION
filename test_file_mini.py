import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


DATA_ROOT = Path(".")                  # run from ~/Downloads/v1.0-trainval
JSON_DIR = DATA_ROOT / "v1.0-trainval"
BLOB_ROOTS = [
    DATA_ROOT,
    DATA_ROOT / "v1.0-trainval02",
    DATA_ROOT / "v1.0-trainval03",
    DATA_ROOT / "v1.0-trainval04",
    DATA_ROOT / "v1.0-trainval05",
    DATA_ROOT / "v1.0-trainval06",
]



OUTPUT_DIR = Path("outputs_intent_pipeline")
VIS_DIR = OUTPUT_DIR / "camera_visualizations"

OUTPUT_DIR.mkdir(exist_ok=True)
VIS_DIR.mkdir(exist_ok=True)

TRAJ_CSV = OUTPUT_DIR / "trajectory_motion_table.csv"
CAMERA_BOX_CSV = OUTPUT_DIR / "camera_box_projection_table.csv"
VISIBLE_CAMERA_BOX_CSV = OUTPUT_DIR / "visible_camera_box_table.csv"
MERGED_CSV = OUTPUT_DIR / "merged_camera_trajectory_table.csv"

TARGET_CATEGORY_PREFIXES = (
    "vehicle.",
    "human.pedestrian",
    "movable_object.",
    "static_object.",
)

LSTM_WINDOW_NPZ = OUTPUT_DIR / "lstm_trajectory_windows.npz"

LSTM_WINDOW_INDEX_CSV = OUTPUT_DIR / "lstm_traj_window_index.csv"
SAVE_COMPRESSED_LSTM_NPZ = False

HISTORY_FRAMES = 6
FUTURE_FRAMES = 4

MAX_NEIGHBORS = 8
SOCIAL_RADIUS = 12.0
OBJECT_SOCIAL_RADIUS = 10.0
MIN_SOCIAL_DISTANCE = 0.25
NEIGHBOR_FEATURE_COLS = [
    "neighbor_rel_x",
    "neighbor_rel_y",
    "neighbor_distance",
    "neighbor_speed",
    "neighbor_acceleration",
    "neighbor_v_forward",
    "neighbor_v_lateral",
    "neighbor_is_vehicle",
    "neighbor_is_pedestrian",
    "neighbor_is_object",
]

FUTURE_FRAMES_BY_LABEL = {
    "vehicle_cut_in": 6,
    "vehicle_braking": 12,
    "pedestrian_crossing": 4,
    "obstacle_approach": 4,
}

LSTM_LABELS = [
    "none",
    "vehicle_cut_in",
    "vehicle_braking",
    "pedestrian_crossing",
    "obstacle_approach",
]

# Use only front camera for bbox projection.
RUN_CAMERA_PROJECTION = False
CAMERA_CHANNELS = [
    "CAM_FRONT",
]

# Visualization settings.
SAVE_CAMERA_VISUALIZATIONS = True
MAX_VIS_IMAGES = 200

TARGET_SCENE_NAME = {
    "scene-0207",
    "scene-0436",
    "scene-0479",
    "scene-0504",
}


# Projection settings.
MIN_DEPTH = 0.1
SKIP_BOX_IF_ANY_CORNER_BEHIND_CAMERA = True

# Simple ego-lane approximation.
SAME_LANE_HALF_WIDTH = 1.8
ADJACENT_LANE_LIMIT = 5.5


def load_json(filename):
    path = JSON_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")

    with open(path, "r") as f:
        return json.load(f)


def resolve_data_file(filename):
    for blob_root in BLOB_ROOTS:
        path = blob_root / filename
        if path.exists():
            return path

    return None


print("Loading nuScenes JSON files...")


#IF NOT FILE NOPT FOUND ERROR
samples = load_json("sample.json")
sample_data = load_json("sample_data.json")
sample_annotations = load_json("sample_annotation.json")
ego_poses = load_json("ego_pose.json")
calibrated_sensors = load_json("calibrated_sensor.json")
sensors = load_json("sensor.json")
instances = load_json("instance.json")
categories = load_json("category.json")
scenes = load_json("scene.json")
attributes = load_json("attribute.json")


print("Loaded JSON files.")



sample_by_token = {s["token"]: s for s in samples}
sample_data_by_token = {sd["token"]: sd for sd in sample_data}
ann_by_token = {a["token"]: a for a in sample_annotations}
ego_pose_by_token = {e["token"]: e for e in ego_poses}
calib_by_token = {c["token"]: c for c in calibrated_sensors}
sensor_by_token = {s["token"]: s for s in sensors}
instance_by_token = {i["token"]: i for i in instances}
category_by_token = {c["token"]: c for c in categories}
scene_by_token = {s["token"]: s for s in scenes}
attribute_by_token = {a["token"]: a for a in attributes}

# sample_token -> list(annotation_token)
sample_to_anns = defaultdict(list)
for ann in sample_annotations:
    sample_to_anns[ann["sample_token"]].append(ann["token"])

# sample_token -> channel -> sample_data_token
sample_to_channel_data = defaultdict(dict)
sample_to_all_sample_data = defaultdict(list)

for sd in sample_data:
    sample_token = sd["sample_token"]
    sample_to_all_sample_data[sample_token].append(sd["token"])

    calib = calib_by_token[sd["calibrated_sensor_token"]]
    sensor = sensor_by_token[calib["sensor_token"]]
    channel = sensor["channel"]

    # Prefer keyframes when duplicated.
    if channel not in sample_to_channel_data[sample_token]:
        sample_to_channel_data[sample_token][channel] = sd["token"]
    else:
        old_sd = sample_data_by_token[sample_to_channel_data[sample_token][channel]]
        old_is_key = old_sd.get("is_key_frame", False)
        new_is_key = sd.get("is_key_frame", False)

        if new_is_key and not old_is_key:
            sample_to_channel_data[sample_token][channel] = sd["token"]


# instance_token -> category info
instance_info = {}

for inst in instances:
    cat = category_by_token[inst["category_token"]]
    instance_info[inst["token"]] = {
        "category": cat["name"],
        "category_description": cat.get("description", ""),
        "first_annotation_token": inst["first_annotation_token"],
        "last_annotation_token": inst["last_annotation_token"],
    }


# ============================================================
# GEOMETRY UTILS
# ============================================================

def quaternion_to_rotation_matrix(q):
    """
    nuScenes quaternion format: [w, x, y, z].

    Matrix convention:
    point_parent = R @ point_child + translation
    """
    w, x, y, z = q

    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def quaternion_to_yaw(q):
    """
    nuScenes quaternion format: [w, x, y, z].
    Returns yaw in radians.
    """
    w, x, y, z = q

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)

    return np.arctan2(siny_cosp, cosy_cosp)


def angle_wrap(angle):
    """
    Wrap angle to [-pi, pi].
    """
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def global_to_ego_xy(global_x, global_y, ego_x, ego_y, ego_yaw):
    """
    Convert global x,y into ego-centric coordinates.

    rel_x > 0 means object is in front of ego.
    rel_y > 0 means object is left of ego.
    """
    dx = global_x - ego_x
    dy = global_y - ego_y

    c = np.cos(ego_yaw)
    s = np.sin(ego_yaw)

    rel_x = c * dx + s * dy
    rel_y = -s * dx + c * dy

    return rel_x, rel_y


def get_category_name(annotation):
    inst = instance_by_token.get(annotation["instance_token"])

    if inst is None:
        return "unknown"

    cat = category_by_token.get(inst["category_token"])

    if cat is None:
        return "unknown"

    return cat["name"]


def get_coarse_actor_type(category_name):
    if category_name.startswith("vehicle."):
        return "vehicle"

    if category_name.startswith("human.pedestrian"):
        return "pedestrian"

    return "other"


def is_target_category(category_name):
    return any(category_name.startswith(prefix) for prefix in TARGET_CATEGORY_PREFIXES)


def get_attribute_names(annotation):
    names = []

    for tok in annotation.get("attribute_tokens", []):
        attr = attribute_by_token.get(tok)
        if attr is not None:
            names.append(attr["name"])

    return "|".join(names)


def get_ego_pose_for_sample(sample_token):
    """
    Prefer LIDAR_TOP ego pose because it is the canonical sample timestamp.
    If unavailable, use the first available sample_data ego pose.
    """
    channel_map = sample_to_channel_data.get(sample_token, {})

    if "LIDAR_TOP" in channel_map:
        sd = sample_data_by_token[channel_map["LIDAR_TOP"]]
        return ego_pose_by_token[sd["ego_pose_token"]]

    sd_tokens = sample_to_all_sample_data.get(sample_token, [])

    if len(sd_tokens) == 0:
        return None

    sd = sample_data_by_token[sd_tokens[0]]
    return ego_pose_by_token[sd["ego_pose_token"]]


def simple_lane_relation(rel_y):
    if abs(rel_y) < SAME_LANE_HALF_WIDTH:
        return "same_lane"

    if SAME_LANE_HALF_WIDTH <= rel_y < ADJACENT_LANE_LIMIT:
        return "left_adjacent"

    if -ADJACENT_LANE_LIMIT < rel_y <= -SAME_LANE_HALF_WIDTH:
        return "right_adjacent"

    return "far"


# ============================================================
# STEP 1 + STEP 2: TRAJECTORY TABLE + EGO-CENTRIC FEATURES
# ============================================================

print("\nBuilding trajectory table with ego-centric features...")

traj_rows = []

for ann in sample_annotations:
    category_name = get_category_name(ann)

    if not is_target_category(category_name):
        continue

    sample_token = ann["sample_token"]
    sample = sample_by_token[sample_token]
    scene = scene_by_token[sample["scene_token"]]

    ego_pose = get_ego_pose_for_sample(sample_token)

    if ego_pose is None:
        continue

    ego_x, ego_y, ego_z = ego_pose["translation"]
    ego_yaw = quaternion_to_yaw(ego_pose["rotation"])

    global_x, global_y, global_z = ann["translation"]
    width, length, height = ann["size"]

    yaw = quaternion_to_yaw(ann["rotation"])
    yaw_relative_to_ego = angle_wrap(yaw - ego_yaw)

    rel_x, rel_y = global_to_ego_xy(
        global_x=global_x,
        global_y=global_y,
        ego_x=ego_x,
        ego_y=ego_y,
        ego_yaw=ego_yaw,
    )

    distance_2d = np.sqrt(rel_x ** 2 + rel_y ** 2)

    timestamp_sec = sample["timestamp"] / 1e6

    traj_rows.append({
        "scene_token": sample["scene_token"],
        "scene_name": scene["name"],
        "sample_token": sample_token,
        "timestamp_us": sample["timestamp"],
        "timestamp_sec": timestamp_sec,

        "instance_token": ann["instance_token"],
        "annotation_token": ann["token"],

        "category": category_name,
        "actor_type": get_coarse_actor_type(category_name),
        "attribute_names": get_attribute_names(ann),

        # global annotation pose
        "global_x": global_x,
        "global_y": global_y,
        "global_z": global_z,
        "yaw_global": yaw,

        # object size
        "width": width,
        "length": length,
        "height": height,

        # ego pose
        "ego_x": ego_x,
        "ego_y": ego_y,
        "ego_z": ego_z,
        "ego_yaw": ego_yaw,

        # ego-relative position
        "rel_x": rel_x,
        "rel_y": rel_y,
        "distance_2d": distance_2d,
        "yaw_relative_to_ego": yaw_relative_to_ego,
        "lane_relation_simple": simple_lane_relation(rel_y),

        # raw annotation metadata
        "visibility_token": int(ann.get("visibility_token", 0)),
        "num_lidar_pts": int(ann.get("num_lidar_pts", 0)),
        "num_radar_pts": int(ann.get("num_radar_pts", 0)),
        "prev_annotation_token": ann.get("prev", ""),
        "next_annotation_token": ann.get("next", ""),
    })

traj_df = pd.DataFrame(traj_rows)

traj_df = traj_df.sort_values(
    ["scene_token", "instance_token", "timestamp_sec"]
).reset_index(drop=True)

print(f"Trajectory rows after filtering: {len(traj_df)}")
print("Actor type counts:")
print(traj_df["actor_type"].value_counts())


# ============================================================
# STEP 3: MOTION FEATURES
# ============================================================

print("\nComputing motion features...")

# dt per agent track
traj_df["dt"] = traj_df.groupby(["scene_token", "instance_token"])["timestamp_sec"].diff()

# Replace invalid dt with NaN.
traj_df.loc[traj_df["dt"] <= 0, "dt"] = np.nan

# Agent global velocity.
traj_df["agent_vx_global_raw"] = (
    traj_df.groupby(["scene_token", "instance_token"])["global_x"].diff() / traj_df["dt"]
)

traj_df["agent_vy_global_raw"] = (
    traj_df.groupby(["scene_token", "instance_token"])["global_y"].diff() / traj_df["dt"]
)

# Fill first velocity values inside each track.
traj_df["agent_vx_global"] = (
    traj_df.groupby(["scene_token", "instance_token"])["agent_vx_global_raw"]
    .transform(lambda s: s.bfill().ffill())
    .fillna(0.0)
)

traj_df["agent_vy_global"] = (
    traj_df.groupby(["scene_token", "instance_token"])["agent_vy_global_raw"]
    .transform(lambda s: s.bfill().ffill())
    .fillna(0.0)
)

traj_df["agent_speed"] = np.sqrt(
    traj_df["agent_vx_global"] ** 2 + traj_df["agent_vy_global"] ** 2
)

# Agent acceleration.
traj_df["agent_acceleration_raw"] = (
    traj_df.groupby(["scene_token", "instance_token"])["agent_speed"].diff() / traj_df["dt"]
)

traj_df["agent_acceleration"] = (
    traj_df.groupby(["scene_token", "instance_token"])["agent_acceleration_raw"]
    .transform(lambda s: s.bfill().ffill())
    .fillna(0.0)
)

# Yaw rate.
traj_df["yaw_diff"] = traj_df.groupby(["scene_token", "instance_token"])["yaw_global"].diff()
traj_df["yaw_diff"] = traj_df["yaw_diff"].apply(lambda x: angle_wrap(x) if pd.notna(x) else np.nan)

traj_df["yaw_rate_raw"] = traj_df["yaw_diff"] / traj_df["dt"]

traj_df["yaw_rate"] = (
    traj_df.groupby(["scene_token", "instance_token"])["yaw_rate_raw"]
    .transform(lambda s: s.bfill().ffill())
    .fillna(0.0)
)

# Agent velocity projected into ego frame.
traj_df["agent_v_forward"] = (
    traj_df["agent_vx_global"] * np.cos(traj_df["ego_yaw"]) +
    traj_df["agent_vy_global"] * np.sin(traj_df["ego_yaw"])
)

traj_df["agent_v_lateral"] = (
    -traj_df["agent_vx_global"] * np.sin(traj_df["ego_yaw"]) +
    traj_df["agent_vy_global"] * np.cos(traj_df["ego_yaw"])
)

# Ego motion table.
ego_rows = []

for sample in samples:
    sample_token = sample["token"]
    ego_pose = get_ego_pose_for_sample(sample_token)

    if ego_pose is None:
        continue

    scene = scene_by_token[sample["scene_token"]]

    ego_x, ego_y, ego_z = ego_pose["translation"]
    ego_yaw = quaternion_to_yaw(ego_pose["rotation"])

    ego_rows.append({
        "scene_token": sample["scene_token"],
        "scene_name": scene["name"],
        "sample_token": sample_token,
        "timestamp_sec": sample["timestamp"] / 1e6,
        "ego_x_for_motion": ego_x,
        "ego_y_for_motion": ego_y,
        "ego_yaw_for_motion": ego_yaw,
    })

ego_df = pd.DataFrame(ego_rows)
ego_df = ego_df.sort_values(["scene_token", "timestamp_sec"]).reset_index(drop=True)

ego_df["ego_dt"] = ego_df.groupby("scene_token")["timestamp_sec"].diff()
ego_df.loc[ego_df["ego_dt"] <= 0, "ego_dt"] = np.nan

ego_df["ego_vx_global_raw"] = ego_df.groupby("scene_token")["ego_x_for_motion"].diff() / ego_df["ego_dt"]
ego_df["ego_vy_global_raw"] = ego_df.groupby("scene_token")["ego_y_for_motion"].diff() / ego_df["ego_dt"]

ego_df["ego_vx_global"] = (
    ego_df.groupby("scene_token")["ego_vx_global_raw"]
    .transform(lambda s: s.bfill().ffill())
    .fillna(0.0)
)

ego_df["ego_vy_global"] = (
    ego_df.groupby("scene_token")["ego_vy_global_raw"]
    .transform(lambda s: s.bfill().ffill())
    .fillna(0.0)
)

ego_df["ego_speed"] = np.sqrt(
    ego_df["ego_vx_global"] ** 2 + ego_df["ego_vy_global"] ** 2
)

ego_df["ego_acceleration_raw"] = ego_df.groupby("scene_token")["ego_speed"].diff() / ego_df["ego_dt"]

ego_df["ego_acceleration"] = (
    ego_df.groupby("scene_token")["ego_acceleration_raw"]
    .transform(lambda s: s.bfill().ffill())
    .fillna(0.0)
)

ego_df["ego_v_forward"] = (
    ego_df["ego_vx_global"] * np.cos(ego_df["ego_yaw_for_motion"]) +
    ego_df["ego_vy_global"] * np.sin(ego_df["ego_yaw_for_motion"])
)

ego_df["ego_v_lateral"] = (
    -ego_df["ego_vx_global"] * np.sin(ego_df["ego_yaw_for_motion"]) +
    ego_df["ego_vy_global"] * np.cos(ego_df["ego_yaw_for_motion"])
)

ego_df["ego_yaw_diff"] = ego_df.groupby("scene_token")["ego_yaw_for_motion"].diff()
ego_df["ego_yaw_diff"] = ego_df["ego_yaw_diff"].apply(
    lambda x: angle_wrap(x) if pd.notna(x) else np.nan
)

ego_df["ego_yaw_rate_raw"] = ego_df["ego_yaw_diff"] / ego_df["ego_dt"]

ego_df["ego_yaw_rate"] = (
    ego_df.groupby("scene_token")["ego_yaw_rate_raw"]
    .transform(lambda s: s.bfill().ffill())
    .fillna(0.0)
)

ego_motion_cols = [
    "sample_token",
    "ego_vx_global",
    "ego_vy_global",
    "ego_speed",
    "ego_acceleration",
    "ego_v_forward",
    "ego_v_lateral",
    "ego_yaw_rate",
]

traj_df = traj_df.merge(
    ego_df[ego_motion_cols],
    on="sample_token",
    how="left",
)

# Relative velocity features.
traj_df["relative_v_forward"] = traj_df["agent_v_forward"] - traj_df["ego_v_forward"]
traj_df["relative_v_lateral"] = traj_df["agent_v_lateral"] - traj_df["ego_v_lateral"]

# Positive closing speed means ego is closing in on the agent.
traj_df["closing_speed"] = traj_df["ego_v_forward"] - traj_df["agent_v_forward"]

# Time-to-collision approximation.
traj_df["ttc"] = np.inf

valid_ttc = (
    (traj_df["rel_x"] > 0.0) &
    (traj_df["closing_speed"] > 0.1)
)

traj_df.loc[valid_ttc, "ttc"] = (
    traj_df.loc[valid_ttc, "rel_x"] / traj_df.loc[valid_ttc, "closing_speed"]
)

# Useful boolean flags for event labeling later.
traj_df["is_in_front"] = traj_df["rel_x"] > 0.0
traj_df["is_behind"] = traj_df["rel_x"] < 0.0
traj_df["is_left"] = traj_df["rel_y"] > 0.0
traj_df["is_right"] = traj_df["rel_y"] < 0.0
traj_df["is_same_lane_simple"] = traj_df["lane_relation_simple"] == "same_lane"
traj_df["is_adjacent_lane_simple"] = traj_df["lane_relation_simple"].isin(["left_adjacent", "right_adjacent"])

# Vehicle cut-in detection.
LOOKBACK_FRAMES = 5

CUT_IN_LATERAL_NEAR_ZERO = 1.5
CUT_IN_MIN_START_LATERAL = 2.0
CUT_IN_MIN_WINDOW_DECREASE = 0.1

MAX_CUT_IN_HEADING_DIFF = np.deg2rad(60)
MIN_TARGET_FORWARD_SPEED = 1.5
MAX_LATERAL_TO_FORWARD_RATIO = 0.7

MAX_EGO_YAW_RATE = np.deg2rad(3.0)

MIN_TARGET_SPEED = 0.5
MIN_TARGET_LATERAL_SPEED = 0.15
MAX_EGO_LATERAL_SPEED = 0.35

MIN_FRONT_X = 1.0
MAX_FRONT_X = 25.0

track_group = traj_df.groupby(["scene_token", "instance_token"], group_keys=False)

traj_df["lateral_distance"] = traj_df["rel_y"].abs()

traj_df["prev_window_max_lateral_distance"] = (
    track_group["lateral_distance"]
    .transform(lambda s: s.shift(1).rolling(LOOKBACK_FRAMES, min_periods=1).max())
)

traj_df["window_lateral_decrease"] = (
    traj_df["prev_window_max_lateral_distance"] - traj_df["lateral_distance"]
)

# Important: use target vehicle velocity, not rel_y_rate.
traj_df["target_lateral_motion_toward_center"] = (
    ((traj_df["rel_y"] > 0.0) & (traj_df["agent_v_lateral"] < -MIN_TARGET_LATERAL_SPEED)) |
    ((traj_df["rel_y"] < 0.0) & (traj_df["agent_v_lateral"] > MIN_TARGET_LATERAL_SPEED))
)

traj_df["recent_target_lateral_motion_toward_center"] = (
    track_group["target_lateral_motion_toward_center"]
    .transform(lambda s: s.shift(1).rolling(LOOKBACK_FRAMES, min_periods=1).max())
    .fillna(False)
    .astype(bool)
)

traj_df["lateral_position_displaced_inward"] = (
    track_group["lateral_distance"]
    .transform(lambda s: s.shift(1).rolling(LOOKBACK_FRAMES, min_periods=1).max() - s)
    > CUT_IN_MIN_WINDOW_DECREASE  # already defined, e.g. 0.1m
)

traj_df["prev_lane_relation_simple"] = (
    track_group["lane_relation_simple"].shift(1)
)


traj_df["is_adjacent_to_same_lane_transition"] = (
    traj_df["prev_lane_relation_simple"].isin(["left_adjacent", "right_adjacent"]) &
    (traj_df["lane_relation_simple"] == "same_lane")
)

# Once the target is in ego lane, it should be roughly aligned with ego heading.
traj_df["same_lane_heading_aligned"] = (
    (traj_df["lane_relation_simple"] != "same_lane") |
    (traj_df["yaw_relative_to_ego"].abs() < MAX_CUT_IN_HEADING_DIFF)
)

# Cut-in vehicles should still move mostly forward, unlike crossing vehicles.
traj_df["mostly_forward_motion"] = (
    (traj_df["agent_v_forward"] > MIN_TARGET_FORWARD_SPEED) &
    (
        traj_df["agent_v_lateral"].abs()
        < MAX_LATERAL_TO_FORWARD_RATIO * traj_df["agent_v_forward"].abs().clip(lower=0.1)
    )
)
traj_df["ego_stable_for_cut_in"] = (
    (traj_df["ego_v_lateral"].abs() < MAX_EGO_LATERAL_SPEED) &
    (traj_df["ego_yaw_rate"].abs() < MAX_EGO_YAW_RATE)
)

traj_df["cut_in_candidate_frame"] = (
    (traj_df["actor_type"] == "vehicle") &

    (traj_df["is_adjacent_to_same_lane_transition"]) &

    (traj_df["same_lane_heading_aligned"]) &

    # Target must be in front after cut-in.
    (traj_df["rel_x"].between(MIN_FRONT_X, MAX_FRONT_X)) &

    # Target was laterally away recently.
    (traj_df["prev_window_max_lateral_distance"] > CUT_IN_MIN_START_LATERAL) &

    # Target is now close to ego lane center.
    (traj_df["lateral_distance"] < CUT_IN_LATERAL_NEAR_ZERO) &

    # Lateral distance reduced over the window.
    (traj_df["window_lateral_decrease"] >= CUT_IN_MIN_WINDOW_DECREASE) &

    # Target itself is moving toward ego lane center.
    (
        traj_df["target_lateral_motion_toward_center"] |
        traj_df["recent_target_lateral_motion_toward_center"] |
        traj_df["lateral_position_displaced_inward"]

     ) &

    # Reject ego lane-change-caused cases.
    (traj_df["ego_stable_for_cut_in"]) &

    # Reject parked/static vehicles.
    (traj_df["agent_speed"] > MIN_TARGET_SPEED)
)
CUT_IN_SCENES_CSV = OUTPUT_DIR / "vehicle_cut_in_scenes.csv"
CUT_IN_FRAMES_CSV = OUTPUT_DIR / "vehicle_cut_in_frames.csv"
BRAKING_FRAMES_CSV = OUTPUT_DIR / "vehicle_braking_frames.csv"
BRAKING_SCENES_CSV = OUTPUT_DIR / "vehicle_braking_scenes.csv"
PEDESTRIAN_CROSSING_FRAMES_CSV = OUTPUT_DIR / "pedestrian_crossing_frames.csv"
PEDESTRIAN_CROSSING_SCENES_CSV = OUTPUT_DIR / "pedestrian_crossing_scenes.csv"
OBSTACLE_APPROACH_FRAMES_CSV = OUTPUT_DIR / "obstacle_approach_frames.csv"
OBSTACLE_APPROACH_SCENES_CSV = OUTPUT_DIR / "obstacle_approach_scenes.csv"

MASTER_LABEL_FRAMES_CSV = OUTPUT_DIR / "trajectory_intent_label_frames.csv"
MASTER_LABEL_SCENES_CSV = OUTPUT_DIR / "trajectory_intent_label_scenes.csv"
MASTER_LABEL_SUMMARY_CSV = OUTPUT_DIR / "trajectory_intent_label_summary.csv"

cut_in_frames_df = traj_df[traj_df["cut_in_candidate_frame"]].copy()
cut_in_frames_df.to_csv(CUT_IN_FRAMES_CSV, index=False)

cut_in_scenes_df = (
    cut_in_frames_df[
        [
            "scene_token",
            "scene_name",
        ]
    ]
    .drop_duplicates()
    .sort_values("scene_name")
)

cut_in_scenes_df.to_csv(CUT_IN_SCENES_CSV, index=False)

# Vehicle braking detection.
BRAKING_MAX_FRONT_X = 30.0
BRAKING_MIN_FRONT_X = 2.0
BRAKING_MIN_SPEED = 3.0
BRAKING_MIN_DECEL = -2.0
BRAKING_MIN_SPEED_DROP = 1.0
BRAKING_LOOKBACK_FRAMES = 3
MAX_EGO_BRAKING_ACCEL = -1.0
MAX_EGO_BRAKING_SPEED_DROP = 1.2
MIN_EGO_SPEED_FOR_BRAKING = 2.0
MAX_BRAKING_HEADING_DIFF = np.deg2rad(35.0)
TARGET_BRAKES_MORE_THAN_EGO_MARGIN = 0.0
TARGET_SPEED_DROP_MORE_THAN_EGO_MARGIN = 0.0
BRAKING_MIN_CLOSING_SPEED = 0.5
BRAKING_MAX_TTC = 30.0

traj_df["prev_window_max_agent_speed_for_braking"] = (
    track_group["agent_speed"]
    .transform(lambda s: s.shift(1).rolling(BRAKING_LOOKBACK_FRAMES, min_periods=1).max())
)

traj_df["braking_window_speed_drop"] = (
    traj_df["prev_window_max_agent_speed_for_braking"] - traj_df["agent_speed"]
)

ego_braking_df = (
    traj_df[["scene_token", "sample_token", "timestamp_sec", "ego_speed"]]
    .drop_duplicates()
    .sort_values(["scene_token", "timestamp_sec"])
)

ego_braking_group = ego_braking_df.groupby("scene_token", group_keys=False)

ego_braking_df["prev_window_max_ego_speed_for_braking"] = (
    ego_braking_group["ego_speed"]
    .transform(lambda s: s.shift(1).rolling(BRAKING_LOOKBACK_FRAMES, min_periods=1).max())
)

ego_braking_df["ego_braking_window_speed_drop"] = (
    ego_braking_df["prev_window_max_ego_speed_for_braking"] - ego_braking_df["ego_speed"]
)

traj_df = traj_df.merge(
    ego_braking_df[
        [
            "sample_token",
            "prev_window_max_ego_speed_for_braking",
            "ego_braking_window_speed_drop",
        ]
    ],
    on="sample_token",
    how="left",
)

traj_df["vehicle_braking_candidate_frame"] = (
    (traj_df["actor_type"] == "vehicle") &

    # Vehicle should be in ego lane.
    (traj_df["is_same_lane_simple"]) &

    # Vehicle should be moving in the same direction as ego.
    (traj_df["yaw_relative_to_ego"].abs() < MAX_BRAKING_HEADING_DIFF) &

    # Vehicle should be ahead of ego and close enough to matter.
    (traj_df["rel_x"].between(BRAKING_MIN_FRONT_X, BRAKING_MAX_FRONT_X)) &

    # Ego should actually be closing on this braking vehicle.
    (traj_df["closing_speed"] > BRAKING_MIN_CLOSING_SPEED) &
    (traj_df["ttc"] < BRAKING_MAX_TTC) &

    # Ego should be moving, but not be the vehicle doing the braking event.
    (traj_df["ego_speed"] > MIN_EGO_SPEED_FOR_BRAKING) &
    (traj_df["ego_acceleration"] > MAX_EGO_BRAKING_ACCEL) &
    (traj_df["ego_braking_window_speed_drop"].fillna(0.0) < MAX_EGO_BRAKING_SPEED_DROP) &

    # Reject parked or crawling vehicles.
    (traj_df["prev_window_max_agent_speed_for_braking"] > BRAKING_MIN_SPEED) &

    # Sudden deceleration.
    (traj_df["agent_acceleration"] < BRAKING_MIN_DECEL) &

    # Target should be braking more strongly than ego.
    (traj_df["agent_acceleration"] < traj_df["ego_acceleration"] - TARGET_BRAKES_MORE_THAN_EGO_MARGIN) &

    # Actual speed drop over recent frames, avoids noisy acceleration false positives.
    (traj_df["braking_window_speed_drop"] >= BRAKING_MIN_SPEED_DROP) &
    (
        traj_df["braking_window_speed_drop"]
        > traj_df["ego_braking_window_speed_drop"].fillna(0.0) + TARGET_SPEED_DROP_MORE_THAN_EGO_MARGIN
    ) &

    # Avoid mixing cut-in and braking labels.
    (~traj_df["cut_in_candidate_frame"])
)
braking_frames_df = traj_df[traj_df["vehicle_braking_candidate_frame"]].copy()
braking_frames_df.to_csv(BRAKING_FRAMES_CSV, index=False)

braking_scenes_df = (
    braking_frames_df[
        [
            "scene_token",
            "scene_name",
        ]
    ]
    .drop_duplicates()
    .sort_values("scene_name")
)

braking_scenes_df.to_csv(BRAKING_SCENES_CSV, index=False)
# Pedestrian crossing detection.
PED_CROSSING_MIN_FRONT_X = 0.0
PED_CROSSING_MAX_FRONT_X = 30.0
PED_CROSSING_MAX_LATERAL_DISTANCE = 3.5

PED_MIN_SPEED = 0.4
PED_MIN_LATERAL_SPEED = 0.35
PED_LATERAL_TO_FORWARD_RATIO = 1.2

traj_df["pedestrian_lateral_motion_toward_ego_path"] = (
    ((traj_df["rel_y"] > 0.0) & (traj_df["agent_v_lateral"] < -PED_MIN_LATERAL_SPEED)) |
    ((traj_df["rel_y"] < 0.0) & (traj_df["agent_v_lateral"] > PED_MIN_LATERAL_SPEED))
)

traj_df["pedestrian_lateral_motion_dominant"] = (
    traj_df["agent_v_lateral"].abs()
    > PED_LATERAL_TO_FORWARD_RATIO * traj_df["agent_v_forward"].abs().clip(lower=0.1)
)

traj_df["pedestrian_crossing_candidate_frame"] = (
    (traj_df["actor_type"] == "pedestrian") &

    # Pedestrian is in front of ego.
    (traj_df["rel_x"].between(PED_CROSSING_MIN_FRONT_X, PED_CROSSING_MAX_FRONT_X)) &

    # Pedestrian is close to ego driving path.
    (traj_df["lateral_distance"] < PED_CROSSING_MAX_LATERAL_DISTANCE) &

    # Pedestrian is actually moving.
    (traj_df["agent_speed"] > PED_MIN_SPEED) &

    # Pedestrian has enough sideways motion.
    (traj_df["agent_v_lateral"].abs() > PED_MIN_LATERAL_SPEED) &

    # Motion is across the road, not along the road.
    (traj_df["pedestrian_lateral_motion_dominant"]) &

    # Pedestrian is moving toward/across ego path.
    (traj_df["pedestrian_lateral_motion_toward_ego_path"])
)
ped_crossing_frames_df = traj_df[traj_df["pedestrian_crossing_candidate_frame"]].copy()
ped_crossing_frames_df.to_csv(PEDESTRIAN_CROSSING_FRAMES_CSV, index=False)

ped_crossing_scenes_df = (
    ped_crossing_frames_df[
        [
            "scene_token",
            "scene_name",
        ]
    ]
    .drop_duplicates()
    .sort_values("scene_name")
)

ped_crossing_scenes_df.to_csv(PEDESTRIAN_CROSSING_SCENES_CSV, index=False)

# Obstacle approach detection.
OBSTACLE_MIN_FRONT_X = 2.0
OBSTACLE_MAX_FRONT_X = 20.0
OBSTACLE_MAX_LATERAL_DISTANCE = 1.5

OBSTACLE_MIN_EGO_SPEED = 2.0
OBSTACLE_MIN_CLOSING_SPEED = 2.0
OBSTACLE_MAX_TTC = 5.0
OBSTACLE_MAX_TARGET_SPEED = 4.0
OBSTACLE_STATIC_SPEED = 0.5
traj_df["is_static_vehicle_obstacle"] = (
    (traj_df["actor_type"] == "vehicle") &
    (traj_df["agent_speed"] < OBSTACLE_STATIC_SPEED)
)

traj_df["is_object_obstacle"] = (
    traj_df["category"].str.startswith("movable_object.") |
    traj_df["category"].str.startswith("static_object.")
)

traj_df["is_obstacle_type"] = (
    traj_df["is_static_vehicle_obstacle"] |
    traj_df["is_object_obstacle"]
)

traj_df["obstacle_approach_candidate_frame"] = (

    (traj_df["is_obstacle_type"]) &
    # Object should be ahead of ego.
    (traj_df["rel_x"].between(OBSTACLE_MIN_FRONT_X, OBSTACLE_MAX_FRONT_X)) &

    # Object should be near ego driving path.
    (traj_df["lateral_distance"] < OBSTACLE_MAX_LATERAL_DISTANCE) &

    # Ego should be moving.
    (traj_df["ego_speed"] > OBSTACLE_MIN_EGO_SPEED) &

    # Ego is closing in on the object.
    (traj_df["closing_speed"] > OBSTACLE_MIN_CLOSING_SPEED) &

    # Time to reach object is low.   
    (traj_df["ttc"] < OBSTACLE_MAX_TTC) &

    # Object is slow/stationary compared to ego.
    (traj_df["agent_speed"] < OBSTACLE_STATIC_SPEED) &

    # Avoid mixing with existing labels.
    (~traj_df["cut_in_candidate_frame"]) &
    (~traj_df["vehicle_braking_candidate_frame"]) &
    (~traj_df["pedestrian_crossing_candidate_frame"])
)

obstacle_approach_frames_df = traj_df[traj_df["obstacle_approach_candidate_frame"]].copy()
obstacle_approach_frames_df.to_csv(OBSTACLE_APPROACH_FRAMES_CSV, index=False)

obstacle_approach_scenes_df = (
    obstacle_approach_frames_df[
        [
            "scene_token",
            "scene_name",
        ]
    ]
    .drop_duplicates()
    .sort_values("scene_name")
)

obstacle_approach_scenes_df.to_csv(OBSTACLE_APPROACH_SCENES_CSV, index=False)
# Save trajectory + motion table.

intent_label_priority = [
    ("vehicle_cut_in", "cut_in_candidate_frame"),
    ("pedestrian_crossing", "pedestrian_crossing_candidate_frame"),  
    ("vehicle_braking", "vehicle_braking_candidate_frame"),
    ("obstacle_approach", "obstacle_approach_candidate_frame"),
]


traj_df["intent_label"] = "none"

for label_name, candidate_col in intent_label_priority:
    traj_df.loc[
        (traj_df["intent_label"] == "none") &
        (traj_df[candidate_col]),
        "intent_label"
    ] = label_name


#future intent generation block

def future_event(series, future_frames):
    return (
        series.iloc[::-1]
        .shift(1)
        .rolling(future_frames, min_periods=1)
        .max()
        .iloc[::-1]
        .fillna(False)
        .astype(bool)
    )

event_label_to_col = [
    ("vehicle_cut_in", "cut_in_candidate_frame"),
    ("pedestrian_crossing", "pedestrian_crossing_candidate_frame"),  
    ("vehicle_braking", "vehicle_braking_candidate_frame"),
    ("obstacle_approach", "obstacle_approach_candidate_frame"),
]

track_group = traj_df.groupby(["scene_token","instance_token"],group_keys=False)

for label_name, event_col in event_label_to_col:
    future_col= f"future_{label_name}"
    future_frames = FUTURE_FRAMES_BY_LABEL.get(label_name, FUTURE_FRAMES)

    traj_df[future_col]= (
        track_group[event_col]
        .transform(lambda s, n=future_frames: future_event(s.astype(bool), n))
        .astype(bool)
    )

# The training label includes the current event frame plus near-future frames.
# This gives rare classes like braking some examples with the actual motion cue visible.
traj_df["future_intent_label"] = traj_df["intent_label"]

for label_name, _ in intent_label_priority:
    future_col = f"future_{label_name}"


    traj_df.loc[
        (traj_df["future_intent_label"] == "none") &
        (traj_df[future_col]),
        "future_intent_label"
    ] = label_name

traj_df["future_intent_label_id"] = traj_df["future_intent_label"].map(
    {name: idx for idx , name in enumerate(LSTM_LABELS)}
)

if traj_df["future_intent_label_id"].isna().any():
    unknown_labels = sorted(traj_df.loc[
        traj_df["future_intent_label_id"].isna(),
        "future_intent_label"
    ].dropna().unique())
    raise ValueError(f"Unknown future intent labels: {unknown_labels}")

traj_df["future_intent_label_id"] = traj_df["future_intent_label_id"].astype(int)


master_label_cols = [
    "scene_token",
    "scene_name",
    "sample_token",
    "timestamp_us",
    "timestamp_sec",
    "instance_token",
    "annotation_token",
    "category",
    "actor_type",

    "rel_x",
    "rel_y",
    "lateral_distance",
    "distance_2d",

    "agent_speed",
    "agent_acceleration",
    "yaw_rate",
    "agent_v_forward",
    "agent_v_lateral",

    "ego_speed",
    "ego_acceleration",
    "ego_v_forward",
    "ego_v_lateral",

    "relative_v_forward",
    "relative_v_lateral",
    "closing_speed",
    "ttc",

    "cut_in_candidate_frame",
    "vehicle_braking_candidate_frame",
    "pedestrian_crossing_candidate_frame",
    "obstacle_approach_candidate_frame",
    "intent_label",
]

master_label_frames_df = traj_df[
    traj_df["intent_label"] != "none"
][master_label_cols].copy()

master_label_frames_df.to_csv(MASTER_LABEL_FRAMES_CSV, index=False)
master_label_scenes_df = (
    master_label_frames_df[
        [
            "scene_token",
            "scene_name",
            "intent_label",
        ]
    ]
    .drop_duplicates()
    .sort_values(["intent_label", "scene_name"])
)

master_label_scenes_df.to_csv(MASTER_LABEL_SCENES_CSV, index=False)

master_label_summary_df = (
    traj_df["intent_label"]
    .value_counts()
    .rename_axis("intent_label")
    .reset_index(name="frame_count")
)

master_label_summary_df.to_csv(MASTER_LABEL_SUMMARY_CSV, index=False)

# ============================================================
# BUILD LSTM TRAJECTORY WINDOW DATASET
# ============================================================

print("\nBuilding LSTM trajectory window dataset...")
print("Future frames by label:", FUTURE_FRAMES_BY_LABEL)

# Convert inf TTC to a capped numeric value for neural network training.
MAX_TTC_FOR_MODEL = 10.0
MAX_JERK_FOR_MODEL = 10.0
MAX_CLOSING_RATIO_FOR_MODEL = 3.0

traj_df["ttc_clipped"] = traj_df["ttc"].replace([np.inf, -np.inf], MAX_TTC_FOR_MODEL)
traj_df["ttc_clipped"] = traj_df["ttc_clipped"].clip(0.0, MAX_TTC_FOR_MODEL)

traj_df["is_vehicle"] = (traj_df["actor_type"] == "vehicle").astype(float)
traj_df["is_pedestrian"] = (traj_df["actor_type"] == "pedestrian").astype(float)
traj_df["is_object"] = traj_df["category"].str.startswith(
    ("movable_object.", "static_object.")
).astype(float)

traj_df["is_in_front_float"] = traj_df["is_in_front"].astype(float)
traj_df["is_same_lane_simple_float"] = traj_df["is_same_lane_simple"].astype(float)
traj_df["is_adjacent_lane_simple_float"] = traj_df["is_adjacent_lane_simple"].astype(float)

lstm_track_group = traj_df.groupby(["scene_token", "instance_token"], group_keys=False)

traj_df["agent_speed_prev_1"] = (
    lstm_track_group["agent_speed"].shift(1).fillna(traj_df["agent_speed"])
)
traj_df["agent_speed_prev_2"] = (
    lstm_track_group["agent_speed"].shift(2).fillna(traj_df["agent_speed_prev_1"])
)
traj_df["agent_speed_prev_4"] = (
    lstm_track_group["agent_speed"].shift(4).fillna(traj_df["agent_speed_prev_2"])
)

traj_df["agent_speed_drop_1"] = traj_df["agent_speed_prev_1"] - traj_df["agent_speed"]
traj_df["agent_speed_drop_2"] = traj_df["agent_speed_prev_2"] - traj_df["agent_speed"]
traj_df["agent_speed_drop_4"] = traj_df["agent_speed_prev_4"] - traj_df["agent_speed"]

traj_df["agent_acceleration_prev_1"] = (
    lstm_track_group["agent_acceleration"].shift(1).fillna(traj_df["agent_acceleration"])
)
traj_df["agent_jerk"] = (
    (traj_df["agent_acceleration"] - traj_df["agent_acceleration_prev_1"])
    / traj_df["dt"].replace(0.0, np.nan)
)
traj_df["agent_jerk"] = (
    traj_df["agent_jerk"]
    .replace([np.inf, -np.inf], 0.0)
    .fillna(0.0)
    .clip(-MAX_JERK_FOR_MODEL, MAX_JERK_FOR_MODEL)
)

traj_df["time_gap"] = np.inf
valid_time_gap = (
    (traj_df["rel_x"] > 0.0) &
    (traj_df["ego_speed"] > 0.1)
)
traj_df.loc[valid_time_gap, "time_gap"] = (
    traj_df.loc[valid_time_gap, "rel_x"] / traj_df.loc[valid_time_gap, "ego_speed"]
)
traj_df["time_gap_clipped"] = (
    traj_df["time_gap"]
    .replace([np.inf, -np.inf], MAX_TTC_FOR_MODEL)
    .clip(0.0, MAX_TTC_FOR_MODEL)
)
traj_df["closing_speed_ratio"] = (
    traj_df["closing_speed"] / traj_df["ego_speed"].clip(lower=0.1)
)
traj_df["closing_speed_ratio"] = (
    traj_df["closing_speed_ratio"]
    .replace([np.inf, -np.inf], 0.0)
    .fillna(0.0)
    .clip(-MAX_CLOSING_RATIO_FOR_MODEL, MAX_CLOSING_RATIO_FOR_MODEL)
)

traj_df["relative_acceleration"] = (
    traj_df["agent_acceleration"] - traj_df["ego_acceleration"]
)

traj_df["speed_drop_minus_ego"] = (
    traj_df["braking_window_speed_drop"].fillna(0.0)
    - traj_df["ego_braking_window_speed_drop"].fillna(0.0)
)

traj_df["prev_window_max_agent_speed_for_braking"] = (
    traj_df["prev_window_max_agent_speed_for_braking"].fillna(traj_df["agent_speed"])
)

traj_df["braking_window_speed_drop"] = traj_df["braking_window_speed_drop"].fillna(0.0)
traj_df["ego_braking_window_speed_drop"] = traj_df["ego_braking_window_speed_drop"].fillna(0.0)
traj_df["prev_window_max_lateral_distance"] = (
    traj_df["prev_window_max_lateral_distance"].fillna(traj_df["lateral_distance"])
)
traj_df["window_lateral_decrease"] = traj_df["window_lateral_decrease"].fillna(0.0)

traj_df["target_lateral_motion_toward_center_float"] = (
    traj_df["target_lateral_motion_toward_center"].astype(float)
)
traj_df["pedestrian_lateral_motion_toward_ego_path_float"] = (
    traj_df["pedestrian_lateral_motion_toward_ego_path"].astype(float)
)
traj_df["pedestrian_lateral_motion_dominant_float"] = (
    traj_df["pedestrian_lateral_motion_dominant"].astype(float)
)

LSTM_FEATURE_COLS = [
    # ego-relative position
    "rel_x",
    "rel_y",
    "lateral_distance",
    "distance_2d",
    "yaw_relative_to_ego",
    "width",
    "length",

    # target motion
    "agent_speed",
    "agent_acceleration",
    "yaw_rate",
    "agent_v_forward",
    "agent_v_lateral",
    "agent_speed_drop_1",
    "agent_speed_drop_2",
    "agent_speed_drop_4",
    "agent_jerk",
    "prev_window_max_agent_speed_for_braking",
    "braking_window_speed_drop",

    # ego motion
    "ego_speed",
    "ego_acceleration",
    "ego_v_forward",
    "ego_v_lateral",
    "ego_yaw_rate",
    "ego_braking_window_speed_drop",

    # relational motion
    "relative_v_forward",
    "relative_v_lateral",
    "relative_acceleration",
    "closing_speed",
    "closing_speed_ratio",
    "speed_drop_minus_ego",
    "ttc_clipped",
    "time_gap_clipped",

    # recent lateral trend
    "prev_window_max_lateral_distance",
    "window_lateral_decrease",
    "target_lateral_motion_toward_center_float",
    "pedestrian_lateral_motion_toward_ego_path_float",
    "pedestrian_lateral_motion_dominant_float",

    # simple geometry flags
    "is_in_front_float",
    "is_same_lane_simple_float",
    "is_adjacent_lane_simple_float",

    # actor type
    "is_vehicle",
    "is_pedestrian",
    "is_object",
]

NEIGHBOR_BASE_FEATURE_COLS = [
    "agent_speed",
    "agent_acceleration",
    "agent_v_forward",
    "agent_v_lateral",
    "is_vehicle",
    "is_pedestrian",
    "is_object",
]


def build_sample_actor_lookup(df):
    lookup = {}

    for sample_token, group_df in df.groupby("sample_token", sort=False):
        rel_xy = (
            group_df[["rel_x", "rel_y"]]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        base_features = (
            group_df[NEIGHBOR_BASE_FEATURE_COLS]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )

        lookup[str(sample_token)] = {
            "instance_tokens": group_df["instance_token"].astype(str).to_numpy(),
            "rel_x": rel_xy[:, 0],
            "rel_y": rel_xy[:, 1],
            "base_features": base_features,
        }

    return lookup


def build_row_neighbor_cache(df, sample_actor_lookup):
    num_rows = len(df)
    neighbor_features = np.zeros(
        (num_rows, MAX_NEIGHBORS, len(NEIGHBOR_FEATURE_COLS)),
        dtype=np.float32,
    )
    neighbor_mask = np.zeros(
        (num_rows, MAX_NEIGHBORS),
        dtype=bool,
    )

    row_sample_tokens = df["sample_token"].astype(str).to_numpy()
    row_instance_tokens = df["instance_token"].astype(str).to_numpy()
    row_rel_xy = (
        df[["rel_x", "rel_y"]]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    for row_idx in range(num_rows):
        if (row_idx + 1) % 100000 == 0:
            print(
                f"Prepared social neighbors for {row_idx + 1}/{num_rows} rows",
                flush=True,
            )

        actors_at_time = sample_actor_lookup.get(row_sample_tokens[row_idx])

        if actors_at_time is None:
            continue

        dx = actors_at_time["rel_x"] - row_rel_xy[row_idx, 0]
        dy = actors_at_time["rel_y"] - row_rel_xy[row_idx, 1]
        distance = np.hypot(dx, dy)
        is_object_neighbor = actors_at_time["base_features"][:, 6] > 0.5

        valid = (
            (actors_at_time["instance_tokens"] != row_instance_tokens[row_idx])
            & (distance >= MIN_SOCIAL_DISTANCE)
            & (distance <= SOCIAL_RADIUS)
            & (~is_object_neighbor | (distance <= OBJECT_SOCIAL_RADIUS))
        )

        if not np.any(valid):
            continue

        valid_indices = np.flatnonzero(valid)
        valid_distances = distance[valid_indices]

        if len(valid_indices) > MAX_NEIGHBORS:
            nearest_local = np.argpartition(
                valid_distances,
                MAX_NEIGHBORS - 1,
            )[:MAX_NEIGHBORS]
            selected_indices = valid_indices[nearest_local]
            selected_distances = valid_distances[nearest_local]
            order = np.argsort(selected_distances)
            selected_indices = selected_indices[order]
            selected_distances = selected_distances[order]
        else:
            order = np.argsort(valid_distances)
            selected_indices = valid_indices[order]
            selected_distances = valid_distances[order]

        count = len(selected_indices)
        neighbor_features[row_idx, :count, 0] = dx[selected_indices]
        neighbor_features[row_idx, :count, 1] = dy[selected_indices]
        neighbor_features[row_idx, :count, 2] = selected_distances
        neighbor_features[row_idx, :count, 3:] = actors_at_time["base_features"][
            selected_indices
        ]
        neighbor_mask[row_idx, :count] = True

    print(f"Prepared social neighbors for {num_rows}/{num_rows} rows", flush=True)

    return neighbor_features, neighbor_mask


def build_lstm_windows(df, feature_cols, history_frames):
    X_list = []
    y_list = []
    index_rows = []
    X_neighbors_list = []
    neighbor_mask_list = []


    label_to_id = {name: idx for idx, name in enumerate(LSTM_LABELS)}

    df = df.sort_values(
        ["scene_token", "instance_token", "timestamp_sec"]
    ).reset_index(drop=True)

    sample_actor_lookup = build_sample_actor_lookup(df)
    row_neighbor_features, row_neighbor_mask = build_row_neighbor_cache(
        df,
        sample_actor_lookup,
    )

    grouped = df.groupby(["scene_token", "instance_token"], group_keys=False, sort=False)
    total_tracks = grouped.ngroups

    for track_idx, ((scene_token, instance_token), g) in enumerate(grouped, start=1):
        g = g.sort_values("timestamp_sec")

        if len(g) < history_frames:
            continue

        # Keep vehicles, pedestrians, and obstacle objects for first version.
        actor_type = g["actor_type"].iloc[0]
        is_object_track = bool(g["is_object"].iloc[0])

        if actor_type not in ["vehicle", "pedestrian"] and not is_object_track:
            continue

        features = g[feature_cols].replace([np.inf, -np.inf], np.nan)

        # Fill minor missing values safely.
        features = features.ffill().bfill().fillna(0.0)

        feature_array = features.to_numpy(dtype=np.float32)

        labels = g["future_intent_label"].fillna("none").tolist()

        timestamps = g["timestamp_sec"].to_numpy()
        row_indices = g.index.to_numpy()

        for end_idx in range(history_frames - 1, len(g)):
            start_idx = end_idx - history_frames + 1

            window_timestamps = timestamps[start_idx:end_idx + 1]

            # Check rough continuity. nuScenes keyframes are about 0.5s apart.
            # Allow up to 0.8s gap to be safe.
            if len(window_timestamps) > 1:
                max_gap = np.max(np.diff(window_timestamps))
                if max_gap > 0.8:
                    continue

            label_name = labels[end_idx]

            if label_name not in label_to_id:
                label_name = "none"

            X_list.append(feature_array[start_idx:end_idx + 1])
            window_row_indices = row_indices[start_idx:end_idx + 1]
            X_neighbors_list.append(row_neighbor_features[window_row_indices])
            neighbor_mask_list.append(row_neighbor_mask[window_row_indices])

            y_list.append(label_to_id[label_name])

            row = g.iloc[end_idx]

            index_rows.append({
                "scene_token": row["scene_token"],
                "scene_name": row["scene_name"],
                "sample_token": row["sample_token"],
                "timestamp_sec": row["timestamp_sec"],
                "instance_token": row["instance_token"],
                "annotation_token": row["annotation_token"],
                "category": row["category"],
                "actor_type": row["actor_type"],
                "current_intent_label": row["intent_label"],
                "future_intent_label": row["future_intent_label"],
                "future_intent_label_id": label_to_id[label_name],
            })

        if track_idx % 500 == 0:
            print(
                f"Built windows for {track_idx}/{total_tracks} tracks; "
                f"windows so far: {len(X_list)}",
                flush=True,
            )

    if len(X_list) == 0:
        raise RuntimeError("No LSTM windows were created. Check filters/history length.")

    X = np.stack(X_list, axis=0).astype(np.float32)
    X_neighbors = np.stack(X_neighbors_list, axis=0).astype(np.float32)
    neighbor_mask = np.stack(neighbor_mask_list, axis=0).astype(bool)

    y = np.array(y_list, dtype=np.int64)
    index_df = pd.DataFrame(index_rows)

    return X, X_neighbors, neighbor_mask, y, index_df


X, X_neighbors, neighbor_mask, y, window_index_df = build_lstm_windows(
    df=traj_df,
    feature_cols=LSTM_FEATURE_COLS,
    history_frames=HISTORY_FRAMES,
)

save_lstm_npz = np.savez_compressed if SAVE_COMPRESSED_LSTM_NPZ else np.savez
save_lstm_npz(
    LSTM_WINDOW_NPZ,
    X=X,
    X_neighbors=X_neighbors,
    neighbor_mask=neighbor_mask,
    y=y,
    feature_cols=np.array(LSTM_FEATURE_COLS),
    neighbor_feature_cols=np.array(NEIGHBOR_FEATURE_COLS),
    label_names=np.array(LSTM_LABELS),
)

window_index_df.to_csv(LSTM_WINDOW_INDEX_CSV, index=False)

print(f"Saved LSTM window dataset: {LSTM_WINDOW_NPZ}")
print(f"Saved LSTM window index: {LSTM_WINDOW_INDEX_CSV}")
print(f"X shape: {X.shape}")
print(f"X_neighbors shape: {X_neighbors.shape}")
print(f"neighbor_mask shape: {neighbor_mask.shape}")
print(f"y shape: {y.shape}")

print("\nLSTM label distribution:")
print(window_index_df["future_intent_label"].value_counts())


traj_df.to_csv(TRAJ_CSV, index=False)

print(f"Saved trajectory + motion table: {TRAJ_CSV}")
print(f"Rows: {len(traj_df)}")
print(f"Unique scenes: {traj_df['scene_token'].nunique()}")
print(f"Unique instances: {traj_df['instance_token'].nunique()}")

print("\nCategory counts:")
print(traj_df["category"].value_counts())

if not RUN_CAMERA_PROJECTION:
    print("\nSkipping camera projection because RUN_CAMERA_PROJECTION = False.")
    print("\nDone.")
    raise SystemExit(0)


# ============================================================
# CAMERA BOUNDING BOX PROJECTION UTILS
# ============================================================

def create_3d_box_corners(center, size, rotation):
    """
    Create 8 global corners of a 3D annotation box.

    nuScenes size:
        [width, length, height]

    Corner order:
        0,1,2,3 top face
        4,5,6,7 bottom face
    """
    width, length, height = size

    x_corners = np.array([
        length / 2, length / 2, -length / 2, -length / 2,
        length / 2, length / 2, -length / 2, -length / 2,
    ])

    y_corners = np.array([
        width / 2, -width / 2, -width / 2, width / 2,
        width / 2, -width / 2, -width / 2, width / 2,
    ])

    z_corners = np.array([
        height / 2, height / 2, height / 2, height / 2,
        -height / 2, -height / 2, -height / 2, -height / 2,
    ])

    corners_local = np.stack([x_corners, y_corners, z_corners], axis=1)

    R_box_to_global = quaternion_to_rotation_matrix(rotation)
    center = np.array(center, dtype=np.float64)

    corners_global = corners_local @ R_box_to_global.T + center

    return corners_global


def global_points_to_camera(points_global, ego_pose, calibrated_sensor):
    """
    Transform points:
        global -> ego -> camera

    nuScenes stores:
        ego_pose rotation = ego -> global
        calibrated_sensor rotation = sensor/camera -> ego
    """
    points_global = np.asarray(points_global, dtype=np.float64)

    ego_translation = np.array(ego_pose["translation"], dtype=np.float64)
    R_ego_to_global = quaternion_to_rotation_matrix(ego_pose["rotation"])

    # Row-vector inverse:
    # p_ego = R_ego_to_global.T @ (p_global - t)
    # row equivalent = (p_global - t) @ R_ego_to_global
    points_ego = (points_global - ego_translation) @ R_ego_to_global

    sensor_translation = np.array(calibrated_sensor["translation"], dtype=np.float64)
    R_camera_to_ego = quaternion_to_rotation_matrix(calibrated_sensor["rotation"])

    # p_camera = R_camera_to_ego.T @ (p_ego - t)
    # row equivalent = (p_ego - t) @ R_camera_to_ego
    points_camera = (points_ego - sensor_translation) @ R_camera_to_ego

    return points_camera


def project_camera_points_to_image(points_camera, camera_intrinsic):
    points_camera = np.asarray(points_camera, dtype=np.float64)
    K = np.array(camera_intrinsic, dtype=np.float64)

    depths = points_camera[:, 2].copy()
    safe_depths = np.where(np.abs(depths) < 1e-6, 1e-6, depths)

    projected = points_camera @ K.T
    points_2d = projected[:, :2] / safe_depths[:, None]

    return points_2d, depths


def get_projection_status(points_2d, depths, image_width, image_height):
    if SKIP_BOX_IF_ANY_CORNER_BEHIND_CAMERA:
        if np.any(depths <= MIN_DEPTH):
            return "behind_camera"
    else:
        if np.all(depths <= MIN_DEPTH):
            return "behind_camera"

    x = points_2d[:, 0]
    y = points_2d[:, 1]

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    overlaps_image = (
        x_max >= 0 and
        x_min < image_width and
        y_max >= 0 and
        y_min < image_height
    )

    if overlaps_image:
        return "visible"

    return "outside_image"


def clip_bbox(x_min, y_min, x_max, y_max, image_width, image_height):
    x_min_c = max(0.0, min(float(x_min), float(image_width - 1)))
    y_min_c = max(0.0, min(float(y_min), float(image_height - 1)))
    x_max_c = max(0.0, min(float(x_max), float(image_width - 1)))
    y_max_c = max(0.0, min(float(y_max), float(image_height - 1)))

    return x_min_c, y_min_c, x_max_c, y_max_c


def draw_projected_3d_box(draw, points_2d, label):
    points = [(float(x), float(y)) for x, y in points_2d]

    edges = [
        # top
        (0, 1), (1, 2), (2, 3), (3, 0),
        # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),
        # vertical
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    for i, j in edges:
        draw.line([points[i], points[j]], fill=(0, 255, 0), width=3)

    # front face in red
    front_edges = [(0, 1), (4, 5), (0, 4), (1, 5)]
    for i, j in front_edges:
        draw.line([points[i], points[j]], fill=(255, 0, 0), width=3)

    x_min = max(0.0, min(p[0] for p in points))
    y_min = max(0.0, min(p[1] for p in points) - 14)

    draw.text((x_min, y_min), label, fill=(255, 255, 0))


def draw_2d_bbox(draw, bbox):
    x_min, y_min, x_max, y_max = bbox
    draw.rectangle([x_min, y_min, x_max, y_max], outline=(255, 255, 0), width=2)


# ============================================================
# CAMERA BOUNDING BOX TABLE + IMAGE VISUALIZATION
# ============================================================

print("\nProjecting 3D boxes to camera images...")

camera_rows = []
vis_saved = 0

# Fast lookup because trajectory table already filtered categories.
target_annotation_tokens = set(traj_df["annotation_token"].tolist())

for sample_idx, sample in enumerate(samples):
    sample_token = sample["token"]
    scene = scene_by_token[sample["scene_token"]]
    scene_name = scene["name"]

    if scene_name not in TARGET_SCENE_NAME:
        continue

    ann_tokens = sample_to_anns.get(sample_token, [])

    for camera_channel in CAMERA_CHANNELS:
        cam_sd_token = sample_to_channel_data[sample_token].get(camera_channel)

        if cam_sd_token is None:
            continue

        cam_sd = sample_data_by_token[cam_sd_token]
        image_path = resolve_data_file(cam_sd["filename"])

        if image_path is None:
            continue

        ego_pose = ego_pose_by_token[cam_sd["ego_pose_token"]]
        calibrated_sensor = calib_by_token[cam_sd["calibrated_sensor_token"]]
        camera_intrinsic = calibrated_sensor.get("camera_intrinsic", None)

        if camera_intrinsic is None:
            continue

        # Open image once for dimension.
        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size

        draw = None
        save_vis_for_this_image = SAVE_CAMERA_VISUALIZATIONS and vis_saved < MAX_VIS_IMAGES

        if save_vis_for_this_image:
            draw = ImageDraw.Draw(image)

        drawn_count = 0

        for ann_token in ann_tokens:
            if ann_token not in target_annotation_tokens:
                continue

            ann = ann_by_token[ann_token]
            category_name = get_category_name(ann)

            corners_global = create_3d_box_corners(
                center=ann["translation"],
                size=ann["size"],
                rotation=ann["rotation"],
            )

            corners_camera = global_points_to_camera(
                corners_global,
                ego_pose,
                calibrated_sensor,
            )

            points_2d, depths = project_camera_points_to_image(
                corners_camera,
                camera_intrinsic,
            )

            status = get_projection_status(
                points_2d=points_2d,
                depths=depths,
                image_width=image_width,
                image_height=image_height,
            )

            if status == "visible":
                x = points_2d[:, 0]
                y = points_2d[:, 1]

                x_min, x_max = float(np.min(x)), float(np.max(x))
                y_min, y_max = float(np.min(y)), float(np.max(y))

                x_min_c, y_min_c, x_max_c, y_max_c = clip_bbox(
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                    image_width=image_width,
                    image_height=image_height,
                )

                bbox_width = x_max_c - x_min_c
                bbox_height = y_max_c - y_min_c

                visible = bbox_width > 1 and bbox_height > 1
            else:
                x_min = y_min = x_max = y_max = np.nan
                x_min_c = y_min_c = x_max_c = y_max_c = np.nan
                bbox_width = bbox_height = np.nan
                visible = False

            row = {
                "scene_token": sample["scene_token"],
                "scene_name": scene_name,
                "sample_token": sample_token,
                "timestamp_us": sample["timestamp"],
                "timestamp_sec": sample["timestamp"] / 1e6,

                "camera_channel": camera_channel,
                "camera_sample_data_token": cam_sd_token,
                "image_filename": cam_sd["filename"],
                "image_resolved_path": str(image_path),
                "image_width": image_width,
                "image_height": image_height,

                "annotation_token": ann_token,
                "instance_token": ann["instance_token"],
                "category": category_name,
                "actor_type": get_coarse_actor_type(category_name),

                "projection_status": status,
                "visible_in_camera": bool(visible),

                "bbox_xmin_raw": x_min,
                "bbox_ymin_raw": y_min,
                "bbox_xmax_raw": x_max,
                "bbox_ymax_raw": y_max,

                "bbox_xmin": x_min_c,
                "bbox_ymin": y_min_c,
                "bbox_xmax": x_max_c,
                "bbox_ymax": y_max_c,
                "bbox_width": bbox_width,
                "bbox_height": bbox_height,

                "min_depth": float(np.min(depths)),
                "max_depth": float(np.max(depths)),
                "mean_depth": float(np.mean(depths)),
            }

            # Save projected 2D corner points.
            for corner_idx in range(8):
                row[f"corner_{corner_idx}_u"] = float(points_2d[corner_idx, 0])
                row[f"corner_{corner_idx}_v"] = float(points_2d[corner_idx, 1])
                row[f"corner_{corner_idx}_depth"] = float(depths[corner_idx])

            camera_rows.append(row)

            if save_vis_for_this_image and visible:
                label = category_name.split(".")[-1]
                draw_projected_3d_box(draw, points_2d, label=label)
                draw_2d_bbox(draw, (x_min_c, y_min_c, x_max_c, y_max_c))
                drawn_count += 1

        if save_vis_for_this_image and drawn_count > 0:
            save_path = VIS_DIR / f"{vis_saved:03d}_{scene_name}_{camera_channel}_{sample_token[:8]}.jpg"
            image.save(save_path)
            print(f"Saved visualization: {save_path} | boxes: {drawn_count}")
            vis_saved += 1

camera_df = pd.DataFrame(camera_rows)
camera_df.to_csv(CAMERA_BOX_CSV, index=False)

visible_camera_df = camera_df[camera_df["visible_in_camera"] == True].copy()
visible_camera_df.to_csv(VISIBLE_CAMERA_BOX_CSV, index=False)

print(f"\nSaved camera projection table: {CAMERA_BOX_CSV}")
print(f"Rows: {len(camera_df)}")

print(f"Saved visible camera box table: {VISIBLE_CAMERA_BOX_CSV}")
print(f"Visible rows: {len(visible_camera_df)}")

if len(camera_df) > 0:
    print("\nCamera projection status counts:")
    print(camera_df["projection_status"].value_counts())

    print("\nVisible boxes by camera:")
    print(visible_camera_df["camera_channel"].value_counts())


# ============================================================
# MERGE CAMERA BBOX TABLE WITH TRAJECTORY + MOTION TABLE
# ============================================================

print("\nMerging visible camera boxes with trajectory/motion features...")

# Merge on annotation_token because one annotation corresponds to one object at one sample.
merged_df = visible_camera_df.merge(
    traj_df,
    on=[
        "scene_token",
        "scene_name",
        "sample_token",
        "timestamp_us",
        "timestamp_sec",
        "annotation_token",
        "instance_token",
        "category",
        "actor_type",
    ],
    how="left",
    suffixes=("_camera", "_traj"),
)

merged_df.to_csv(MERGED_CSV, index=False)

print(f"Saved merged camera + trajectory table: {MERGED_CSV}")
print(f"Merged rows: {len(merged_df)}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\nDone.")
print("\nGenerated files:")
print(f"1. {TRAJ_CSV}")
print(f"2. {CAMERA_BOX_CSV}")
print(f"3. {VISIBLE_CAMERA_BOX_CSV}")
print(f"4. {MERGED_CSV}")
print(f"5. {VIS_DIR}/")
print(f"6. {CUT_IN_FRAMES_CSV}")
print(f"7. {CUT_IN_SCENES_CSV}")
print(f"8. {BRAKING_FRAMES_CSV}")
print(f"9. {BRAKING_SCENES_CSV}")
print(f"10. {PEDESTRIAN_CROSSING_FRAMES_CSV}")
print(f"11. {PEDESTRIAN_CROSSING_SCENES_CSV}")
print(f"12. {OBSTACLE_APPROACH_FRAMES_CSV}")
print(f"13. {OBSTACLE_APPROACH_SCENES_CSV}")
print(f"14. {MASTER_LABEL_FRAMES_CSV}")
print(f"15. {MASTER_LABEL_SCENES_CSV}")
print(f"16. {MASTER_LABEL_SUMMARY_CSV}")

print("\nUse this file for event labeling first:")
print(f"   {TRAJ_CSV}")

print("\nUse this file when you want RGB image boxes + motion features together:")
print(f"   {MERGED_CSV}")
