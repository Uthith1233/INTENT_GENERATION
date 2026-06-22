# ============================================================
# BUILD LSTM TRAJECTORY WINDOW DATASET
# ============================================================

print("\nBuilding LSTM trajectory window dataset...")

# Convert inf TTC to a capped numeric value for neural network training.
MAX_TTC_FOR_MODEL = 10.0

traj_df["ttc_clipped"] = traj_df["ttc"].replace([np.inf, -np.inf], MAX_TTC_FOR_MODEL)
traj_df["ttc_clipped"] = traj_df["ttc_clipped"].clip(0.0, MAX_TTC_FOR_MODEL)

traj_df["is_vehicle"] = (traj_df["actor_type"] == "vehicle").astype(float)
traj_df["is_pedestrian"] = (traj_df["actor_type"] == "pedestrian").astype(float)

traj_df["is_in_front_float"] = traj_df["is_in_front"].astype(float)
traj_df["is_same_lane_simple_float"] = traj_df["is_same_lane_simple"].astype(float)
traj_df["is_adjacent_lane_simple_float"] = traj_df["is_adjacent_lane_simple"].astype(float)

LSTM_FEATURE_COLS = [
    # ego-relative position
    "rel_x",
    "rel_y",
    "distance_2d",
    "yaw_relative_to_ego",

    # target motion
    "agent_speed",
    "agent_acceleration",
    "yaw_rate",
    "agent_v_forward",
    "agent_v_lateral",

    # ego motion
    "ego_speed",
    "ego_acceleration",
    "ego_v_forward",
    "ego_v_lateral",
    "ego_yaw_rate",

    # relational motion
    "relative_v_forward",
    "relative_v_lateral",
    "closing_speed",
    "ttc_clipped",

    # simple geometry flags
    "is_in_front_float",
    "is_same_lane_simple_float",
    "is_adjacent_lane_simple_float",

    # actor type
    "is_vehicle",
    "is_pedestrian",
]


def build_lstm_windows(df, feature_cols, history_frames):
    X_list = []
    y_list = []
    index_rows = []

    label_to_id = {name: idx for idx, name in enumerate(LSTM_LABEL_NAMES)}

    df = df.sort_values(
        ["scene_token", "instance_token", "timestamp_sec"]
    ).reset_index(drop=True)

    grouped = df.groupby(["scene_token", "instance_token"], group_keys=False)

    for (scene_token, instance_token), g in grouped:
        g = g.sort_values("timestamp_sec").reset_index(drop=True)

        if len(g) < history_frames:
            continue

        # Only train vehicle and pedestrian windows for first version.
        actor_type = g["actor_type"].iloc[0]

        if actor_type not in ["vehicle", "pedestrian"]:
            continue

        features = g[feature_cols].replace([np.inf, -np.inf], np.nan)

        # Fill minor missing values safely.
        features = features.ffill().bfill().fillna(0.0)

        feature_array = features.to_numpy(dtype=np.float32)

        labels = g["future_intent_label"].fillna("none").tolist()

        timestamps = g["timestamp_sec"].to_numpy()

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

    if len(X_list) == 0:
        raise RuntimeError("No LSTM windows were created. Check filters/history length.")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    index_df = pd.DataFrame(index_rows)

    return X, y, index_df


X, y, window_index_df = build_lstm_windows(
    df=traj_df,
    feature_cols=LSTM_FEATURE_COLS,
    history_frames=HISTORY_FRAMES,
)

np.savez_compressed(
    LSTM_WINDOW_NPZ,
    X=X,
    y=y,
    feature_cols=np.array(LSTM_FEATURE_COLS),
    label_names=np.array(LSTM_LABEL_NAMES),
)

window_index_df.to_csv(LSTM_WINDOW_INDEX_CSV, index=False)

print(f"Saved LSTM window dataset: {LSTM_WINDOW_NPZ}")
print(f"Saved LSTM window index: {LSTM_WINDOW_INDEX_CSV}")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

print("\nLSTM label distribution:")
print(window_index_df["future_intent_label"].value_counts())