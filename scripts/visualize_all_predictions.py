"""
visualize_all_predictions.py
Visualizes ALL scenes/samples from nuscenes_action_tokens_with_predictions.csv.

Usage:
  # Full (with LiDAR, slow - ~hours):
  python visualize_all_predictions.py

  # Fast (no LiDAR, useful for quick trajectory comparison):
  python visualize_all_predictions.py --no-lidar

  # Specific scene only:
  python visualize_all_predictions.py --scene <scene_token>

Output:
  rerun_logs/all_predictions.rrd   (open with `rerun` anytime without rerunning)
"""

import argparse
import csv
import ast
import os
import numpy as np
import matplotlib.style as mpltstyle
import matplotlib.pyplot as plt

# Fix for NuScenes map-api matplotlib style issue
def _safe_style(style):
    try:
        mpltstyle.original_use(style)
    except:
        try:
            mpltstyle.original_use('seaborn-v0_8-whitegrid')
        except:
            pass

if not hasattr(mpltstyle, 'original_use'):
    mpltstyle.original_use = mpltstyle.use
    mpltstyle.use = _safe_style
    plt.style.use = _safe_style

import rerun as rr
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion

# ── Config ────────────────────────────────────────────────────────────────────
NUSCENES_ROOT = "/media/santhru/Extreme SSD1/Nuscenes Dataset/Dataset/train"
VERSION       = "v1.0-trainval"
CSV_PATH      = ("/home/santhru/FYP38_First Experiment/NuScenesVQA-/"
                 "QWEN_VL_AD/output/nuscenes_action_tokens_with_predictions.csv")
OUTPUT_RRD    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "rerun_logs", "all_predictions.rrd")
# ─────────────────────────────────────────────────────────────────────────────


def setup_maps(nusc_root):
    """Symlink expansion maps to /tmp so NuScenesMap can find them."""
    tmp = "/tmp/nuscenes_maps_link"
    os.makedirs(os.path.join(tmp, "maps"), exist_ok=True)
    src = os.path.join(nusc_root, "maps/expansion")
    if os.path.exists(src):
        for f in os.listdir(src):
            if f.endswith(".json"):
                dst = os.path.join(tmp, "maps", f)
                if not os.path.exists(dst):
                    try:
                        os.symlink(os.path.join(src, f), dst)
                    except:
                        pass
    return os.path.exists(os.path.join(tmp, "maps")) and tmp or nusc_root


def transform_to_ego(points, ego_trans, ego_rot_q):
    qt = Quaternion(ego_rot_q)
    p_centered = points - np.array(ego_trans)
    return np.dot(p_centered, qt.inverse.rotation_matrix.T)


def correct_pred_to_ego(points_global, csv_ego_x, csv_ego_y, csv_ego_yaw,
                         actual_ego_trans, actual_ego_rot):
    """Undo CSV yaw, re-apply actual NuScenes yaw, then transform to ego."""
    if not points_global:
        return np.zeros((0, 3))
    pts = np.array([[p[0], p[1]] for p in points_global], dtype=np.float64)
    # 1. Undo csv ego_to_global
    pts -= np.array([csv_ego_x, csv_ego_y])
    c, s = np.cos(-csv_ego_yaw), np.sin(-csv_ego_yaw)
    pts = pts @ np.array([[c, -s], [s, c]]).T
    # 2. Re-apply actual yaw
    actual_yaw = Quaternion(actual_ego_rot).yaw_pitch_roll[0]
    c2, s2 = np.cos(actual_yaw), np.sin(actual_yaw)
    pts = pts @ np.array([[c2, -s2], [s2, c2]]).T
    pts += np.array([actual_ego_trans[0], actual_ego_trans[1]])
    pts_3d = np.column_stack([pts, np.zeros(len(pts))])
    return transform_to_ego(pts_3d, actual_ego_trans, actual_ego_rot)


def safe_parse(s):
    if not s or str(s).strip() in ("", "nan", "[]"):
        return None
    try:
        return ast.literal_eval(str(s))
    except:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-lidar", action="store_true",
                        help="Skip LiDAR loading (much faster)")
    parser.add_argument("--scene", default=None,
                        help="Visualize only this scene_token")
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="Limit number of scenes (e.g. --max-scenes 5)")
    args = parser.parse_args()

    print("Initializing Rerun…")
    rr.init("NuScenes All Predictions", spawn=False)

    print("Loading NuScenes…")
    nusc = NuScenes(version=VERSION, dataroot=NUSCENES_ROOT, verbose=False)

    print(f"Reading {CSV_PATH}…")
    rows = []
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            rows.append(row)

    # Group by scene
    scenes_map: dict[str, list] = {}
    for row in rows:
        scenes_map.setdefault(row['scene_token'], []).append(row)

    # Filter / limit scenes
    if args.scene:
        scenes_map = {args.scene: scenes_map[args.scene]}
    elif args.max_scenes:
        keys = list(scenes_map.keys())[:args.max_scenes]
        scenes_map = {k: scenes_map[k] for k in keys}

    print(f"Visualizing {len(scenes_map)} scene(s), "
          f"{sum(len(v) for v in scenes_map.values())} sample(s)…")
    if args.no_lidar:
        print("  [LiDAR DISABLED for speed]")

    maps = {}
    map_root = setup_maps(NUSCENES_ROOT)

    global_step = 0

    for scene_idx, (scene_token, scene_rows) in enumerate(scenes_map.items()):

        # Try to get scene description
        try:
            scene_meta = nusc.get('scene', scene_token)
            scene_name = scene_meta['name']
        except:
            scene_name = scene_token[:8]

        print(f"  Scene {scene_idx+1}/{len(scenes_map)}: {scene_name} "
              f"({len(scene_rows)} samples)")

        for row in scene_rows:
            rr.set_time("step", sequence=global_step)
            # Also tag which scene this step belongs to
            rr.set_time("scene_index", sequence=scene_idx)
            global_step += 1

            sample_token = row['sample_token']
            try:
                sample = nusc.get('sample', sample_token)
            except:
                continue

            # Ego pose
            lidar_token = sample['data']['LIDAR_TOP']
            sd_record   = nusc.get('sample_data', lidar_token)
            pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
            ego_trans   = pose_record['translation']
            ego_rot     = pose_record['rotation']

            q_xyzw = [ego_rot[1], ego_rot[2], ego_rot[3], ego_rot[0]]
            rr.log("world/ego", rr.Transform3D(
                translation=ego_trans,
                rotation=rr.Quaternion(xyzw=q_xyzw)
            ))

            # Map lanes
            log_record = nusc.get('log', nusc.get('scene', scene_token)['log_token'])
            map_name   = log_record['location']
            if map_name not in maps:
                try:
                    maps[map_name] = NuScenesMap(dataroot=map_root, map_name=map_name)
                except:
                    maps[map_name] = None

            nusc_map = maps.get(map_name)
            if nusc_map:
                radius = 80
                box = (ego_trans[0]-radius, ego_trans[1]-radius,
                       ego_trans[0]+radius, ego_trans[1]+radius)
                avail = [l for l in ['lane', 'lane_connector']
                         if l in nusc_map.layer_names]
                if avail:
                    try:
                        recs = nusc_map.get_records_in_patch(
                            box, layer_names=avail, mode='intersect')
                        tokens = []
                        for l in avail:
                            tokens.extend(recs.get(l, []))
                        if tokens:
                            geom = nusc_map.discretize_lanes(tokens, 0.5)
                            strips = []
                            for pts in geom.values():
                                pts_ego = transform_to_ego(
                                    np.array(pts), ego_trans, ego_rot)
                                strips.append(pts_ego)
                            rr.log("world/ego/map",
                                   rr.LineStrips3D(strips, colors=[100, 100, 255]))
                    except:
                        pass

            # Annotations (bounding boxes)
            centers, sizes, quats, labels_ann = [], [], [], []
            qt_inv = Quaternion(ego_rot).inverse
            for ann_token in sample['anns']:
                ann   = nusc.get('sample_annotation', ann_token)
                c_ego = transform_to_ego(
                    np.array(ann['translation']).reshape(1, 3), ego_trans, ego_rot)[0]
                s     = ann['size']
                q_ego = qt_inv * Quaternion(ann['rotation'])
                centers.append(c_ego)
                sizes.append([s[1], s[0], s[2]])
                quats.append([q_ego[1], q_ego[2], q_ego[3], q_ego[0]])
                labels_ann.append(ann['category_name'])
            if centers:
                rr.log("world/ego/annotations", rr.Boxes3D(
                    centers=centers, sizes=sizes, quaternions=quats,
                    labels=labels_ann, colors=[255, 100, 100]))

            # LiDAR (optional)
            if not args.no_lidar:
                try:
                    pcl_path = os.path.join(nusc.dataroot, sd_record['filename'])
                    pc = LidarPointCloud.from_file(pcl_path)
                    cs = nusc.get('calibrated_sensor',
                                  sd_record['calibrated_sensor_token'])
                    pc.rotate(Quaternion(cs['rotation']).rotation_matrix)
                    pc.translate(np.array(cs['translation']))
                    pts = pc.points[:3, ::5].T
                    rr.log("world/ego/lidar",
                           rr.Points3D(pts, colors=[200, 200, 200], radii=0.05))
                except:
                    pass

            # ── Waypoints ────────────────────────────────────────────────────
            goal_name    = row.get('goal_name', '')
            action_token = row.get('action_token', 'N/A')
            csv_x   = float(row['ego_x'])
            csv_y   = float(row['ego_y'])
            csv_yaw = float(row.get('ego_yaw', 0.0))

            goal_color = ([0, 255, 0]   if "STRAIGHT" in goal_name else
                          [0, 255, 255] if "LEFT"     in goal_name else
                          [255, 165, 0] if "RIGHT"    in goal_name else
                          [200, 200, 200])

            # Ground truth
            wps = safe_parse(row.get('waypoints'))
            if wps:
                wps_ego = transform_to_ego(
                    np.array([[p[0], p[1], 0] for p in wps]), ego_trans, ego_rot)
                rr.log("world/ego/waypoints/gt",
                       rr.LineStrips3D([wps_ego], colors=goal_color,
                                       radii=0.2, labels=["GT"]))

            # Predicted (corrected heading)
            p_wps = safe_parse(row.get('predicted_waypoints'))
            if p_wps:
                p_ego = correct_pred_to_ego(
                    p_wps, csv_x, csv_y, csv_yaw, ego_trans, ego_rot)
                rr.log("world/ego/waypoints/predicted",
                       rr.LineStrips3D([p_ego], colors=[255, 0, 0],
                                       radii=0.15, labels=["Predicted"]))

            # Reference (corrected heading)
            r_wps = safe_parse(row.get('reference_trajectory'))
            if r_wps:
                r_ego = correct_pred_to_ego(
                    r_wps, csv_x, csv_y, csv_yaw, ego_trans, ego_rot)
                rr.log("world/ego/waypoints/reference",
                       rr.LineStrips3D([r_ego], colors=[0, 0, 255],
                                       radii=0.1, labels=["Reference"]))

            # Past
            past = safe_parse(row.get('past_waypoints'))
            if past:
                past_ego = transform_to_ego(
                    np.array([[p[0], p[1], 0] for p in past]), ego_trans, ego_rot)
                rr.log("world/ego/waypoints/past",
                       rr.LineStrips3D([past_ego], colors=[150, 150, 150], radii=0.2))

            # Floating label above ego (yellow)
            label_text = f"{action_token} | {goal_name}"
            rr.log("world/ego/label", rr.Points3D(
                [[0, 0, 4.0]],
                labels=[label_text],
                radii=0.01,
                colors=[[255, 255, 0]]
            ))

            # HUD panel
            rr.log("world/hud/info", rr.TextDocument(
                f"**Scene:** `{scene_name}`\n"
                f"**Scene Token:** `{scene_token}`\n"
                f"**Action Token:** `{action_token}`\n"
                f"**Goal:** `{goal_name}`\n"
                f"**Sample:** `{sample_token}`"
            ))

    # Blueprint
    import rerun.blueprint as rbl
    rr.send_blueprint(rbl.Blueprint(
        rbl.Spatial3DView(origin="world/ego", name="All Predictions")
    ))

    os.makedirs(os.path.dirname(OUTPUT_RRD), exist_ok=True)
    rr.save(OUTPUT_RRD)
    print(f"\n✅ Saved → {OUTPUT_RRD}")
    print(f"   Open with:")
    print(f"   rerun \"{OUTPUT_RRD}\"")


if __name__ == "__main__":
    main()
