"""
Convert raw recorded sessions directly into a LeRobot dataset.

This script aligns timestamps, extracts frames from recorded videos, and writes episodes into a LeRobot dataset.

Key assumptions:
- Video frames are read with OpenCV (BGR), converted to RGB and resized
    to 224x224 (uint8) to match LeRobot image expectations.
- Single-arm sessions write images to `observation.images.robot_0`.
    Dual-arm sessions populate both `robot_0` and `robot_1` image fields.

Minimal usage example (required flags shown):

    python raw_2_lerobot_V4.0.py \
        --task-root /path/to/raw/task_root \
        --repo-id myorg/myrepo \
        --task-name "Pick and place" \
        --traj-source merge \
        --target-fps 20 \
        --source_camera_fps 20 \
        --mode video \
        --output /path/to/save/lerobot
                       
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Literal

import dataclasses
import shutil
import cv2
import numpy as np
import pandas as pd
import subprocess
import json

# LeRobotDataset is provided by the `lerobot` package; we import it directly
# from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME as LEROBOT_HOME

from lerobot.datasets.lerobot_dataset import LeRobotDataset #0.4.1
# from lerobot.common.datasets.dataset_utils import write_image_file

def _ensure_module_path():
    this_dir = Path(__file__).resolve().parent
    if str(this_dir) not in sys.path:
        sys.path.insert(0, str(this_dir))


def _to_rgb_and_resize(img, size=(224, 224)):
    # img: BGR numpy array from cv2
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8)


# ---------------------------------------------------------------------------
# Module-level helpers (moved out of convert_raw_to_lerobot)
# ---------------------------------------------------------------------------

# Assume source camera recorded at 60 FPS unless specified otherwise.
# SOURCE_CAMERA_FPS = 60

def extract_frames_sequential(video_path: str, frame_indices):
    """Read selected frames from a video in sequential order.

    This avoids random seeks which are slow for GOP codecs (H.264/H.265).

    Args:
        video_path: Path to the video file.
        frame_indices: Iterable of integer frame indices to extract.

    Returns:
        dict mapping frame_index -> BGR numpy array (as returned by cv2.read()).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    frames = {}
    sorted_indices = sorted(set(frame_indices))

    current_frame = 0
    for target in sorted_indices:
        while current_frame < target:
            cap.grab()
            current_frame += 1
        ret, frame = cap.read()
        current_frame += 1
        if ret:
            frames[target] = frame

    cap.release()
    return frames


def find_nearest_indices(sorted_arr: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Find nearest indices in a sorted array for each target using binary search.

    This is a vectorized replacement for repeated argmin(abs(arr - t)).
    """
    indices = np.searchsorted(sorted_arr, targets)
    indices = np.clip(indices, 1, len(sorted_arr) - 1)
    left = sorted_arr[indices - 1]
    right = sorted_arr[indices]
    indices = indices - (np.abs(targets - left) < np.abs(targets - right)).astype(int)
    return indices


def clamp_txt_to_csv(txt_path: str, csv_path: str) -> bool:
    """Convert clamp_data text file to CSV if present."""
    try:
        df = pd.read_csv(txt_path, sep=r"\s+", header=None)
        df.columns = ["timestamp", "clamp"]
        df.to_csv(csv_path, index=False)
        return True
    except Exception:
        return False


def ensure_clamp_csv_for_path(data_path: str) -> None:
    """Ensure `Clamp_Data/clamp.csv` exists in a data path (convert if needed)."""
    clamp_dir = os.path.join(data_path, "Clamp_Data")
    if not os.path.exists(clamp_dir):
        return
    clamp_txt_path = os.path.join(clamp_dir, "clamp_data_tum.txt")
    clamp_csv_path = os.path.join(clamp_dir, "clamp.csv")
    if os.path.exists(clamp_csv_path):
        return
    if os.path.exists(clamp_txt_path):
        clamp_txt_to_csv(clamp_txt_path, clamp_csv_path)


def detect_layout(session_path: str):
    """Detect whether a session is single-arm or dual-arm.

    Returns:
        tuple(mode, paths) where mode in {'dual','single','invalid'} and
        paths is a dict with keys pointing to the arm directories.
    """
    if not os.path.isdir(session_path):
        return 'invalid', {}
    subdirs = [d for d in os.listdir(session_path) if os.path.isdir(os.path.join(session_path, d))]
    left_dir_name = next((d for d in subdirs if d.startswith('left_hand')), None)
    right_dir_name = next((d for d in subdirs if d.startswith('right_hand')), None)
    if left_dir_name and right_dir_name:
        return 'dual', {'left': os.path.join(session_path, left_dir_name), 'right': os.path.join(session_path, right_dir_name)}
    if os.path.exists(os.path.join(session_path, "RGB_Images")):
        return 'single', {'single': session_path}
    return 'invalid', {}


def get_video_state(video_path: str) -> bool:
    """Check whether a video file is valid using ffprobe (duration > 0)."""
    if not os.path.isfile(video_path):
        return False
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.returncode == 0 and float(result.stdout.strip()) > 0
    except Exception:
        return False


def read_trj_txt(txt_path: str) -> pd.DataFrame:
    """Read a trajectory text file into a DataFrame with 8 columns.

    Expected columns: timestamp PosX PosY PosZ Qx Qy Qz Qw
    """
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"trajectory txt not found: {txt_path}")
    df = pd.read_csv(txt_path, sep=r"\s+", header=None)
    if df.shape[1] < 8:
        raise ValueError("trajectory txt expects >=8 columns")
    df = df.iloc[:, :8]
    df.columns = ['timestamp', 'Pos X', 'Pos Y', 'Pos Z', 'Q_X', 'Q_Y', 'Q_Z', 'Q_W']
    return df


def load_trajectory(session_path: str, traj_source: str) -> pd.DataFrame:
    """Load trajectory data from one of the supported sources."""
    if traj_source == 'merge':
        trj_dir = os.path.join(session_path, "Merged_Trajectory")
        trj_csv = os.path.join(trj_dir, "merged_trajectory.csv")
        trj_txt = os.path.join(trj_dir, "merged_trajectory.txt")
        if os.path.exists(trj_csv):
            df = pd.read_csv(trj_csv)
            expected_cols = ['timestamp', 'Pos X', 'Pos Y', 'Pos Z', 'Q_X', 'Q_Y', 'Q_Z', 'Q_W']
            if all(c in df.columns for c in expected_cols):
                return df[expected_cols]
        if os.path.exists(trj_txt):
            return read_trj_txt(trj_txt)
        raise FileNotFoundError(f"No merged trajectory found in {session_path}")
    elif traj_source == 'slam':
        return read_trj_txt(os.path.join(session_path, "SLAM_Poses", "slam_processed.txt"))
    elif traj_source == 'vive':
        return read_trj_txt(os.path.join(session_path, "Vive_Poses", "vive_data_tum.txt"))
    else:
        raise ValueError(f"Unknown traj_source={traj_source}")


def load_arm_data(data_path: str, traj_source: str):
    """Load video, timestamps, clamp and trajectory for a single arm path.

    Returns a dict with keys: 'traj', 'clamp', 'timestamps', 'video_path', or None when missing.
    """
    if not os.path.isdir(data_path):
        return None
    image_dir = os.path.join(data_path, "RGB_Images")
    clamp_dir = os.path.join(data_path, "Clamp_Data")
    video_path = os.path.join(image_dir, "video.mp4")
    timestamps_path = os.path.join(image_dir, "timestamps.csv")
    clamp_path = os.path.join(clamp_dir, "clamp.csv")
    ensure_clamp_csv_for_path(data_path)
    if not (os.path.exists(video_path) and os.path.exists(timestamps_path) and os.path.exists(clamp_path)):
        return None
    if not get_video_state(video_path):
        return None
    try:
        traj = load_trajectory(data_path, traj_source)
        clamp = pd.read_csv(clamp_path)
        timestamps = pd.read_csv(timestamps_path)
    except Exception:
        return None
    
    # if 'aligned_stamp' in timestamps.columns:
    #     timestamps['timestamp'] = timestamps['aligned_stamp']
    # elif 'timestamp' not in timestamps.columns:
    #     timestamps['timestamp'] = np.arange(len(timestamps), dtype=float)
    
    if  'header_stamp' in timestamps.columns:
        timestamps['timestamp'] = timestamps['header_stamp']
    elif 'aligned_stamp' in timestamps.columns:
        timestamps['timestamp'] = timestamps['aligned_stamp']
        
    if 'frame_index' not in timestamps.columns:
        timestamps['frame_index'] = np.arange(len(timestamps), dtype=int)
    return {"traj": traj, "clamp": clamp, "timestamps": timestamps, "video_path": video_path}


def find_all_sessions(root_path: str) -> List[str]:
    """Recursively find all directories that start with 'session' under root_path."""
    found_sessions = []
    for root, dirs, _ in os.walk(root_path):
        for d in dirs:
            if d.startswith("session"):
                found_sessions.append(os.path.join(root, d))
    return sorted(found_sessions)


def prepare_bimanual_alignment(left: dict, right: dict, target_fps: int, source_camera_fps: int):
    """Prepare timestamp/trajectory indices and pre-extract frames for dual-arm sessions.

    Args:
        left: dict returned by `load_arm_data` for left arm.
        right: dict returned by `load_arm_data` for right arm.
        source_camera_fps: original camera frequency in Hz.
        target_fps: desired output frequency in Hz.

    Returns:
        Tuple containing:
        - master_ts (np.ndarray): timestamps used as master timeline
        - master_fidx (np.ndarray): frame indices on the master camera
        - l_traj_arr, l_clamp_arr: left trajectory/clamp arrays
        - r_traj_arr, r_clamp_arr: right trajectory/clamp arrays
        - l_traj_idx, l_clamp_idx, r_traj_idx, r_clamp_idx: precomputed index arrays
        - left_frames, right_frames: dicts mapping frame_index->frame (BGR)
        - r_cam_fidx, r_cam_idx: right camera frame indices and alignment indices
    """
    master = left["timestamps"].iloc[:: max(1, int(source_camera_fps / target_fps))].reset_index(drop=True)
    master_ts = master["timestamp"].to_numpy()
    master_fidx = master["frame_index"].to_numpy().astype(int)

    # Left arrays
    l_traj_ts = left["traj"]["timestamp"].to_numpy()
    l_clamp_ts = left["clamp"]["timestamp"].to_numpy()
    l_traj_arr = left["traj"][['Pos X','Pos Y','Pos Z','Q_X','Q_Y','Q_Z','Q_W']].to_numpy()
    l_clamp_arr = left["clamp"]["clamp"].to_numpy()

    # Right arrays
    r_traj_ts = right["traj"]["timestamp"].to_numpy()
    r_clamp_ts = right["clamp"]["timestamp"].to_numpy()
    r_traj_arr = right["traj"][['Pos X','Pos Y','Pos Z','Q_X','Q_Y','Q_Z','Q_W']].to_numpy()
    r_clamp_arr = right["clamp"]["clamp"].to_numpy()

    r_cam_ts = right["timestamps"]["timestamp"].to_numpy()
    r_cam_fidx = right["timestamps"]["frame_index"].to_numpy()

    # Precompute indices
    l_traj_idx = find_nearest_indices(np.sort(l_traj_ts), master_ts)
    l_clamp_idx = find_nearest_indices(np.sort(l_clamp_ts), master_ts)
    r_cam_idx = find_nearest_indices(np.sort(r_cam_ts), master_ts)
    r_cam_ts_aligned = r_cam_ts[r_cam_idx]
    r_traj_idx = find_nearest_indices(np.sort(r_traj_ts), r_cam_ts_aligned)
    r_clamp_idx = find_nearest_indices(np.sort(r_clamp_ts), r_cam_ts_aligned)

    # Extract frames sequentially
    left_indices = master_fidx.tolist()
    right_indices = r_cam_fidx[r_cam_idx].tolist()
    left_frames = extract_frames_sequential(left["video_path"], left_indices)
    right_frames = extract_frames_sequential(right["video_path"], right_indices)

    return (
        master_ts,
        master_fidx,
        l_traj_arr,
        l_clamp_arr,
        r_traj_arr,
        r_clamp_arr,
        l_traj_idx,
        l_clamp_idx,
        r_traj_idx,
        r_clamp_idx,
        left_frames,
        right_frames,
        r_cam_fidx,
        r_cam_idx,
    )


def prepare_single_alignment(single: dict, target_fps: int, source_camera_fps: int):
    """Prepare timestamp/trajectory indices and pre-extract frames for single-arm sessions.

    Args:
        single: dict returned by `load_arm_data` for the single arm.
        target_fps: desired output frequency in Hz.

    Returns:
        Tuple containing:
        - master_ts (np.ndarray): timestamps used as master timeline
        - master_fidx (np.ndarray): frame indices on the master camera
        - traj_arr: trajectory array (N,7)
        - clamp_arr: clamp values (N,)
        - traj_idx, clamp_idx: precomputed index arrays mapping master_ts -> traj/clamp rows
        - frames: dict mapping frame_index->frame (BGR)
    """
    master = single["timestamps"].iloc[:: max(1, int(source_camera_fps / target_fps))].reset_index(drop=True)
    master_ts = master["timestamp"].to_numpy()
    master_fidx = master["frame_index"].to_numpy().astype(int)

    traj_ts = single["traj"]["timestamp"].to_numpy()
    clamp_ts = single["clamp"]["timestamp"].to_numpy()
    traj_arr = single["traj"][['Pos X','Pos Y','Pos Z','Q_X','Q_Y','Q_Z','Q_W']].to_numpy()
    clamp_arr = single["clamp"]["clamp"].to_numpy()

    traj_idx = find_nearest_indices(np.sort(traj_ts), master_ts)
    clamp_idx = find_nearest_indices(np.sort(clamp_ts), master_ts)

    frame_indices = master_fidx.tolist()
    frames = extract_frames_sequential(single["video_path"], frame_indices)

    return master_ts, master_fidx, traj_arr, clamp_arr, traj_idx, clamp_idx, frames

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 1e-4
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = "pyav"


# DEFAULT_DATASET_CONFIG = DatasetConfig()

STATE_NAMES_8 = [
    "x",
    "y",
    "z",
    "qx",
    "qy",
    "qz",
    "qw",
    "gripper_width",
]


def create_empty_dataset(
    *,
    repo_id: str,
    repo_output: Path,
    fps: int,
    mode: Literal["video", "image"] = "video",
    robot_type: str = "fastumi",
    bimanual: bool = False,
    dataset_config: DatasetConfig = DatasetConfig(),
) -> LeRobotDataset:
    """Create an empty LeRobotDataset with the expected features for this repo.

    The function mirrors the implementation used in the HDF5->LeRobot converter
    so the produced dataset is compatible.
    """
    features: dict = {}
    if not bimanual:
        features["observation.state"] = {
            "dtype": "float32",
            "shape": (8,),
            "names": STATE_NAMES_8,
        }
        features["action"] = {
            "dtype": "float32",
            "shape": (8,),
            "names": STATE_NAMES_8,
        }
    else:
        state_names_16 = [f"robot_0_{n}" for n in STATE_NAMES_8] + [f"robot_1_{n}" for n in STATE_NAMES_8]
        features["observation.state"] = {
            "dtype": "float32",
            "shape": (16,),
            "names": state_names_16,
        }
        features["action"] = {
            "dtype": "float32",
            "shape": (16,),
            "names": state_names_16,
        }
        features["robot_0_action"] = {"dtype": "float32", "shape": (8,), "names": STATE_NAMES_8}
        features["robot_1_action"] = {"dtype": "float32", "shape": (8,), "names": STATE_NAMES_8}

    cams = ["robot_0", "robot_1"] if bimanual else ["robot_0"]
    for cam in cams:
        feat = {
            "dtype": mode,
            "shape": (224, 224, 3),
            "names": ["height", "width", "channels"],
        }
        if mode == "video":
            feat["info"] = {
                "video.height": 224,
                "video.width": 224,
                "video.codec": "libx264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
            }
        features[f"observation.images.{cam}"] = feat

    dataset_dir = LEROBOT_HOME / repo_id
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    return LeRobotDataset.create(
        root = dataset_dir,
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def convert_raw_to_lerobot(
    task_root: Path,
    repo_id: str,
    task_name: str,
    # camera_names: List[str],
    traj_source: str,
    target_fps: int,
    source_camera_fps: int,
    repo_output: Optional[Path],
    mode: str = "video",
):
    """Main conversion routine.

    This will:
    - discover sessions using `find_all_sessions` from `raw2hdf5_new.py`;
    - for each session, load arm data and align timestamps; extract frames
      sequentially and build per-frame records; add frames to the
      LeRobot dataset and save episodes.
    """
    _ensure_module_path()

    # Inline necessary helpers (copied from raw2hdf5_new.py) so this script
    # is self-contained and does not import raw2hdf5_new at runtime.

    # SOURCE_CAMERA_FPS = 60
    task_root = Path(task_root).expanduser().resolve()
    sessions = find_all_sessions(str(task_root))
    if not sessions:
        raise RuntimeError(f"No sessions found under {task_root}")

    # Detect bimanual by checking first session layout
    first_layout, _ = detect_layout(sessions[0])
    bimanual = first_layout == "dual"

    # Create dataset
    dataset = create_empty_dataset(repo_id=repo_id,repo_output=repo_output, fps=target_fps, mode=mode, bimanual=bimanual)

    total_frames = 0

    for idx, session_path in enumerate(sessions):
        layout, paths = detect_layout(session_path)
        if layout == "invalid":
            print(f"[SKIP] Invalid session layout: {session_path}")
            continue

        if layout == "dual":
            left = load_arm_data(paths["left"], traj_source)
            right = load_arm_data(paths["right"], traj_source)
            if left is None or right is None:
                print(f"[SKIP] missing data in dual session: {session_path}")
                continue

            (
                master_ts,    # np.ndarray, master timeline timestamps (T,)
                master_fidx,  # np.ndarray[int], master frame indices corresponding to master_ts
                l_traj_arr,   # np.ndarray, left trajectory array (N_left, 7: pos + quat)
                l_clamp_arr,  # np.ndarray, left clamp values (N_left,)
                r_traj_arr,   # np.ndarray, right trajectory array (N_right, 7: pos + quat)
                r_clamp_arr,  # np.ndarray, right clamp values (N_right,)
                l_traj_idx,   # np.ndarray[int], indices mapping master_ts -> left traj rows
                l_clamp_idx,  # np.ndarray[int], indices mapping master_ts -> left clamp rows
                r_traj_idx,   # np.ndarray[int], indices mapping right-aligned cam ts -> right traj rows
                r_clamp_idx,  # np.ndarray[int], indices mapping right-aligned cam ts -> right clamp rows
                left_frames,  # dict[int->ndarray], extracted left camera frames keyed by frame index (BGR)
                right_frames, # dict[int->ndarray], extracted right camera frames keyed by frame index (BGR)
                r_cam_fidx,   # np.ndarray[int], frame indices for right camera timestamps
                r_cam_idx,    # np.ndarray[int], indices mapping master_ts -> right camera timestamp indices
            ) = prepare_bimanual_alignment(left, right, target_fps,source_camera_fps)

            valid = 0
            for i in range(len(master_ts)):
                l_fidx = master_fidx[i]
                r_fidx = r_cam_fidx[r_cam_idx[i]]
                if l_fidx not in left_frames or r_fidx not in right_frames:
                    continue
                frame_l = _to_rgb_and_resize(left_frames[l_fidx])
                frame_r = _to_rgb_and_resize(right_frames[r_fidx])

                l_idx_t = l_traj_idx[i]
                l_idx_c = l_clamp_idx[i]
                r_idx_t = r_traj_idx[i]
                r_idx_c = r_clamp_idx[i]

                l_pos = list(l_traj_arr[l_idx_t]) + [float(l_clamp_arr[l_idx_c])]
                r_pos = list(r_traj_arr[r_idx_t]) + [float(r_clamp_arr[r_idx_c])]

                # Build per-frame dict compatible with create_empty_dataset
                state16 = np.asarray(l_pos + r_pos, dtype=np.float32)
                action16 = state16.copy()

                frame = {
                    "task": task_name,
                    "observation.state": state16,
                    "action": action16,
                    "observation.images.robot_0": frame_l,
                    "observation.images.robot_1": frame_r,
                    "robot_0_action": np.asarray(l_pos, dtype=np.float32),
                    "robot_1_action": np.asarray(r_pos, dtype=np.float32),
                }

                dataset.add_frame(frame)
                valid += 1

            if valid > 0:
                dataset.save_episode()
                total_frames += valid

        else:  # single
            single = load_arm_data(paths["single"], traj_source)
            if single is None:
                print(f"[SKIP] missing data in single session: {session_path}")
                continue

            # `prepare_single_alignment` returns the following tuple:
            # - master_ts: np.ndarray, master timeline timestamps sampled at `target_fps`
            # - master_fidx: np.ndarray[int], frame indices on the camera corresponding to `master_ts`
            # - traj_arr: np.ndarray (N,7), trajectory rows with [Pos X,Pos Y,Pos Z,Qx,Qy,Qz,Qw]
            # - clamp_arr: np.ndarray (N,), gripper/clamp values aligned with trajectory timestamps
            # - traj_idx: np.ndarray[int], indices mapping each `master_ts` entry -> row in `traj_arr`
            # - clamp_idx: np.ndarray[int], indices mapping each `master_ts` entry -> row in `clamp_arr`
            # - frames: dict[int->ndarray], pre-extracted BGR frames keyed by original frame index
            (
                master_ts,
                master_fidx,
                traj_arr,
                clamp_arr,
                traj_idx,
                clamp_idx,
                frames,
            ) = prepare_single_alignment(single, target_fps, source_camera_fps)

            valid = 0
            for i in range(len(master_ts)):
                fidx = master_fidx[i]
                if fidx not in frames:
                    continue
                img = _to_rgb_and_resize(frames[fidx])
                pos_quat = list(traj_arr[traj_idx[i]]) + [float(clamp_arr[clamp_idx[i]])]

                state8 = np.asarray(pos_quat, dtype=np.float32)
                frame = {
                     "task": task_name,
                    "observation.state": state8,
                    "action": state8,
                    "observation.images.robot_0": img,
                }
                dataset.add_frame(frame)
                valid += 1

            if valid > 0:
                dataset.save_episode(task=task_name, encode_videos=True)
                total_frames += valid

        print(f"Processed session {idx+1}/{len(sessions)}: {session_path} — frames added: {valid}")

    # Finalize dataset
    dataset.stop_image_writer()

    # dataset.consolidate(run_compute_stats=True, keep_image_files=False)

    
    if repo_output is not None:
        target = Path(repo_output) / repo_id
        if target.exists():
            shutil.rmtree(target)
 
        shutil.copytree(LEROBOT_HOME / repo_id, target, dirs_exist_ok=True)
        print(f"Dataset copied to: {target}")

    print(f"Done. Total frames added: {total_frames}")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--task-name", type=str, default="Converted task")
    # parser.add_argument("--camera-names", type=str, default="front")
    parser.add_argument("--traj-source", type=str, default="merge")
    parser.add_argument("--target-fps", type=int, default=20)
    parser.add_argument("--source_camera_fps", type=int, default=60)
    parser.add_argument("--mode", type=str, default="video", choices=["video", "image"])
    parser.add_argument("--output", type=Path, default=None, help="optional copy destination for final dataset")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    convert_raw_to_lerobot(
        task_root=args.task_root,
        repo_id=args.repo_id,
        task_name=args.task_name,
        # camera_names=cam_names,
        traj_source=args.traj_source,
        target_fps=args.target_fps,
        source_camera_fps=args.source_camera_fps,
        repo_output=args.output,
        mode=args.mode,
    )

    # # Example usage:
    # convert_raw_to_lerobot(
    #     task_root=r'C:\project\data_process\task_test\background_01\multi_sessions_20251231_174451',
    #     repo_id='fastumi/task_test_20260128',
    #     task_name= 'Place the solid glue stick into the pen holder！',
    #     # camera_names=cam_names,
    #     traj_source= 'merge',
    #     target_fps=20,
    #     source_camera_fps = 60,
    #     repo_output= r'C:\project\data_process\task_test2',
    #     mode= 'video',
    # )
