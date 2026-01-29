
## 💻 Usage

Convert raw recorded sessions directly into a LeRobot dataset.

This script aligns timestamps, extracts frames from recorded videos, and writes episodes into a LeRobot dataset.

Key assumptions:
- Video frames are read with OpenCV (BGR), converted to RGB and resized
    to 224x224 (uint8) to match LeRobot image expectations.
- Single-arm sessions write images to `observation.images.robot_0`.
    Dual-arm sessions populate both `robot_0` and `robot_1` image fields.

Minimal usage example (required flags shown):
```code
    python raw_2_lerobot_V4.0.py \
        --task-root /path/to/raw/task_root \
        --repo-id myorg/myrepo \
        --task-name "Pick and place" \
        --traj-source merge \
        --target-fps 20 \
        --source_camera_fps 20 \
        --mode video \
        --output /path/to/save/lerobot
```

Notes:
- The script expects session directories named like `session*` under
Notes:
- The script expects session directories named like `session*` under
  `--task-root` and standard subfolders (`RGB_Images`, `Clamp_Data`,
  `Merged_Trajectory`, etc.). See the code for exact expectations.
- Required Python packages: `opencv-python`, `numpy`, `pandas`, `h5py`,
  and the `lerobot` package providing `LeRobotDataset`.

Recommended `lerobot` dependency (git + pinned revision):
```code
pip install lerobot==0.4.0 -i https://pypi.tuna.tsinghua.edu.cn/simple     
```
