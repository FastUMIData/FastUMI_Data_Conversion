## 🚀 Key Features

* **Recursive Scanning**: Automatically finds all `session_*` folders across multiple directory levels.
* **Auto Layout Detection**: Switches between Single-Arm and Dual-Arm processing logic based on folder structure.
* **Frequency Alignment**: Downsamples and aligns 60fps raw data to target frequencies (**20Hz**, **30Hz**, or **60Hz**).
* **Parallel Processing**: Uses multi-processing to significantly speed up video decoding and HDF5 compression.
* **Robustness**: Automatically handles `txt` to `csv` conversion for gripper data and validates video integrity via `ffprobe`.

---

## 📂 Required Data Structure

The script expects the following organization:

### raw data Structure
Single-Arm Layout
```text
session_001/
├── RGB_Images/
│   ├── video.mp4
│   └── timestamps.csv
├── Clamp_Data/
│   └── clamp_data_tum.txt
└── Merged_Trajectory/
    └── merged_trajectory.txt
```

Dual-Arm Layout
```text
session_001/
├── left_hand_data/   (folder starting with 'left_hand')
│   ├── RGB_Images/ ...
│   ├── Clamp_Data/ ...
│   └── Merged_Trajectory/ ...
└── right_hand_data/  (folder starting with 'right_hand')
    └── ...
```

## 🛠 Prerequisites
Install the required Python libraries:
```bash
pip install pandas numpy h5py opencv-python tqdm
```
Note: Ensure FFmpeg is installed on your system, as the script uses ffprobe to verify video health.

