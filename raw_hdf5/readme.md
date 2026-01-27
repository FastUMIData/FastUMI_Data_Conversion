
## 💻 Usage

### Basic Execution
Specify the input root and the output directory:

```bash
python raw2hdf5.py --task_root /path/to/raw_data --output /path/to/hdf5_output
```
### Arguments
| Argument         | Description                                              | Default |
| ---------------- | -------------------------------------------------------- | ------- |
| `--task_root`    | (Required) The root directory containing session folders | `-`     |
| `--output`       | (Required) Path to save the generated HDF5 files         | `-`     |
| `--frequency`    | Target alignment frequency: `20`, `30`, or `60`          | `20`    |
| `--camera_names` | Names of the camera datasets inside the HDF5             | `front` |
| `--num_workers`  | Number of parallel processes for conversion              | `8`     |


## 📊 HDF5 Structure

The output `.hdf5` files are structured based on the detected arm layout.

### Single-Arm Layout

```text
observations/
  images/
    <camera_name>: (T, H, W, C) uint8
  qpos:   (T, 8) float32  # [x, y, z, qx, qy, qz, qw, clamp]
action:  (T, 8) float32
```
### Dual-Arm Layout
```text
robot_0/  # Left Arm
  observations/
    images/
      <camera_name>: (T, H, W, C) uint8
    qpos:   (T, 8) float32
  action:  (T, 8) float32

robot_1/  # Right Arm
  observations/
    images/
      <camera_name>: (T, H, W, C) uint8
    qpos:   (T, 8) float32
  action:  (T, 8) float32
```