# PointCloud Processing using ZED Stereo Camera & Open3D

This repository contains a complete pipeline for:
- Capturing point clouds using the **ZED Stereo Camera**
- Streaming and saving `.ply` point cloud data
- Processing and registering point clouds using **Open3D**

---

## Project Structure

```

PointCloud/
│
├── PointCloudStream/
│   ├── get_python_api.py
│   ├── depth_sensing.py
│   ├── streaming_receiver.py
│   ├── streaming_sender.py
│   ├── ogl_viewer/
│       ├── viewer.py
│
├── Open3D_Processing/
│   ├── desk1.ply
│   ├── desk2.ply
│   ├── desk.ipynb
│   ├── TutorialExamples/
│
├── requirements.txt
├── README.md

```

---

## System Requirements

Before installing Python dependencies, install the following:

1. **ZED SDK**  
   https://www.stereolabs.com/en-gb/developers/release

2. **NVIDIA CUDA Toolkit**  
   https://developer.nvidia.com/cuda-downloads

> ⚠️ Make sure CUDA version is compatible with your GPU and ZED SDK version.

---

## Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
````

---

## Setup & Execution Workflow

### Step 1: Install ZED Python API

Run:

```bash
python get_python_api.py
```

This will:

* Detect your system configuration
* Download and install the correct `pyzed` wheel file

---

### Step 2: Test Camera Streaming

Connect your ZED camera to the system.

#### Terminal 1:

```bash
python streaming_sender.py
```

* This starts streaming
* Note the **port number** displayed

#### Terminal 2:

```bash
python streaming_receiver.py --ip_address 127.0.0.1:<port>
```

* Replace `<port>` with the port from sender

If successful, you should see the live camera feed.

---

### Step 3: Capture Point Cloud

Run:

```bash
python depth_sensing.py
```

* This opens live point cloud visualization
* Press **`s`** to save the scene

Output:

```
PointCloud/Pointcloud.ply
```

> ⚠️ Important:

* Every new capture overwrites `Pointcloud.ply`
* Rename previous files before capturing new ones

---

## Open3D Processing

Navigate to:

```
Open3D_Processing/
```

### 📌 Objective

Capture **two point clouds** of the same scene from different angles:

* `desk1.ply`
* `desk2.ply`

---

## Processing Pipeline (desk.ipynb)

The notebook performs the following:

### 1. Read Point Clouds

```python
desk1 = o3d.io.read_point_cloud("desk1.ply")
desk2 = o3d.io.read_point_cloud("desk2.ply")
```

---

### 2. Visualisation

* Interactive 3D visualization using Plotly

---

### 3. Downsampling

```python
voxel_size = 0.02

```
* Reduces noise and computation
* Increasing the voxel size more can downsample too much, sometimes leading to loss of fine data features.

---

### 4. Global Registration (RANSAC)

* Uses FPFH features
* Estimates rough alignment

```python
registration_ransac_based_on_feature_matching(...)
```

* Outputs transformation matrix (`trans_init`)

---

### 5. Local Registration (ICP)

Refines alignment using:

#### ➤ Point-to-Point ICP

```python
TransformationEstimationPointToPoint()
```

#### ➤ Point-to-Plane ICP

```python
TransformationEstimationPointToPlane()
```

* Requires good initial transformation from RANSAC

---

### 6. Evaluation

```python
evaluate_registration()
```

* Measures alignment quality

---

## Key Notes

* **Global registration → gives rough alignment**
* **ICP → refines alignment**
* Always use **RANSAC result as `trans_init` for ICP**

---

## Output

* Registered / stitched point clouds
* Transformation matrices
* Visual comparison of alignment

---

## Troubleshooting

* `pyzed not found` → Run `get_python_api.py`
* Camera not detected → Check ZED SDK installation
* CUDA errors → Verify GPU + CUDA compatibility
* Plotly not working → `pip install plotly`

---

## Acknowledgements

* Stereolabs ZED SDK
* Open3D Library
