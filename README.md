# ONNX Runtime Sample Apps
This repository contains sample apps for running ONNX models efficiently using [ONNX Runtime](https://onnxruntime.ai/), specifically targeting Qualcomm Hexagon NPU with [QNN Execution Provider](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)
## Table of Contents
1. [Available Apps](#available-apps)
2. [Project Status](#project-status)
3. [Getting Started](#getting-started)
4. [Quick Start](#quick-start)
5. [Contributing](#contributing)
6. [Directory Structure](#directory-structure)
7. [License](#license)

## Available Apps
| App Name               | Model Used | Providers | Quick Start                                                                          | Notebook                                                                                          | Notes |
|------------------------|------------|-----------|--------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|-------|
| 'HRNet Pose Detection' | HRNetPose  | CPU       |[Complete](https://github.com/DerrickJ1612/qnn_sample_apps/tree/main/src/hrnet_pose)  |[Complete](https://github.com/DerrickJ1612/qnn_sample_apps/tree/main/notebooks/pose_detection)     | None  |
| 'DeepSeek Local'       | DeepSeek   | QNN       |[Complete](https://github.com/DerrickJ1612/qnn_sample_apps/tree/main/src/deepseek_r1) |[Complete](https://github.com/DerrickJ1612/qnn_sample_apps/tree/main/notebooks/reasoning_llm)      | None  |

These apps demonstrate end-to-end inference using ONNX Runtime on devices with Hexagon NPUs. Each app includes:
- Input preprocessing
- Onnx model inference
- Output postprocessing
  
Supported features:
- CPU fallback (if you don't have access to Hexagon NPU)
- Hexagon QNN Execution

## Getting Started
#### General Requirements
- Python (version 3.11.+)
   - If targeting Hexagon ensure you install ARM64 compatible Python version
- C and C++ support in Visual Studio
   - [Installation Instructions](https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=msvc-170)
- Rust Support
   - [Installation Instructions](https://rustup.rs/)
#### 1. Clone the Repository
```
>> git clone https://github.com/qualcomm/snapdragon-compute-samples.git
```
#### 2. Setup Virtual Environment
```
>> python -m venv venv
>> venv\Scripts\activate.ps1 # Linux: >> source venv/bin/activate
>> pip install -r \src\<App Name>\requirements.txt # Ex: pip install -r \src\deepseek_r1\requirements.txt
```
#### 3. Download Models
| Model Name  | Description              | Download Source                                                                                               |
|-------------|--------------------------|---------------------------------------------------------------------------------------------------------------|
| HRNetPose   | Human pose estimation    | [AI Hub](https://aihub.qualcomm.com/compute/models/hrnet_pose?domain=Computer+Vision&useCase=Pose+Estimation) |
| DeepSeek R1 | Reasoning Language Model | [s3 Bucket](tbd)                                                                                              | 

#### 4. Run models.py
models.py will automatically place models in appropriate destination
```
>> python models.py --model_directory (absolute path to directory where models were downloaded)
```

## Quick Start

| App Name               | CLI Command                                 |
|------------------------|---------------------------------------------|
| 'HRNet Pose Detection' | ` >> python ./src/hrnet_pose/main.py `      |
| 'DeepSeek Local'       | ` >> python ./src/deepseek_r1/main.py `     |

## Contributing
We welcome contributions to this repository! Please refer to our [contributing guide](CONTRIBUTING.md) for how to contribute.

## Directory Structure
## License
This project is licensed under the [MIT](https://github.com/DerrickJ1612/qnn_sample_apps/blob/main/LICENSE.txt)

