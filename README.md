# 双目相机标定系统 (YOLO-Enhanced)

基于圆形标定板的高精度双目相机标定系统。系统结合深度学习（YOLO 分割）与亚像素几何算法，提供鲁棒且精确的标定能力，支持多种标定板布局及外部坐标导入。

## 系统特性

- **YOLO 智能分割**: 集成 ONNX Runtime，使用 YOLO 分割模型预定位标定板，显著提升复杂背景下的检测鲁棒性和处理效率。
- **高精度椭圆检测**:
  - 基于 ED (Edge Drawing) 算法的高速边缘提取。
  - 采用 Gauss-Newton 迭代结合 **Tukey Loss** 的亚像素级椭圆精修。
  - 支持自适应径向边缘修正，消除光照不均引起的边缘偏移。
- **灵活的标定板配置**:
  - **自动生成模式**: 仅需设置行、列、间距即可自动生成规则网格。
  - **外部文件模式**: 支持导入第三方测量（如测量仪得到的高精度坐标）得到的 3D 坐标文件。
- **鲁棒的特征识别**:
  - **指纹匹配**: 基于几何指纹的旋转不变性特征识别，支持标定板任意角度摆放。
  - **缺失圆推算**: 通过单应性变换 (Homography) 自动补全被遮挡或未检出的缺失圆心。
- **统一参数管理**: 核心算法阈值归口至 `include/param.h`，支持根据工况深度调优。

## 编译要求

- **C++ 标准**: C++17
- **CMake**: >= 3.16
- **依赖库**:
  - **OpenCV**: 4.x
  - **ONNX Runtime**: 必须安装，用于加载 `.onnx` 模型

### onnx安装
下载包路径：https://github.com/microsoft/onnxruntime/releases
Linux安装指令：解压tar文件
```shell
sudo cp -r include/* /usr/local/include
sudo cp -P lib/libonnxruntime*.so* /usr/local/lib/
sudo ldconfig
```

### 编译步骤

```bash
cmake --build build -j$(nproc)
```

## 使用方法

目前系统统一使用配置文件启动：

```bash
./build/stereo_calibrator board.yaml ./images
```

### 配置文件说明 (`board.yaml`)

```yaml
%YAML:1.0
---
board:
   rows: 9               # 标定板行数
   cols: 11              # 标定板列数
   auto_generate_coords: 1 # 1: 自动生成网格; 0: 使用 coords_file
   circle_spacing: 60.0  # 圆心间距 (mm)，仅自动模式使用
   coords_file: "LV-CB-251011.txt" # 外部坐标文件，仅外部模式使用

# 特征锚点位置 (Row, Col) - 必须配置 5 个
anchor_circle:
   - [2, 5]
   - [4, 2]
   - [4, 8]
   - [6, 5]
   - [6, 6]

# YOLO 分割模型配置
segmentor:
   model_path: "models/calibration_board_seg_dynamic.onnx"
   inference_size: 1024
   confidence: 0.25
```

## 图像目录结构

```text
images/
├── left/
│   ├── image_0.bmp
│   └── ...
└── right/
    ├── image_0.bmp
    └── ...
```

## 参数调优 (`include/param.h`)

如需调整边缘检测或椭圆拟合性能，参考以下结构体：

- `EdgeParams`: 调整 `gradThresh`（梯度阈值）和 `sigma`（平滑系数）。
- `EllipseParams`: 调整 `minEllipseScore1`（内点率要求）和 `radialSearchRange`（径向搜索范围）。
- `DetectorParams`: 设置 `debug = true` 可保存各阶段诊断图到 `debug_diag/`。

## 输出结果

- `calibration.yaml`: 相机内参及双目外参。
- `points_3d.txt`: 标定点云数据。

## 技术亮点

1. **ROI 提取**: YOLO 分割快速排除环境背景干扰。
2. **边缘建模**: 使用 ED 算法替代普通的轮廓查找，获得更连续的质感。
3. **鲁棒精修**: Tukey Loss 能够有效抑制边缘毛刺和噪声对圆心的影响。
4. **单应性补全**: 基于 5 个特征锚点建立的单应性模型，对局部缺失具有极强的鲁棒性。
