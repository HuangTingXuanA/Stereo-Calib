# 双目相机标定误差计算原理分析报告

本报告基于 `src` 目录下的代码实现（特别是 `StereoCalibrator.cpp`），从**第一性原理**剖析项目当前的误差计算逻辑。系统通过三个层面的指标来联合评估标定质量。

---

## 1. 像素重投影误差 (Reprojection Error)

### 1.1 数学原理
像素重投影误差是衡量**数学参数模型与实际图像特征**之间契合度的最基础指标。
基于针孔相机模型，世界坐标系中的点 $\mathbf{P}_w = [X, Y, Z, 1]^T$ 投射到像素平面 $\mathbf{p} = [u, v]^T$ 的公式为：
$$\mathbf{p}_{proj} = K \cdot \text{Distort}(R \cdot \mathbf{P}_w + \mathbf{t})$$
其中 $K$ 为内参矩阵，$R, \mathbf{t}$ 为该帧图像的位姿（外参），$\text{Distort}$ 为畸变模型（本项目使用了含有 8 个参数的有理畸变模型）。

### 1.2 代码实现
在 `computeReprojectionErrors()` 函数中实现：
1. **输入量**：已知的标定板物理点 `world_points`，检测到的图像圆心 `left_centers` / `right_centers`。
2. **计算过程**：
   - 使用 `cv::projectPoints` 将物理点投影回图像生成 `projected_left`。
   - 计算欧式距离：`err = cv::norm(detected - projected)`。
3. **输出量**：
   - 每组图像的平均误差（px）。
   - 所有图像的总平均重投影误差 `reproj_err_left/right`。

---

## 2. 标定 RMS 误差 (RMS Error)

### 2.1 定义
RMS（Root Mean Square，均方根误差）是优化算法（Levenberg-Marquardt）在收敛时的目标函数残差。它对大误差（离群点）更为敏感，公式为：
$$RMS = \sqrt{\frac{1}{N} \sum_{i=1}^N \| \mathbf{p}_{detected,i} - \mathbf{p}_{proj,i} \|^2}$$

### 2.2 双目约束 (Stereo Constraint)
在 `calibrate()` 函数中，除了左右相机的单目 RMS，还计算了 `rms_stereo`：
- **第一性原理意义**：双目标定不是简单的两个单目标定的组合，它引入了左右相机相对位姿 $(R, \mathbf{T})$ 的**刚性约束**。
- **计算逻辑**：`cv::stereoCalibrate` 同时优化内外参，使得两幅图像中的对应点必须满足对极几何约束。`rms_stereo` 反映了整个双目系统（包括同步、支架结构、对应匹配）的全局一致性。

---

## 3. 几何一致性误差 (Geometric Consistency)

这是本项目中最具特色的、基于**第一性原理**的误差评估方法，在 `evaluateGeometricConsistency()` 中实现。

### 3.1 原理：平面单应性 (Planar Homography)
不同于重投影误差依赖于相机参数模型（容易产生过拟合），几何一致性利用了**标定板是理想平面**这一物理先验。

1. **拟合单应性矩阵**：假设图像去畸变后是理想的，那么物理坐标 $(X, Y)$ 到像素坐标 $(u, v)$ 之间存在唯一的线性变换（单应性矩阵 $H$）。
   $$\mathbf{p}_{px} \approx H \cdot \mathbf{P}_{mm}$$
2. **反求物理空间误差**：
   - 将图像圆心通过 $H^{-1}$ 映射回物理空间，得到 $\mathbf{P}_{back\_proj}$（单位：mm）。
   - 计算 $\mathbf{P}_{back\_proj}$ 与理想网格坐标 $[0, 20, 40, ...]$ 的物理距离。

### 3.2 现实指导意义
- **RMSE (px)**：告诉你“模型拟合得好不好”。
- **RMSE (mm)**：告诉你“检测到的圆心到底准不准”。
- **预警机制**：代码中设置了 `if (rmse_mm > 0.5) std::cout << "[警告: 几何一致性差]"`。这能直接定位出是因为光照、反光或分割不准导致的检测偏移，而非标定算法本身的问题。

---

## 4. 误差评估总结表

| 指标 | 代码位置 | 单位 | 反映问题 |
| :--- | :--- | :--- | :--- |
| **重投影误差** | `computeReprojectionErrors` | px | 相机参数建模的准确度 |
| **单目 RMS** | `cv::calibrateCamera` 返回值 | px | 单次标定优化的数值稳定性 |
| **双目 RMS** | `cv::stereoCalibrate` 返回值 | px | 左右相机相对位置的结构鲁棒性 |
| **平面拟合误差** | `evaluateGeometricConsistency` | **mm** | **原始检测点（圆心）的物理可信度** |

---
**结论**：
该项目通过“**像素级模型对齐 + 双目结构约束 + 物理平面验证**”三位一体的方案，构建了完整的标定质量评价体系。特别是通过单应性反求物理坐标偏移的方法，实现了对检测精度的量化质量回溯。
