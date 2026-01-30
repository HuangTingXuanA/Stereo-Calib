# YOLO26n 分割 - 标定板实例分割

## 方案说明

使用**实例分割**检测标定板区域，输出精确的多边形边界，避免OBB的冗余区域。

## 文件结构

```
python/
├── data.yaml           # 分割数据集配置
├── train.py            # 分割训练脚本
├── export_onnx.py      # ONNX导出
├── debug_onnx.py       # ONNX调试
├── inference.py        # 推理测试
├── split_dataset.py    # 数据集划分
├── models/             # 模型文件
├── runs/               # 训练输出
└── dataset/
    ├── classes.txt
    ├── images/{train,val}/
    └── labels/{train,val}/
```

## 分割标签格式

```
class_index x1 y1 x2 y2 x3 y3 ... xn yn
```
多边形顶点坐标（归一化0-1），至少3个点。

## 使用流程

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 用X-AnyLabeling标注（使用polygon模式）
x_anylabeling

# 3. 训练
python train.py --data data.yaml --epochs 100

# 4. 推理测试
python inference.py --model runs/calibration_board/weights/best.pt --source test.jpg --show

# 5. 导出ONNX
python export_onnx.py --model runs/calibration_board/weights/best.pt

# 6. 调试ONNX
python debug_onnx.py --model models/calibration_board_seg_dynamic.onnx --image test_images/left/image_0.bmp
```

## 场景
```bash
# 场景 1：新增样本后的“增量学习”（微调）
# 如果往原始的数据集里添加了新的标定板图像或负样本，模型会继承之前的“记忆”，只针对新加入的特征进行针对性强化
python train.py --model runs/calibration_board/weights/best.pt --epochs 50 --imgsz 1024

# 场景 2：训练意外中断后的恢复
# 会自动读取中断时的 epoch、优化器状态和学习率，无缝接续。
python train.py --model runs/calibration_board/weights/last.pt --resume
```

## 模型

- **yolo26n-seg.pt** - 轻量级分割模型
