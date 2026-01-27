#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO26n 分割 ONNX 导出脚本

用途：将训练好的分割模型导出为ONNX格式

使用方法：
    # 默认即为动态尺寸导出（支持不同分辨率输入）
    python export_onnx.py --model runs/calibration_board/weights/best.pt
    
    # 固定尺寸导出
    python export_onnx.py --model runs/calibration_board/weights/best.pt --no-dynamic
"""

import argparse
import os
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description='分割模型 ONNX 导出')
    parser.add_argument('--model', type=str, required=True,
                        help='分割模型路径 (.pt文件)')
    parser.add_argument('--imgsz', type=int, default=1024,
                        help='输入图像尺寸 (用于固定尺寸导出)')
    parser.add_argument('--opset', type=int, default=12,
                        help='ONNX opset版本')
    parser.add_argument('--dynamic', action='store_true', default=True,
                        help='启用动态输入尺寸 (默认: True)')
    parser.add_argument('--no-dynamic', action='store_false', dest='dynamic',
                        help='禁用动态输入尺寸 (使用固定 imgsz)')
    parser.add_argument('--output', type=str, default='models',
                        help='输出目录')
    parser.add_argument('--name', type=str, default='',
                        help='输出文件名（默认自动生成）')
    return parser.parse_args()


def export_onnx(args):
    print("=" * 50)
    print("YOLO26n 分割模型 ONNX 导出")
    print("=" * 50)
    
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        return None
    
    print(f"\n加载模型: {args.model}")
    model = YOLO(args.model)
    
    os.makedirs(args.output, exist_ok=True)
    
    # 确定输出文件名
    if args.name:
        output_name = args.name
    else:
        suffix = "_dynamic" if args.dynamic else f"_{args.imgsz}"
        output_name = f"calibration_board_seg{suffix}.onnx"
    
    print(f"\n导出配置:")
    print(f"  图像尺寸: {args.imgsz}")
    print(f"  Opset版本: {args.opset}")
    print(f"  动态尺寸: {'是' if args.dynamic else '否'}")
    
    if args.dynamic:
        print("\n注意: 动态尺寸模式支持不同分辨率输入")
        print("      但推理速度可能略慢于固定尺寸")
    
    # 执行导出
    export_path = model.export(
        format='onnx',
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=True,
        dynamic=args.dynamic,  # 动态尺寸
    )
    
    if export_path:
        import shutil
        output_path = Path(args.output) / output_name
        if Path(export_path).exists():
            shutil.move(export_path, output_path)
            print(f"\n✅ 导出成功: {output_path}")
            
            # 打印使用说明
            print(f"\n使用说明:")
            if args.dynamic:
                print("  - 支持任意分辨率输入")
                print("  - 推理时会自动letterbox到32的倍数")
            else:
                print(f"  - 固定输入尺寸: {args.imgsz}x{args.imgsz}")
                print("  - 推理时需先resize到该尺寸")
            
            return str(output_path)
    
    print("❌ 导出失败!")
    return None


if __name__ == '__main__':
    export_onnx(parse_args())
