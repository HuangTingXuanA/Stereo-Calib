#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO26n 分割训练脚本 - 标定板实例分割

用途：训练YOLO26n-seg模型精确分割标定板区域

使用方法：
    python train.py --data data.yaml --epochs 100
    
    # CLI
    # 情况1：从零开始训练
    python train.py --data data.yaml --epochs 100
    
    # 情况2：微调（基于上一次最好的结果继续训练新样本）
    python train.py --model runs/calibration_board/weights/best.pt --epochs 50
    
    # 情况3：中断恢复（如果训练意外中断，继续跑完剩下的epochs）
    python train.py --model runs/calibration_board/weights/last.pt --resume
"""

import argparse
import os
from ultralytics import YOLO


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLO26n 分割训练脚本')
    
    parser.add_argument('--model', type=str, default='yolo26n-seg.pt',
                        help='预训练模型 (默认: yolo26n-seg.pt)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数 (默认: 100)')
    parser.add_argument('--imgsz', type=int, default=1024,
                        help='输入图像尺寸 (默认: 1024)')
    parser.add_argument('--batch', type=int, default=16,
                        help='批量大小 (默认: 16)')
    parser.add_argument('--device', type=str, default='0',
                        help='训练设备 (默认: 0)')
    parser.add_argument('--data', type=str, default='data.yaml',
                        help='数据集配置文件路径')
    parser.add_argument('--project', type=str, default='runs',
                        help='项目保存路径 (默认: runs)')
    parser.add_argument('--name', type=str, default='calibration_board',
                        help='实验名称')
    parser.add_argument('--resume', action='store_true',
                        help='断点续训 (继续未完成的训练任务)')
    
    return parser.parse_args()


def train(args):
    """执行分割训练"""
    print("=" * 50)
    print("YOLO26n 分割训练 - 标定板实例分割")
    print("=" * 50)
    
    if not os.path.exists(args.data):
        print(f"错误: 数据集配置文件不存在: {args.data}")
        return None
    
    print(f"\n加载预训练模型: {args.model}")
    model = YOLO(args.model)
    
    print(f"\n开始分割训练...")
    print(f"  数据集: {args.data}")
    print(f"  轮数: {args.epochs}")
    print(f"  图像尺寸: {args.imgsz}")
    print(f"  批量大小: {args.batch}")
    
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume, # 支持断点续训
        patience=50,      # 增加早停阈值
        save=True,
        plots=True,
        # 数据增强优化
        perspective=0.001, # 增加透视变换鲁棒性
        degrees=15.0,      # 增加旋转旋转鲁棒性
        scale=0.5,         # 增加缩放缩放鲁棒性
        mosaic=1.0,        # 增强小目标检测
        mixup=0.1,         # 适当使用mixup
    )
    
    print(f"\n训练完成! 模型保存在: {args.project}/{args.name}/weights/best.pt")
    return results


if __name__ == '__main__':
    train(parse_args())
