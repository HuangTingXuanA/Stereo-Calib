#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO26n 分割推理测试脚本

用途：测试分割模型在测试图像上的效果

使用方法：
    python inference.py --model runs/segment/calibration_board/weights/best.pt --source test.jpg --show
"""

import argparse
import os
import cv2
import numpy as np
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description='分割模型推理测试')
    parser.add_argument('--model', type=str, required=True,
                        help='模型路径 (.pt 或 .onnx)')
    parser.add_argument('--source', type=str, required=True,
                        help='输入图像/目录路径')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='输入图像尺寸')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='置信度阈值')
    parser.add_argument('--save', action='store_true',
                        help='保存结果')
    parser.add_argument('--output', type=str, default='results',
                        help='结果保存目录')
    parser.add_argument('--show', action='store_true',
                        help='显示结果（按任意键继续）')
    return parser.parse_args()


def run_inference(args):
    print("=" * 50)
    print("YOLO26n 分割推理测试")
    print("=" * 50)
    
    print(f"\n加载模型: {args.model}")
    model = YOLO(args.model)
    
    print(f"推理中... (置信度阈值: {args.conf})")
    
    # 执行推理，不使用内置show（会闪退）
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        save=args.save,
        project=args.output if args.save else None,
        show=False,  # 禁用内置show
    )
    
    # 打印分割结果
    print("\n检测结果:")
    print("-" * 60)
    for i, result in enumerate(results):
        print(f"\n图像 {i+1}: {result.path}")
        
        if result.masks is not None and len(result.masks) > 0:
            for j in range(len(result.masks)):
                conf = result.boxes.conf[j].cpu().numpy()
                cls = int(result.boxes.cls[j].cpu().numpy())
                
                print(f"\n  分割 {j+1}:")
                print(f"    类别: {cls} ({result.names[cls]})")
                print(f"    置信度: {conf:.4f}")
                
                # 获取边界框 (x1, y1, x2, y2)
                box = result.boxes.xyxy[j].cpu().numpy()
                print(f"    BBox (x1, y1, x2, y2): {box}")
                
                # 获取掩码信息
                mask = result.masks.data[j].cpu().numpy()
                print(f"    掩码尺寸: {mask.shape}")
                print(f"    掩码面积: {np.sum(mask > 0.5)} 像素")
                
                # 获取多边形轮廓
                if hasattr(result.masks, 'xy') and j < len(result.masks.xy):
                    polygon = result.masks.xy[j]
                    print(f"    多边形顶点数: {len(polygon)}")
        else:
            print("  未检测到目标")
        
        # 手动显示结果图像
        if args.show:
            # 获取绘制了结果的图像
            plot_img = result.plot()
            
            # 如果图像太大，缩放显示
            h, w = plot_img.shape[:2]
            max_size = 1200
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                plot_img = cv2.resize(plot_img, (int(w*scale), int(h*scale)))
            
            window_name = f"Result {i+1} - Press any key to continue"
            cv2.imshow(window_name, plot_img)
            print(f"\n显示结果图像，按任意键继续...")
            cv2.waitKey(0)  # 等待按键
            cv2.destroyAllWindows()
    
    if args.save:
        print(f"\n结果已保存到: {args.output}")
    
    return results


if __name__ == '__main__':
    args = parse_args()
    
    if not os.path.exists(args.model):
        print(f"错误: 模型不存在: {args.model}")
    elif not os.path.exists(args.source):
        print(f"错误: 输入路径不存在: {args.source}")
    else:
        run_inference(args)
