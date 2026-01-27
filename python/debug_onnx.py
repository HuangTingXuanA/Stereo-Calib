#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX分割模型输出格式调试脚本

用于分析ONNX模型的输出格式，帮助C++代码正确解析
"""

import numpy as np
import onnxruntime as ort
import cv2
import os

def letterbox(img, new_shape=640, color=(114, 114, 114), stride=32):
    """保持纵横比的缩放并添加padding（Minimum Rectangle）"""
    shape = img.shape[:2]  # [height, width]
    
    # 计算缩放比例
    r = min(new_shape / shape[0], new_shape / shape[1])
    
    # 计算新尺寸
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    
    # Minimum Rectangle: 
    # 只要满足stride倍数即可，不一定非要补齐到new_shape
    # 如果是动态推理，这里的dw, dh是相对于new_shape的差值
    # 我们需要根据new_unpad重新计算dw, dh使其满足stride
    
    dw = (new_unpad[0] + stride - 1) // stride * stride - new_unpad[0]
    dh = (new_unpad[1] + stride - 1) // stride * stride - new_unpad[1]
    
    # 平均分配padding
    dw /= 2
    dh /= 2
    
    # resize
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    # 添加padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return img, r, (dw, dh)


def debug_onnx_output(model_path, image_path, imgsz=1024):
    """调试ONNX模型输出格式"""
    
    print("=" * 60)
    print("ONNX模型输出格式调试")
    print("=" * 60)
    
    # 加载模型
    print(f"\n加载模型: {model_path}")
    session = ort.InferenceSession(model_path)
    
    # 打印输入输出信息
    print("\n输入节点:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.shape}, {inp.type}")
    
    print("\n输出节点:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.shape}, {out.type}")
    
    # 加载图像
    print(f"\n加载图像: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print("错误: 无法加载图像")
        return
    
    orig_h, orig_w = img.shape[:2]
    print(f"原始尺寸: {orig_w} x {orig_h}")
    
    # 预处理
    img_letterbox, scale, (pad_w, pad_h) = letterbox(img, imgsz)
    print(f"缩放比例: {scale}, padding: ({pad_w}, {pad_h})")
    
    # BGR -> RGB, HWC -> CHW, 归一化
    img_input = img_letterbox[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    img_input = np.expand_dims(img_input, axis=0)  # [1, 3, 1024, 1024]
    
    # 推理
    print("\n执行推理...")
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_input})
    
    # 分析输出
    print("\n输出分析:")
    for i, out in enumerate(outputs):
        print(f"\n输出 {i}:")
        print(f"  形状: {out.shape}")
        print(f"  数据类型: {out.dtype}")
        print(f"  值范围: [{out.min():.4f}, {out.max():.4f}]")
        
        if i == 0:
            # 检测输出 [1, num_dets, 38]
            print(f"\n  检测输出详细分析:")
            print(f"    - 检测数量: {out.shape[1]}")
            print(f"    - 每个检测的维度: {out.shape[2]}")
            
            # 找置信度最高的检测
            confs = out[0, :, 4]  # 置信度在第4个位置
            best_idx = np.argmax(confs)
            best_conf = confs[best_idx]
            best_det = out[0, best_idx]
            
            print(f"\n  最高置信度检测 (idx={best_idx}):")
            print(f"    置信度: {best_conf:.4f}")
            print(f"    前10个值: {best_det[:10]}")
            print(f"    bbox (cx,cy,w,h): ({best_det[0]:.1f}, {best_det[1]:.1f}, {best_det[2]:.1f}, {best_det[3]:.1f})")
            print(f"    类别ID: {best_det[5]:.0f}")
            
            # 转换边界框到原图坐标
            x1_in, y1_in, x2_in, y2_in = best_det[0], best_det[1], best_det[2], best_det[3]
            x1 = (x1_in - pad_w) / scale
            y1 = (y1_in - pad_h) / scale
            x2 = (x2_in - pad_w) / scale
            y2 = (y2_in - pad_h) / scale
            print(f"    BBox (x1,y1,x2,y2): ({x1_in:.1f}, {y1_in:.1f}, {x2_in:.1f}, {y2_in:.1f})")
            print(f"    原图坐标: ({x1:.1f}, {y1:.1f}) - ({x2:.1f}, {y2:.1f})")
            
        elif i == 1:
            # Proto输出 [1, 32, 160, 160]
            print(f"\n  Proto输出详细分析:")
            print(f"    - Batch: {out.shape[0]}")
            print(f"    - 通道数: {out.shape[1]}")
            print(f"    - 高度: {out.shape[2]}")
            print(f"    - 宽度: {out.shape[3]}")
    
    # 可视化最佳检测的掩码
    print("\n生成并保存掩码可视化...")
    
    det_output = outputs[0]
    proto_output = outputs[1]
    
    confs = det_output[0, :, 4]
    best_idx = np.argmax(confs)
    best_det = det_output[0, best_idx]
    
    # mask系数从索引6开始
    mask_coef = best_det[6:38]  # 32个系数
    print(f"Mask系数: 共{len(mask_coef)}个, 范围[{mask_coef.min():.4f}, {mask_coef.max():.4f}]")
    
    # 计算mask
    proto = proto_output[0]  # [32, proto_h, proto_w]
    _, proto_h, proto_w = proto.shape
    proto_flat = proto.reshape(32, -1).T  # [proto_h*proto_w, 32]
    mask_logits = proto_flat @ mask_coef  # [proto_h*proto_w]
    mask_logits = mask_logits.reshape(proto_h, proto_w)
    
    # sigmoid
    mask_sigmoid = 1 / (1 + np.exp(-mask_logits))
    
    # 可视化
    mask_vis = (mask_sigmoid * 255).astype(np.uint8)
    cv2.imwrite("debug_mask_160.png", mask_vis)
    print(f"保存: debug_mask_160.png ({proto_w}x{proto_h} sigmoid mask)")
    
    # 放大到输入图像尺寸 (基于 letterbox 后的输入尺寸，这里通常是 640 或动态尺寸)
    input_h, input_w = img_letterbox.shape[:2]
    mask_resized = cv2.resize(mask_vis, (input_w, input_h))
    cv2.imwrite("debug_mask_640.png", mask_resized)
    print(f"保存: debug_mask_640.png ({input_w}x{input_h} mask)")
    
    # 应用边界框裁剪 (防止掩码溢出到边界框外)
    x1_in, y1_in, x2_in, y2_in = best_det[0], best_det[1], best_det[2], best_det[3]
    x1 = max(0, int(x1_in))
    y1 = max(0, int(y1_in))
    x2 = min(input_w, int(x2_in))
    y2 = min(input_h, int(y2_in))
    
    mask_cropped = np.zeros((input_h, input_w), dtype=np.uint8)
    mask_cropped[y1:y2, x1:x2] = mask_resized[y1:y2, x1:x2]
    cv2.imwrite("debug_mask_cropped.png", mask_cropped)
    print(f"保存: debug_mask_cropped.png (bbox裁剪: ({x1},{y1})-({x2},{y2}))")
    
    # 去除padding并缩放到原图
    scaled_w = int(orig_w * scale)
    scaled_h = int(orig_h * scale)
    pad_w_int = int(pad_w)
    pad_h_int = int(pad_h)
    
    mask_no_pad = mask_cropped[pad_h_int:pad_h_int+scaled_h, pad_w_int:pad_w_int+scaled_w]
    mask_original = cv2.resize(mask_no_pad, (orig_w, orig_h))
    cv2.imwrite("debug_mask_original.png", mask_original)
    print(f"保存: debug_mask_original.png (原图尺寸 {orig_w}x{orig_h})")
    
    # 叠加到原图
    img_vis = img.copy()
    mask_color = np.zeros_like(img_vis)
    mask_color[:, :, 1] = mask_original  # 绿色通道
    img_vis = cv2.addWeighted(img_vis, 0.7, mask_color, 0.3, 0)
    
    # 绘制边界框
    bx1 = int((x1_in - pad_w) / scale)
    by1 = int((y1_in - pad_h) / scale)
    bx2 = int((x2_in - pad_w) / scale)
    by2 = int((y2_in - pad_h) / scale)
    cv2.rectangle(img_vis, (bx1, by1), (bx2, by2), (0, 255, 255), 3)
    
    cv2.imwrite("debug_result.png", img_vis)
    print("保存: debug_result.png (叠加可视化)")
    
    print("\n调试完成!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ONNX模型输出格式调试脚本')
    parser.add_argument('--model', '-m', type=str, required=True, help='ONNX模型路径')
    parser.add_argument('--image', '-i', type=str, required=True, help='测试图像路径')
    parser.add_argument('--imgsz', '-s', type=int, default=1024, help='推理长边尺寸')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"错误: 模型不存在: {args.model}")
    elif not os.path.exists(args.image):
        print(f"错误: 图像不存在: {args.image}")
    else:
        debug_onnx_output(args.model, args.image, args.imgsz)
