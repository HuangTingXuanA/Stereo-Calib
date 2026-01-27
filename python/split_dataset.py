#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集划分脚本

用途：根据data.yaml配置，将标注好的图像和标签划分为训练集和验证集

使用方法：
    python split_dataset.py
"""

import os
import shutil
import random
from pathlib import Path
import yaml


def load_config(config_path='data.yaml'):
    """加载data.yaml配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_image_extensions():
    """支持的图像扩展名"""
    return {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


def split_dataset():
    """根据data.yaml配置划分数据集"""
    print("=" * 50)
    print("数据集划分工具")
    print("=" * 50)
    
    # 加载配置
    config = load_config()
    
    # 获取配置参数
    output_dir = Path(config.get('path', 'dataset'))
    source_images = config.get('source_images')
    source_labels = config.get('source_labels')
    split_ratio = config.get('split_ratio', 0.8)
    
    # 检查必要配置
    if not source_images or source_images == '/path/to/your/images':
        print("错误: 请在data.yaml中设置source_images（原始图像目录）")
        return
    if not source_labels or source_labels == '/path/to/your/labels':
        print("错误: 请在data.yaml中设置source_labels（原始标签目录）")
        return
    
    images_dir = Path(source_images)
    labels_dir = Path(source_labels)
    
    # 检查输入目录
    if not images_dir.exists():
        print(f"错误: 图像目录不存在: {images_dir}")
        return
    if not labels_dir.exists():
        print(f"错误: 标签目录不存在: {labels_dir}")
        return
    
    print(f"\n配置信息:")
    print(f"  原始图像: {images_dir}")
    print(f"  原始标签: {labels_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  训练比例: {split_ratio}")
    
    # 创建输出目录
    train_images_dir = output_dir / 'images' / 'train'
    val_images_dir = output_dir / 'images' / 'val'
    train_labels_dir = output_dir / 'labels' / 'train'
    val_labels_dir = output_dir / 'labels' / 'val'
    
    for d in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件
    image_extensions = get_image_extensions()
    image_files = [f for f in images_dir.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    # 找到有对应标签的图像
    valid_pairs = []
    for img_file in image_files:
        label_file = labels_dir / (img_file.stem + '.txt')
        if label_file.exists():
            valid_pairs.append((img_file, label_file))
    
    if not valid_pairs:
        print("\n错误: 没有找到有效的图像-标签对")
        print(f"  图像目录有 {len(image_files)} 个图像文件")
        print(f"  请确保标签文件与图像文件同名（仅扩展名不同）")
        return
    
    print(f"\n找到 {len(valid_pairs)} 对有效的图像-标签")
    
    # 随机打乱
    random.seed(42)
    random.shuffle(valid_pairs)
    
    # 划分
    split_idx = int(len(valid_pairs) * split_ratio)
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]
    
    print(f"\n划分结果:")
    print(f"  训练集: {len(train_pairs)} 对 ({split_ratio*100:.0f}%)")
    print(f"  验证集: {len(val_pairs)} 对 ({(1-split_ratio)*100:.0f}%)")
    
    # 复制文件
    print(f"\n复制文件中...")
    
    for img_file, label_file in train_pairs:
        shutil.copy2(str(img_file), str(train_images_dir / img_file.name))
        shutil.copy2(str(label_file), str(train_labels_dir / label_file.name))
    
    for img_file, label_file in val_pairs:
        shutil.copy2(str(img_file), str(val_images_dir / img_file.name))
        shutil.copy2(str(label_file), str(val_labels_dir / label_file.name))
    
    print("\n完成!")
    print(f"\n输出目录结构:")
    print(f"  {output_dir}/")
    print(f"  ├── images/")
    print(f"  │   ├── train/ ({len(train_pairs)} 张)")
    print(f"  │   └── val/   ({len(val_pairs)} 张)")
    print(f"  └── labels/")
    print(f"      ├── train/ ({len(train_pairs)} 个)")
    print(f"      └── val/   ({len(val_pairs)} 个)")


if __name__ == '__main__':
    split_dataset()
