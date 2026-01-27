/**
 * @file BoardConfig.h
 * @brief 标定板配置定义
 * 
 * 支持从配置文件加载标定板参数和YOLO分割模型路径
 */

#ifndef BOARD_CONFIG_H
#define BOARD_CONFIG_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

#include <map>

/**
 * @brief 分割器配置
 */
struct SegmentorConfig {
    std::string model_path;   // ONNX模型路径
    int inference_size;       // 推理尺寸（长边）
    float confidence;         // 置信度阈值
    
    SegmentorConfig() : inference_size(1024), confidence(0.5f) {}
};

/**
 * @brief 标定板配置结构体
 */
struct BoardConfig {
    // 标定板基本参数
    int rows;                              // 标定板行数
    int cols;                              // 标定板列数
    double circle_spacing;                 // 圆心间距（毫米）
    
    // 特征大圆位置映射: ID -> (Row, Col)
    // 注意: Point2i.x = col, Point2i.y = row
    std::map<int, cv::Point2i> anchors;
    
    // 分割器配置
    SegmentorConfig segmentor;             // YOLO分割器配置
    
    BoardConfig() : rows(0), cols(0), circle_spacing(0.0) {}
    
    /**
     * @brief 从YAML文件加载配置
     * @param filepath YAML配置文件路径
     * @return 成功返回true，失败返回false
     */
    bool loadFromFile(const std::string& filepath);
    
    /**
     * @brief 保存配置到YAML文件
     * @param filepath YAML配置文件路径
     * @return 成功返回true，失败返回false
     */
    bool saveToFile(const std::string& filepath) const;
    
    /**
     * @brief 验证配置有效性
     * @return 配置有效返回true
     */
    bool isValid() const;
};

#endif // BOARD_CONFIG_H
