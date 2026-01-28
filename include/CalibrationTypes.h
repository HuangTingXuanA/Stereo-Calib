/**
 * @file CalibrationTypes.h
 * @brief 双目标定系统的数据结构定义
 * 
 * 包含标定结果和图像对数据的精简结构体定义
 */

#ifndef CALIBRATION_TYPES_H
#define CALIBRATION_TYPES_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

/**
 * @brief 标定结果结构体（精简版）
 * 
 * 只保留双目标定必需的字段，移除不必要的复杂性
 */
struct CalibrationResult {
    // 相机内参和畸变
    cv::Mat camera_matrix_left;      // 左相机内参矩阵 3x3
    cv::Mat camera_matrix_right;     // 右相机内参矩阵 3x3
    cv::Mat dist_coeffs_left;        // 左相机畸变系数
    cv::Mat dist_coeffs_right;       // 右相机畸变系数
    
    // 双目外参
    cv::Mat R;                       // 旋转矩阵 3x3
    cv::Mat T;                       // 平移向量 3x1
    cv::Mat E;                       // 本质矩阵 3x3
    cv::Mat F;                       // 基础矩阵 3x3
    
    // 标定质量指标
    double rms_left;                 // 左相机RMS误差
    double rms_right;                // 右相机RMS误差
    double rms_stereo;               // 双目RMS误差
    
    // 构造函数：初始化误差值为0
    CalibrationResult() 
        : rms_left(0.0), rms_right(0.0), rms_stereo(0.0) {}
};

/**
 * @brief 图像对数据结构体
 * 
 * 存储一对双目图像的圆心坐标和世界坐标
 */
struct ImagePairData {
    std::vector<cv::Point2f> left_centers;   // 左图圆心坐标
    std::vector<cv::Point2f> right_centers;  // 右图圆心坐标
    std::vector<cv::Point3f> world_points;   // 世界坐标
    std::string left_path;                   // 左图路径
    std::string right_path;                  // 右图路径
    
    // 构造函数
    ImagePairData() {}
    
    // 判断是否为有效数据（圆心不为空）
    bool isValid() const {
        return !left_centers.empty() && !right_centers.empty();
    }
};

#endif // CALIBRATION_TYPES_H
