/**
 * @file CommonTypes.h
 * @brief 通用类型定义（已废弃）
 * 
 * 注意：此文件已被 CalibrationTypes.h 替代
 * 保留此文件仅为向后兼容
 */

#pragma once
#include <opencv2/core.hpp>
#include <vector>

// 双目标定将结果
struct CalibrationResult {
    // 0-左相机，1-右相机
    std::array<cv::Mat, 2> camera_matrix;      // 单目相机内参矩阵
    std::array<cv::Mat, 2> dist_coeffs;        // 单目相机畸变系数
    std::array<std::vector<cv::Mat>, 2> rvecs; // 左/右相机的旋转向量（每个图像一个）
    std::array<std::vector<cv::Mat>, 2> tvecs; // 左/右相机的平移向量（每个图像一个）
    cv::Mat R;                  // 双目旋转矩阵
    cv::Mat T;                  // 双目平移矩阵
    cv::Mat E;                  // 双目本质矩阵
    cv::Mat F;                  // 双目基础矩阵
    cv::Mat rectify_matrix[2];  // 极线校正后的3*3内参矩阵
    cv::Mat P[2];               // 极线校正后的3*4投影矩阵，P2 的基线非零（单位：毫米）
    cv::Mat rectify_R[2];
    cv::Mat Q;                  // 视差图到深度图的映射矩阵
    std::vector<std::array<cv::Mat, 2>> remap;
    double baseline;            // 双目基线距离（单位：毫米）
    double l_rmse;                // 左侧相机标定精度
    double r_rmse;                // 右侧相机标定精度
    double rmse;                  // 双目标定精度
    double l_reproj_avge;        // 左相机重投影平均误差
    double r_reproj_avge;        // 右相机重投影平均误差
};
