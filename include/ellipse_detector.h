#pragma once

#include "param.h"
#include <vector>

// ============================================================================
// 椭圆结构体
// ============================================================================

/**
 * @brief 椭圆结构体
 * 
 * 表示检测到的椭圆，包含亚像素级中心坐标
 */
struct Ellipse {
    cv::Point2d center;  // 椭圆中心（亚像素级）
    double a;            // 长半轴
    double b;            // 短半轴
    double phi;          // 旋转角度（弧度）
    double goodness;     // 质量评分 [0, 1]
    
    Ellipse() : center(0, 0), a(0), b(0), phi(0), goodness(0) {}
    
    Ellipse(const cv::Point2d& c, double _a, double _b, double _phi, double g = 0)
        : center(c), a(_a), b(_b), phi(_phi), goodness(g) {}
    
    /**
     * @brief 从 OpenCV RotatedRect 构造
     */
    Ellipse(const cv::RotatedRect& rect, double g = 0) {
        center = rect.center;
        a = rect.size.width / 2.0;
        b = rect.size.height / 2.0;
        phi = rect.angle * CV_PI / 180.0;
        goodness = g;
        
        // 确保 a >= b
        if (a < b) {
            std::swap(a, b);
            phi += CV_PI / 2.0;
        }
        
        // 归一化角度到 [0, π)
        while (phi < 0) phi += CV_PI;
        while (phi >= CV_PI) phi -= CV_PI;
    }
    
    /**
     * @brief 转换为 OpenCV RotatedRect
     */
    cv::RotatedRect toRotatedRect() const {
        return cv::RotatedRect(
            cv::Point2f(static_cast<float>(center.x), static_cast<float>(center.y)),
            cv::Size2f(static_cast<float>(a * 2), static_cast<float>(b * 2)),
            static_cast<float>(phi * 180.0 / CV_PI)
        );
    }
};

// ============================================================================
// 主检测函数
// ============================================================================

/**
 * @brief 检测图像中的椭圆
 * 
 * @param image  输入图像（灰度或彩色）
 * @param params 检测参数
 * @return       检测到的椭圆列表
 */
std::vector<Ellipse> detectEllipses(const cv::Mat& image,
                                     const DetectorParams& params = DetectorParams());
