#pragma once

#include "param.h"
#include <vector>
#include <opencv2/core.hpp>

// ============================================================================
// 常量定义
// ============================================================================

#define ANCHOR_PIXEL  254
#define EDGE_PIXEL    255

// ============================================================================
// 边缘检测类
// ============================================================================

/**
 * @brief 边缘检测器类
 * 
 * 基于 Edge Drawing (ED) 算法的边缘检测实现
 */
class EdgeDetector {
public:
    /**
     * @brief 构造函数
     * @param srcImage 源图像
     * @param params 边缘检测参数
     */
    EdgeDetector(const cv::Mat& srcImage, const EdgeParams& params = EdgeParams());
    
    /**
     * @brief 获取边缘段列表
     */
    std::vector<std::vector<cv::Point>> getSegments() const { return segmentPoints_; }
    
    /**
     * @brief 获取平滑后的图像
     */
    cv::Mat getSmoothImage() const { return smoothImage_; }
    
    /**
     * @brief 获取图像尺寸
     */
    int getWidth() const { return width_; }
    int getHeight() const { return height_; }

private:
    // 核心算法步骤
    void computeGradient();
    void computeAnchorPoints();
    void joinAnchorPointsUsingSortedAnchors();
    void sortAnchorsByGradValue();
    
private:
    int width_;                 // 图像宽度
    int height_;                // 图像高度
    EdgeParams params_;         // 检测参数
    
    cv::Mat srcImage_;          // 源图像
    cv::Mat smoothImage_;       // 平滑图像
    cv::Mat edgeImage_;         // 边缘图像
    cv::Mat gradImage_;         // 梯度图像
    cv::Mat dirImage_;          // 方向图像
    
    uchar* srcImg_;             // 数据指针
    uchar* smoothImg_;
    uchar* edgeImg_;
    short* gradImg_;
    uchar* dirImg_;
    
    int anchorNos_;
    int segmentNos_;
    std::vector<cv::Point> anchorPoints_;
    std::vector<std::vector<cv::Point>> segmentPoints_;
};
