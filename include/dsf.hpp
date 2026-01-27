/**
 * @file dsf.hpp
 * @brief 边缘检测与平滑圆弧提取模块
 * 
 * 从 EDSF 项目严格移植的边缘检测算法，包含：
 * - 基于梯度的边缘检测（ED算法）
 * - RDP 多边形逼近
 * - 基于拐点检测的圆弧分割
 */

#ifndef DSF_HPP
#define DSF_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include "param.h"

// ============================================================================
// 常量定义
// ============================================================================

#define ANCHOR_PIXEL  254
#define EDGE_PIXEL    255

#define LEFT  1
#define RIGHT 2
#define UP    3
#define DOWN  4

#define EDGE_VERTICAL   1
#define EDGE_HORIZONTAL 2

// ============================================================================
// 数据结构
// ============================================================================

/**
 * @brief 边缘检测结果
 */
struct EdgeResult {
    std::vector<std::vector<cv::Point>> edges;  // 边缘段列表
    cv::Mat smoothImage;                         // 平滑后的图像
    cv::Mat gradImage;                           // 梯度图像
    cv::Mat dirImage;                            // 方向图像
    int width;                                   // 图像宽度
    int height;                                  // 图像高度
};

/**
 * @brief 圆弧提取结果
 */
struct ArcResult {
    std::vector<std::vector<cv::Point>> arcs;                    // 圆弧列表
    std::vector<std::vector<std::pair<int, int>>> arcSegs;       // 每个圆弧的线段索引
    std::vector<int> polarities;                                 // 每个圆弧的极性（1:内部，-1:外部，0:未知）
};

// ============================================================================
// 链结构（用于边缘链接）
// ============================================================================

/**
 * @brief 栈节点，用于边缘追踪
 */
struct StackNode {
    int r, c;       // 起始像素坐标
    int parent;     // 父链索引（-1表示无父节点）
    int dir;        // 追踪方向
};

/**
 * @brief 链结构，用于边缘链接
 */
struct Chain {
    int dir;            // 链方向
    int len;            // 像素数量
    int parent;         // 父节点索引
    int children[2];    // 子节点索引
    cv::Point* pixels;  // 像素数组指针
};

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
     * @param srcImage 源图像（灰度或彩色）
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
     * @brief 获取梯度图像
     */
    cv::Mat getGradImage() const { return gradImage_; }
    
    /**
     * @brief 获取方向图像
     */
    cv::Mat getDirImage() const { return dirImage_; }
    
    /**
     * @brief 获取图像尺寸
     */
    int getWidth() const { return width_; }
    int getHeight() const { return height_; }

private:
    // 核心算法
    void computeGradient();
    void computeAnchorPoints();
    void joinAnchorPointsUsingSortedAnchors();
    void sortAnchorsByGradValue();
    
    // 辅助函数
    static int longestChain(Chain* chains, int root);
    static int retrieveChainNos(Chain* chains, int root, int chainNos[]);

private:
    int width_;                 // 图像宽度
    int height_;                // 图像高度
    EdgeParams params_;         // 检测参数
    
    cv::Mat srcImage_;          // 源图像
    cv::Mat smoothImage_;       // 平滑图像
    cv::Mat edgeImage_;         // 边缘图像
    cv::Mat gradImage_;         // 梯度图像
    cv::Mat dirImage_;          // 方向图像（Mat封装）
    
    uchar* srcImg_;             // 源图像数据指针
    uchar* smoothImg_;          // 平滑图像数据指针
    uchar* edgeImg_;            // 边缘图像数据指针
    short* gradImg_;            // 梯度数据指针
    uchar* dirImg_;             // 方向数据指针
    
    int anchorNos_;             // 锚点数量
    int segmentNos_;            // 边缘段数量
    std::vector<cv::Point> anchorPoints_;              // 锚点列表
    std::vector<std::vector<cv::Point>> segmentPoints_; // 边缘段列表
};

// ============================================================================
// RDP 并行处理类
// ============================================================================

/**
 * @brief RDP 多边形逼近并行处理类
 */
class ParallelRDP : public cv::ParallelLoopBody {
public:
    ParallelRDP(std::vector<cv::Point>* edgeLists,
                std::vector<std::pair<int, int>>* segLists,
                double epsilon, int num, int threads);
    
    void operator()(const cv::Range& r) const override;

private:
    static double perpendicularDistance2(const cv::Point& pt,
                                         const cv::Point& lineStart,
                                         const cv::Point& lineEnd);
    
    void rdp(const std::vector<cv::Point>& edge,
             int l, int r, double epsilon, int id) const;

private:
    std::vector<cv::Point>* edgeLists_;
    std::vector<std::pair<int, int>>* segLists_;
    double epsilon_;
    int threads_;
    int num_;
    int range_;
};

// ============================================================================
// 圆弧提取器
// ============================================================================

/**
 * @brief 圆弧提取器类
 * 
 * 从边缘段中提取平滑圆弧
 */
class ArcExtractor {
public:
    /**
     * @brief 构造函数
     * @param edges 边缘段列表
     * @param params 圆弧提取参数
     */
    ArcExtractor(const std::vector<std::vector<cv::Point>>& edges,
                 const ArcParams& params = ArcParams());
    
    /**
     * @brief 执行圆弧提取
     */
    void extract();

    /**
     * @brief 计算圆弧极性
     * @param gradImage 梯度图像
     */
    void computePolarity(const cv::Mat& gradImage);
    
    /**
     * @brief 获取圆弧列表
     */
    std::vector<std::vector<cv::Point>> getArcs() const { return arcs_; }
    
    /**
     * @brief 获取圆弧线段索引
     */
    std::vector<std::vector<std::pair<int, int>>> getArcSegs() const { return arcSegs_; }

    /**
     * @brief 获取圆弧极性
     */
    std::vector<int> getPolarities() const { return polarities_; }

private:
    void runRDP();
    void splitEdge();

private:
    ArcParams params_;
    std::vector<std::vector<cv::Point>> edges_;     // 输入边缘
    std::vector<std::vector<std::pair<int, int>>> segList_;  // RDP 分段结果
    
    std::vector<std::vector<cv::Point>> arcs_;      // 输出圆弧
    std::vector<std::vector<std::pair<int, int>>> arcSegs_;  // 圆弧线段索引
    std::vector<int> polarities_;                   // 圆弧极性
};

// ============================================================================
// 便捷函数接口
// ============================================================================

/**
 * @brief 边缘检测便捷函数
 * @param image 输入灰度图像
 * @param params 检测参数
 * @return 边缘检测结果
 */
EdgeResult detectEdges(const cv::Mat& image, const EdgeParams& params = EdgeParams());

/**
 * @brief 圆弧提取便捷函数
 * @param edgeResult 边缘检测简结果
 * @param params 提取参数
 * @return 圆弧提取结果
 */
ArcResult extractArcs(const EdgeResult& edgeResult,
                      const ArcParams& params = ArcParams());

#endif // DSF_HPP
