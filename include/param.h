#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

/**
 * @file param.h
 * @brief 椭圆检测项目所有可调参数与阈值的统一汇总
 */

// ============================================================================
// 枚举类型
// ============================================================================

/**
 * @brief 梯度算子类型
 */
enum class GradientOperator {
    PREWITT = 101,
    SOBEL = 102,
    SCHARR = 103
};

// ============================================================================
// 边缘检测参数 (ED 算法)
// ============================================================================

struct EdgeParams {
    GradientOperator op = GradientOperator::SOBEL;  // 梯度算子
    int gradThresh = 20;       // 梯度阈值：下调提升对模糊/低对比度边缘的提取
    int anchorThresh = 8;       // 锚点阈值：下调提升平缓边缘的捕获能力
    int scanInterval = 1;      // 扫描间隔：寻找锚点时的行/列步长
    int minPathLen = 5;       // 最小路径长度：丢弃过短的边缘段
    
    // 平滑参数：设置为 0 时将关闭对应步骤的平滑
    double sigma = 0.5;        // 高斯平滑参数
    int preBlurSize = 3;       // 初步平滑核大小 (设置为 0 关闭)
    double preBlurSigma = 1.0; // 初步平滑标准差
    int smoothBlurSize = 3;    // ED 算法内部平滑核大小 (设置为 0 关闭)
    
    // 辅助段参数
    int minAuxSegmentLen = 5; // joinAnchorPoints 中提取辅助链的最小长度
};

// ============================================================================
// 圆弧提取参数
// ============================================================================

struct ArcParams {
    int minArcLength = 6;       // 最小圆弧长度：丢弃点数过少的弧段
    int closedDistThresh = 4;   // 闭合判定曼哈顿距离阈值 (单位: 像素)
    double convexRatioThresh = 0.50; // 凸性占比阈值：用于筛选单向弯曲的弧段
};

// ============================================================================
// 椭圆检测与精修参数
// ============================================================================

struct EllipseParams {
    // ---------- 评分与验证阈值 ----------
    double minEllipseScore1 = 0.45; // 椭圆评分阈值（拟合内点率）
    double inlierDist = 1.5;        // 内点判定距离阈值 (单位: 像素)
    
    // ---------- 几何约束 ----------
    double minAspectRatio = 0.45;           // 最小长短轴比
    double minMinorAxis = 1.5;                  // 最小短轴长度限制
    
    // ---------- 聚类与去重 ----------
    double clusterDist = 10.0;             // 聚类距离阈值：用于非极大值抑制 (NMS)
    
    // ---------- 亮度/极性约束 (可选) ----------
    int polarity = 0;                      // 边缘极性约束 (0: 无, 1: 内亮外暗, -1: 内暗外亮)

    // ---------- 几何精修 (LM + Tukey) ----------
    int refineMaxIter = 15;                // 最大迭代次数
    double tukeyConstant = 4.685;          // Tukey Loss 参数 (典型值 4.685)
    double lmLambda = 0.01;                // LM 阻尼项初始值
    double jacobianEps = 1e-4;             // 数值求导步长

    // ---------- 径向边缘修正 (自适应百分位) ----------
    double radialSearchRange = 5.0;        // 径向搜索范围 (像素)
    double radialSearchStep = 0.5;         // 径向采样步长 (像素)
    
    double contrastHigh = 80.0;            // 高对比度阈值
    double contrastMid = 50.0;             // 中对比度阈值
    
    double ratioInnerBrightHigh = 0.17;    // 内亮外暗 - 高对比度模式下的阈值比例
    double ratioInnerBrightMid = 0.12;     // 内亮外暗 - 中对比度模式下的阈值比例
    double ratioInnerBrightLow = 0.08;     // 内亮外暗 - 低对比度模式下的阈值比例
    
    double ratioInnerDarkHigh = 0.75;      // 内暗外亮 - 高对比度模式下的阈值比例
    double ratioInnerDarkMid = 0.82;       // 内暗外亮 - 中对比度模式下的阈值比例
    double ratioInnerDarkLow = 0.88;       // 内暗外亮 - 低对比度模式下的阈值比例
};

// ============================================================================
// 顶层聚合参数结构
// ============================================================================

struct DetectorParams {
    EdgeParams edge;
    ArcParams arc;
    EllipseParams ellipse;
    
    int threads = 8; // 全局并行线程数
    bool debug = false; // 调试模式：开启后将保存中间诊断图像
    
    /**
     * @brief 默认构造函数
     */
    DetectorParams() = default;
};
