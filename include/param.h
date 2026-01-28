/**
 * @file param.h
 * @brief 椭圆检测统一参数定义
 * 
 * 将所有可调参数集中管理，按功能划分结构体：
 * - EdgeParams: 边缘检测参数
 * - ArcParams: 圆弧提取参数  
 * - FitParams: 拟合与验证参数
 * - DetectorParams: 聚合所有参数
 */

#ifndef PARAM_H
#define PARAM_H

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
// 边缘检测参数
// ============================================================================

/**
 * @brief 边缘检测参数
 * 
 * 用于 Edge Drawing (ED) 算法的边缘提取
 */
struct EdgeParams {
    GradientOperator op = GradientOperator::SOBEL;  // 梯度算子
    int gradThresh = 10;       // 梯度阈值
    int anchorThresh = 10;      // 锚点阈值
    int scanInterval = 1;      // 扫描间隔
    int minPathLen = 3;       // 最小路径长度
    double sigma = 0.5;        // 高斯平滑参数（0: 禁用, >0: 启用并指定标准差）
};

// ============================================================================
// 圆弧提取参数
// ============================================================================

/**
 * @brief 圆弧提取参数
 * 
 * 用于从边缘段中提取平滑圆弧
 */
struct ArcParams {
    double epsilon = 0.5;      // RDP 多边形逼近阈值
    double sharpAngle = 100.0;  // 尖角阈值（度）
    int minArcLength = 3;     // 最小圆弧长度
    int threads = 8;           // 并行线程数
};

// ============================================================================
// 拟合与验证参数
// ============================================================================

/**
 * @brief 拟合与验证参数
 * 
 * 用于椭圆拟合、评分和梯度验证
 */
struct FitParams {
    // ========== 配对约束参数 ==========
    double thLengthRatio = 6.0;    // 弧长比限制
    double thDistance = 1.5;       // 距离限制系数
    
    // ========== 评分参数 ==========
    double minEdgeScore = 0.35;    // 最小边缘评分
    double minEllipseScore1 = 0.35; // 椭圆评分阈值1（内点率）
    double minEllipseScore2 = 0.35;// 椭圆评分阈值2（覆盖率）
    double inlierDist = 1.5;       // 内点距离阈值
    
    // ========== 验证参数 ==========
    double remainScore = 0.9;     // 梯度验证保留分数阈值
    int sampleNum = 48;           // 椭圆采样点数
    int gradRadius = 1;            // 梯度平均半径
    double gradAngleThresh = 35.0; // 梯度角度阈值（度）
    
    // ========== 聚类参数 ==========
    double clusterDist = 10.0;     // 聚类距离阈值
    
    // ========== 极性约束参数 ==========
    int polarity = 1;              // 边缘极性约束（0:无, 1:内亮外暗, -1:内暗外亮）
    double minAspectRatio = 0.4;   // 最小长短轴比（偏心率约束）
    int centerIntensityThresh = 1; // 圆心灰度阈值（0:自适应）
    int brightCenterThresh = 150;  // 若 polarity=1: 检查 Center > brightCenterThresh
    int darkCenterThresh = 100;    // 若 polarity=-1: 检查 Center < darkCenterThresh
};

// ============================================================================
// 二次拟合参数
// ============================================================================

/**
 * @brief 二次拟合参数
 * 
 * 用于在初步检测后基于 ROI 和原始梯度进行精细化拟合
 */
struct RefineParams {
    bool useRefine = true;         // 是否启用二次拟合
    double roiRatio = 1.6;         // ROI 截取比例（相对于长轴）
    int refineIter = 8;            // 迭代次数
    double tukeyAlpha = 4;       // Tukey Loss 常数
    int minGradient = 30;          // 最小梯度幅值阈值
    int samplePoints = 0;          // 采样点数限制 (0表示使用所有高梯度点)
};

// ============================================================================
// 统一检测参数
// ============================================================================

/**
 * @brief 椭圆检测器统一参数
 * 
 * 聚合所有子模块参数，便于统一管理和调试
 */
struct DetectorParams {
    EdgeParams edge;    // 边缘检测参数
    ArcParams arc;      // 圆弧提取参数
    FitParams fit;      // 拟合验证参数
    RefineParams refine;// 二次拟合参数
    int threads = 8;    // 并行线程数
};

#endif // PARAM_H
